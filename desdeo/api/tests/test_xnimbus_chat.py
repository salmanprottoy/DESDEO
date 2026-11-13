"""End-to-end tests for the grounded XNIMBUS chat endpoint, with a fake language model."""

import json

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel import select

from desdeo.api.deepseek import DeepSeekProviderError
from desdeo.api.grounded_chat import GroundedAnswer, ModelGeneration, ModelProfile
from desdeo.api.llm_config import CloudConsentRequiredError
from desdeo.api.models import ChatMessage, InteractiveSessionDB, NIMBUSInitializationRequest
from desdeo.api.routers import xnimbus_chat

from .conftest import login, post_json

PROFILE = ModelProfile(profile_id="fake-1", label="Fake", provider="fake", model_id="fake-1", capabilities=[])


@pytest.fixture(autouse=True)
def clear_chat_rate_limit():
    """Keep endpoint tests independent while production requests share the limiter."""
    with xnimbus_chat._CHAT_RATE_LOCK:
        xnimbus_chat._CHAT_RATE_EVENTS.clear()
    yield
    with xnimbus_chat._CHAT_RATE_LOCK:
        xnimbus_chat._CHAT_RATE_EVENTS.clear()


class FakeClient:
    """Answers from the first objective fact in the prompt, so its answer is always grounded."""

    def __init__(self, *, fabricate: bool = False, fail: bool = False):
        """Optionally return a wrong number, or fail like a provider would."""
        self.prompts: list[str] = []
        self.fabricate = fabricate
        self.fail = fail

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Answer with the first objective value of solution 0 found in the prompt."""
        self.prompts.append(prompt)
        if self.fail:
            raise DeepSeekProviderError("boom")
        facts = json.loads(prompt.split("Evidence JSON:\n", 1)[1])
        fact = next(f for f in facts if f["kind"] == "objective_value" and f["source"]["solution_index"] == 0)
        value = 123.456 if self.fabricate else fact["value"]["value"]
        text = f"{fact['value']['symbol']} is {value:.3f}."
        return ModelGeneration(
            answer=GroundedAnswer(
                answer=text,
                claims=[
                    {
                        "text": text,
                        "claim_type": "objective_value",
                        "evidence_ids": [fact["evidence_id"]],
                    }
                ],
                uncertainty="none",
                follow_up_questions=[],
            ),
            latency_ms=1,
        )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    """Replace the configured provider with a fake one and return it."""
    fake = FakeClient()
    monkeypatch.setattr(xnimbus_chat, "get_llm_configuration", lambda *args, **kwargs: (fake, PROFILE))
    return fake


def _start(client: TestClient) -> tuple[str, int]:
    access_token = login(client)
    response = post_json(
        client, "/method/xnimbus/initialize", NIMBUSInitializationRequest(problem_id=1).model_dump(), access_token
    )
    assert response.status_code == status.HTTP_200_OK
    return access_token, response.json()["state_id"]


def _ask(client: TestClient, token: str, state_id: int, message: str, **extra):
    body = {"problem_id": 1, "state_id": state_id, "solution_index": 0, "message": message, **extra}
    return post_json(client, "/method/xnimbus/chat", body, token)


def test_chat_returns_verified_answer_with_cited_evidence(
    client: TestClient,
    fake_llm: FakeClient,
    session_and_user,
):
    """A grounded answer is verified, and the response carries exactly the cited evidence."""
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is the value of the first objective?")

    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["status"] == "verified"
    assert data["verified"] is True
    assert [fact["evidence_id"] for fact in data["evidence"]] == data["claims"][0]["evidence_ids"]
    messages = session_and_user["session"].exec(select(ChatMessage)).all()
    audit = messages[-1].model_answer_json["audit"]
    assert audit["provider"] == "fake"
    assert audit["prompt_version"] == "xnimbus-grounded-v4"
    assert audit["fact_sheet_hash"]
    assert audit["cited_evidence_ids"] == data["claims"][0]["evidence_ids"]


def test_chat_keeps_conversation_history(client: TestClient, fake_llm: FakeClient):
    """Follow-up questions reuse the conversation and the model sees the earlier turns."""
    token, state_id = _start(client)
    first = _ask(client, token, state_id, "What is the value of the first objective?").json()

    second = _ask(client, token, state_id, "And what is f_2?", conversation_id=first["conversation_id"]).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert "Conversation so far" not in fake_llm.prompts[0]
    assert "Decision maker: What is the value of the first objective?" in fake_llm.prompts[1]
    assert first["answer"] in fake_llm.prompts[1]


def test_chat_rejects_fabricated_numbers(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """A model answer with numbers that are not in the evidence is replaced by an abstention."""
    monkeypatch.setattr(
        xnimbus_chat,
        "get_llm_configuration",
        lambda *args, **kwargs: (FakeClient(fabricate=True), PROFILE),
    )
    token, state_id = _start(client)

    data = _ask(client, token, state_id, "What is the value of the first objective?").json()

    assert data["status"] == "fallback"
    assert data["verified"] is False
    assert data["claims"]
    assert data["evidence"]


def test_chat_blocks_actions_without_calling_the_model(client: TestClient, fake_llm: FakeClient):
    """Requests to change the session are blocked before reaching the model."""
    token, state_id = _start(client)

    data = _ask(client, token, state_id, "Solve the problem again").json()

    assert data["status"] == "blocked"
    assert fake_llm.prompts == []


def test_chat_validates_solution_index_before_calling_the_model(client: TestClient, fake_llm: FakeClient):
    """An unknown solution index is rejected without spending a model call."""
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?", solution_index=99)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert fake_llm.prompts == []


def test_chat_records_provider_errors_as_computed_fallback(
    client: TestClient, session_and_user, monkeypatch: pytest.MonkeyPatch
):
    """Provider failures are stored and return deterministic facts instead of a raw error."""
    monkeypatch.setattr(
        xnimbus_chat,
        "get_llm_configuration",
        lambda *args, **kwargs: (FakeClient(fail=True), PROFILE),
    )
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "fallback"
    statuses = [m.verification_status for m in session_and_user["session"].exec(select(ChatMessage)).all()]
    assert statuses == ["received", "fallback"]


def test_chat_rejects_unknown_model_profile(client: TestClient):
    """An arbitrary client-provided model profile is rejected."""
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?", model_profile_id="unknown:model")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "unknown:model" in response.json()["detail"]


def test_chat_rejects_unknown_conversation(client: TestClient, fake_llm: FakeClient):
    """A conversation id that does not belong to the user is not found."""
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?", conversation_id=999)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert fake_llm.prompts == []


def test_chat_requires_explicit_cloud_consent(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Selecting a configured cloud model is not sufficient without per-turn consent."""

    def reject_without_consent(*args, **kwargs):
        raise CloudConsentRequiredError("explicit consent is required")

    monkeypatch.setattr(xnimbus_chat, "get_llm_configuration", reject_without_consent)
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?", model_profile_id="openai:test")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "consent" in response.json()["detail"]


def test_chat_rate_limits_authenticated_users(client: TestClient, fake_llm: FakeClient):
    """Repeated valid requests cannot consume provider capacity without a bound."""
    token, state_id = _start(client)
    for index in range(xnimbus_chat.CHAT_RATE_LIMIT):
        response = _ask(client, token, state_id, f"What is f_1? request {index}")
        assert response.status_code == status.HTTP_200_OK

    limited = _ask(client, token, state_id, "What is f_1 once more?")

    assert limited.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert int(limited.headers["Retry-After"]) >= 1


def test_chat_history_query_is_bounded(client: TestClient, fake_llm: FakeClient):
    """Long conversations send only the configured recent message window to the model."""
    token, state_id = _start(client)
    conversation_id = None
    for index in range(5):
        response = _ask(
            client,
            token,
            state_id,
            f"What is f_1? turn {index}",
            conversation_id=conversation_id,
        )
        assert response.status_code == status.HTTP_200_OK
        conversation_id = response.json()["conversation_id"]

    assert "turn 0" not in fake_llm.prompts[-1]
    assert "turn 3" in fake_llm.prompts[-1]


def test_model_catalog_returns_only_server_configured_profiles(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """The browser receives safe profile metadata, never credentials or arbitrary URLs."""
    monkeypatch.setattr(xnimbus_chat, "available_model_profiles", lambda: [PROFILE])
    monkeypatch.setattr(xnimbus_chat, "default_profile_id", lambda profiles: PROFILE.profile_id)
    token = login(client)

    response = client.get(
        "/method/xnimbus/chat/models",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["models"][0]["profile_id"] == PROFILE.profile_id
    assert "api_key" not in response.text


def test_chat_rejects_state_from_another_active_session(
    client: TestClient,
    session_and_user,
    fake_llm: FakeClient,
):
    """An owned problem does not authorize a state outside the user's active session."""
    token, state_id = _start(client)
    db = session_and_user["session"]
    user = session_and_user["user"]
    other_session = InteractiveSessionDB(user_id=user.id)
    db.add(other_session)
    db.commit()
    db.refresh(other_session)
    user.active_session_id = other_session.id
    db.add(user)
    db.commit()

    response = _ask(client, token, state_id, "What is f_1?")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert fake_llm.prompts == []
