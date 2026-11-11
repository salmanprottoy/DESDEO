"""End-to-end tests for the grounded XNIMBUS chat endpoint, with a fake language model."""

import json

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel import select

from desdeo.api.deepseek import DeepSeekProviderError
from desdeo.api.grounded_chat import GroundedAnswer, ModelProfile
from desdeo.api.models import ChatMessage, NIMBUSInitializationRequest
from desdeo.api.routers import xnimbus_chat

from .conftest import login, post_json

PROFILE = ModelProfile(profile_id="fake-1", label="Fake", provider="fake", model_id="fake-1", capabilities=[])


class FakeClient:
    """Answers from the first objective fact in the prompt, so its answer is always grounded."""

    def __init__(self, *, fabricate: bool = False, fail: bool = False):
        self.prompts: list[str] = []
        self.fabricate = fabricate
        self.fail = fail

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        self.prompts.append(prompt)
        if self.fail:
            raise DeepSeekProviderError("boom")
        facts = json.loads(prompt.split("Evidence JSON:\n", 1)[1])
        fact = next(f for f in facts if f["kind"] == "objective_value" and f["source"]["solution_index"] == 0)
        value = 123.456 if self.fabricate else fact["value"]["value"]
        text = f"{fact['value']['symbol']} is {value:.3f}."
        return GroundedAnswer(
            answer=text,
            claims=[{"text": text, "evidence_ids": [fact["evidence_id"]]}],
            uncertainty="none",
            follow_up_questions=[],
        )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    """Replace the configured provider with a fake one and return it."""
    fake = FakeClient()
    monkeypatch.setattr(xnimbus_chat, "get_llm_configuration", lambda: (fake, PROFILE))
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


def test_chat_returns_verified_answer_with_cited_evidence(client: TestClient, fake_llm: FakeClient):
    """A grounded answer is verified, and the response carries exactly the cited evidence."""
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is the value of the first objective?")

    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()
    assert data["status"] == "verified"
    assert data["verified"] is True
    assert [fact["evidence_id"] for fact in data["evidence"]] == data["claims"][0]["evidence_ids"]


def test_chat_keeps_conversation_history(client: TestClient, fake_llm: FakeClient):
    """Follow-up questions reuse the conversation and the model sees the earlier turns."""
    token, state_id = _start(client)
    first = _ask(client, token, state_id, "What is the value of the first objective?").json()

    second = _ask(client, token, state_id, "And why is that?", conversation_id=first["conversation_id"]).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert "Conversation so far" not in fake_llm.prompts[0]
    assert "Decision maker: What is the value of the first objective?" in fake_llm.prompts[1]
    assert first["answer"] in fake_llm.prompts[1]


def test_chat_rejects_fabricated_numbers(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """A model answer with numbers that are not in the evidence is replaced by an abstention."""
    monkeypatch.setattr(xnimbus_chat, "get_llm_configuration", lambda: (FakeClient(fabricate=True), PROFILE))
    token, state_id = _start(client)

    data = _ask(client, token, state_id, "What is the value of the first objective?").json()

    assert data["status"] == "rejected"
    assert data["verified"] is False
    assert data["claims"] == []
    assert data["evidence"] == []


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


def test_chat_records_provider_errors(client: TestClient, session_and_user, monkeypatch: pytest.MonkeyPatch):
    """Provider failures return 502 and are stored, so they can be counted in evaluation."""
    monkeypatch.setattr(xnimbus_chat, "get_llm_configuration", lambda: (FakeClient(fail=True), PROFILE))
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?")

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    statuses = [m.verification_status for m in session_and_user["session"].exec(select(ChatMessage)).all()]
    assert statuses == ["received", "provider_error"]


def test_chat_rejects_unknown_provider(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """A misspelled provider name is a configuration error, not a silent fallback."""
    monkeypatch.setenv("XNIMBUS_LLM_PROVIDER", "gemni")
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "gemni" in response.json()["detail"]


def test_chat_rejects_unknown_conversation(client: TestClient, fake_llm: FakeClient):
    """A conversation id that does not belong to the user is not found."""
    token, state_id = _start(client)

    response = _ask(client, token, state_id, "What is f_1?", conversation_id=999)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert fake_llm.prompts == []
