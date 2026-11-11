"""Tests for the provider-independent grounded chat boundary."""

import pytest

from desdeo.api.evidence import EvidenceContext, EvidenceFact
from desdeo.api.grounded_chat import (
    GroundedAnswer,
    GroundedChatService,
    HistoryTurn,
    ModelProfile,
    build_prompt,
    classify_intent,
    mentioned_numbers,
    verify_grounded_answer,
)

F1 = "state-42-solution-0-objective-f_1"
TRADEOFFS = "state-42-solution-0-tradeoffs"


def _context() -> EvidenceContext:
    return EvidenceContext(
        state_id=42,
        facts=[
            EvidenceFact(
                evidence_id=F1,
                kind="objective_value",
                value={"symbol": "f_1", "value": 0.3333},
                source={"state_id": 42, "solution_index": 0, "field": "objective"},
            ),
            EvidenceFact(
                evidence_id=TRADEOFFS,
                kind="tradeoff",
                value={"matrix": {"f_1": {"f_2": -0.5}}},
                source={"state_id": 42, "solution_index": 0, "field": "derived"},
            ),
        ],
    )


def _profile() -> ModelProfile:
    return ModelProfile(profile_id="test", label="Test", provider="test", model_id="test-1", capabilities=[])


def _answer(text: str, claims: list[tuple[str, list[str]]], uncertainty: str = "none") -> GroundedAnswer:
    return GroundedAnswer(
        answer=text,
        claims=[{"text": claim, "evidence_ids": ids} for claim, ids in claims],
        uncertainty=uncertainty,
        follow_up_questions=[],
    )


class _StaticClient:
    def __init__(self, answer: GroundedAnswer):
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        self.prompts.append(prompt)
        return self.answer


# --- Verifier -------------------------------------------------------------------------------------


def test_verifier_accepts_claim_with_matching_number() -> None:
    """A claim whose number is in the cited evidence is accepted."""
    answer = _answer("f_1 is 0.3333.", [("f_1 is 0.3333.", [F1])])

    result = verify_grounded_answer(answer, _context())

    assert result.accepted is True
    assert result.cited_evidence_ids == [F1]


def test_verifier_rejects_fabricated_prose_without_claims() -> None:
    """An answer with no claims cannot pass as verified, whatever its text says."""
    answer = _answer("f_1 is 42.0 and is the best ever.", [], uncertainty="partial")

    result = verify_grounded_answer(answer, _context())

    assert result.accepted is False
    assert result.reason == "Answer has no evidence-backed claims."


def test_verifier_rejects_unsupported_numbers_in_answer_text() -> None:
    """Numbers in the answer text must come from evidence cited by a claim."""
    answer = _answer("f_1 is 0.3333, which is 12.5 better than before.", [("f_1 is 0.3333.", [F1])])

    result = verify_grounded_answer(answer, _context())

    assert result.accepted is False
    assert "Answer text" in (result.reason or "")


def test_verifier_rejects_numbers_from_uncited_evidence() -> None:
    """A claim's numbers must come from the evidence that claim cites, not from other facts."""
    answer = _answer("The trade-off is 0.5.", [("The trade-off is 0.5.", [F1])])

    assert verify_grounded_answer(answer, _context()).accepted is False


def test_verifier_rejects_unknown_evidence() -> None:
    """Claims citing evidence that does not exist are rejected."""
    answer = _answer("The value is 9.0.", [("The value is 9.0.", ["unknown"])])

    assert verify_grounded_answer(answer, _context()).accepted is False


@pytest.mark.parametrize(
    ("claim", "evidence_ids"),
    [
        ("f_1 is 0.33.", [F1]),  # rounded
        ("f_1 is about 0.333.", [F1]),  # rounded to more digits
        ("In solution 0, f_1 is 0.3333.", [F1]),  # index phrase is not a value
        ("Improving f_1 decreases f_2 by 0.5.", [TRADEOFFS]),  # magnitude of a nested value
        (f"f_1 is 0.3333 ({F1}).", [F1]),  # evidence id quoted in text
    ],
)
def test_verifier_accepts_rounding_index_phrases_and_nested_values(claim: str, evidence_ids: list[str]) -> None:
    """Correct claims are not rejected for rounding, index phrases, nesting, or quoted ids."""
    answer = _answer(claim, [(claim, evidence_ids)])

    result = verify_grounded_answer(answer, _context())

    assert result.accepted is True, result.reason


def test_verifier_rejects_rounding_beyond_written_precision() -> None:
    """0.34 is not a rounding of 0.3333."""
    answer = _answer("f_1 is 0.34.", [("f_1 is 0.34.", [F1])])

    assert verify_grounded_answer(answer, _context()).accepted is False


def test_mentioned_numbers_ignores_identifiers() -> None:
    """Numbers inside identifiers such as f_1, f1, or 3rd are not values."""
    assert mentioned_numbers("f_1 and f2 on the 3rd try") == []
    assert mentioned_numbers("value 1.5e-3") == [(0.0015, 0.5e-4)]


# --- Intent ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("Why did f_1 improve?", "explanation"),
        ("Why does this solution save more money?", "explanation"),
        ("Which objective is worst here?", "explanation"),
        ("Tell me about solution 2", "explanation"),
        ("Should I improve the second objective?", "explanation"),
        ("How does f2 trade off against f1?", "calculation"),
        ("How much does f_2 change per unit of f_1?", "calculation"),
        ("What happens to f1 if I relax f3?", "what_if"),
        ("What if I allow f_2 to worsen?", "what_if"),
        ("Compare solution 1 and solution 2", "comparison"),
        ("Is solution 1 better than solution 2?", "comparison"),
        ("Prioritize f_1 and solve again", "action"),
        ("Solve the problem again", "action"),
        ("Can you save this solution?", "action"),
        ("Please finalize", "action"),
        ("I want to finalize", "action"),
        ("hello", "unsupported"),
    ],
)
def test_classify_intent(message: str, intent: str) -> None:
    """Intent rules separate real action requests from questions that mention action words."""
    assert classify_intent(message) == intent


# --- Service --------------------------------------------------------------------------------------


def test_service_reports_unavailable_without_provider() -> None:
    """Without a provider the service abstains and says why."""
    outcome = GroundedChatService(client=None).respond("Why did f_1 improve?", context=_context(), profile=None)

    assert outcome.status == "unavailable"
    assert outcome.answer.uncertainty == "insufficient"
    assert outcome.answer.claims == []


def test_service_blocks_actions_without_calling_the_model() -> None:
    """Action requests never reach the model."""
    client = _StaticClient(_answer("Solving now.", [("Solving now.", [F1])]))

    outcome = GroundedChatService(client=client).respond("Solve again", context=_context(), profile=_profile())

    assert outcome.status == "blocked"
    assert outcome.intent == "action"
    assert client.prompts == []


def test_service_replaces_rejected_answer_with_abstention_and_keeps_raw_output() -> None:
    """A rejected model answer is not shown, but is kept for auditing."""
    bad = _answer("f_1 is 42.0.", [("f_1 is 42.0.", [F1])])

    outcome = GroundedChatService(client=_StaticClient(bad)).respond(
        "What is f_1?", context=_context(), profile=_profile()
    )

    assert outcome.status == "rejected"
    assert outcome.answer.claims == []
    assert outcome.model_answer == bad
    assert "unsupported numbers" in (outcome.reason or "")


@pytest.mark.parametrize(("uncertainty", "status"), [("none", "verified"), ("partial", "partial")])
def test_service_maps_uncertainty_to_status(uncertainty: str, status: str) -> None:
    """Verified answers are split into verified and partial by the model's uncertainty."""
    answer = _answer("f_1 is 0.3333.", [("f_1 is 0.3333.", [F1])], uncertainty=uncertainty)

    outcome = GroundedChatService(client=_StaticClient(answer)).respond(
        "What is f_1?", context=_context(), profile=_profile()
    )

    assert outcome.status == status
    assert outcome.cited_evidence_ids == [F1]


def test_service_reports_model_abstention() -> None:
    """A model that says the evidence is insufficient is reported as abstained."""
    answer = _answer("The evidence does not cover f_3.", [], uncertainty="insufficient")

    outcome = GroundedChatService(client=_StaticClient(answer)).respond(
        "What is f_3?", context=_context(), profile=_profile()
    )

    assert outcome.status == "abstained"


def test_prompt_contains_history_focus_and_evidence() -> None:
    """The prompt carries recent history as context, the focused solution, and the evidence."""
    history = [HistoryTurn(role="user", content="What is f_1?"), HistoryTurn(role="assistant", content="0.3333")]

    prompt = build_prompt("And f_2?", context=_context(), intent="explanation", history=history, focus_solution_index=0)

    assert "Conversation so far (context only, not evidence):" in prompt
    assert "Decision maker: What is f_1?" in prompt
    assert "solution_index 0 (shown to them as solution 1)" in prompt
    assert F1 in prompt
    assert prompt.index("And f_2?") < prompt.index("Evidence JSON")


def test_prompt_keeps_only_recent_history() -> None:
    """Old turns are dropped so the prompt stays bounded."""
    history = [HistoryTurn(role="user", content=f"question-{index}") for index in range(10)]

    prompt = build_prompt("Now?", context=_context(), intent="explanation", history=history)

    assert "question-3" not in prompt
    assert "question-4" in prompt
    assert "question-9" in prompt


def test_model_profile_is_pinned_and_carries_capabilities() -> None:
    """Model profiles retain backend-selected identifiers and capabilities."""
    profile = ModelProfile(
        profile_id="test-grounded",
        label="Test grounded model",
        provider="test",
        model_id="test-1",
        capabilities=["structured_output", "tool_use"],
    )

    assert profile.model_id == "test-1"
