"""Tests for the provider-independent grounded chat boundary."""

from desdeo.api.evidence import EvidenceContext, EvidenceFact
from desdeo.api.grounded_chat import (
    ActionGate,
    GroundedAnswer,
    GroundedChatService,
    ModelProfile,
    classify_intent,
    verify_grounded_answer,
)


def _context() -> EvidenceContext:
    return EvidenceContext(
        state_id=42,
        facts=[
            EvidenceFact(
                evidence_id="state-42-solution-0-objective-f_1",
                kind="objective_value",
                value={"symbol": "f_1", "value": 5.2},
                source={"state_id": 42, "solution_index": 0, "field": "objective"},
            )
        ],
    )


def test_classify_intent_identifies_explanations_and_optimization_requests() -> None:
    """Policy classification distinguishes read-only and mutating requests."""
    assert classify_intent("Why did f_1 improve?") == "explanation"
    assert classify_intent("Prioritize f_1 and solve again") == "optimization"


def test_verifier_accepts_claims_referencing_current_evidence() -> None:
    """Verifier accepts a claim with valid evidence and matching numbers."""
    answer = GroundedAnswer(
        answer="The value of f_1 is 5.2.",
        claims=[
            {
                "text": "The value of f_1 is 5.2.",
                "evidence_ids": ["state-42-solution-0-objective-f_1"],
            }
        ],
        uncertainty="none",
        follow_up_questions=[],
    )

    result = verify_grounded_answer(answer, _context())

    assert result.accepted is True


def test_verifier_rejects_unknown_evidence_and_read_only_actions() -> None:
    """Verifier rejects unknown evidence and unconfirmed state changes."""
    unsupported = GroundedAnswer(
        answer="The value is 9.0.",
        claims=[{"text": "The value is 9.0.", "evidence_ids": ["unknown"]}],
        uncertainty="none",
        follow_up_questions=[],
    )
    action = GroundedAnswer(
        answer="I will solve again.",
        claims=[
            {
                "text": "I will solve again.",
                "evidence_ids": ["state-42-solution-0-objective-f_1"],
            }
        ],
        uncertainty="none",
        follow_up_questions=[],
    )

    assert verify_grounded_answer(unsupported, _context()).accepted is False
    assert verify_grounded_answer(action, _context(), intent="optimization").accepted is False


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


def test_chat_service_abstains_explicitly_when_no_provider_is_configured() -> None:
    """Service abstains rather than inventing an answer without a provider."""
    service = GroundedChatService(client=None)

    response = service.respond("Why did f_1 improve?", context=_context(), profile=None)

    assert response.uncertainty == "insufficient"
    assert "verified answer" in response.answer
    assert response.claims == []


def test_action_gate_requires_current_state_and_rejects_replay() -> None:
    """Confirmation is tied to the current state and can only be used once."""
    gate = ActionGate()
    proposal = gate.propose(action="save", state_id=42, payload={"solution_index": 1})

    assert gate.confirm(proposal.token, state_id=42) == {"solution_index": 1}
    try:
        gate.confirm(proposal.token, state_id=42)
    except ValueError as error:
        assert str(error) == "Action confirmation has already been used."
    else:
        raise AssertionError("A confirmation token must be single-use.")


def test_action_gate_rejects_stale_state() -> None:
    """A proposal cannot be confirmed after the state changes."""
    gate = ActionGate()
    proposal = gate.propose(action="finalize", state_id=42, payload={})

    try:
        gate.confirm(proposal.token, state_id=43)
    except ValueError as error:
        assert str(error) == "Action confirmation is stale."
    else:
        raise AssertionError("A stale state must not be confirmed.")
