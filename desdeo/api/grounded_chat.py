"""Provider-independent policy and verification for grounded XNIMBUS chat."""

import json
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from desdeo.api.evidence import EvidenceContext, EvidenceFact

Intent = Literal["explanation", "comparison", "calculation", "what_if", "optimization", "unsupported"]


class EvidenceReference(BaseModel):
    """Reference to one evidence fact."""

    evidence_id: str


class AnswerClaim(BaseModel):
    """A claim that must be supported by evidence."""

    text: str
    evidence_ids: list[str] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    """Structured answer returned by a grounded model."""

    answer: str
    claims: list[AnswerClaim]
    uncertainty: Literal["none", "partial", "insufficient"]
    follow_up_questions: list[str]


class ModelProfile(BaseModel):
    """Backend-owned, pinned model metadata."""

    profile_id: str
    label: str
    provider: str
    model_id: str
    capabilities: list[str]
    enabled: bool = True


class VerificationResult(BaseModel):
    """Result of deterministic answer verification."""

    accepted: bool
    reason: str | None = None


class ChatRequest(BaseModel):
    """Authenticated XNIMBUS chat request."""

    problem_id: int
    state_id: int
    solution_index: int = Field(default=0, ge=0)
    message: str = Field(min_length=1, max_length=4000)
    model_profile_id: str | None = None
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    """Verified chat response returned by the API."""

    answer: str
    claims: list[AnswerClaim]
    evidence: list[EvidenceFact]
    uncertainty: Literal["none", "partial", "insufficient"]
    follow_up_questions: list[str]
    verified: bool
    conversation_id: int


class ActionProposal(BaseModel):
    """A pending state-changing action awaiting explicit confirmation."""

    token: str
    action: Literal["solve", "save", "finalize"]
    state_id: int
    payload: dict[str, object]
    expires_at: datetime


class GroundedLLMClient(Protocol):
    """Interface implemented by a provider adapter."""

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        """Generate a structured answer using the selected pinned profile."""


class ActionGate:
    """Create and consume short-lived, single-use action confirmations."""

    def __init__(self, *, ttl_seconds: int = 300):
        """Create an in-memory gate for one process."""
        self._ttl = timedelta(seconds=ttl_seconds)
        self._proposals: dict[str, ActionProposal] = {}
        self._used_tokens: set[str] = set()

    def propose(
        self,
        *,
        action: Literal["solve", "save", "finalize"],
        state_id: int,
        payload: dict[str, object],
    ) -> ActionProposal:
        """Create a confirmation proposal without executing the action."""
        proposal = ActionProposal(
            token=secrets.token_urlsafe(32),
            action=action,
            state_id=state_id,
            payload=payload,
            expires_at=datetime.now(UTC) + self._ttl,
        )
        self._proposals[proposal.token] = proposal
        return proposal

    def confirm(self, token: str, *, state_id: int) -> dict[str, object]:
        """Consume a valid confirmation token and return its action payload."""
        if token in self._used_tokens:
            raise ValueError("Action confirmation has already been used.")
        proposal = self._proposals.get(token)
        if proposal is None:
            raise ValueError("Action confirmation was not found.")
        if proposal.state_id != state_id:
            raise ValueError("Action confirmation is stale.")
        if proposal.expires_at <= datetime.now(UTC):
            raise ValueError("Action confirmation has expired.")
        self._used_tokens.add(token)
        return proposal.payload


class GroundedChatService:
    """Apply policy and verification around an optional provider adapter."""

    def __init__(self, client: GroundedLLMClient | None):
        """Create a service without owning provider configuration."""
        self._client = client

    def respond(
        self,
        message: str,
        *,
        context: EvidenceContext,
        profile: ModelProfile | None,
    ) -> GroundedAnswer:
        """Return a verified answer or an explicit abstention."""
        intent = classify_intent(message)
        if intent == "optimization":
            return GroundedAnswer(
                answer="I cannot perform state-changing actions without explicit confirmation.",
                claims=[],
                uncertainty="insufficient",
                follow_up_questions=["Would you like an explanation of the current solution instead?"],
            )
        if self._client is None or profile is None:
            return GroundedAnswer(
                answer="I could not produce a verified answer from the available XNIMBUS evidence.",
                claims=[],
                uncertainty="insufficient",
                follow_up_questions=[],
            )
        prompt = (
            f"Question:\n{message}\n\n"
            "Evidence JSON:\n"
            f"{json.dumps([fact.model_dump() for fact in context.facts], sort_keys=True)}"
        )
        answer = self._client.generate(prompt=prompt, profile=profile)
        verification = verify_grounded_answer(answer, context, intent=intent)
        if not verification.accepted:
            return GroundedAnswer(
                answer="I could not produce a verified answer from the available XNIMBUS evidence.",
                claims=[],
                uncertainty="insufficient",
                follow_up_questions=[],
            )
        return answer


def classify_intent(message: str) -> Intent:
    """Classify a message into the supported policy categories."""
    normalized = message.casefold()
    if any(term in normalized for term in ("solve", "prioritize", "optimize", "finalize", "save")):
        return "optimization"
    if any(term in normalized for term in ("what if", "what would happen", "suppose")):
        return "what_if"
    if any(term in normalized for term in ("compare", "difference", "versus", " vs ")):
        return "comparison"
    if any(term in normalized for term in ("trade-off", "tradeoff", "calculate", "how much")):
        return "calculation"
    if any(term in normalized for term in ("why", "explain", "what is", "value", "objective")):
        return "explanation"
    return "unsupported"


def _numbers(text: str) -> set[float]:
    return {float(value) for value in re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", text)}


def _fact_numbers(facts: list[EvidenceFact]) -> set[float]:
    values: set[float] = set()
    for fact in facts:
        for value in fact.value.values():
            if isinstance(value, int | float) and not isinstance(value, bool):
                values.add(float(value))
    return values


def verify_grounded_answer(
    answer: GroundedAnswer,
    context: EvidenceContext,
    *,
    intent: Intent = "explanation",
) -> VerificationResult:
    """Reject unsupported, unreferenced, numerically inconsistent answers."""
    facts_by_id = {fact.evidence_id: fact for fact in context.facts}
    if intent == "optimization":
        return VerificationResult(accepted=False, reason="State-changing actions require confirmation.")
    for claim in answer.claims:
        referenced = [facts_by_id.get(evidence_id) for evidence_id in claim.evidence_ids]
        if any(fact is None for fact in referenced):
            return VerificationResult(accepted=False, reason="Claim references unknown evidence.")
        supporting_facts = [fact for fact in referenced if fact is not None]
        if not _numbers(claim.text).issubset(_fact_numbers(supporting_facts)):
            return VerificationResult(accepted=False, reason="Claim contains unsupported numeric values.")
    return VerificationResult(accepted=True)
