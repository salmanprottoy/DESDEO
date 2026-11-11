"""Provider-independent policy and verification for grounded XNIMBUS chat.

The language model is only an explanation layer. Every answer it returns is
checked against the deterministic evidence built from the XNIMBUS state before
it can be shown to the decision maker. The chat never changes the optimization
session: requests to solve, save, or finalize are blocked and the decision
maker is pointed to the regular XNIMBUS controls instead.
"""

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from desdeo.api.evidence import EvidenceContext, EvidenceFact

Intent = Literal["explanation", "comparison", "calculation", "what_if", "action", "unsupported"]
"""Policy category of a question. ``action`` means the DM asked the chat to change the session."""

ChatStatus = Literal["verified", "partial", "abstained", "rejected", "blocked", "unavailable"]
"""Outcome of one chat turn.

- ``verified``: every claim and number is backed by cited evidence, and the model reported no uncertainty.
- ``partial``: verified, but the model reported that the evidence only partly answers the question.
- ``abstained``: the model said the evidence does not answer the question.
- ``rejected``: the model's answer failed verification and was replaced by an abstention.
- ``blocked``: the question asked the chat to change the session.
- ``unavailable``: no language model provider is configured.
"""

ANSWERED_STATUSES: frozenset[str] = frozenset({"verified", "partial"})

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1000

ABSTENTION_TEXT = "I could not produce a verified answer from the available XNIMBUS evidence."
BLOCKED_TEXT = (
    "I can only explain the current XNIMBUS results; I cannot solve, save, or finalize for you. "
    "Use the XNIMBUS controls to do that."
)
UNAVAILABLE_TEXT = "The XNIMBUS assistant is not configured on this server."

SYSTEM_INSTRUCTION = (
    "You explain results of the XNIMBUS interactive multiobjective optimization method to a decision maker. "
    "Answer only from the supplied DESDEO evidence JSON. Return JSON with exactly these fields: "
    "answer, claims, uncertainty, follow_up_questions. "
    "Every factual statement in 'answer' must also appear as a claim, and each claim must list the "
    "evidence_ids that support it. Copy numbers from the evidence; you may round them, but do not "
    "compute new numbers. Use uncertainty='partial' when the evidence only partly answers the question "
    "and uncertainty='insufficient' when it does not answer it. "
    "The conversation history is context only and is never evidence."
)


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
    cited_evidence_ids: list[str] = Field(default_factory=list)


class HistoryTurn(BaseModel):
    """One earlier message of the conversation, given to the model as context."""

    role: Literal["user", "assistant"]
    content: str


class ChatOutcome(BaseModel):
    """What the service decided for one chat turn."""

    answer: GroundedAnswer
    """The answer to show. Replaced by an abstention when the model output was not accepted."""
    status: ChatStatus
    intent: Intent
    reason: str | None = None
    """Why the model output was rejected, if it was."""
    cited_evidence_ids: list[str] = Field(default_factory=list)
    model_answer: GroundedAnswer | None = None
    """The raw model output, kept for auditing and evaluation."""


class ChatRequest(BaseModel):
    """Authenticated XNIMBUS chat request."""

    problem_id: int
    state_id: int
    solution_index: int = Field(default=0, ge=0)
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    """Chat response returned by the API."""

    answer: str
    claims: list[AnswerClaim]
    evidence: list[EvidenceFact]
    """Exactly the evidence cited by the claims."""
    uncertainty: Literal["none", "partial", "insufficient"]
    follow_up_questions: list[str]
    status: ChatStatus
    intent: Intent
    verified: bool
    """True only when ``status == "verified"``."""
    conversation_id: int


class GroundedLLMClient(Protocol):
    """Interface implemented by a provider adapter."""

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        """Generate a structured answer using the selected pinned profile."""


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
        history: Sequence[HistoryTurn] = (),
        focus_solution_index: int | None = None,
    ) -> ChatOutcome:
        """Return a verified answer or an explicit abstention.

        Provider errors are not caught here; the caller decides how to report them.
        """
        intent = classify_intent(message)
        if intent == "action":
            return ChatOutcome(
                answer=_abstention(BLOCKED_TEXT, ["Would you like an explanation of the current solution instead?"]),
                status="blocked",
                intent=intent,
            )
        if self._client is None or profile is None:
            return ChatOutcome(answer=_abstention(UNAVAILABLE_TEXT), status="unavailable", intent=intent)

        prompt = build_prompt(
            message,
            context=context,
            intent=intent,
            history=history,
            focus_solution_index=focus_solution_index,
        )
        model_answer = self._client.generate(prompt=prompt, profile=profile)
        verification = verify_grounded_answer(model_answer, context)
        if not verification.accepted:
            return ChatOutcome(
                answer=_abstention(ABSTENTION_TEXT),
                status="rejected",
                intent=intent,
                reason=verification.reason,
                model_answer=model_answer,
            )
        status: ChatStatus = {"none": "verified", "partial": "partial", "insufficient": "abstained"}[
            model_answer.uncertainty
        ]
        return ChatOutcome(
            answer=model_answer,
            status=status,
            intent=intent,
            cited_evidence_ids=verification.cited_evidence_ids,
            model_answer=model_answer,
        )


def _abstention(text: str, follow_up_questions: list[str] | None = None) -> GroundedAnswer:
    return GroundedAnswer(
        answer=text,
        claims=[],
        uncertainty="insufficient",
        follow_up_questions=follow_up_questions or [],
    )


_INTENT_GUIDANCE: dict[str, str] = {
    "explanation": "Explain using the objective values, preferences, and solver status in the evidence.",
    "comparison": "Compare solutions only through objective values present in the evidence.",
    "calculation": (
        "Use trade-off and Lagrange multiplier evidence when it exists; do not compute numbers that are not "
        "in the evidence."
    ),
    "what_if": (
        "A what-if question can only be answered as a local estimate from trade-off evidence. Say that it is a "
        "local estimate. If there is no trade-off evidence, answer with uncertainty='insufficient'."
    ),
    "unsupported": "If the evidence does not answer the question, use uncertainty='insufficient'.",
}


def build_prompt(
    message: str,
    *,
    context: EvidenceContext,
    intent: Intent,
    history: Sequence[HistoryTurn] = (),
    focus_solution_index: int | None = None,
) -> str:
    """Build the user prompt: guidance, recent history, question, and evidence."""
    parts = [f"Question type: {intent}. {_INTENT_GUIDANCE.get(intent, '')}"]
    if focus_solution_index is not None:
        parts.append(
            f"The decision maker is looking at solution {focus_solution_index}. "
            "Assume questions are about it unless they name another solution."
        )
    recent = list(history)[-MAX_HISTORY_TURNS:]
    if recent:
        lines = [
            f"{'Decision maker' if turn.role == 'user' else 'Assistant'}: {turn.content[:MAX_HISTORY_CHARS]}"
            for turn in recent
        ]
        parts.append("Conversation so far (context only, not evidence):\n" + "\n".join(lines))
    parts.append(f"Question:\n{message}")
    parts.append(
        "Evidence JSON:\n" + json.dumps([fact.model_dump() for fact in context.facts], sort_keys=True)
    )
    return "\n\n".join(parts)


_ACTION_VERBS = r"(?:re-?solve|solve|optimi[sz]e|finali[sz]e|save|iterate|prioriti[sz]e|run)"
_INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (
        "action",
        re.compile(
            rf"^\s*(?:please\s+)?{_ACTION_VERBS}\b"
            rf"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?{_ACTION_VERBS}\b"
            r"|\b(?:re-?solve|solve|optimi[sz]e|run)\s+(?:it\s+|this\s+|the\s+problem\s+)?again\b"
            r"|\band\s+(?:re-?solve|solve|optimi[sz]e|save|finali[sz]e)\b"
            r"|\b(?:save|finali[sz]e)\s+(?:this|that|it|the\s+(?:current\s+)?(?:solution|result))\b"
            r"|\bi\s+(?:want|would\s+like)\s+to\s+(?:save|finali[sz]e|stop)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "what_if",
        re.compile(
            r"\bwhat\s+if\b|\bwhat\s+(?:would|will)\s+happen\b|\bwhat\s+happens\b|\bsuppose\b"
            r"|\bif\s+i\s+(?:relax|improve|worsen|change|increase|decrease|allow|lower|raise)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "comparison",
        re.compile(
            r"\bcompar\w*|\bdifferen\w*|\bdiffer\b|\bversus\b|\bvs\.?(?=\s)|\b(?:better|worse)\s+than\b",
            re.IGNORECASE,
        ),
    ),
    (
        "calculation",
        re.compile(r"\btrade[\s-]?offs?\b|\bcalculat\w*|\bhow\s+(?:much|many)\b|\bmultipliers?\b", re.IGNORECASE),
    ),
    (
        "explanation",
        re.compile(
            r"\bwhy\b|\bexplain\w*|\bwhat\b|\bwhich\b|\bhow\b|\btell\s+me\b|\bdescribe\b|\bshould\b"
            r"|\bvalues?\b|\bobjectives?\b|\bsolutions?\b|\bpreferences?\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_intent(message: str) -> Intent:
    """Classify a message into the supported policy categories.

    Only explicit requests to change the session count as ``action``; a question that merely contains
    a word like "save" (for example "why does this save money?") is not an action.
    """
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message):
            return intent
    return "unsupported"


# Phrases where a number is a label, not a value: "solution 2", "objective #3", "solutions 1 and 2".
_INDEX_PHRASE = re.compile(
    r"\b(?:solutions?|objectives?|options?|candidates?|alternatives?|states?|iterations?|steps?|rounds?)"
    r"\s*(?:#|no\.?|number)?\s*\d+(?![.\d])(?:\s*(?:,|and|or|&)\s*\d+(?![.\d]))*",
    re.IGNORECASE,
)
# A number that is not part of an identifier such as f_1, f1, x2, or 3rd.
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z_\d])")


def mentioned_numbers(text: str, *, evidence_ids: Iterable[str] = ()) -> list[tuple[float, float]]:
    """Return ``(value, tolerance)`` for each number stated in ``text``.

    Evidence identifiers and index phrases are removed first. The tolerance allows rounding to the
    precision the number was written with: ``0.33`` matches any value within 0.005.
    """
    for evidence_id in sorted(evidence_ids, key=len, reverse=True):
        text = text.replace(evidence_id, " ")
    text = _INDEX_PHRASE.sub(" ", text)
    numbers: list[tuple[float, float]] = []
    for match in _NUMBER.finditer(text):
        token = match.group()
        mantissa = re.split(r"[eE]", token)[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        exponent = int(re.split(r"[eE]", token)[1]) if re.search(r"[eE]", token) else 0
        numbers.append((float(token), 0.5 * 10.0 ** (exponent - decimals)))
    return numbers


def fact_numbers(facts: Iterable[EvidenceFact]) -> list[float]:
    """Collect every numeric value in the facts, including nested ones such as trade-off matrices."""
    values: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int | float):
            values.append(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list | tuple):
            for item in value:
                collect(item)

    for fact in facts:
        collect(fact.value)
    return values


def unsupported_numbers(
    text: str,
    facts: Sequence[EvidenceFact],
    *,
    evidence_ids: Iterable[str] = (),
) -> list[float]:
    """Return the numbers in ``text`` that do not match any number in ``facts``.

    A number matches if it equals a fact value, or its magnitude, after rounding to the precision it
    was written with. Magnitudes are accepted because answers often say "decreases by 0.5" for -0.5.
    """
    known = fact_numbers(facts)
    missing: list[float] = []
    for value, tolerance in mentioned_numbers(text, evidence_ids=evidence_ids):
        slack = tolerance + 1e-9 * max(1.0, abs(value))
        if not any(abs(value - fact) <= slack or abs(abs(value) - abs(fact)) <= slack for fact in known):
            missing.append(value)
    return missing


def verify_grounded_answer(answer: GroundedAnswer, context: EvidenceContext) -> VerificationResult:
    """Accept an answer only if all of its claims and numbers are backed by cited evidence.

    Rules:
    1. An answer that is not an abstention must contain at least one claim.
    2. Every claim must cite only evidence that exists in the context.
    3. Every number in a claim must match a number in the evidence that claim cites.
    4. Every number in the answer text must match a number in the evidence cited by some claim.
    """
    facts_by_id = {fact.evidence_id: fact for fact in context.facts}
    if answer.uncertainty != "insufficient" and not answer.claims:
        return VerificationResult(accepted=False, reason="Answer has no evidence-backed claims.")

    cited_ids: list[str] = []
    for claim in answer.claims:
        unknown = [evidence_id for evidence_id in claim.evidence_ids if evidence_id not in facts_by_id]
        if unknown:
            return VerificationResult(accepted=False, reason=f"Claim references unknown evidence: {unknown}.")
        cited = [facts_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        if missing := unsupported_numbers(claim.text, cited, evidence_ids=facts_by_id):
            return VerificationResult(accepted=False, reason=f"Claim contains unsupported numbers: {missing}.")
        cited_ids.extend(evidence_id for evidence_id in claim.evidence_ids if evidence_id not in cited_ids)

    cited_facts = [facts_by_id[evidence_id] for evidence_id in cited_ids]
    if missing := unsupported_numbers(answer.answer, cited_facts, evidence_ids=facts_by_id):
        return VerificationResult(accepted=False, reason=f"Answer text contains unsupported numbers: {missing}.")
    return VerificationResult(accepted=True, cited_evidence_ids=cited_ids)
