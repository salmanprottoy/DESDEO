"""Provider-independent policy and verification for grounded XNIMBUS chat.

The model may select typed evidence-backed claims, but model prose is never
shown directly. After verification, the server renders every displayed claim
from deterministic evidence. This keeps a fluent but unsupported model claim
from crossing the trusted boundary.
"""

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from desdeo.api.evidence import EvidenceContext, EvidenceFact

Intent = Literal["explanation", "comparison", "calculation", "what_if", "history", "action", "unsupported"]
ClaimType = Literal[
    "objective_value",
    "variable_value",
    "problem_context",
    "preference",
    "solver_status",
    "lagrange_multiplier",
    "tradeoff",
    "active_objective",
    "explanation_availability",
    "comparison",
    "experiment_summary",
    "iteration",
    "preference_history",
    "experiment_solution",
    "history_comparison",
]
ChatStatus = Literal["verified", "partial", "abstained", "rejected", "blocked", "fallback", "unavailable"]
DeploymentType = Literal["local", "cloud"]

ANSWERED_STATUSES: frozenset[str] = frozenset({"verified", "partial", "fallback"})
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1000
MAX_GENERATION_ATTEMPTS = 2
PROMPT_VERSION = "xnimbus-grounded-v10"
TRADEOFFS_UNAVAILABLE_REASON = "XNIMBUS reports no reliable trade-offs for this solution."

ABSTENTION_TEXT = "I could not produce a verified answer from the available XNIMBUS evidence."
BLOCKED_TEXT = (
    "I can only explain the current XNIMBUS results; I cannot solve, save, or finalize for you. "
    "Use the XNIMBUS controls to do that."
)
UNAVAILABLE_TEXT = "The XNIMBUS assistant has no configured local model and no cloud model was selected."

SYSTEM_INSTRUCTION = (
    "You select evidence-backed claims about XNIMBUS results. Return JSON with exactly these fields: "
    "answer, claims, uncertainty, follow_up_questions. Each claim has text, claim_type, and evidence_ids. "
    "Allowed claim_type values are objective_value, variable_value, problem_context, preference, "
    "solver_status, lagrange_multiplier, tradeoff, active_objective, explanation_availability, comparison, "
    "experiment_summary, iteration, preference_history, experiment_solution, and history_comparison. "
    "The claim_type must exactly match every cited fact kind: preference facts require preference, "
    "problem_context facts require problem_context, and comparison facts require comparison. "
    "Ideal and nadir values, objective names, and maximize/minimize directions are problem_context facts; "
    "objective_value is only the computed value of an objective in a solution. "
    "Answer only from the supplied evidence JSON. "
    "Do not calculate, infer causes, recommend preferences, claim optimality, or interpret a raw trade-off "
    "matrix entry as an improvement cost. Use uncertainty='none' when your claims answer the question, "
    "'partial' when they answer only part of it, and 'insufficient' only with an empty claims list when the "
    "evidence cannot answer. "
    "Conversation history is context only and is never evidence. The server will discard your prose and "
    "render accepted claims deterministically from the cited evidence."
)


class ProviderResponseError(RuntimeError):
    """Raised when a model provider cannot return a schema-valid answer."""


class AnswerClaim(BaseModel):
    """A typed claim whose displayed text is rendered by the server."""

    model_config = ConfigDict(extra="forbid")

    text: str
    claim_type: ClaimType = Field(description="Must exactly match the kind of every cited evidence fact.")
    evidence_ids: list[str] = Field(
        min_length=1,
        description="Stable IDs copied verbatim from the supplied evidence JSON.",
    )


class GroundedAnswer(BaseModel):
    """Structured answer selected by a model or rendered by the server."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    claims: list[AnswerClaim]
    uncertainty: Literal["none", "partial", "insufficient"] = Field(
        description=(
            "Use none when the claims answer the question, partial when they answer part of it, and insufficient "
            "only with an empty claims list when the evidence cannot answer."
        ),
    )
    follow_up_questions: list[str]


class ModelProfile(BaseModel):
    """Backend-owned model metadata safe to expose to the frontend."""

    profile_id: str
    label: str
    provider: str
    model_id: str
    deployment: DeploymentType = "local"
    capabilities: list[str]
    decoding_parameters: dict[str, int | float | str | bool] = Field(default_factory=dict)
    enabled: bool = True


class ModelGeneration(BaseModel):
    """One provider generation plus reproducibility metadata."""

    answer: GroundedAnswer
    latency_ms: int
    token_usage: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of deterministic answer verification."""

    accepted: bool
    reason: str | None = None
    cited_evidence_ids: list[str] = Field(default_factory=list)


class HistoryTurn(BaseModel):
    """One earlier message supplied as non-evidentiary context."""

    role: Literal["user", "assistant"]
    content: str


class ChatOutcome(BaseModel):
    """Trusted result of one chat turn."""

    answer: GroundedAnswer
    status: ChatStatus
    intent: Intent
    reason: str | None = None
    cited_evidence_ids: list[str] = Field(default_factory=list)
    model_answer: GroundedAnswer | None = None
    model_answers: list[GroundedAnswer] = Field(default_factory=list)
    verifier_reasons: list[str] = Field(default_factory=list)
    attempts: int = 0
    latency_ms: int | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)
    provider_error: bool = False


class ChatRequest(BaseModel):
    """Authenticated XNIMBUS chat request."""

    problem_id: int
    state_id: int
    solution_index: int = Field(default=0, ge=0)
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None
    model_profile_id: str | None = None
    use_model: bool = True
    cloud_consent: bool = False


class ChatResponse(BaseModel):
    """Trusted chat response returned by the API."""

    answer: str
    claims: list[AnswerClaim]
    evidence: list[EvidenceFact]
    uncertainty: Literal["none", "partial", "insufficient"]
    follow_up_questions: list[str]
    status: ChatStatus
    intent: Intent
    verified: bool
    conversation_id: int
    state_id: int
    solution_index: int
    model_profile: ModelProfile | None = None


class ModelCatalogResponse(BaseModel):
    """Configured model profiles available to the current server."""

    models: list[ModelProfile]
    default_profile_id: str | None = None


class GroundedLLMClient(Protocol):
    """Interface implemented by provider adapters."""

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Generate one schema-valid candidate answer."""


class GroundedChatService:
    """Apply policy, bounded repair, and deterministic rendering around a provider."""

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
        """Return a server-rendered answer, deterministic fallback, or abstention."""
        intent = classify_intent(message)
        if intent == "action":
            return ChatOutcome(
                answer=_abstention(BLOCKED_TEXT, ["Would you like the current objective values instead?"]),
                status="blocked",
                intent=intent,
            )
        if policy_reason := _policy_abstention_reason(message, context):
            return ChatOutcome(
                answer=_abstention(ABSTENTION_TEXT, _safe_followups("unsupported")),
                status="abstained",
                intent="unsupported",
                reason=policy_reason,
            )
        if _tradeoffs_unavailable_for(message, context, intent=intent, focus_solution_index=focus_solution_index):
            # XNIMBUS itself reports no reliable trade-offs here, so true but unrelated facts (raw multipliers,
            # problem context) must not be presented as an answer. Decided before, and independent of, the model.
            availability = _focused_facts(context, focus_solution_index, {"explanation_availability"})
            return _render_fallback(intent, "explanation_availability", availability[:1], TRADEOFFS_UNAVAILABLE_REASON)
        if self._client is None or profile is None:
            return _fallback_outcome(
                context,
                message=message,
                intent=intent,
                focus_solution_index=focus_solution_index,
                reason=UNAVAILABLE_TEXT,
                unavailable_when_empty=True,
            )

        base_prompt = build_prompt(
            message,
            context=context,
            intent=intent,
            history=history,
            focus_solution_index=focus_solution_index,
        )
        model_context = context.model_copy(
            update={
                "facts": _select_prompt_facts(
                    context,
                    intent=intent,
                    focus_solution_index=focus_solution_index,
                    message=message,
                )
            }
        )
        prompt = base_prompt
        model_answers: list[GroundedAnswer] = []
        verifier_reasons: list[str] = []
        total_latency = 0
        token_usage: dict[str, Any] = {}
        provider_error: str | None = None

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                generation = self._client.generate(prompt=prompt, profile=profile)
            except ProviderResponseError as error:
                provider_error = str(error)
                if attempt < MAX_GENERATION_ATTEMPTS:
                    continue
                break

            total_latency += generation.latency_ms
            token_usage = generation.token_usage
            model_answer = generation.answer
            model_answers.append(model_answer)
            if model_answer.uncertainty == "insufficient" and not model_answer.claims:
                fallback = _fallback_outcome(
                    context,
                    message=message,
                    intent=intent,
                    focus_solution_index=focus_solution_index,
                    reason="Model reported insufficient evidence.",
                )
                if fallback.status == "rejected":
                    fallback.status = "abstained"
                fallback.model_answer = model_answer
                fallback.model_answers = model_answers
                fallback.verifier_reasons = verifier_reasons
                fallback.attempts = attempt
                fallback.latency_ms = total_latency
                fallback.token_usage = token_usage
                return fallback
            # Small models often label grounded claims "insufficient". Such claims are verified like any other
            # answer, but can at most produce a partial answer; the raw label stays in the audit log.
            candidate = (
                model_answer.model_copy(update={"uncertainty": "partial"})
                if model_answer.uncertainty == "insufficient"
                else model_answer
            )
            verification = verify_grounded_answer(candidate, model_context, intent=intent, question=message)
            if verification.accepted:
                trusted_answer = render_verified_answer(candidate, model_context, intent=intent)
                computed_abstention = _cites_unavailable_explanation(
                    verification, model_context, answer=candidate, intent=intent
                )
                if computed_abstention:
                    trusted_answer = trusted_answer.model_copy(update={"uncertainty": "insufficient"})
                    status: ChatStatus = "abstained"
                else:
                    status = "verified" if candidate.uncertainty == "none" else "partial"
                return ChatOutcome(
                    answer=trusted_answer,
                    status=status,
                    intent=intent,
                    cited_evidence_ids=verification.cited_evidence_ids,
                    model_answer=model_answer,
                    model_answers=model_answers,
                    verifier_reasons=verifier_reasons,
                    attempts=attempt,
                    latency_ms=total_latency,
                    token_usage=token_usage,
                )

            reason = verification.reason or "Answer failed verification."
            verifier_reasons.append(reason)
            prompt = (
                "The previous candidate was rejected by the deterministic verifier: "
                + reason
                + " Return a corrected JSON answer.\n\n"
                + base_prompt
            )

        reason = provider_error or (verifier_reasons[-1] if verifier_reasons else "Provider unavailable.")
        fallback = _fallback_outcome(
            context,
            message=message,
            intent=intent,
            focus_solution_index=focus_solution_index,
            reason=reason,
        )
        fallback.model_answer = model_answers[-1] if model_answers else None
        fallback.model_answers = model_answers
        fallback.verifier_reasons = verifier_reasons
        fallback.attempts = MAX_GENERATION_ATTEMPTS
        fallback.latency_ms = total_latency or None
        fallback.token_usage = token_usage
        fallback.provider_error = provider_error is not None
        return fallback


def _tradeoffs_unavailable_for(
    message: str,
    context: EvidenceContext,
    *,
    intent: Intent,
    focus_solution_index: int | None,
) -> bool:
    """True for trade-off and what-if questions about a solution whose trade-offs XNIMBUS marks unavailable.

    A request for the raw Lagrange multipliers is a factual lookup and stays answerable.
    """
    if intent not in {"calculation", "what_if"}:
        return False
    if intent == "calculation" and re.search(r"\b(?:lagrange|kkt|multipliers?)\b", message, re.IGNORECASE):
        return False
    availability = _focused_facts(context, focus_solution_index, {"explanation_availability"})
    return bool(availability) and all(fact.value.get("available") is False for fact in availability)


def _cites_unavailable_explanation(
    verification: VerificationResult,
    context: EvidenceContext,
    *,
    answer: GroundedAnswer,
    intent: Intent,
) -> bool:
    """Let deterministic availability override model confidence for trade-off answers.

    A trade-off or what-if answer that cites ``available=false`` is an abstention whatever the model says.
    Other answers (for example "which objectives are active?") may cite availability as context; they are
    only abstentions when availability and problem context are all they cite.
    """
    facts_by_id = {fact.evidence_id: fact for fact in context.facts}
    cites_unavailable = any(
        facts_by_id[evidence_id].kind == "explanation_availability"
        and facts_by_id[evidence_id].value.get("available") is False
        for evidence_id in verification.cited_evidence_ids
    )
    if not cites_unavailable:
        return False
    if intent in {"calculation", "what_if"}:
        return True
    return all(claim.claim_type in {"explanation_availability", "problem_context"} for claim in answer.claims)


def _abstention(text: str, follow_up_questions: list[str] | None = None) -> GroundedAnswer:
    return GroundedAnswer(
        answer=text,
        claims=[],
        uncertainty="insufficient",
        follow_up_questions=follow_up_questions or [],
    )


def _fallback_outcome(
    context: EvidenceContext,
    *,
    message: str,
    intent: Intent,
    focus_solution_index: int | None,
    reason: str,
    unavailable_when_empty: bool = False,
) -> ChatOutcome:
    requested_symbol = _requested_objective_symbol(message, context)
    fallback: ChatOutcome | None = None
    if intent == "comparison":
        fallback = _comparison_fallback(context, requested_symbol, reason)
    elif intent in {"calculation", "what_if"}:
        fallback = _calculation_fallback(
            context,
            intent=intent,
            message=message,
            focus_solution_index=focus_solution_index,
            requested_symbol=requested_symbol,
            reason=reason,
        )
    elif intent == "explanation":
        fallback = _explanation_fallback(context, message, focus_solution_index, requested_symbol, reason)
    elif intent == "history":
        fallback = _history_fallback(context, message, reason)
    if fallback is not None:
        return fallback

    return ChatOutcome(
        answer=_abstention(UNAVAILABLE_TEXT if unavailable_when_empty else ABSTENTION_TEXT),
        status="unavailable" if unavailable_when_empty else "rejected",
        intent=intent,
        reason=reason,
    )


def _comparison_fallback(
    context: EvidenceContext,
    requested_symbol: str | None,
    reason: str,
) -> ChatOutcome | None:
    facts = [
        fact
        for fact in context.facts
        if fact.kind == "comparison" and (requested_symbol is None or fact.value.get("symbol") == requested_symbol)
    ]
    return _render_fallback("comparison", "comparison", facts, reason) if facts else None


def _calculation_fallback(
    context: EvidenceContext,
    *,
    intent: Intent,
    message: str,
    focus_solution_index: int | None,
    requested_symbol: str | None,
    reason: str,
) -> ChatOutcome | None:
    if re.search(r"\bmultipliers?\b", message, re.IGNORECASE):
        multipliers = _focused_facts(context, focus_solution_index, {"lagrange_multiplier"})
        if multipliers:
            return _render_fallback(intent, "lagrange_multiplier", multipliers, reason)
    facts = _focused_facts(context, focus_solution_index, {"tradeoff"})
    if requested_symbol is not None:
        facts = [
            fact
            for fact in facts
            if requested_symbol in {fact.value.get("row_symbol"), fact.value.get("column_symbol")}
        ]
    if facts:
        return _render_fallback(intent, "tradeoff", facts, reason)
    availability = _focused_facts(context, focus_solution_index, {"explanation_availability"})
    return _render_fallback(intent, "explanation_availability", availability[:1], reason) if availability else None


def _explanation_fallback(
    context: EvidenceContext,
    message: str,
    focus_solution_index: int | None,
    requested_symbol: str | None,
    reason: str,
) -> ChatOutcome | None:
    global_rules: list[tuple[str, ClaimType, str]] = [
        (r"\b(?:ideal|nadir|maximi[sz]e|minimi[sz]e|direction)\b", "problem_context", "problem_context"),
        (r"\b(?:preference|aspiration|reference point)\b", "preference", "preference"),
    ]
    for pattern, claim_type, fact_kind in global_rules:
        if re.search(pattern, message, re.IGNORECASE):
            facts = [fact for fact in context.facts if fact.kind == fact_kind]
            return _render_fallback("explanation", claim_type, facts[:1], reason) if facts else None

    focused_rules: list[tuple[str, ClaimType, str]] = [
        (r"\bactive\b", "active_objective", "active_objective"),
        (r"\bsolver\b|\bsuccess", "solver_status", "solver_status"),
    ]
    for pattern, claim_type, fact_kind in focused_rules:
        if re.search(pattern, message, re.IGNORECASE):
            facts = _focused_facts(context, focus_solution_index, {fact_kind})
            return _render_fallback("explanation", claim_type, facts, reason) if facts else None

    objective_question = requested_symbol is not None or re.search(
        r"\bobjective\s+values?\b|\bvalue\s+of\s+(?:the\s+)?(?:first|second|third|fourth)\s+objective\b",
        message,
        re.IGNORECASE,
    )
    if not objective_question:
        return None
    facts = _focused_facts(context, focus_solution_index, {"objective_value"})
    if requested_symbol is not None:
        facts = [fact for fact in facts if fact.value.get("symbol") == requested_symbol]
    return _render_fallback("explanation", "objective_value", facts, reason) if facts else None


def _history_fallback(
    context: EvidenceContext,
    message: str,
    reason: str,
) -> ChatOutcome | None:
    if re.search(r"\bpreference|aspiration|reference point", message, re.IGNORECASE):
        facts = [fact for fact in context.facts if fact.kind == "preference_history"]
        return _render_fallback("history", "preference_history", facts, reason) if facts else None
    if (
        re.search(r"\b(?:previous|last|earlier)\s+(?:iteration|round|state)\b", message, re.IGNORECASE)
        and re.search(r"\bsolution\b", message, re.IGNORECASE)
        and not re.search(r"\bcompar|\bdiffer", message, re.IGNORECASE)
    ):
        facts = _previous_solution_facts(context, message)
        if facts:
            return _render_fallback("history", "experiment_solution", facts, reason)
    if re.search(r"\bcompar|\bdiffer|\bprevious|\blast|\bearlier", message, re.IGNORECASE):
        facts = [fact for fact in context.facts if fact.kind == "history_comparison"]
        return _render_fallback("history", "history_comparison", facts, reason) if facts else None
    summary = [fact for fact in context.facts if fact.kind == "experiment_summary"]
    if summary:
        return _render_fallback("history", "experiment_summary", summary[:1], reason)
    iterations = [fact for fact in context.facts if fact.kind == "iteration"]
    return _render_fallback("history", "iteration", iterations, reason) if iterations else None


def _previous_solution_facts(context: EvidenceContext, message: str) -> list[EvidenceFact]:
    """Resolve the requested solution from the latest earlier state on the selected lineage."""
    summary = next((fact for fact in context.facts if fact.kind == "experiment_summary"), None)
    selected_state_id = summary.value.get("selected_state_id") if summary is not None else None
    lineage_states = {
        fact.value.get("state_id"): fact.value.get("sequence")
        for fact in context.facts
        if fact.kind == "iteration"
        and fact.value.get("on_selected_lineage") is True
        and fact.value.get("state_id") != selected_state_id
    }
    if not lineage_states:
        return []
    latest_sequence = max(sequence for sequence in lineage_states.values() if isinstance(sequence, int))
    requested = re.search(r"\bsolution\s+(\d+)\b", message, re.IGNORECASE)
    requested_index = int(requested.group(1)) - 1 if requested else 0
    return [
        fact
        for fact in context.facts
        if fact.kind == "experiment_solution"
        and fact.value.get("sequence") == latest_sequence
        and fact.value.get("solution_index") == requested_index
    ]


def _focused_facts(
    context: EvidenceContext,
    focus_solution_index: int | None,
    kinds: set[str],
) -> list[EvidenceFact]:
    """Return selected-solution facts of the requested kinds."""
    return [
        fact
        for fact in context.facts
        if fact.kind in kinds
        and (focus_solution_index is None or fact.source.get("solution_index") == focus_solution_index)
    ]


def _requested_objective_symbol(message: str, context: EvidenceContext) -> str | None:
    """Resolve an objective symbol from either its symbol or authoritative problem name."""
    symbol_match = re.search(r"\bf_?(\d+)\b", message, re.IGNORECASE)
    if symbol_match:
        return f"f_{symbol_match.group(1)}"
    folded = message.casefold()
    for fact in context.facts:
        if fact.kind != "problem_context":
            continue
        objectives = fact.value.get("objectives")
        if not isinstance(objectives, list):
            continue
        for objective in objectives:
            if not isinstance(objective, dict):
                continue
            name = objective.get("name")
            symbol = objective.get("symbol")
            if isinstance(name, str) and isinstance(symbol, str) and name.casefold() in folded:
                return symbol
    return None


def _render_fallback(
    intent: Intent,
    claim_type: ClaimType,
    facts: list[EvidenceFact],
    reason: str,
) -> ChatOutcome:
    """Render a deterministic fallback from preselected evidence facts."""
    claim = AnswerClaim(
        text="",
        claim_type=claim_type,
        evidence_ids=[fact.evidence_id for fact in facts],
    )
    rendered_claim = _render_claim(claim, facts)
    is_computed_abstention = claim_type == "explanation_availability" and all(
        fact.value.get("available") is False for fact in facts
    )
    answer = GroundedAnswer(
        answer=(
            "Computed abstention: " + rendered_claim
            if is_computed_abstention
            else "Computed fallback (model output withheld): " + rendered_claim
        ),
        claims=[claim.model_copy(update={"text": rendered_claim})],
        uncertainty="insufficient" if is_computed_abstention else "partial",
        follow_up_questions=_safe_followups(intent),
    )
    return ChatOutcome(
        answer=answer,
        status="abstained" if is_computed_abstention else "fallback",
        intent=intent,
        reason=reason,
        cited_evidence_ids=[fact.evidence_id for fact in facts],
    )


_INTENT_GUIDANCE: dict[str, str] = {
    "explanation": (
        "Match fact kind exactly: preference -> preference, problem_context -> problem_context, "
        "objective_value -> objective_value, and solver_status -> solver_status."
    ),
    "comparison": "Use claim_type='comparison' and cite a server-computed comparison fact.",
    "calculation": "Use only server-computed trade-off or multiplier facts.",
    "what_if": "Use only reliable local trade-off facts, or abstain.",
    "history": (
        "Use only experiment_summary, iteration, preference_history, experiment_solution, or "
        "history_comparison facts. These are stored DM inputs and solver results; never propose new solutions."
    ),
    "unsupported": "Use uncertainty='insufficient' when the ledger cannot answer.",
}


def build_prompt(
    message: str,
    *,
    context: EvidenceContext,
    intent: Intent,
    history: Sequence[HistoryTurn] = (),
    focus_solution_index: int | None = None,
) -> str:
    """Build the versioned prompt from recent context and deterministic evidence."""
    parts = [
        f"Prompt version: {PROMPT_VERSION}.",
        f"Question type: {intent}. {_INTENT_GUIDANCE.get(intent, '')}",
        "Solution indices in evidence start at 0; the decision maker sees them starting at 1.",
    ]
    if focus_solution_index is not None:
        parts.append(
            f"The selected solution_index is {focus_solution_index} (displayed as solution {focus_solution_index + 1})."
        )
    recent = list(history)[-MAX_HISTORY_TURNS:]
    if recent:
        lines = [
            f"{'Decision maker' if turn.role == 'user' else 'Assistant'}: {turn.content[:MAX_HISTORY_CHARS]}"
            for turn in recent
        ]
        parts.append("Conversation so far (context only, not evidence):\n" + "\n".join(lines))
    parts.append(f"Question:\n{message}")
    prompt_facts = _select_prompt_facts(
        context,
        intent=intent,
        focus_solution_index=focus_solution_index,
        message=message,
    )
    parts.append("Evidence JSON:\n" + json.dumps([fact.model_dump() for fact in prompt_facts], sort_keys=True))
    return "\n\n".join(parts)


def _select_prompt_facts(
    context: EvidenceContext,
    *,
    intent: Intent,
    focus_solution_index: int | None,
    message: str,
) -> list[EvidenceFact]:
    """Minimize the model-visible ledger to facts needed for the classified intent."""
    global_kinds = {"problem_context"}
    if intent == "explanation":
        global_kinds.add("preference")

    if intent == "history":
        experiment_kinds = {"experiment_summary", "iteration"}
        if re.search(r"\bpreference|aspiration|reference point", message, re.IGNORECASE):
            experiment_kinds.add("preference_history")
        elif re.search(r"\bcompar|\bdiffer|\bprevious|\blast|\bearlier", message, re.IGNORECASE):
            experiment_kinds.update({"history_comparison", "experiment_solution"})
        else:
            experiment_kinds.update({"preference_history", "experiment_solution", "history_comparison"})
        return [fact for fact in context.facts if fact.kind in experiment_kinds]
    if intent == "comparison":
        solution_kinds = {"comparison"}
        include_all_solutions = True
    elif intent in {"calculation", "what_if"}:
        solution_kinds = {
            "lagrange_multiplier",
            "tradeoff",
            "active_objective",
            "explanation_availability",
        }
        include_all_solutions = False
    else:
        solution_kinds = {
            "objective_value",
            "variable_value",
            "solver_status",
            "active_objective",
            "explanation_availability",
        }
        include_all_solutions = False

    selected: list[EvidenceFact] = []
    for fact in context.facts:
        if fact.kind in global_kinds:
            selected.append(fact)
            continue
        if fact.kind not in solution_kinds:
            continue
        index = fact.source.get("solution_index")
        if include_all_solutions or focus_solution_index is None or index == focus_solution_index:
            selected.append(fact)
    return selected


_ACTION_VERBS = r"(?:re-?solve|solve|optimi[sz]e|finali[sz]e|save|iterate|prioriti[sz]e|run|recompute)"
_INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (
        "action",
        re.compile(
            # A leading verb is a command unless it is an idiom such as "run me through" or "save me time".
            rf"^\s*(?:please\s+)?{_ACTION_VERBS}\b(?!\s+(?:me|us)\b|\s+through\b)"
            rf"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?{_ACTION_VERBS}\b"
            r"|\b(?:re-?solve|solve|optimi[sz]e|run|recompute)\s+(?:it\s+|this\s+|the\s+problem\s+)?again\b"
            r"|\band\s+(?:re-?solve|solve|optimi[sz]e|save|finali[sz]e)\b"
            r"|\b(?:save|finali[sz]e)\s+(?:this|that|it|the\s+(?:current\s+)?(?:solution|result))\b"
            r"|\bi\s+(?:want|would\s+like)\s+to\s+(?:save|finali[sz]e|stop)\b"
            r"|\blet(?:'s| us)\s+(?:do|run|solve|iterate|recompute)\b"
            r"|\bmake\s+\w+\s+better\s+and\s+(?:recompute|solve|run)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "history",
        re.compile(
            r"\b(?:previous|last|earlier|latest)\s+(?:iteration|round|state|solution)\b"
            r"|\bacross\s+iterations?\b|\bexperiment\s+history\b|\bpreference\s+history\b"
            r"|\bover\s+time\b|\bhow\s+(?:have|did)\s+.*\bchange[sd]?\b"
            r"|\bhow\s+many\s+(?:experiment\s+)?(?:steps?|iterations?|states?|stored\s+solutions?)\b",
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
            r"\bcompar\w*|\bdifferen\w*|\bdiffer\b|\bversus\b|\bvs\.?(?=\s)|\b(?:better|worse)\s+than\b"
            r"|\bwhich\s+solution\s+(?:has|is)\s+(?:the\s+)?(?:higher|lower|better|worse)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "calculation",
        re.compile(
            r"\btrade[\s-]?offs?\b|\bcalculat\w*|\bhow\s+(?:much|many)\b|\bmultipliers?\b"
            # Implicit trade-off questions: "how does improving f_1 affect f_2?", "what does it cost in f_2?"
            r"|\b(?:improv|worsen|chang|increas|decreas|rais|lower|reduc)\w*\b.*\b(?:affect|impact|cost)\w*\b"
            r"|\bgive\s+up\b|\bsacrific\w*|\bat\s+the\s+(?:cost|expense)\s+of\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explanation",
        re.compile(
            r"\bwhy\b|\bexplain\w*|\bwhat\b|\bwhich\b|\bhow\b|\btell\s+me\b|\bdescribe\b|\bshould\b"
            r"|\bvalues?\b|\bobjectives?\b|\bsolutions?\b|\bpreferences?\b"
            r"|\bideal\b|\bnadir\b|\bmaximi[sz]\w*|\bminimi[sz]\w*|\bdirection\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_intent(message: str) -> Intent:
    """Classify a message into the supported read-only policy categories."""
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message):
            return intent
    return "unsupported"


def _policy_abstention_reason(message: str, context: EvidenceContext) -> str | None:
    """Fail closed for requests that the v1 evidence model cannot support."""
    policies = [
        (r"\bignore\b.*\b(?:rules?|instructions?)\b|\binvent\b", "prompt_injection_or_invention"),
        (r"\b(?:because|caused?|causal)\b|\bwhy\s+(?:did|does|is|was|were)\b", "unsupported_causality"),
        (
            r"\bshould\s+i\b|\brecommend\b|\bwhich\s+solution\s+should\b|\bmake\s+the\s+decision\b",
            "decision_advice",
        ),
        (r"\b(?:euros?|dollars?|pounds?|kilograms?|meters?|liters?)\b", "unit_metadata_unavailable"),
        (r"\bweather\b|\bother\s+stakeholders?\b", "external_information_unavailable"),
        # A local trade-off is a first-order rate at the current solution; it cannot predict the exact
        # objective values of a new solution, so "exactly what would f_1 be" needs a new solve.
        (
            r"\bexact(?:ly)?\b.*\bwould\b|\bwould\b.*\bexact(?:ly)?\b|\bexact\s+value\b.*\bif\b",
            "exact_prediction_unavailable",
        ),
    ]
    for pattern, reason in policies:
        if re.search(pattern, message, re.IGNORECASE):
            return reason
    if _mentions_unknown_objective(message, context):
        return "unknown_objective"
    asks_for_history = re.search(
        r"\b(?:previous|last|earlier|latest)\s+(?:iteration|round|state|solution)\b"
        r"|\bacross\s+iterations?\b|\bhow\s+many\s+iterations?\b|\bexperiment\s+history\b",
        message,
        re.IGNORECASE,
    )
    has_history = any(
        fact.kind in {"experiment_summary", "iteration", "preference_history", "experiment_solution"}
        for fact in context.facts
    )
    if asks_for_history and not has_history:
        return "decision_history_unavailable"
    return None


def _mentions_unknown_objective(message: str, context: EvidenceContext) -> bool:
    """True when the question names an objective symbol (f_5, f5) that the problem does not define.

    Without this, a model that correctly says "not available" but cites the problem context is shown as a
    partial answer listing the real objectives, and the decision maker never learns the objective does not exist.
    Only applies when the evidence lists the problem's objectives.
    """
    known: set[str] = set()
    for fact in context.facts:
        if fact.kind != "problem_context":
            continue
        objectives = fact.value.get("objectives")
        if isinstance(objectives, list):
            known.update(o["symbol"] for o in objectives if isinstance(o, dict) and isinstance(o.get("symbol"), str))
    if not known:
        return False
    mentioned = {f"f_{number}" for number in re.findall(r"\bf_?(\d+)\b", message, re.IGNORECASE)}
    return bool(mentioned - known)


_INDEX_PHRASE = re.compile(
    r"\b(?:solutions?|objectives?|options?|candidates?|alternatives?|states?|iterations?|steps?|rounds?"
    r"|index|indices|solution_index|state_id)"
    r"\s*(?:#|no\.?|number|[=:])?\s*\d+(?!\d|\.\d)(?:\s*(?:,|and|or|&)\s*\d+(?!\d|\.\d))*",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z_\d])")


def mentioned_numbers(text: str, *, evidence_ids: Iterable[str] = ()) -> list[tuple[float, float]]:
    """Return ``(value, tolerance)`` for each numeric value stated in text."""
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
    """Collect numeric values in fact values, excluding identifiers in provenance."""
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
    """Return numeric tokens that do not match cited fact values at displayed precision."""
    known = fact_numbers(facts)
    missing: list[float] = []
    for value, tolerance in mentioned_numbers(text, evidence_ids=evidence_ids):
        slack = tolerance + 1e-9 * max(1.0, abs(value))
        if not any(abs(value - fact) <= slack for fact in known):
            missing.append(value)
    return missing


_CLAIM_FACT_KINDS: dict[ClaimType, frozenset[str]] = {
    "objective_value": frozenset({"objective_value"}),
    "variable_value": frozenset({"variable_value"}),
    "problem_context": frozenset({"problem_context"}),
    "preference": frozenset({"preference"}),
    "solver_status": frozenset({"solver_status"}),
    "lagrange_multiplier": frozenset({"lagrange_multiplier"}),
    "tradeoff": frozenset({"tradeoff"}),
    "active_objective": frozenset({"active_objective"}),
    "explanation_availability": frozenset({"explanation_availability"}),
    "comparison": frozenset({"comparison"}),
    "experiment_summary": frozenset({"experiment_summary"}),
    "iteration": frozenset({"iteration"}),
    "preference_history": frozenset({"preference_history"}),
    "experiment_solution": frozenset({"experiment_solution"}),
    "history_comparison": frozenset({"history_comparison"}),
}


def verify_grounded_answer(
    answer: GroundedAnswer,
    context: EvidenceContext,
    *,
    intent: Intent | None = None,
    question: str = "",
) -> VerificationResult:
    """Validate typed claim/evidence compatibility and every model-written number.

    Numbers the decision maker wrote in ``question`` may be repeated in the answer text (for example when
    correcting "f_1 is 999"); claim texts must still match their cited evidence.
    """
    facts_by_id = {fact.evidence_id: fact for fact in context.facts}
    if answer.uncertainty == "insufficient":
        if answer.claims:
            return VerificationResult(accepted=False, reason="An insufficient answer must not contain claims.")
        return VerificationResult(accepted=True)
    if not answer.claims:
        return VerificationResult(accepted=False, reason="Answer has no evidence-backed claims.")
    if intent == "comparison" and not any(claim.claim_type == "comparison" for claim in answer.claims):
        return VerificationResult(accepted=False, reason="A comparison answer must cite a comparison fact.")
    if intent in {"calculation", "what_if"} and not any(
        claim.claim_type in {"tradeoff", "lagrange_multiplier", "explanation_availability"} for claim in answer.claims
    ):
        return VerificationResult(
            accepted=False,
            reason="A calculation answer must cite computed explanation or availability evidence.",
        )
    if intent == "history" and not any(
        claim.claim_type
        in {"experiment_summary", "iteration", "preference_history", "experiment_solution", "history_comparison"}
        for claim in answer.claims
    ):
        return VerificationResult(
            accepted=False,
            reason="An experiment-history answer must cite experiment evidence.",
        )

    cited_ids: list[str] = []
    for claim in answer.claims:
        unknown = [evidence_id for evidence_id in claim.evidence_ids if evidence_id not in facts_by_id]
        if unknown:
            return VerificationResult(accepted=False, reason=f"Claim references unknown evidence: {unknown}.")
        cited = [facts_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        allowed_kinds = _CLAIM_FACT_KINDS[claim.claim_type]
        wrong_kinds = sorted({fact.kind for fact in cited if fact.kind not in allowed_kinds})
        if wrong_kinds:
            return VerificationResult(
                accepted=False,
                reason=f"Claim type {claim.claim_type!r} cannot cite fact kinds {wrong_kinds}.",
            )
        if claim.claim_type == "comparison" and (error := _validate_comparison(cited)) is not None:
            return VerificationResult(accepted=False, reason=error)
        if missing := unsupported_numbers(claim.text, cited, evidence_ids=facts_by_id):
            return VerificationResult(accepted=False, reason=f"Claim contains unsupported numbers: {missing}.")
        cited_ids.extend(evidence_id for evidence_id in claim.evidence_ids if evidence_id not in cited_ids)

    cited_facts = [facts_by_id[evidence_id] for evidence_id in cited_ids]
    quoted = [value for value, _ in mentioned_numbers(question)]
    missing = [
        value
        for value in unsupported_numbers(answer.answer, cited_facts, evidence_ids=facts_by_id)
        if not any(abs(value - quoted_value) <= 1e-9 * max(1.0, abs(value)) for quoted_value in quoted)
    ]
    if missing:
        return VerificationResult(accepted=False, reason=f"Answer text contains unsupported numbers: {missing}.")
    return VerificationResult(accepted=True, cited_evidence_ids=cited_ids)


def _validate_comparison(facts: list[EvidenceFact]) -> str | None:
    if not facts or any(fact.kind != "comparison" for fact in facts):
        return "A comparison claim must cite one or more deterministic comparison facts."
    return None


def render_verified_answer(answer: GroundedAnswer, context: EvidenceContext, *, intent: Intent) -> GroundedAnswer:
    """Discard model prose and render accepted claims from deterministic evidence."""
    facts_by_id = {fact.evidence_id: fact for fact in context.facts}
    rendered_claims: list[AnswerClaim] = []
    seen_claims: set[tuple[ClaimType, tuple[str, ...]]] = set()
    for claim in answer.claims:
        identity = (claim.claim_type, tuple(sorted(claim.evidence_ids)))
        if identity in seen_claims:
            continue
        seen_claims.add(identity)
        facts = [facts_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        text = _render_claim(claim, facts)
        rendered_claims.append(claim.model_copy(update={"text": text}))
    return GroundedAnswer(
        answer=" ".join(claim.text for claim in rendered_claims),
        claims=rendered_claims,
        uncertainty=answer.uncertainty,
        follow_up_questions=_safe_followups(intent),
    )


def _render_claim(claim: AnswerClaim, facts: list[EvidenceFact]) -> str:
    if claim.claim_type in {"objective_value", "variable_value"}:
        label = "Objective" if claim.claim_type == "objective_value" else "Variable"
        return "; ".join(
            f"{label} {_objective_label(fact)} is {_display_value(fact.value.get('value'))}"
            f" for {_solution_label(fact)}."
            for fact in facts
        )
    if claim.claim_type == "comparison":
        return "; ".join(_render_comparison_fact(fact) for fact in facts)
    if claim.claim_type == "problem_context":
        objectives = facts[0].value.get("objectives", [])
        rendered = []
        for objective in objectives if isinstance(objectives, list) else []:
            if not isinstance(objective, dict):
                continue
            direction = "maximize" if objective.get("maximize") else "minimize"
            rendered.append(
                f"{objective.get('symbol')} ({objective.get('name')}): {direction}, "
                f"ideal {_display_value(objective.get('ideal'))}, nadir {_display_value(objective.get('nadir'))}"
            )
        return "Problem objectives: " + "; ".join(rendered) + "."
    if claim.claim_type == "preference":
        return "Recorded preference: " + _display_value(facts[0].value) + "."
    if claim.claim_type == "solver_status":
        return "; ".join(
            f"For {_solution_label(fact)}, solver success is {str(fact.value.get('success')).lower()} "
            f"with message {_display_value(fact.value.get('message'))}."
            for fact in facts
        )
    if claim.claim_type == "lagrange_multiplier":
        return "; ".join(
            f"The multiplier for {_objective_label(fact)} at {_solution_label(fact)} is "
            f"{_display_value(fact.value.get('value'))}; reliable for ratios: "
            f"{str(fact.value.get('reliable_for_tradeoffs')).lower()}."
            for fact in facts
        )
    if claim.claim_type == "tradeoff":
        return "; ".join(
            f"The raw local trade-off matrix entry "
            f"[{_named(fact.value.get('row_symbol'), fact.value.get('row_name'))}]"
            f"[{_named(fact.value.get('column_symbol'), fact.value.get('column_name'))}] "
            f"is {_display_value(fact.value.get('value'))}, computed as "
            "-lambda_column/lambda_row. It is not interpreted as an improvement cost until the convention "
            "has been validated."
            for fact in facts
        )
    if claim.claim_type == "active_objective":
        symbols = ", ".join(_objective_label(fact) for fact in facts)
        return f"The computed active objectives are {symbols}."
    if claim.claim_type == "explanation_availability":
        fact = facts[0]
        return (
            f"Trade-off explanations available for {_solution_label(fact)}: "
            f"{str(fact.value.get('available')).lower()}; reason: {fact.value.get('reason')}."
        )
    if claim.claim_type in {
        "experiment_summary",
        "iteration",
        "preference_history",
        "experiment_solution",
        "history_comparison",
    }:
        return _render_experiment_claim(claim.claim_type, facts)
    raise ValueError(f"Unsupported claim type: {claim.claim_type}")


def _named(symbol: object, name: object) -> str:
    """Render an objective as its symbol, followed by its problem name when the evidence carries one."""
    text = str(symbol)
    return f"{text} ({name})" if isinstance(name, str) and name and name != text else text


def _objective_label(fact: EvidenceFact) -> str:
    """Render the objective of a single-objective fact with its name."""
    return _named(fact.value.get("symbol"), fact.value.get("name"))


def _objective_mapping(values: object, names: object) -> str:
    """Render a symbol-to-number mapping as named objectives, e.g. ``f_1 (DO city) = 4.0``."""
    if not isinstance(values, dict):
        return _display_value(values)
    lookup = names if isinstance(names, dict) else {}
    rendered = [f"{_named(symbol, lookup.get(symbol))} = {_display_value(value)}" for symbol, value in values.items()]
    return ", ".join(rendered) if rendered else "none"


def _objective_list(symbols: object, names: object) -> str:
    """Render a list of objective symbols with their names."""
    if not isinstance(symbols, list):
        return _display_value(symbols)
    lookup = names if isinstance(names, dict) else {}
    return ", ".join(_named(symbol, lookup.get(symbol)) for symbol in symbols) if symbols else "none"


def _render_comparison_fact(fact: EvidenceFact) -> str:
    """Render one precomputed same-objective comparison without model arithmetic."""
    value = fact.value
    first_index = value.get("first_solution_index")
    second_index = value.get("second_solution_index")
    better_index = value.get("better_solution_index")
    if not isinstance(first_index, int) or not isinstance(second_index, int):
        raise TypeError("Comparison evidence is missing solution indices.")
    if better_index is not None and not isinstance(better_index, int):
        raise TypeError("Comparison evidence has an invalid better-solution index.")
    better_text = (
        "the recorded values are equal"
        if better_index is None
        else f"solution {better_index + 1} is better for this objective"
    )
    return (
        f"For objective {_objective_label(fact)}, solution {first_index + 1} is "
        f"{_display_value(value.get('first_value'))} and solution {second_index + 1} is "
        f"{_display_value(value.get('second_value'))}; the server-computed second-minus-first difference is "
        f"{_display_value(value.get('delta_second_minus_first'))}, and {better_text}."
    )


def _render_experiment_claim(claim_type: ClaimType, facts: list[EvidenceFact]) -> str:
    """Render experiment-wide facts without allowing model-authored prose through."""
    if claim_type == "experiment_summary":
        value = facts[0].value
        return (
            f"This read-only decision experiment contains {value.get('state_count')} stored states and "
            f"{value.get('solution_count')} solver-generated solutions. The selected context is state "
            f"{value.get('selected_state_id')}, solution {int(value.get('selected_solution_index', 0)) + 1}."
        )
    if claim_type == "iteration":
        return "; ".join(
            f"Experiment step {fact.value.get('sequence')} is state {fact.value.get('state_id')} "
            f"({fact.value.get('kind')}) with {fact.value.get('solution_count')} stored solutions; "
            f"on selected lineage: {str(fact.value.get('on_selected_lineage')).lower()}."
            for fact in facts
        )
    if claim_type == "preference_history":
        return "; ".join(
            f"At experiment step {fact.value.get('sequence')} (state {fact.value.get('state_id')}), "
            f"the DM provided {_display_value(fact.value.get('preference'))}."
            for fact in facts
        )
    if claim_type == "experiment_solution":
        return "; ".join(
            f"Experiment step {fact.value.get('sequence')}, state {fact.value.get('state_id')}, "
            f"solution {int(fact.value.get('solution_index', 0)) + 1} has objective values "
            f"{_objective_mapping(fact.value.get('objective_values'), fact.value.get('objective_names'))}."
            for fact in facts
        )
    return "; ".join(_render_history_comparison(fact) for fact in facts)


def _render_history_comparison(fact: EvidenceFact) -> str:
    """Render a selected-vs-historical solution comparison computed by the server."""
    value = fact.value
    names = value.get("objective_names")
    return (
        f"Compared with experiment step {value.get('other_sequence')} state {value.get('other_state_id')} "
        f"solution {int(value.get('other_solution_index', 0)) + 1}, the selected-minus-other objective "
        f"deltas are {_objective_mapping(value.get('objective_deltas_selected_minus_other'), names)}. "
        f"The selected solution is better on {_objective_list(value.get('selected_better_objectives'), names)}, "
        f"worse on {_objective_list(value.get('selected_worse_objectives'), names)}, and equal on "
        f"{_objective_list(value.get('equal_objectives'), names)}."
    )


def _solution_label(fact: EvidenceFact) -> str:
    index = fact.source.get("solution_index")
    return f"solution {index + 1}" if isinstance(index, int) else "the selected state"


def _display_value(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _safe_followups(intent: Intent) -> list[str]:
    if intent == "history":
        return ["How did my preferences change?", "Compare with the previous iteration"]
    if intent in {"calculation", "what_if"}:
        return ["Which trade-off estimates are available?", "Which objectives are active?"]
    if intent == "comparison":
        return ["What are the objective values of the selected solution?"]
    return ["What are the objective values?", "Which objectives are active?"]
