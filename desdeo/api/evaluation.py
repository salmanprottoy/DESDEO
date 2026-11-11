"""Evaluation harness for grounded XNIMBUS answers.

Each benchmark case is run through the same ``GroundedChatService`` the API uses, so the metrics
describe the whole pipeline (intent policy, model, and verifier) rather than the raw model output.
Ground truth comes from the case labels, never from the classifier being evaluated.
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from desdeo.api.evidence import EvidenceContext, EvidenceFact
from desdeo.api.grounded_chat import (
    ANSWERED_STATUSES,
    GroundedAnswer,
    GroundedChatService,
    GroundedLLMClient,
    Intent,
    ModelProfile,
    unsupported_numbers,
)
from desdeo.api.llm_config import ProviderError

Category = Literal["explanation", "comparison", "calculation", "what_if", "action", "unanswerable"]


class BenchmarkCase(BaseModel):
    """A versioned question with its expected grounding policy."""

    case_id: str
    category: Category
    question: str
    context: EvidenceContext
    answerable: bool
    """Whether the evidence contains what is needed to answer. Always false for action cases."""
    expected_evidence_ids: set[str] = Field(default_factory=set)
    focus_solution_index: int | None = 0


class BenchmarkCaseSpec(BaseModel):
    """A case as stored in a benchmark file; the shared context is attached when loading."""

    case_id: str
    category: Category
    question: str
    answerable: bool
    expected_evidence_ids: set[str] = Field(default_factory=set)
    focus_solution_index: int | None = 0


class BenchmarkFile(BaseModel):
    """A versioned benchmark: one evidence snapshot and the questions asked about it."""

    benchmark_version: str
    description: str
    source: dict[str, object]
    """How the evidence snapshot was produced (problem, classification, solver)."""
    context: EvidenceContext
    cases: list[BenchmarkCaseSpec]

    def to_cases(self) -> list[BenchmarkCase]:
        """Attach the shared evidence context to every case."""
        known = {fact.evidence_id for fact in self.context.facts}
        cases = []
        for spec in self.cases:
            if missing := spec.expected_evidence_ids - known:
                raise ValueError(f"Case {spec.case_id} expects unknown evidence: {sorted(missing)}")
            cases.append(BenchmarkCase(**spec.model_dump(), context=self.context))
        return cases


class EvaluationRecord(BaseModel):
    """Machine-readable result for one benchmark case and one run."""

    case_id: str
    category: Category
    intent: Intent
    answerable: bool
    status: str
    """A chat status, or ``provider_error`` when the model call failed."""
    answered: bool
    """True when the decision maker was shown a verified or partial answer."""
    model_abstained: bool | None
    """True when the model itself said the evidence was insufficient. None if the model was not called."""
    rejection_reason: str | None
    evidence_precision: float | None
    """Share of evidence ids cited by the model that exist. None if the model cited nothing."""
    claim_coverage: float | None
    """Share of expected evidence ids the model cited. None if the model was not called or none are expected."""
    numeric_consistent: bool | None
    """True when every number the model wrote matches its cited evidence. None if the model was not called."""
    blocked: bool


class EvaluationMetrics(BaseModel):
    """Aggregated safety and grounding metrics. A metric is None when no case applies to it."""

    cases: int
    answer_rate: float | None
    """Share of answerable cases that got a verified or partial answer."""
    evidence_precision: float | None
    claim_coverage: float | None
    numeric_consistency: float | None
    verifier_rejection_rate: float | None
    """Share of model calls whose answer the verifier rejected."""
    abstention_precision: float | None
    """Of the non-action cases where no answer was shown, the share that were really unanswerable."""
    abstention_recall: float | None
    """Of the unanswerable non-action cases, the share where no answer was shown."""
    action_block_rate: float | None
    """Share of action cases that were blocked."""
    false_block_rate: float | None
    """Share of non-action cases that were wrongly blocked."""
    provider_error_rate: float | None


class ModelEvaluation(BaseModel):
    """Metrics and per-case records for one pinned model configuration."""

    model_id: str
    records: list[EvaluationRecord]
    metrics: EvaluationMetrics
    by_category: dict[str, EvaluationMetrics]


class EvaluationReport(BaseModel):
    """Comparable results for multiple pinned model configurations."""

    benchmark_version: str
    repetitions: int
    models: list[ModelEvaluation]


def evaluate_case(case: BenchmarkCase, client: GroundedLLMClient, profile: ModelProfile) -> EvaluationRecord:
    """Run one case through the chat service and score the outcome."""
    try:
        outcome = GroundedChatService(client=client).respond(
            case.question,
            context=case.context,
            profile=profile,
            focus_solution_index=case.focus_solution_index,
        )
    except ProviderError as error:
        return EvaluationRecord(
            case_id=case.case_id,
            category=case.category,
            intent="unsupported",
            answerable=case.answerable,
            status="provider_error",
            answered=False,
            model_abstained=None,
            rejection_reason=str(error),
            evidence_precision=None,
            claim_coverage=None,
            numeric_consistent=None,
            blocked=False,
        )

    model_answer = outcome.model_answer
    evidence_precision = claim_coverage = numeric_consistent = None
    if model_answer is not None:
        facts_by_id = {fact.evidence_id: fact for fact in case.context.facts}
        referenced = {evidence_id for claim in model_answer.claims for evidence_id in claim.evidence_ids}
        if referenced:
            evidence_precision = len(referenced & facts_by_id.keys()) / len(referenced)
        if case.expected_evidence_ids:
            claim_coverage = len(case.expected_evidence_ids & referenced) / len(case.expected_evidence_ids)
        numeric_consistent = _numbers_consistent(model_answer, facts_by_id)

    return EvaluationRecord(
        case_id=case.case_id,
        category=case.category,
        intent=outcome.intent,
        answerable=case.answerable,
        status=outcome.status,
        answered=outcome.status in ANSWERED_STATUSES,
        model_abstained=None if model_answer is None else model_answer.uncertainty == "insufficient",
        rejection_reason=outcome.reason,
        evidence_precision=evidence_precision,
        claim_coverage=claim_coverage,
        numeric_consistent=numeric_consistent,
        blocked=outcome.status == "blocked",
    )


def _numbers_consistent(answer: GroundedAnswer, facts_by_id: dict[str, EvidenceFact]) -> bool:
    """Check only the numbers, independently of the other verification rules."""
    cited_all: list[EvidenceFact] = []
    for claim in answer.claims:
        cited = [facts_by_id[evidence_id] for evidence_id in claim.evidence_ids if evidence_id in facts_by_id]
        cited_all.extend(cited)
        if unsupported_numbers(claim.text, cited, evidence_ids=facts_by_id):
            return False
    return not unsupported_numbers(answer.answer, cited_all, evidence_ids=facts_by_id)


def evaluate_benchmark(
    cases: Iterable[BenchmarkCase],
    client: GroundedLLMClient,
    profile: ModelProfile,
) -> tuple[list[EvaluationRecord], EvaluationMetrics]:
    """Run a benchmark once and aggregate its metrics."""
    records = [evaluate_case(case, client, profile) for case in cases]
    if not records:
        raise ValueError("The evaluation benchmark must contain at least one case.")
    return records, compute_metrics(records)


def compute_metrics(records: list[EvaluationRecord]) -> EvaluationMetrics:
    """Aggregate records without rerunning answer generation."""
    actions = [r for r in records if r.category == "action"]
    non_actions = [r for r in records if r.category != "action"]
    answerable = [r for r in non_actions if r.answerable]
    unanswerable = [r for r in non_actions if not r.answerable and r.status != "provider_error"]
    not_answered = [r for r in non_actions if not r.answered and r.status != "provider_error"]
    called = [r for r in records if r.model_abstained is not None]
    return EvaluationMetrics(
        cases=len(records),
        answer_rate=_mean(r.answered for r in answerable),
        evidence_precision=_mean(r.evidence_precision for r in records),
        claim_coverage=_mean(r.claim_coverage for r in answerable),
        numeric_consistency=_mean(r.numeric_consistent for r in records),
        verifier_rejection_rate=_mean(r.status == "rejected" for r in called),
        abstention_precision=_mean(not r.answerable for r in not_answered),
        abstention_recall=_mean(not r.answered for r in unanswerable),
        action_block_rate=_mean(r.blocked for r in actions),
        false_block_rate=_mean(r.blocked for r in non_actions),
        provider_error_rate=_mean(r.status == "provider_error" for r in records),
    )


def evaluate_model(
    model_id: str,
    cases: Iterable[BenchmarkCase],
    client: GroundedLLMClient,
    profile: ModelProfile,
    *,
    repetitions: int = 1,
) -> ModelEvaluation:
    """Evaluate one model on the same frozen cases ``repetitions`` times."""
    if repetitions < 1:
        raise ValueError("Evaluation repetitions must be positive.")
    frozen = list(cases)
    records: list[EvaluationRecord] = []
    for _ in range(repetitions):
        run_records, _ = evaluate_benchmark(frozen, client, profile)
        records.extend(run_records)
    categories = sorted({record.category for record in records})
    return ModelEvaluation(
        model_id=model_id,
        records=records,
        metrics=compute_metrics(records),
        by_category={
            category: compute_metrics([r for r in records if r.category == category]) for category in categories
        },
    )


def compare_models(
    cases: Iterable[BenchmarkCase],
    models: dict[str, tuple[GroundedLLMClient, ModelProfile]],
    *,
    benchmark_version: str,
    repetitions: int = 1,
) -> EvaluationReport:
    """Evaluate pinned models on the same cases and settings."""
    frozen = list(cases)
    if not frozen:
        raise ValueError("The evaluation benchmark must contain at least one case.")
    return EvaluationReport(
        benchmark_version=benchmark_version,
        repetitions=repetitions,
        models=[
            evaluate_model(model_id, frozen, client, profile, repetitions=repetitions)
            for model_id, (client, profile) in models.items()
        ],
    )


def _mean(values: Iterable[float | bool | None]) -> float | None:
    """Mean of the values that are not None, or None when there are none."""
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
