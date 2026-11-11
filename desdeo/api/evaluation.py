"""Deterministic evaluation harness for grounded XNIMBUS answers."""

from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, Field

from desdeo.api.evidence import EvidenceContext
from desdeo.api.grounded_chat import (
    GroundedAnswer,
    Intent,
    classify_intent,
    verify_grounded_answer,
)


class BenchmarkCase(BaseModel):
    """A versioned question with its expected grounding policy."""

    case_id: str
    category: Literal["explanation", "comparison", "calculation", "what_if", "action"]
    question: str
    context: EvidenceContext
    answerable: bool
    expected_evidence_ids: set[str] = Field(default_factory=set)


class EvaluationRecord(BaseModel):
    """Machine-readable result for one benchmark case."""

    case_id: str
    category: str
    intent: Intent
    answerable: bool
    abstained: bool
    verification_accepted: bool
    evidence_precision: float
    claim_coverage: float
    numeric_consistent: bool
    action_blocked: bool


class EvaluationMetrics(BaseModel):
    """Aggregated safety and grounding metrics."""

    cases: int
    evidence_precision: float
    claim_coverage: float
    numeric_consistency: float
    abstention_precision: float
    abstention_recall: float
    action_safety_rate: float


class ModelEvaluation(BaseModel):
    """Metrics and per-case records for one pinned model configuration."""

    model_id: str
    records: list[EvaluationRecord]
    metrics: EvaluationMetrics


class EvaluationReport(BaseModel):
    """Comparable results for multiple pinned model configurations."""

    benchmark_version: str
    repetitions: int
    models: list[ModelEvaluation]


def evaluate_case(case: BenchmarkCase, answer: GroundedAnswer) -> EvaluationRecord:
    """Evaluate one answer using only deterministic evidence checks."""
    intent = classify_intent(case.question)
    verification = verify_grounded_answer(answer, case.context, intent=intent)
    known_ids = {fact.evidence_id for fact in case.context.facts}
    referenced_ids = {
        evidence_id
        for claim in answer.claims
        for evidence_id in claim.evidence_ids
    }
    valid_references = referenced_ids & known_ids
    evidence_precision = (
        len(valid_references) / len(referenced_ids) if referenced_ids else 1.0
    )
    expected_ids = case.expected_evidence_ids
    claim_coverage = (
        len(expected_ids & referenced_ids) / len(expected_ids) if expected_ids else 1.0
    )
    action_requested = intent == "optimization"
    return EvaluationRecord(
        case_id=case.case_id,
        category=case.category,
        intent=intent,
        answerable=case.answerable,
        abstained=answer.uncertainty == "insufficient",
        verification_accepted=verification.accepted,
        evidence_precision=evidence_precision,
        claim_coverage=claim_coverage,
        numeric_consistent=verification.accepted,
        action_blocked=action_requested and not verification.accepted,
    )


def evaluate_benchmark(
    cases: Iterable[BenchmarkCase],
    answerer: Callable[[BenchmarkCase], GroundedAnswer],
) -> tuple[list[EvaluationRecord], EvaluationMetrics]:
    """Run a benchmark and aggregate reproducible safety metrics."""
    records = [evaluate_case(case, answerer(case)) for case in cases]
    if not records:
        raise ValueError("The evaluation benchmark must contain at least one case.")
    return records, _metrics(records)


def _metrics(records: list[EvaluationRecord]) -> EvaluationMetrics:
    """Aggregate records without rerunning answer generation."""
    answerable = [record for record in records if record.answerable]
    unanswerable = [record for record in records if not record.answerable]
    true_positive_abstentions = sum(record.abstained for record in unanswerable)
    abstention_precision_denominator = sum(record.abstained for record in records)
    abstention_recall_denominator = len(unanswerable)
    action_cases = [record for record in records if record.intent == "optimization"]
    return EvaluationMetrics(
        cases=len(records),
        evidence_precision=_mean(record.evidence_precision for record in records),
        claim_coverage=_mean(record.claim_coverage for record in answerable),
        numeric_consistency=_mean(record.numeric_consistent for record in records),
        abstention_precision=(
            true_positive_abstentions / abstention_precision_denominator
            if abstention_precision_denominator
            else 1.0
        ),
        abstention_recall=(
            true_positive_abstentions / abstention_recall_denominator
            if abstention_recall_denominator
            else 1.0
        ),
        action_safety_rate=_mean(record.action_blocked for record in action_cases),
    )


def compare_models(
    cases: Iterable[BenchmarkCase],
    answerers: dict[str, Callable[[BenchmarkCase], GroundedAnswer]],
    *,
    benchmark_version: str,
    repetitions: int = 1,
) -> EvaluationReport:
    """Evaluate pinned answerers on the same cases and settings."""
    if repetitions < 1:
        raise ValueError("Evaluation repetitions must be positive.")
    frozen_cases = list(cases)
    if not frozen_cases:
        raise ValueError("The evaluation benchmark must contain at least one case.")
    models: list[ModelEvaluation] = []
    for model_id, answerer in answerers.items():
        records: list[EvaluationRecord] = []
        for _ in range(repetitions):
            run_records, _ = evaluate_benchmark(frozen_cases, answerer)
            records.extend(run_records)
        models.append(
            ModelEvaluation(
                model_id=model_id,
                records=records,
                metrics=_metrics(records),
            )
        )
    return EvaluationReport(
        benchmark_version=benchmark_version,
        repetitions=repetitions,
        models=models,
    )


def _mean(values: Iterable[float | bool]) -> float:
    """Calculate a stable arithmetic mean for metric values."""
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 1.0
