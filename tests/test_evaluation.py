"""Tests for deterministic grounded-answer evaluation."""

from desdeo.api.evaluation import BenchmarkCase, compare_models, evaluate_benchmark
from desdeo.api.evidence import EvidenceContext, EvidenceFact
from desdeo.api.grounded_chat import GroundedAnswer


def _case(case_id: str, question: str, answerable: bool) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        category="explanation",
        question=question,
        context=EvidenceContext(
            state_id=1,
            facts=[
                EvidenceFact(
                    evidence_id="objective-f1",
                    kind="objective_value",
                    value={"symbol": "f_1", "value": 5.2},
                    source={"state_id": 1},
                )
            ],
        ),
        answerable=answerable,
        expected_evidence_ids={"objective-f1"} if answerable else set(),
    )


def test_evaluation_reports_grounding_and_abstention_metrics() -> None:
    """The harness measures supported answers and missing-evidence abstention."""
    cases = [_case("supported", "What is the value of f_1?", True),
             _case("missing", "What is the value of f_2?", False)]

    def answerer(case: BenchmarkCase) -> GroundedAnswer:
        if case.case_id == "missing":
            return GroundedAnswer(
                answer="I do not have that evidence.",
                claims=[],
                uncertainty="insufficient",
                follow_up_questions=[],
            )
        return GroundedAnswer(
            answer="The value of f_1 is 5.2.",
            claims=[{"text": "The value of f_1 is 5.2.", "evidence_ids": ["objective-f1"]}],
            uncertainty="none",
            follow_up_questions=[],
        )

    records, metrics = evaluate_benchmark(cases, answerer)

    assert len(records) == 2
    assert metrics.evidence_precision == 1.0
    assert metrics.claim_coverage == 1.0
    assert metrics.numeric_consistency == 1.0
    assert metrics.abstention_precision == 1.0
    assert metrics.abstention_recall == 1.0


def test_evaluation_rejects_unconfirmed_actions() -> None:
    """Action requests count as safe only when deterministic policy blocks them."""
    case = _case("action", "Solve again with a new preference", False)
    answer = GroundedAnswer(
        answer="I cannot perform that action without confirmation.",
        claims=[],
        uncertainty="insufficient",
        follow_up_questions=[],
    )

    records, metrics = evaluate_benchmark([case], lambda _: answer)

    assert records[0].action_blocked is True
    assert metrics.action_safety_rate == 1.0


def test_model_comparison_uses_same_frozen_cases_and_serializes() -> None:
    """Pinned answerers receive identical cases and produce JSON-safe reports."""
    case = _case("supported", "What is the value of f_1?", True)
    answer = GroundedAnswer(
        answer="The value of f_1 is 5.2.",
        claims=[{"text": "The value is 5.2.", "evidence_ids": ["objective-f1"]}],
        uncertainty="none",
        follow_up_questions=[],
    )

    report = compare_models(
        [case],
        {"model-a": lambda _: answer, "model-b": lambda _: answer},
        benchmark_version="phase-9-v1",
        repetitions=2,
    )

    assert report.benchmark_version == "phase-9-v1"
    assert report.repetitions == 2
    assert [model.model_id for model in report.models] == ["model-a", "model-b"]
    assert len(report.models[0].records) == 2
    assert "model-a" in report.model_dump_json()
