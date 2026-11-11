"""Checks that the committed XNIMBUS chat benchmark is well-formed."""

from collections import Counter
from pathlib import Path

from desdeo.api.evaluation import BenchmarkFile

BENCHMARK = Path(__file__).resolve().parents[1] / "datasets" / "xnimbus_chat_benchmark_v1.json"


def test_benchmark_loads_and_references_existing_evidence() -> None:
    """Every case has a unique id and only expects evidence that is in the snapshot."""
    benchmark = BenchmarkFile.model_validate_json(BENCHMARK.read_text(encoding="utf-8"))

    cases = benchmark.to_cases()

    assert len({case.case_id for case in cases}) == len(cases)
    assert set(Counter(case.category for case in cases)) == {
        "explanation",
        "comparison",
        "calculation",
        "what_if",
        "unanswerable",
        "action",
    }
    assert all(not case.answerable for case in cases if case.category in {"action", "unanswerable"})
    assert all(case.expected_evidence_ids for case in cases if case.answerable)
