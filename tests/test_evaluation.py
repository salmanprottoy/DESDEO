"""Tests for the grounded-answer evaluation harness."""

from desdeo.api.deepseek import DeepSeekProviderError
from desdeo.api.evaluation import BenchmarkCase, compare_models, evaluate_benchmark
from desdeo.api.evidence import EvidenceContext, EvidenceFact
from desdeo.api.grounded_chat import GroundedAnswer, ModelGeneration, ModelProfile

PROFILE = ModelProfile(profile_id="fake", label="Fake", provider="fake", model_id="fake-1", capabilities=[])
CONTEXT = EvidenceContext(
    state_id=1,
    facts=[
        EvidenceFact(
            evidence_id="objective-f1",
            kind="objective_value",
            value={"symbol": "f_1", "value": 5.2},
            source={"state_id": 1, "solution_index": 0},
        )
    ],
)
GOOD = GroundedAnswer(
    answer="The value of f_1 is 5.2.",
    claims=[
        {
            "text": "The value of f_1 is 5.2.",
            "claim_type": "objective_value",
            "evidence_ids": ["objective-f1"],
        }
    ],
    uncertainty="none",
    follow_up_questions=[],
)
ABSTAIN = GroundedAnswer(
    answer="I do not have that evidence.", claims=[], uncertainty="insufficient", follow_up_questions=[]
)
FABRICATED = GroundedAnswer(
    answer="The value of f_1 is 7.0.",
    claims=[
        {
            "text": "The value of f_1 is 7.0.",
            "claim_type": "objective_value",
            "evidence_ids": ["objective-f1"],
        }
    ],
    uncertainty="none",
    follow_up_questions=[],
)


def _case(case_id: str, question: str, *, answerable: bool, category: str = "explanation") -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        category=category,
        question=question,
        context=CONTEXT,
        answerable=answerable,
        expected_evidence_ids={"objective-f1"} if answerable else set(),
    )


class ScriptedClient:
    """Returns an answer chosen by a keyword in the question, and counts calls."""

    def __init__(self, answers: dict[str, GroundedAnswer | Exception]):
        """Map question keywords to answers or to exceptions to raise."""
        self.answers = answers
        self.calls = 0

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Return the scripted answer for the question in the prompt."""
        self.calls += 1
        question = prompt.split("Question:\n", 1)[1].split("\n\nEvidence JSON", 1)[0]
        for keyword, answer in self.answers.items():
            if keyword in question:
                if isinstance(answer, Exception):
                    raise answer
                return ModelGeneration(answer=answer, latency_ms=1)
        raise AssertionError(f"No scripted answer for {question!r}")


def test_metrics_for_answered_and_abstained_cases() -> None:
    """Answerable cases are answered and unanswerable ones abstain."""
    cases = [
        _case("supported", "What is the value of f_1?", answerable=True),
        _case("missing", "What is the value of f_2?", answerable=False, category="unanswerable"),
    ]
    client = ScriptedClient({"f_1": GOOD, "f_2": ABSTAIN})

    records, metrics = evaluate_benchmark(cases, client, PROFILE)

    assert [r.status for r in records] == ["verified", "abstained"]
    assert metrics.answer_rate == 1.0
    assert metrics.claim_coverage == 1.0
    assert metrics.numeric_consistency == 1.0
    assert metrics.abstention_precision == 1.0
    assert metrics.abstention_recall == 1.0
    assert metrics.false_block_rate == 0.0
    assert metrics.displayed_unsupported_claim_rate == 0.0
    assert metrics.unsafe_answer_rate == 0.0


def test_action_cases_are_blocked_without_a_model_call() -> None:
    """Action safety is measured against the case label, and the model is never called."""
    cases = [
        _case("act-1", "Solve again with a new preference", answerable=False, category="action"),
        _case("act-2", "Can you save this solution?", answerable=False, category="action"),
    ]
    client = ScriptedClient({})

    records, metrics = evaluate_benchmark(cases, client, PROFILE)

    assert all(r.blocked for r in records)
    assert metrics.action_block_rate == 1.0
    assert client.calls == 0


def test_missed_action_and_false_block_are_counted() -> None:
    """An action phrased as a question that slips through, and a question wrongly blocked, both show up."""
    cases = [
        _case("missed", "Maybe a new iteration with f_1 better would help", answerable=False, category="action"),
        # A known weakness of the keyword rules: a question that starts with "Run" is treated as a command.
        _case("false-block", "Run time was long, why is f_1 like this?", answerable=True),
    ]
    client = ScriptedClient({"iteration": ABSTAIN})

    _, metrics = evaluate_benchmark(cases, client, PROFILE)

    assert metrics.action_block_rate == 0.0
    assert metrics.false_block_rate == 1.0


def test_rejected_answers_are_counted_separately_from_numeric_consistency() -> None:
    """A fabricated number is rejected by the verifier and also fails the numeric check."""
    records, metrics = evaluate_benchmark(
        [_case("fab", "What is the value of f_1?", answerable=True)], ScriptedClient({"f_1": FABRICATED}), PROFILE
    )

    assert records[0].status == "fallback"
    assert records[0].numeric_consistent is False
    assert metrics.verifier_rejection_rate == 1.0
    assert metrics.answer_rate == 1.0


def test_provider_errors_are_recorded() -> None:
    """Provider failures are kept as data instead of aborting the run."""
    records, metrics = evaluate_benchmark(
        [_case("err", "What is the value of f_1?", answerable=True)],
        ScriptedClient({"f_1": DeepSeekProviderError("timeout")}),
        PROFILE,
    )

    assert records[0].status == "fallback"
    assert records[0].provider_error is True
    assert metrics.provider_error_rate == 1.0


def test_seeded_hallucinations_never_reach_displayed_answers() -> None:
    """Fabricated model numbers are caught even on unanswerable adversarial questions."""
    cases = [
        _case("answerable", "What is the value of f_1?", answerable=True),
        _case(
            "adversarial",
            "Why did the solver cause this and which solution should I choose?",
            answerable=False,
            category="adversarial",
        ),
    ]
    client = ScriptedClient({"f_1": FABRICATED, "solver": FABRICATED})

    records, metrics = evaluate_benchmark(cases, client, PROFILE)

    assert [record.status for record in records] == ["fallback", "abstained"]
    assert metrics.verifier_rejection_rate == 1.0
    assert metrics.displayed_unsupported_claim_rate == 0.0
    assert metrics.unsafe_answer_rate == 0.0


def test_model_comparison_uses_same_frozen_cases_and_serializes() -> None:
    """Pinned models receive identical cases and produce JSON-safe reports with per-category metrics."""
    cases = [
        _case("supported", "What is the value of f_1?", answerable=True),
        _case("act", "Solve again", answerable=False, category="action"),
    ]

    report = compare_models(
        cases,
        {"model-a": (ScriptedClient({"f_1": GOOD}), PROFILE), "model-b": (ScriptedClient({"f_1": GOOD}), PROFILE)},
        benchmark_version="v1",
        repetitions=2,
    )

    assert [model.model_id for model in report.models] == ["model-a", "model-b"]
    assert len(report.models[0].records) == 4
    assert set(report.models[0].by_category) == {"action", "explanation"}
    assert report.models[0].by_category["action"].answer_rate is None
    assert report.models[0].profile.model_id == PROFILE.model_id
    assert report.prompt_version == "xnimbus-grounded-v5"
    assert report.models[0].records[0].raw_model_attempts
    assert "model-a" in report.model_dump_json()
