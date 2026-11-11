"""Run the XNIMBUS chat benchmark against the configured language model providers.

The questions and the evidence snapshot are sent to each provider, so running this costs API
credits. Keys come from the environment (DEEPSEEK_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY);
a provider without a key is skipped. Pin models with DEEPSEEK_MODEL and GOOGLE_AI_MODEL.

Usage:
    uv run python scripts/xnimbus_eval/run_benchmark.py --providers deepseek google_ai --repetitions 3
    uv run python scripts/xnimbus_eval/run_benchmark.py --dry-run   # no API calls; checks the pipeline
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from desdeo.api.deepseek import deepseek_configuration
from desdeo.api.evaluation import BenchmarkFile, EvaluationMetrics, compare_models
from desdeo.api.google_ai import google_ai_configuration
from desdeo.api.grounded_chat import GroundedAnswer, ModelProfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "datasets" / "xnimbus_chat_benchmark_v1.json"
DEFAULT_OUT_DIR = ROOT / "scripts" / "xnimbus_eval" / "results"
CONFIGURATIONS = {"deepseek": deepseek_configuration, "google_ai": google_ai_configuration}
METRICS = list(EvaluationMetrics.model_fields)


class AbstainingClient:
    """Always abstains. Used by --dry-run to exercise the pipeline without API calls."""

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        """Return an abstention without looking at the prompt."""
        return GroundedAnswer(answer="No evidence.", claims=[], uncertainty="insufficient", follow_up_questions=[])


def _format(value: float | int | None) -> str:
    if value is None:
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.2f}"


def main() -> None:
    """Run the benchmark and write a timestamped JSON report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--providers", nargs="+", choices=sorted(CONFIGURATIONS), default=sorted(CONFIGURATIONS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="use an always-abstaining fake model")
    args = parser.parse_args()

    benchmark = BenchmarkFile.model_validate_json(args.benchmark.read_text(encoding="utf-8"))
    cases = benchmark.to_cases()

    models = {}
    if args.dry_run:
        models["dry-run"] = (
            AbstainingClient(),
            ModelProfile(profile_id="dry-run", label="Dry run", provider="none", model_id="none", capabilities=[]),
        )
    else:
        for provider in args.providers:
            client, profile = CONFIGURATIONS[provider]()
            if client is None or profile is None:
                print(f"Skipping {provider}: no API key in the environment.")  # noqa: T201
                continue
            models[profile.profile_id] = (client, profile)
    if not models:
        raise SystemExit("No provider is configured. Set an API key or use --dry-run.")

    report = compare_models(cases, models, benchmark_version=benchmark.benchmark_version, repetitions=args.repetitions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{benchmark.benchmark_version}-{stamp}.json"
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    width = max(len(name) for name in METRICS)
    print(f"{'metric':<{width}}  " + "  ".join(f"{model.model_id:>18}" for model in report.models))  # noqa: T201
    for name in METRICS:
        values = [_format(getattr(model.metrics, name)) for model in report.models]
        print(f"{name:<{width}}  " + "  ".join(f"{value:>18}" for value in values))  # noqa: T201
    print(f"\nFull report: {out}")  # noqa: T201


if __name__ == "__main__":
    main()
