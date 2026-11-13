"""Run the XNIMBUS chat benchmark against the configured language model providers.

The questions and evidence snapshot are sent to every selected cloud provider, so a real run can
cost API credits. Only server-configured, explicitly pinned model profiles are evaluated.

Usage:
    uv run python scripts/xnimbus_eval/run_benchmark.py --providers local openai anthropic google_ai deepseek groq
    uv run python scripts/xnimbus_eval/run_benchmark.py --dry-run   # no API calls; checks the pipeline
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from desdeo.api.evaluation import BenchmarkFile, EvaluationMetrics, EvaluationReport, compare_models
from desdeo.api.grounded_chat import GroundedAnswer, ModelGeneration, ModelProfile
from desdeo.api.llm_config import available_model_profiles, get_llm_configuration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "datasets" / "xnimbus_chat_benchmark_v1.json"
DEFAULT_OUT_DIR = ROOT / "scripts" / "xnimbus_eval" / "results"
PROVIDERS = {"local", "openai", "anthropic", "google_ai", "deepseek", "groq"}
METRICS = list(EvaluationMetrics.model_fields)


class AbstainingClient:
    """Always abstains. Used by --dry-run to exercise the pipeline without API calls."""

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Return an abstention without looking at the prompt."""
        return ModelGeneration(
            answer=GroundedAnswer(
                answer="No evidence.",
                claims=[],
                uncertainty="insufficient",
                follow_up_questions=[],
            ),
            latency_ms=0,
        )


def _format(value: float | int | None) -> str:
    if value is None:
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.2f}"


def _markdown_report(report: EvaluationReport) -> str:
    """Render a concise human-readable safety report beside the full JSON audit log."""
    lines = [
        f"# XNIMBUS grounded-chat evaluation — {report.benchmark_version}",
        "",
        f"Prompt version: `{report.prompt_version}`  ",
        f"Repetitions: {report.repetitions}",
        "",
        "Automated verifier metrics measure the deployed deterministic boundary. They do not replace human ",
        "annotation of semantic helpfulness or relevance.",
        "",
    ]
    for model in report.models:
        lines.extend(
            [
                f"## {model.profile.label}",
                "",
                f"Provider: `{model.profile.provider}` · deployment: `{model.profile.deployment}` · "
                f"model: `{model.profile.model_id}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        lines.extend(f"| {name.replace('_', ' ')} | {_format(getattr(model.metrics, name))} |" for name in METRICS)
        failures = [
            record
            for record in model.records
            if record.provider_error
            or record.verifier_rejected
            or (not record.answerable and record.answered)
            or record.displayed_claims_supported is False
        ]
        lines.extend(["", "### Safety-relevant cases", ""])
        if not failures:
            lines.append("No automated safety failure was recorded.")
        for record in failures[:20]:
            raw = record.raw_model_answer.answer if record.raw_model_answer is not None else "(model not called)"
            lines.extend(
                [
                    f"- **{record.case_id}** (`{record.category}`, status `{record.status}`): {record.question}",
                    f"  - Raw candidate: {raw}",
                    f"  - Displayed: {record.displayed_answer}",
                    f"  - Verifier: {record.rejection_reason or 'accepted'}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Run the benchmark and write a timestamped JSON report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS), default=sorted(PROVIDERS))
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
        configured = [profile for profile in available_model_profiles() if profile.provider in args.providers]
        for configured_profile in configured:
            client, profile = get_llm_configuration(configured_profile.profile_id, cloud_consent=True)
            if client is None or profile is None:  # pragma: no cover - registry guarantees both
                continue
            models[profile.profile_id] = (client, profile)
    if not models:
        raise SystemExit("No provider is configured. Set an API key or use --dry-run.")

    report = compare_models(cases, models, benchmark_version=benchmark.benchmark_version, repetitions=args.repetitions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{benchmark.benchmark_version}-{stamp}.json"
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    markdown_out = out.with_suffix(".md")
    markdown_out.write_text(_markdown_report(report), encoding="utf-8")

    width = max(len(name) for name in METRICS)
    print(f"{'metric':<{width}}  " + "  ".join(f"{model.model_id:>18}" for model in report.models))  # noqa: T201
    for name in METRICS:
        values = [_format(getattr(model.metrics, name)) for model in report.models]
        print(f"{name:<{width}}  " + "  ".join(f"{value:>18}" for value in values))  # noqa: T201
    print(f"\nFull report: {out}")  # noqa: T201
    print(f"Summary: {markdown_out}")  # noqa: T201


if __name__ == "__main__":
    main()
