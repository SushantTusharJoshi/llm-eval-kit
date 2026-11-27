"""CLI entry point for llm-eval-kit.

Exit codes: 0 gate passed, 1 gate failed or error, 3 stopped early by a run
limit (--max-cost, --max-calls, --max-run-tokens).
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

from llm_eval_kit import __version__
from llm_eval_kit.cache import DEFAULT_CACHE_DIR, ResultCache
from llm_eval_kit.calls import ModelCaller, provider_region
from llm_eval_kit.config import RegressionTolerance
from llm_eval_kit.guardrails import guardrail_from_env
from llm_eval_kit.loader import load_suite
from llm_eval_kit.pricing import lookup_price
from llm_eval_kit.providers.base import ProviderError
from llm_eval_kit.providers.factory import PROVIDER_NAMES, build_provider
from llm_eval_kit.providers.mock import MockEmbedder
from llm_eval_kit.regression import evaluate_regression, load_baseline, regression_summary, write_baseline
from llm_eval_kit.reliability import Budget, RunLimits
from llm_eval_kit.runner import run_suite
from llm_eval_kit.security import redact

EXIT_STOPPED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval-kit",
        description="Evaluate LLM outputs against golden examples with CI-friendly pass/fail gating.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("suite", help="Path to eval suite file (YAML or JSON)")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock provider and embedder for testing (same as --provider mock)",
    )
    parser.add_argument("--provider", choices=PROVIDER_NAMES, help="Model under test")
    parser.add_argument("--model", help="Model ID (default: GROQ_MODEL / BEDROCK_MODEL_ID / OPENAI_MODEL)")
    parser.add_argument("--base-url", help="Base URL for --provider openai (default: OPENAI_BASE_URL)")
    parser.add_argument("--region", help="AWS region for --provider bedrock (default: AWS_REGION)")
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Embedding dimension for mock embedder (default: 64)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-example results",
    )
    parser.add_argument("--report", metavar="PATH", help="Write the JSON report here")

    gate = parser.add_argument_group("regression gate")
    gate.add_argument("--baseline", metavar="PATH", help="Baseline JSON in git to compare against")
    gate.add_argument("--update-baseline", action="store_true", help="Write this run as the new baseline")
    gate.add_argument("--tolerance-pass-rate", type=float, help="Allowed non-critical pass-rate drop (e.g. 0.05)")
    gate.add_argument("--tolerance-score", type=float, help="Allowed non-critical mean-score drop (e.g. 0.05)")
    parser.add_argument(
        "--no-guardrails", action="store_true", help="Skip guardrail_examples even if BEDROCK_GUARDRAIL_ID is set"
    )

    cache = parser.add_argument_group("cache")
    cache.add_argument("--no-cache", action="store_true", help="Bypass the result cache")
    cache.add_argument(
        "--cache-dir",
        default=os.environ.get("LLM_EVAL_CACHE_DIR", DEFAULT_CACHE_DIR),
        help=f"Cache directory (default: LLM_EVAL_CACHE_DIR or {DEFAULT_CACHE_DIR})",
    )

    limits = parser.add_argument_group("run limits")
    limits.add_argument("--max-cost", type=float, metavar="USD", help="Stop when estimated live spend reaches this")
    limits.add_argument("--max-calls", type=int, help="Stop after this many live model calls")
    limits.add_argument("--max-run-tokens", type=int, help="Stop after this many live tokens (in + out)")
    limits.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds (default: 60)")
    limits.add_argument("--retries", type=int, default=2, help="Retries for throttles/timeouts/5xx (default: 2)")
    return parser


def _write_report(path: str, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        suite = load_suite(args.suite)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {redact(str(e))}", file=sys.stderr)
        return 1

    provider_name = "mock" if args.mock else args.provider
    if provider_name is None:
        print("Error: no provider chosen. Use --mock (only --mock runs offline without keys)", file=sys.stderr)
        print(f"or --provider one of: {', '.join(PROVIDER_NAMES)}", file=sys.stderr)
        return 1

    try:
        provider = build_provider(
            provider_name,
            model=args.model,
            base_url=args.base_url,
            region=args.region,
            timeout_s=args.timeout,
            retries=args.retries,
        )
    except (ValueError, ProviderError) as e:
        print(f"Error: {redact(str(e))}", file=sys.stderr)
        return 1

    if args.update_baseline and not args.baseline:
        print("Error: --update-baseline needs --baseline PATH", file=sys.stderr)
        return 1
    for flag in ("tolerance_pass_rate", "tolerance_score"):
        value = getattr(args, flag)
        if value is not None and not 0.0 <= value <= 1.0:
            print(f"Error: --{flag.replace('_', '-')} must be between 0 and 1", file=sys.stderr)
            return 1
    baseline = None
    if args.baseline and not args.update_baseline:
        try:
            baseline = load_baseline(args.baseline)
        except FileNotFoundError:
            print(f"Error: baseline not found: {args.baseline} (create it with --update-baseline)", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    needs_judge = suite.judge is not None and any(e.comparator == "judge" for e in suite.examples)
    judge_provider = None
    if needs_judge:
        j = suite.judge
        try:
            judge_provider = build_provider(
                j.provider, model=j.model, base_url=j.base_url, region=j.region,
                timeout_s=args.timeout, retries=args.retries,
            )
        except (ValueError, ProviderError) as e:
            print(f"Error: judge: {redact(str(e))}", file=sys.stderr)
            return 1

    guardrail, guardrail_skip = None, "disabled with --no-guardrails"
    if suite.guardrail_examples and not args.no_guardrails:
        guardrail, guardrail_skip = guardrail_from_env(args.region)

    embedder = MockEmbedder(dim=args.embed_dim) if provider_name == "mock" else None

    # Mock calls are free and deterministic, so an all-mock run is never cached.
    all_mock = provider_name == "mock" and (judge_provider is None or judge_provider.name == "mock")
    cache = None if (args.no_cache or all_mock) else ResultCache(args.cache_dir)
    budget = Budget(RunLimits(max_calls=args.max_calls, max_tokens=args.max_run_tokens, max_cost_usd=args.max_cost))
    if args.max_cost is not None:
        model = str(getattr(provider, "model", provider.name))
        if lookup_price(provider.name, model, provider_region(provider)) is None:
            print(
                f"Warning: no price for {provider.name}/{model}; --max-cost cannot be enforced. "
                "Use --max-calls or --max-run-tokens as well.",
                file=sys.stderr,
            )
    caller = ModelCaller(cache=cache, budget=budget)

    report = run_suite(
        suite, provider, embedder, caller=caller,
        judge_provider=judge_provider, guardrail=guardrail, guardrail_skip_reason=guardrail_skip,
    )

    reg = None
    if baseline is not None:
        tol = RegressionTolerance(
            pass_rate_drop=(
                suite.regression.pass_rate_drop if args.tolerance_pass_rate is None else args.tolerance_pass_rate
            ),
            mean_score_drop=suite.regression.mean_score_drop if args.tolerance_score is None else args.tolerance_score,
        )
        reg = evaluate_regression(report, baseline, tol)
        report.regression = asdict(reg)

    if args.verbose:
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            cached = " (cached)" if r.cached else ""
            print(redact(f"[{status}] {r.input[:60]} -> {r.detail}{cached}"))
        print()

    print(report.summary())
    if reg is not None:
        print(regression_summary(reg))

    if args.report:
        _write_report(args.report, report.to_json())

    if report.stopped_reason:
        return EXIT_STOPPED

    if args.update_baseline:
        if report.critical_failures:
            print("Baseline not written: critical cases fail; fix them first.", file=sys.stderr)
            return 1
        write_baseline(report, args.baseline)
        print(f"Baseline written: {args.baseline}")

    return 0 if (report.gate_passed and (reg is None or reg.passed)) else 1


if __name__ == "__main__":
    sys.exit(main())
