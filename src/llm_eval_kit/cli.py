"""CLI entry point for llm-eval-kit.

Exit codes: 0 gate passed, 1 gate failed or error, 3 stopped early by a run
limit (--max-cost, --max-calls, --max-run-tokens).
"""

import argparse
import os
import sys
from pathlib import Path

from llm_eval_kit import __version__
from llm_eval_kit.cache import DEFAULT_CACHE_DIR, ResultCache
from llm_eval_kit.calls import ModelCaller, provider_region
from llm_eval_kit.loader import load_suite
from llm_eval_kit.pricing import lookup_price
from llm_eval_kit.providers.base import ProviderError
from llm_eval_kit.providers.factory import PROVIDER_NAMES, build_provider
from llm_eval_kit.providers.mock import MockEmbedder
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

    embedder = MockEmbedder(dim=args.embed_dim) if provider_name == "mock" else None

    # The mock provider is free and deterministic, so it is never cached.
    cache = None if (args.no_cache or provider_name == "mock") else ResultCache(args.cache_dir)
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

    report = run_suite(suite, provider, embedder, caller=caller)

    if args.verbose:
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            cached = " (cached)" if r.cached else ""
            print(redact(f"[{status}] {r.input[:60]} -> {r.detail}{cached}"))
        print()

    print(report.summary())

    if args.report:
        _write_report(args.report, report.to_json())

    if report.stopped_reason:
        return EXIT_STOPPED
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
