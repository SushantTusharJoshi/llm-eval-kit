"""CLI entry point for llm-eval-kit."""

import argparse
import sys

from llm_eval_kit import __version__
from llm_eval_kit.loader import load_suite
from llm_eval_kit.providers.mock import MockEmbedder, MockProvider
from llm_eval_kit.runner import run_suite


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
        help="Use mock provider and embedder for testing",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        suite = load_suite(args.suite)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.mock:
        provider = MockProvider()
        embedder = MockEmbedder(dim=args.embed_dim)
    else:
        print("Error: only --mock provider is supported in v0.1.0", file=sys.stderr)
        print("Provide your own LLMProvider via the Python API", file=sys.stderr)
        return 1

    report = run_suite(suite, provider, embedder)

    if args.verbose:
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.input[:60]} -> {r.detail}")
        print()

    print(report.summary())

    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
