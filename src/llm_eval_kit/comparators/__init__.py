"""Comparator registry: resolve comparator names to instances."""

from llm_eval_kit.comparators.cosine import CosineComparator
from llm_eval_kit.comparators.exact import CompareResult, ExactComparator
from llm_eval_kit.comparators.judge import JudgeComparator, JudgeContext
from llm_eval_kit.comparators.numeric import NumericComparator
from llm_eval_kit.config import GoldenExample
from llm_eval_kit.providers.base import Embedder


def get_comparator(
    example: GoldenExample,
    embedder: Embedder | None = None,
    judge: JudgeContext | None = None,
) -> ExactComparator | NumericComparator | CosineComparator | JudgeComparator:
    name = example.comparator

    if name == "exact":
        return ExactComparator(
            case_sensitive=example.case_sensitive,
            normalize_whitespace=example.normalize_whitespace,
        )

    if name == "numeric":
        return NumericComparator(
            abs_tol=example.abs_tol,
            rel_tol=example.rel_tol,
        )

    if name == "cosine":
        if embedder is None:
            raise ValueError("Cosine comparator requires an Embedder instance")
        return CosineComparator(embedder, threshold=example.threshold)

    if name == "judge":
        if judge is None:
            raise ValueError("Judge comparator requires a judge provider (add a judge: block to the suite)")
        rubric = example.rubric or judge.config.rubric
        if not rubric:
            raise ValueError("Judge comparator requires a rubric on the case or in the judge: block")
        return JudgeComparator(judge, rubric=rubric, question=example.input)

    raise ValueError(f"Unknown comparator '{name}'. Available: cosine, exact, judge, numeric")


__all__ = [
    "CompareResult",
    "CosineComparator",
    "ExactComparator",
    "JudgeComparator",
    "JudgeContext",
    "NumericComparator",
    "get_comparator",
]
