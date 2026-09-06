"""Comparator registry — resolve comparator names to instances."""

from llm_eval_kit.comparators.cosine import CosineComparator
from llm_eval_kit.comparators.exact import CompareResult, ExactComparator
from llm_eval_kit.comparators.numeric import NumericComparator
from llm_eval_kit.config import GoldenExample
from llm_eval_kit.providers.base import Embedder


def get_comparator(
    example: GoldenExample, embedder: Embedder | None = None,
) -> ExactComparator | NumericComparator | CosineComparator:
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

    raise ValueError(f"Unknown comparator '{name}'. Available: cosine, exact, numeric")


__all__ = [
    "CompareResult",
    "CosineComparator",
    "ExactComparator",
    "NumericComparator",
    "get_comparator",
]
