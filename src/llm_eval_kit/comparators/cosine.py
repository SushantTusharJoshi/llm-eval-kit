"""Cosine similarity comparator using embedding vectors."""

import math

from llm_eval_kit.comparators.exact import CompareResult
from llm_eval_kit.providers.base import Embedder


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CosineComparator:
    """Compare LLM output to expected via cosine similarity of embeddings."""

    def __init__(self, embedder: Embedder, threshold: float = 0.85):
        self._embedder = embedder
        self._threshold = threshold

    def compare(self, actual: str, expected: str) -> CompareResult:
        vecs = self._embedder.embed([actual, expected])
        score = _cosine_similarity(vecs[0], vecs[1])

        if score >= self._threshold:
            return CompareResult(
                passed=True,
                score=score,
                detail=f"Cosine similarity {score:.4f} >= threshold {self._threshold}",
            )

        return CompareResult(
            passed=False,
            score=score,
            detail=f"Cosine similarity {score:.4f} < threshold {self._threshold}",
        )
