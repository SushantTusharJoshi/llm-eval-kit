"""Cosine similarity comparator using embedding vectors."""

from llm_eval_kit.comparators.exact import CompareResult
from llm_eval_kit.math_utils import cosine_similarity as _cosine_similarity
from llm_eval_kit.providers.base import Embedder


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
