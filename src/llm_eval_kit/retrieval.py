"""Retrieval confidence checker for RAG pipeline evaluation."""

from dataclasses import dataclass

from llm_eval_kit.math_utils import cosine_similarity as _cosine_similarity
from llm_eval_kit.providers.base import Embedder


@dataclass
class RetrievalResult:
    passed: bool
    avg_score: float
    chunk_scores: list[float]
    detail: str


def check_retrieval(
    query: str,
    context_chunks: list[str],
    embedder: Embedder,
    threshold: float = 0.7,
) -> RetrievalResult:
    if not context_chunks:
        return RetrievalResult(
            passed=False,
            avg_score=0.0,
            chunk_scores=[],
            detail="No context chunks provided",
        )

    all_texts = [query] + context_chunks
    embeddings = embedder.embed(all_texts)

    query_vec = embeddings[0]
    chunk_vecs = embeddings[1:]

    scores = [_cosine_similarity(query_vec, cv) for cv in chunk_vecs]
    avg = sum(scores) / len(scores)

    if avg >= threshold:
        return RetrievalResult(
            passed=True,
            avg_score=avg,
            chunk_scores=scores,
            detail=f"Avg retrieval score {avg:.4f} >= threshold {threshold}",
        )

    return RetrievalResult(
        passed=False,
        avg_score=avg,
        chunk_scores=scores,
        detail=f"Avg retrieval score {avg:.4f} < threshold {threshold}",
    )
