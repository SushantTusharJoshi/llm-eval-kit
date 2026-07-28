"""Tests for retrieval confidence checker."""

from llm_eval_kit.providers.mock import MockEmbedder
from llm_eval_kit.retrieval import check_retrieval


class TestRetrievalChecker:
    def test_identical_chunk_passes(self):
        embedder = MockEmbedder(dim=32)
        result = check_retrieval("what is Python?", ["what is Python?"], embedder, threshold=0.9)
        assert result.passed
        assert abs(result.avg_score - 1.0) < 1e-10
        assert len(result.chunk_scores) == 1

    def test_low_threshold_passes(self):
        embedder = MockEmbedder(dim=32)
        result = check_retrieval(
            "machine learning",
            ["deep learning", "neural networks"],
            embedder,
            threshold=0.0,
        )
        assert result.passed

    def test_high_threshold_may_fail(self):
        embedder = MockEmbedder(dim=32)
        result = check_retrieval(
            "quantum physics",
            ["cooking recipes", "gardening tips"],
            embedder,
            threshold=0.99,
        )
        assert not result.passed
        assert result.avg_score < 0.99

    def test_empty_chunks_fails(self):
        embedder = MockEmbedder(dim=32)
        result = check_retrieval("query", [], embedder)
        assert not result.passed
        assert result.avg_score == 0.0
        assert "No context chunks" in result.detail

    def test_multiple_chunks_averaged(self):
        embedder = MockEmbedder(dim=32)
        result = check_retrieval(
            "test query",
            ["chunk one", "chunk two", "chunk three"],
            embedder,
            threshold=0.0,
        )
        assert len(result.chunk_scores) == 3
        expected_avg = sum(result.chunk_scores) / 3
        assert abs(result.avg_score - expected_avg) < 1e-10

    def test_chunk_scores_are_bounded(self):
        embedder = MockEmbedder(dim=32)
        result = check_retrieval("test", ["a", "b", "c"], embedder, threshold=0.0)
        for score in result.chunk_scores:
            assert -1.0 <= score <= 1.0
