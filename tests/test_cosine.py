"""Tests for cosine similarity comparator using mock embedder."""

from llm_eval_kit.comparators.cosine import CosineComparator, _cosine_similarity
from llm_eval_kit.providers.mock import MockEmbedder


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestCosineComparator:
    def test_identical_text_passes(self):
        embedder = MockEmbedder(dim=32)
        c = CosineComparator(embedder, threshold=0.9)
        result = c.compare("hello world", "hello world")
        assert result.passed
        assert result.score == 1.0

    def test_similar_text_above_threshold(self):
        embedder = MockEmbedder(dim=32)
        c = CosineComparator(embedder, threshold=0.1)
        result = c.compare("The capital of France is Paris", "Paris is the capital of France")
        assert result.passed
        assert result.score >= 0.1

    def test_different_text_below_high_threshold(self):
        embedder = MockEmbedder(dim=32)
        c = CosineComparator(embedder, threshold=0.99)
        result = c.compare("apples", "quantum mechanics")
        assert not result.passed
        assert result.score < 0.99

    def test_score_in_result(self):
        embedder = MockEmbedder(dim=32)
        c = CosineComparator(embedder, threshold=0.5)
        result = c.compare("test input", "test output")
        assert 0.0 <= result.score <= 1.0
        assert "Cosine similarity" in result.detail

    def test_low_threshold_passes_most(self):
        embedder = MockEmbedder(dim=32)
        c = CosineComparator(embedder, threshold=0.0)
        result = c.compare("anything", "something else")
        assert result.passed
