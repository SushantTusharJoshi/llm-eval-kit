"""Tests for exact match comparator."""

from llm_eval_kit.comparators.exact import ExactComparator


class TestExactMatch:
    def test_identical_strings(self):
        c = ExactComparator()
        result = c.compare("Paris", "Paris")
        assert result.passed
        assert result.score == 1.0

    def test_mismatch(self):
        c = ExactComparator()
        result = c.compare("paris", "Paris")
        assert not result.passed
        assert result.score == 0.0

    def test_case_insensitive(self):
        c = ExactComparator(case_sensitive=False)
        result = c.compare("PARIS", "paris")
        assert result.passed

    def test_whitespace_normalization(self):
        c = ExactComparator(normalize_whitespace=True)
        result = c.compare("hello   world\n", "hello world")
        assert result.passed

    def test_strips_leading_trailing(self):
        c = ExactComparator()
        result = c.compare("  Paris  ", "Paris")
        assert result.passed

    def test_empty_strings_match(self):
        c = ExactComparator()
        result = c.compare("", "")
        assert result.passed
