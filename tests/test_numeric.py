"""Tests for numeric tolerance comparator."""

from llm_eval_kit.comparators.numeric import NumericComparator


class TestAbsoluteTolerance:
    def test_exact_numeric_match(self):
        c = NumericComparator(abs_tol=0.01)
        result = c.compare("42", "42")
        assert result.passed
        assert result.score == 1.0

    def test_within_tolerance(self):
        c = NumericComparator(abs_tol=0.5)
        result = c.compare("The answer is 3.14", "Expected: 3.0")
        assert result.passed
        assert result.score > 0.0

    def test_outside_tolerance(self):
        c = NumericComparator(abs_tol=0.01)
        result = c.compare("3.5", "4.0")
        assert not result.passed
        assert result.score == 0.0

    def test_negative_numbers(self):
        c = NumericComparator(abs_tol=0.1)
        result = c.compare("-2.5", "-2.55")
        assert result.passed


class TestRelativeTolerance:
    def test_within_relative_tolerance(self):
        c = NumericComparator(rel_tol=0.1)
        result = c.compare("105", "100")
        assert result.passed

    def test_outside_relative_tolerance(self):
        c = NumericComparator(rel_tol=0.01)
        result = c.compare("200", "100")
        assert not result.passed

    def test_zero_expected_skips_rel_check(self):
        c = NumericComparator(rel_tol=0.1)
        result = c.compare("0.5", "0")
        assert not result.passed


class TestEdgeCases:
    def test_no_number_in_actual(self):
        c = NumericComparator()
        result = c.compare("no numbers here", "42")
        assert not result.passed
        assert "No number found in output" in result.detail

    def test_no_number_in_expected(self):
        c = NumericComparator()
        result = c.compare("42", "no numbers")
        assert not result.passed
        assert "No number found in expected" in result.detail

    def test_default_tolerance(self):
        c = NumericComparator()
        result = c.compare("1.005", "1.0")
        assert result.passed

    def test_numbers_in_surrounding_text(self):
        c = NumericComparator(abs_tol=1.0)
        result = c.compare("The model scored 95.5 on the test", "Target score: 95.0")
        assert result.passed
