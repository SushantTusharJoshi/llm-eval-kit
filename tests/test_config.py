"""Tests for Pydantic configuration models."""

import pytest
from pydantic import ValidationError

from llm_eval_kit.config import EvalSuite, GoldenExample


class TestGoldenExample:
    def test_valid_construction(self):
        ex = GoldenExample(input="What is 2+2?", expected="4")
        assert ex.input == "What is 2+2?"
        assert ex.expected == "4"

    def test_defaults(self):
        ex = GoldenExample(input="q", expected="a")
        assert ex.comparator == "exact"
        assert ex.threshold == 0.85
        assert ex.case_sensitive is True
        assert ex.normalize_whitespace is False
        assert ex.validators == []
        assert ex.tags == []
        assert ex.abs_tol is None
        assert ex.rel_tol is None

    def test_custom_fields(self):
        ex = GoldenExample(
            input="q",
            expected="a",
            comparator="cosine",
            threshold=0.9,
            case_sensitive=False,
            normalize_whitespace=True,
            tags=["smoke"],
        )
        assert ex.comparator == "cosine"
        assert ex.threshold == 0.9
        assert ex.case_sensitive is False
        assert ex.normalize_whitespace is True
        assert ex.tags == ["smoke"]

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            GoldenExample(input="q")  # missing expected

        with pytest.raises(ValidationError):
            GoldenExample(expected="a")  # missing input

    def test_model_fields_set_tracks_explicit(self):
        ex = GoldenExample(input="q", expected="a")
        assert "comparator" not in ex.model_fields_set
        assert "threshold" not in ex.model_fields_set

        ex2 = GoldenExample(input="q", expected="a", comparator="exact")
        assert "comparator" in ex2.model_fields_set


class TestEvalSuite:
    def test_valid_construction(self):
        suite = EvalSuite(name="my-suite")
        assert suite.name == "my-suite"

    def test_defaults(self):
        suite = EvalSuite()
        assert suite.name == "default"
        assert suite.description == ""
        assert suite.examples == []
        assert suite.retrieval_examples == []
        assert suite.default_comparator == "exact"
        assert suite.default_threshold == 0.85
        assert suite.pass_rate == 1.0

    def test_with_examples(self):
        suite = EvalSuite(
            name="test",
            examples=[
                GoldenExample(input="q1", expected="a1"),
                GoldenExample(input="q2", expected="a2", comparator="cosine"),
            ],
            default_comparator="cosine",
            default_threshold=0.9,
            pass_rate=0.8,
        )
        assert len(suite.examples) == 2
        assert suite.default_comparator == "cosine"
        assert suite.default_threshold == 0.9
        assert suite.pass_rate == 0.8

    def test_invalid_example_in_suite(self):
        with pytest.raises(ValidationError):
            EvalSuite(
                name="bad",
                examples=[{"input": "q"}],  # missing expected
            )
