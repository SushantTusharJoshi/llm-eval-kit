"""Tests for structural validators."""

from llm_eval_kit.config import ValidatorConfig
from llm_eval_kit.validators.structural import (
    validate_json,
    validate_json_keys,
    validate_length,
    validate_regex,
)


class TestJsonValidator:
    def test_valid_json(self):
        cfg = ValidatorConfig(type="json")
        result = validate_json('{"key": "value"}', cfg)
        assert result.passed

    def test_invalid_json(self):
        cfg = ValidatorConfig(type="json")
        result = validate_json("not json at all", cfg)
        assert not result.passed

    def test_valid_json_array(self):
        cfg = ValidatorConfig(type="json")
        result = validate_json("[1, 2, 3]", cfg)
        assert result.passed

    def test_invert_valid_json(self):
        cfg = ValidatorConfig(type="json", invert=True)
        result = validate_json('{"key": "value"}', cfg)
        assert not result.passed


class TestJsonKeysValidator:
    def test_all_keys_present(self):
        cfg = ValidatorConfig(type="json_keys", keys=["name", "age"])
        result = validate_json_keys('{"name": "Alice", "age": 30}', cfg)
        assert result.passed

    def test_missing_key(self):
        cfg = ValidatorConfig(type="json_keys", keys=["name", "email"])
        result = validate_json_keys('{"name": "Alice"}', cfg)
        assert not result.passed
        assert "email" in result.detail

    def test_invalid_json_input(self):
        cfg = ValidatorConfig(type="json_keys", keys=["name"])
        result = validate_json_keys("not json", cfg)
        assert not result.passed

    def test_non_object_json(self):
        cfg = ValidatorConfig(type="json_keys", keys=["name"])
        result = validate_json_keys("[1, 2, 3]", cfg)
        assert not result.passed

    def test_invert_missing_key(self):
        cfg = ValidatorConfig(type="json_keys", keys=["missing"], invert=True)
        result = validate_json_keys('{"name": "Alice"}', cfg)
        assert result.passed


class TestLengthValidator:
    def test_within_bounds(self):
        cfg = ValidatorConfig(type="length", min_length=5, max_length=20)
        result = validate_length("hello world", cfg)
        assert result.passed

    def test_too_short(self):
        cfg = ValidatorConfig(type="length", min_length=10)
        result = validate_length("hi", cfg)
        assert not result.passed

    def test_too_long(self):
        cfg = ValidatorConfig(type="length", max_length=5)
        result = validate_length("this is way too long", cfg)
        assert not result.passed

    def test_exact_boundary(self):
        cfg = ValidatorConfig(type="length", min_length=5, max_length=5)
        result = validate_length("hello", cfg)
        assert result.passed

    def test_no_bounds_always_passes(self):
        cfg = ValidatorConfig(type="length")
        result = validate_length("anything", cfg)
        assert result.passed


class TestRegexValidator:
    def test_pattern_matches(self):
        cfg = ValidatorConfig(type="regex", pattern=r"\d{3}-\d{4}")
        result = validate_regex("Call 555-1234 today", cfg)
        assert result.passed

    def test_pattern_no_match(self):
        cfg = ValidatorConfig(type="regex", pattern=r"\d{3}-\d{4}")
        result = validate_regex("no phone number here", cfg)
        assert not result.passed

    def test_no_pattern_skips(self):
        cfg = ValidatorConfig(type="regex")
        result = validate_regex("anything", cfg)
        assert result.passed

    def test_invert_match(self):
        cfg = ValidatorConfig(type="regex", pattern=r"error", invert=True)
        result = validate_regex("all good", cfg)
        assert result.passed

    def test_invert_match_fails(self):
        cfg = ValidatorConfig(type="regex", pattern=r"error", invert=True)
        result = validate_regex("an error occurred", cfg)
        assert not result.passed
