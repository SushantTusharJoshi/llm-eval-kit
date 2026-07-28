"""Tests for YAML/JSON suite loader."""

import pytest
import yaml

from llm_eval_kit.config import EvalSuite
from llm_eval_kit.loader import load_suite


class TestLoadYAML:
    def test_loads_valid_yaml(self, sample_suite_yaml):
        suite = load_suite(sample_suite_yaml)
        assert isinstance(suite, EvalSuite)
        assert suite.name == "test-suite"
        assert len(suite.examples) == 2

    def test_first_example_is_exact(self, sample_suite_yaml):
        suite = load_suite(sample_suite_yaml)
        ex = suite.examples[0]
        assert ex.input == "What is 2+2?"
        assert ex.expected == "4"
        assert ex.comparator == "exact"

    def test_second_example_has_threshold(self, sample_suite_yaml):
        suite = load_suite(sample_suite_yaml)
        ex = suite.examples[1]
        assert ex.comparator == "cosine"
        assert ex.threshold == 0.8


class TestLoadJSON:
    def test_loads_valid_json(self, sample_suite_json):
        suite = load_suite(sample_suite_json)
        assert suite.name == "json-suite"
        assert len(suite.examples) == 1
        assert suite.examples[0].expected == "Paris"


class TestLoadErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_suite(tmp_path / "nonexistent.yaml")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "suite.txt"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_suite(path)

    def test_malformed_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("[ this is not a mapping ]")
        with pytest.raises(ValueError, match="Expected a mapping"):
            load_suite(path)

    def test_missing_required_fields(self, tmp_path):
        path = tmp_path / "incomplete.yaml"
        path.write_text(yaml.dump({"examples": [{"input": "test"}]}))
        with pytest.raises((ValueError, TypeError)):
            load_suite(path)
