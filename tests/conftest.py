"""Shared test fixtures for llm-eval-kit."""

import pytest
import yaml


@pytest.fixture
def tmp_path_factory_custom(tmp_path):
    return tmp_path


@pytest.fixture
def sample_suite_yaml(tmp_path):
    suite = {
        "name": "test-suite",
        "examples": [
            {
                "input": "What is 2+2?",
                "expected": "4",
                "comparator": "exact",
            },
            {
                "input": "Summarize gravity",
                "expected": "Gravity is a fundamental force",
                "comparator": "cosine",
                "threshold": 0.8,
            },
        ],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(suite))
    return path


@pytest.fixture
def sample_suite_json(tmp_path):
    import json

    suite = {
        "name": "json-suite",
        "examples": [
            {
                "input": "Capital of France?",
                "expected": "Paris",
                "comparator": "exact",
            }
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    return path
