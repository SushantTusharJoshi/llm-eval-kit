"""Shared test fixtures for llm-eval-kit."""

import pytest
import yaml

from llm_eval_kit.providers.base import Completion, LLMProvider


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



class StubProvider(LLMProvider):
    """New-style provider that reports token counts. Pretends to be Groq gpt-oss-120b so prices apply."""

    name = "groq"
    model = "openai/gpt-oss-120b"

    def __init__(self, responses=None, input_tokens=1000, output_tokens=500):
        self.responses = responses or {}
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = []

    def generate(self, prompt):
        return self.complete(prompt).text

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.0):
        self.calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        return Completion(
            text=self.responses.get(prompt, f"stub: {prompt}"),
            model=self.model,
            provider=self.name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=12.5,
        )
