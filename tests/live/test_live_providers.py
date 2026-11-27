"""Smoke tests against real providers. They spend real (small) money.

Deselected by default (pyproject addopts ``-m 'not live'``). Run with
``pytest -m live``; each test also skips itself when its credentials are absent.
"""

import os

import pytest

from llm_eval_kit.calls import ModelCaller
from llm_eval_kit.config import EvalSuite, GoldenExample
from llm_eval_kit.reliability import Budget, RunLimits
from llm_eval_kit.runner import run_suite

pytestmark = pytest.mark.live


def _aws_credentials_present() -> bool:
    try:
        import boto3
    except ImportError:
        return False
    return boto3.Session().get_credentials() is not None


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_groq_smoke():
    from llm_eval_kit.providers.openai_compat import groq_provider

    provider = groq_provider()
    c = provider.complete("Reply with the single word: pong", max_tokens=256)
    assert c.text.strip()
    assert c.input_tokens and c.output_tokens
    suite = EvalSuite(
        name="groq-live",
        examples=[GoldenExample(input="What is 2+2? Reply with only the number.", expected="4", comparator="numeric")],
    )
    report = run_suite(suite, provider, caller=ModelCaller(budget=Budget(RunLimits(max_calls=3, max_cost_usd=0.01))))
    assert report.usage.live_calls == 1
    assert report.results[0].cost_usd is not None


@pytest.mark.skipif(
    not (os.environ.get("BEDROCK_MODEL_ID") and _aws_credentials_present()),
    reason="BEDROCK_MODEL_ID or AWS credentials not set",
)
def test_bedrock_smoke():
    from llm_eval_kit.providers.bedrock import BedrockConverseProvider

    provider = BedrockConverseProvider()
    c = provider.complete("Reply with the single word: pong", max_tokens=256)
    assert c.text.strip()
    assert c.input_tokens and c.output_tokens
