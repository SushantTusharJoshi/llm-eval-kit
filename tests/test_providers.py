"""Provider tests: shared interface, OpenAI-compatible (httpx mock transport), Bedrock (fake client, Stubber)."""

import json

import httpx
import pytest

from llm_eval_kit.providers.base import Completion, LLMProvider, ProviderError
from llm_eval_kit.providers.bedrock import BedrockConverseProvider
from llm_eval_kit.providers.factory import build_provider
from llm_eval_kit.providers.mock import MockProvider
from llm_eval_kit.providers.openai_compat import OpenAICompatibleProvider, groq_provider

FAKE_KEY = "gsk_PLANTED_FAKE_KEY_0123456789abcdef"


class OldStyle(LLMProvider):
    def generate(self, prompt: str) -> str:
        return f"echo:{prompt}"


class TestSharedInterface:
    def test_old_provider_gets_complete_for_free(self):
        c = OldStyle().complete("hi")
        assert isinstance(c, Completion)
        assert c.text == "echo:hi"
        assert c.provider == "custom"
        assert c.input_tokens is None and c.output_tokens is None
        assert c.latency_ms is not None and c.cached is False

    def test_default_complete_prepends_system(self):
        assert OldStyle().complete("q", system="be brief").text == "echo:be brief\n\nq"

    def test_mock_generate_unchanged(self):
        assert MockProvider().generate("x") == "Mock response to: x"
        assert MockProvider().complete("x").provider == "mock"


def _ok_body(text="hello", prompt_tokens=11, completion_tokens=7):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": 18},
    }


def _transport(handler):
    return httpx.MockTransport(handler)


class TestOpenAICompatible:
    def test_groq_request_shape_and_usage(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
        seen = {}

        def handler(request: httpx.Request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        p = groq_provider(transport=_transport(handler))
        c = p.complete("ping", system="sys", max_tokens=64, temperature=0.0)

        assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert seen["auth"] == f"Bearer {FAKE_KEY}"
        body = seen["body"]
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["max_completion_tokens"] == 64
        assert body["reasoning_effort"] == "low"
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        assert c.text == "hello"
        assert (c.input_tokens, c.output_tokens) == (11, 7)
        assert c.provider == "groq" and c.model == "openai/gpt-oss-120b"
        assert p.generate("ping") == "hello"

    def test_no_reasoning_effort_for_other_models(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_body())

        p = OpenAICompatibleProvider("llama-3.1-8b", api_key=FAKE_KEY, transport=_transport(handler))
        p.complete("x")
        assert "reasoning_effort" not in seen["body"]
        assert seen["body"]["max_tokens"] == 512

    def test_retries_429_then_succeeds(self):
        attempts = []
        sleeps = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(429, json={"error": "slow down"})
            return httpx.Response(200, json=_ok_body())

        p = OpenAICompatibleProvider(
            "m", api_key=FAKE_KEY, transport=_transport(handler), retries=2, backoff_s=0.1, sleep=sleeps.append
        )
        assert p.complete("x").text == "hello"
        assert len(attempts) == 3
        assert sleeps == [0.1, 0.2]

    def test_retries_are_bounded(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(503, text="unavailable")

        p = OpenAICompatibleProvider("m", api_key=FAKE_KEY, transport=_transport(handler), retries=2, sleep=lambda s: None)
        with pytest.raises(ProviderError) as ei:
            p.complete("x")
        assert len(attempts) == 3
        assert ei.value.status == 503

    def test_400_is_not_retried(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(400, json={"error": "bad request"})

        p = OpenAICompatibleProvider("m", api_key=FAKE_KEY, transport=_transport(handler), sleep=lambda s: None)
        with pytest.raises(ProviderError):
            p.complete("x")
        assert len(attempts) == 1

    def test_timeout_becomes_retryable_provider_error(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        p = OpenAICompatibleProvider("m", api_key=FAKE_KEY, transport=_transport(handler), retries=1, sleep=lambda s: None)
        with pytest.raises(ProviderError, match="timed out"):
            p.complete("x")

    def test_error_text_never_contains_key(self):
        def handler(request):
            # A hostile or buggy server echoing the credential back.
            return httpx.Response(401, text=f"invalid key {request.headers['authorization']}")

        p = OpenAICompatibleProvider("m", api_key=FAKE_KEY, transport=_transport(handler))
        with pytest.raises(ProviderError) as ei:
            p.complete("x")
        assert FAKE_KEY not in str(ei.value)
        assert FAKE_KEY not in repr(p)

    def test_missing_key_message_names_env_not_value(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            groq_provider()

    def test_malformed_body_is_error(self):
        p = OpenAICompatibleProvider(
            "m", api_key=FAKE_KEY, transport=_transport(lambda r: httpx.Response(200, json={"nope": 1}))
        )
        with pytest.raises(ProviderError, match="no choices"):
            p.complete("x")


class FakeBedrockClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


CONVERSE_OK = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"reasoningContent": {"reasoningText": {"text": "thinking"}}}, {"text": "Paris"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 21, "outputTokens": 3, "totalTokens": 24},
    "metrics": {"latencyMs": 120},
}


class TestBedrock:
    def test_converse_request_and_usage(self):
        client = FakeBedrockClient(CONVERSE_OK)
        p = BedrockConverseProvider("amazon.nova-lite-v1:0", "us-east-1", client=client)
        c = p.complete("Capital of France?", system="one word", max_tokens=32, temperature=0.0)

        assert client.kwargs["modelId"] == "amazon.nova-lite-v1:0"
        assert client.kwargs["system"] == [{"text": "one word"}]
        assert client.kwargs["inferenceConfig"] == {"maxTokens": 32, "temperature": 0.0}
        assert client.kwargs["messages"][0]["content"] == [{"text": "Capital of France?"}]
        assert c.text == "Paris"  # reasoning block skipped
        assert (c.input_tokens, c.output_tokens) == (21, 3)
        assert c.provider == "bedrock"

    def test_model_id_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
        p = BedrockConverseProvider(client=FakeBedrockClient(CONVERSE_OK), region="us-east-1")
        assert p.model == "amazon.nova-micro-v1:0"

    def test_missing_model_id(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        with pytest.raises(ValueError, match="BEDROCK_MODEL_ID"):
            BedrockConverseProvider(client=FakeBedrockClient(CONVERSE_OK))

    def test_client_error_is_redacted_provider_error(self, monkeypatch):
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "PLANTEDsecretVALUE123456")
        err = RuntimeError("AccessDenied for PLANTEDsecretVALUE123456")
        p = BedrockConverseProvider("m", "us-east-1", client=FakeBedrockClient(error=err))
        with pytest.raises(ProviderError) as ei:
            p.complete("x")
        assert "PLANTEDsecretVALUE123456" not in str(ei.value)

    def test_with_botocore_stubber(self):
        boto3 = pytest.importorskip("boto3")
        from botocore.stub import Stubber

        client = boto3.client(
            "bedrock-runtime", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
        )
        with Stubber(client) as stub:
            stub.add_response(
                "converse",
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "4"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                    "metrics": {"latencyMs": 50},
                },
                {
                    "modelId": "openai.gpt-oss-120b-1:0",
                    "messages": [{"role": "user", "content": [{"text": "2+2?"}]}],
                    "inferenceConfig": {"maxTokens": 512, "temperature": 0.0},
                },
            )
            p = BedrockConverseProvider("openai.gpt-oss-120b-1:0", "us-east-1", client=client)
            c = p.complete("2+2?")
        assert c.text == "4" and c.input_tokens == 5 and c.output_tokens == 1


class TestFactory:
    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            build_provider("nope")

    def test_openai_needs_model(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        with pytest.raises(ValueError, match="OPENAI_MODEL"):
            build_provider("openai")
