"""Amazon Bedrock Converse provider over boto3 (optional extra ``[bedrock]``).

boto3 is imported only when a client is actually created, so the core
package works without it. Retries and timeouts use botocore's own
``standard`` retry mode, which backs off on throttling and 5xx errors.
"""

import os
import time
from typing import Any

from llm_eval_kit.providers.base import Completion, LLMProvider, ProviderError
from llm_eval_kit.security import redact

DEFAULT_REGION = "us-east-1"


def default_region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION


def make_aws_client(service: str, region: str, *, timeout_s: float = 60.0, retries: int = 2) -> Any:
    """Create a boto3 client lazily. Raises ProviderError if boto3 is missing."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ProviderError("boto3 is not installed; run: pip install 'llm-eval-kit[bedrock]'") from None
    config = Config(
        connect_timeout=min(timeout_s, 10.0),
        read_timeout=timeout_s,
        retries={"max_attempts": retries + 1, "mode": "standard"},
    )
    return boto3.client(service, region_name=region, config=config)


class BedrockConverseProvider(LLMProvider):
    """Calls ``bedrock-runtime`` Converse. Model ID from ``model_id`` or BEDROCK_MODEL_ID."""

    name = "bedrock"

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        *,
        client: Any = None,
        timeout_s: float = 60.0,
        retries: int = 2,
    ):
        model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not model_id:
            raise ValueError("bedrock: no model ID; pass --model or set BEDROCK_MODEL_ID")
        self.model = model_id
        self.region = region or default_region()
        self._client = client or make_aws_client("bedrock-runtime", self.region, timeout_s=timeout_s, retries=retries)

    def __repr__(self) -> str:
        return f"BedrockConverseProvider(model={self.model!r}, region={self.region!r})"

    def cache_settings(self) -> dict[str, Any]:
        return {"region": self.region}

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        start = time.perf_counter()
        try:
            resp = self._client.converse(**kwargs)
        except Exception as e:  # botocore raises many exception types; report them uniformly
            raise ProviderError(f"bedrock: converse failed: {type(e).__name__}: {redact(str(e))[:300]}") from None
        latency_ms = (time.perf_counter() - start) * 1000

        blocks = (((resp.get("output") or {}).get("message") or {}).get("content")) or []
        # Some models also return reasoningContent blocks; only text blocks are the answer.
        text = "".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b)
        usage = resp.get("usage") or {}
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            latency_ms=latency_ms,
        )
