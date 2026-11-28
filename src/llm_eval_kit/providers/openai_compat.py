"""OpenAI-compatible chat completions provider over httpx.

Works with any server that speaks ``POST {base_url}/chat/completions``,
including Groq (``groq_provider()``), OpenAI and local servers such as vLLM.
"""

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from llm_eval_kit.providers.base import Completion, LLMProvider, ProviderError
from llm_eval_kit.reliability import call_with_retries
from llm_eval_kit.security import redact

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _retry_after_seconds(headers) -> float | None:
    """Seconds from a Retry-After header (numeric form only); None when absent or unparseable."""
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None


def is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in model


class OpenAICompatibleProvider(LLMProvider):
    """Chat completions over httpx with timeouts and bounded retries.

    The API key is read from ``api_key`` or the ``api_key_env`` variable and is
    only ever placed in the Authorization header; it is masked out of every
    error message this class raises.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        name: str = "openai",
        timeout_s: float = 60.0,
        retries: int = 2,
        backoff_s: float = 0.5,
        max_tokens_param: str = "max_tokens",
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"{name}: no API key; set the {api_key_env} environment variable")
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens_param = max_tokens_param
        self.extra_body = dict(extra_body or {})
        self._retries = retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._key = key
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r})"

    def reasoning_effort(self) -> str | None:
        return "low" if is_gpt_oss(self.model) else None

    def cache_settings(self) -> dict[str, Any]:
        """Provider settings that change the output, so they belong in the cache key."""
        return {
            "base_url": self.base_url,
            "max_tokens_param": self.max_tokens_param,
            "reasoning_effort": self.reasoning_effort(),
            "extra_body": self.extra_body,
        }

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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_param: max_tokens,
            "temperature": temperature,
        }
        effort = self.reasoning_effort()
        if effort:
            body["reasoning_effort"] = effort
        body.update(self.extra_body)

        start = time.perf_counter()
        data = call_with_retries(
            lambda: self._post(body), retries=self._retries, backoff_s=self._backoff_s, sleep=self._sleep
        )
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ProviderError(f"{self.name}: response has no choices[0].message") from e
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: request timed out ({type(e).__name__})", retryable=True) from None
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: network error ({type(e).__name__})", retryable=True) from None

        if resp.status_code >= 400:
            snippet = redact(resp.text[:300], [self._key])
            raise ProviderError(
                f"{self.name}: HTTP {resp.status_code}: {snippet}",
                retryable=resp.status_code in _RETRYABLE_STATUS,
                status=resp.status_code,
                retry_after=_retry_after_seconds(resp.headers),
            )
        try:
            return resp.json()
        except ValueError:
            raise ProviderError(f"{self.name}: response was not JSON") from None


def groq_provider(model: str | None = None, *, api_key: str | None = None, **kwargs: Any) -> OpenAICompatibleProvider:
    """Groq's OpenAI-compatible endpoint. Key from GROQ_API_KEY, model from GROQ_MODEL or the default."""
    return OpenAICompatibleProvider(
        model or os.environ.get("GROQ_MODEL") or GROQ_DEFAULT_MODEL,
        base_url=os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL),
        api_key=api_key,
        api_key_env="GROQ_API_KEY",
        name="groq",
        # Groq documents max_tokens as deprecated in favour of max_completion_tokens.
        max_tokens_param="max_completion_tokens",
        **kwargs,
    )
