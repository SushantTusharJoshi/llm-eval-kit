"""Build a provider from a name, as the CLI and suite ``judge:`` blocks do."""

import os
from typing import Any

from llm_eval_kit.providers.base import LLMProvider
from llm_eval_kit.providers.mock import MockProvider

PROVIDER_NAMES = ("mock", "openai", "groq", "bedrock")


def build_provider(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    region: str | None = None,
    timeout_s: float = 60.0,
    retries: int = 2,
    responses: dict[str, str] | None = None,
) -> LLMProvider:
    """Raises ValueError for unknown names or missing settings (key, model ID)."""
    if name == "mock":
        return MockProvider(responses=responses)
    if name == "groq":
        from llm_eval_kit.providers.openai_compat import groq_provider

        return groq_provider(model, timeout_s=timeout_s, retries=retries)
    if name == "openai":
        from llm_eval_kit.providers.openai_compat import OpenAICompatibleProvider

        model = model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ValueError("openai: no model; pass --model or set OPENAI_MODEL")
        kwargs: dict[str, Any] = {"timeout_s": timeout_s, "retries": retries}
        return OpenAICompatibleProvider(
            model,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            **kwargs,
        )
    if name == "bedrock":
        from llm_eval_kit.providers.bedrock import BedrockConverseProvider

        return BedrockConverseProvider(model, region, timeout_s=timeout_s, retries=retries)
    raise ValueError(f"Unknown provider '{name}'. Available: {', '.join(PROVIDER_NAMES)}")
