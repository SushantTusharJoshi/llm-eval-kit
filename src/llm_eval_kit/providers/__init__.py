"""Provider interfaces and built-in implementations.

The OpenAI-compatible and Bedrock providers are imported from their own
modules (``providers.openai_compat``, ``providers.bedrock``) so that importing
this package never needs boto3.
"""

from llm_eval_kit.providers.base import Completion, Embedder, LLMProvider, ProviderError
from llm_eval_kit.providers.factory import build_provider
from llm_eval_kit.providers.mock import MockEmbedder, MockProvider

__all__ = [
    "Completion",
    "Embedder",
    "LLMProvider",
    "MockEmbedder",
    "MockProvider",
    "ProviderError",
    "build_provider",
]
