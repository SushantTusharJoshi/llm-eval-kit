"""Mock implementations for testing without API keys or network access."""

import hashlib
import math

from llm_eval_kit.providers.base import Embedder, LLMProvider


class MockProvider(LLMProvider):
    """Returns canned responses from a dict. Falls back to echoing the input."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}

    def generate(self, prompt: str) -> str:
        if prompt in self._responses:
            return self._responses[prompt]
        return f"Mock response to: {prompt}"


class MockEmbedder(Embedder):
    """Produces deterministic embeddings by hashing text into a fixed-dim vector.

    Useful for testing cosine comparator logic without a real embedding model.
    Similar strings produce similar (but not identical) vectors.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        raw = [int(h[i : i + 2], 16) / 255.0 for i in range(0, min(len(h), self._dim * 2), 2)]
        while len(raw) < self._dim:
            raw.append(0.0)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]
        return raw
