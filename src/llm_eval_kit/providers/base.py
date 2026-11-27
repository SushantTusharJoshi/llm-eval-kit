"""Abstract interfaces for LLM providers and embedding models.

Users implement these to wire their own models (Claude, OpenAI, local).
The eval runner depends only on these interfaces, never on a specific SDK.

Two ways to implement a provider:

* Old style: implement ``generate(prompt) -> str`` only. The default
  ``complete()`` wraps it and reports no token counts.
* New style: implement ``complete()`` and return a ``Completion`` with the
  token counts the API reported, then make ``generate()`` return
  ``self.complete(prompt).text``.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Completion:
    """One model response plus the accounting the report needs."""

    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cached: bool = False


class ProviderError(Exception):
    """A provider call failed. ``retryable`` marks throttles, timeouts and 5xx."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class LLMProvider(ABC):
    """Generate text from a prompt."""

    name: str = "custom"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the text response."""
        ...

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        """Return a Completion. The default wraps ``generate()`` for old providers.

        Old providers cannot take a separate system prompt, so it is prepended
        to the prompt. ``max_tokens`` and ``temperature`` are not passed through
        because ``generate()`` has no way to receive them.
        """
        full_prompt = prompt if system is None else f"{system}\n\n{prompt}"
        start = time.perf_counter()
        text = self.generate(full_prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        return Completion(
            text=text,
            model=str(getattr(self, "model", self.name)),
            provider=self.name,
            latency_ms=latency_ms,
        )


class Embedder(ABC):
    """Produce embedding vectors for text. Used by cosine comparator and retrieval checks."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...
