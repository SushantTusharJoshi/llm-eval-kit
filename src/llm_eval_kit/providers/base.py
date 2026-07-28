"""Abstract interfaces for LLM providers and embedding models.

Users implement these to wire their own models (Claude, OpenAI, local).
The eval runner depends only on these interfaces, never on a specific SDK.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Generate text from a prompt. One method to implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the text response."""
        ...


class Embedder(ABC):
    """Produce embedding vectors for text. Used by cosine comparator and retrieval checks."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...
