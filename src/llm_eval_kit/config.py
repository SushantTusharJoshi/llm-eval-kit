"""Pydantic models for eval suite configuration.

Validates YAML/JSON structure at load time so bad configs fail fast
with clear error messages instead of crashing mid-eval.
"""

import hashlib
import re

from pydantic import BaseModel, Field, field_validator

MAX_REGEX_PATTERN_LEN = 500


class ValidatorConfig(BaseModel):
    """Configuration for a structural validator applied to LLM output."""

    type: str
    keys: list[str] = Field(default_factory=list)
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    invert: bool = False

    @field_validator("pattern")
    @classmethod
    def _pattern_is_safe(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) > MAX_REGEX_PATTERN_LEN:
            raise ValueError(f"regex pattern longer than {MAX_REGEX_PATTERN_LEN} characters")
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"invalid regex pattern: {e}") from None
        return v


class GoldenExample(BaseModel):
    """A single input/expected-output pair with comparison settings."""

    id: str | None = None
    input: str
    expected: str
    comparator: str = "exact"
    threshold: float = 0.85
    abs_tol: float | None = None
    rel_tol: float | None = None
    case_sensitive: bool = True
    normalize_whitespace: bool = False
    validators: list[ValidatorConfig] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def example_id(example: GoldenExample) -> str:
    """Stable case ID: the explicit ``id`` or a short hash of input and expected."""
    if example.id:
        return example.id
    digest = hashlib.sha256(f"{example.input}\x00{example.expected}".encode()).hexdigest()
    return f"case-{digest[:12]}"


class RetrievalExample(BaseModel):
    """A query with retrieved context chunks for confidence checking."""

    query: str
    context_chunks: list[str]
    threshold: float = 0.7


class EvalSuite(BaseModel):
    """Top-level eval suite loaded from YAML or JSON."""

    name: str = "default"
    description: str = ""
    examples: list[GoldenExample] = Field(default_factory=list)
    retrieval_examples: list[RetrievalExample] = Field(default_factory=list)
    default_comparator: str = "exact"
    default_threshold: float = 0.85
    pass_rate: float = 1.0
    # Settings for the model under test. temperature 0 lowers variance but does
    # not guarantee identical outputs across calls.
    system: str | None = None
    max_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
