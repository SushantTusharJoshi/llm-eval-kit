"""Pydantic models for eval suite configuration.

Validates YAML/JSON structure at load time so bad configs fail fast
with clear error messages instead of crashing mid-eval.
"""

from pydantic import BaseModel, Field


class ValidatorConfig(BaseModel):
    """Configuration for a structural validator applied to LLM output."""

    type: str
    keys: list[str] = Field(default_factory=list)
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    invert: bool = False


class GoldenExample(BaseModel):
    """A single input/expected-output pair with comparison settings."""

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
