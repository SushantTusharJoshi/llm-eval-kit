"""Small dated price table for cost estimates.

Prices are USD per 1M tokens, checked on PRICES_CHECKED_ON. They are an
estimate for a CI report, not a bill. Anything not in the table reports
cost as None (null in JSON) rather than a guess.

Bedrock prices are for on-demand us-east-1 only; other regions report None.
Model IDs are matched after stripping a geo inference-profile prefix
(``us.``, ``eu.``, ``apac.``, ``global.``) and any ARN prefix. The IDs below
are the patterns this table expects; check the exact ID your account uses
in the Bedrock console.
"""

import re
from dataclasses import dataclass

PRICES_CHECKED_ON = "2026-11-27"


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


# (provider, region or None for any, model ID prefix after normalisation) -> price
_TABLE: list[tuple[str, str | None, str, Price]] = [
    ("groq", None, "openai/gpt-oss-120b", Price(0.15, 0.60)),
    ("bedrock", "us-east-1", "openai.gpt-oss-120b", Price(0.15, 0.60)),
    ("bedrock", "us-east-1", "amazon.nova-lite-v1", Price(0.06, 0.24)),
    ("bedrock", "us-east-1", "amazon.nova-micro-v1", Price(0.035, 0.14)),
]

_GEO_PREFIX = re.compile(r"^(us|eu|apac|global|us-gov|ca|jp|au)\.")


def normalise_model_id(model: str) -> str:
    model = model.rsplit("/", 1)[-1] if model.startswith("arn:") else model
    return _GEO_PREFIX.sub("", model)


def lookup_price(provider: str, model: str, region: str | None = None) -> Price | None:
    norm = normalise_model_id(model)
    for p_name, p_region, prefix, price in _TABLE:
        if p_name != provider:
            continue
        if p_region is not None and p_region != region:
            continue
        if norm == prefix or norm.startswith(prefix):
            return price
    return None


# What it does: turns token counts into an estimated USD cost, or None when the model or counts are unknown.
# Catch it: "Estimate what this eval run cost, and say what you do when you do not know the price."
# Card: docs/cards/cost-estimate.md
def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    region: str | None = None,
) -> float | None:
    """Estimated USD for one call, or None if the price or a token count is unknown."""
    # DECISION: no token counts from the API means no honest estimate
    if input_tokens is None or output_tokens is None:
        return None
    # STEP: find the dated price for this provider, model and region
    price = lookup_price(provider, model, region)
    # DECISION: unknown model reports null instead of a guess
    if price is None:
        return None
    # RETURN: tokens times price per million
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000
