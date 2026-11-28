"""Bounded retries and per-run limits (calls, tokens, USD)."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from llm_eval_kit.providers.base import Completion, ProviderError

T = TypeVar("T")


# What it does: calls a function, retrying only retryable failures with capped exponential backoff.
# Catch it: "Your provider returns 429 under load. How do you retry without hammering it or looping forever?"
# Card: docs/cards/retry-backoff.md
MAX_RETRY_AFTER_S = 60.0  # never wait longer than this on a provider's say-so


def call_with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 2,
    backoff_s: float = 0.5,
    max_backoff_s: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``fn``; on a retryable ProviderError wait and try again, at most ``retries`` extra times."""
    attempt = 0
    # LOOP: one pass per attempt, bounded by retries + 1
    while True:
        try:
            # STEP: try the call
            return fn()
        except ProviderError as err:
            # DECISION: permanent error or out of attempts, so give up and raise it
            if not err.retryable or attempt >= retries:
                raise
            # STEP: wait backoff * 2^attempt (capped), or longer if the provider sent Retry-After
            wait = min(max_backoff_s, backoff_s * (2**attempt))
            asked = getattr(err, "retry_after", None)
            if asked is not None:
                wait = max(wait, min(asked, MAX_RETRY_AFTER_S))
            sleep(wait)
            attempt += 1


class BudgetExceeded(Exception):
    """A run limit was reached; the runner stops making calls."""


@dataclass
class RunLimits:
    max_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None


class Budget:
    """Tracks live (non cached) calls against RunLimits.

    Limits are checked before each call, so a single call can push the total
    past a limit by at most that one call. The next call is then refused.
    """

    def __init__(self, limits: RunLimits | None = None):
        self.limits = limits or RunLimits()
        self.calls = 0
        self.tokens = 0
        self.cost_usd = 0.0

    def check(self) -> None:
        lim = self.limits
        if lim.max_calls is not None and self.calls >= lim.max_calls:
            raise BudgetExceeded(f"max calls reached ({self.calls}/{lim.max_calls})")
        if lim.max_tokens is not None and self.tokens >= lim.max_tokens:
            raise BudgetExceeded(f"max run tokens reached ({self.tokens}/{lim.max_tokens})")
        if lim.max_cost_usd is not None and self.cost_usd >= lim.max_cost_usd:
            raise BudgetExceeded(f"spend cap reached (${self.cost_usd:.6f} >= ${lim.max_cost_usd:.6f})")

    def record(self, completion: Completion, cost_usd: float | None) -> None:
        self.calls += 1
        self.tokens += (completion.input_tokens or 0) + (completion.output_tokens or 0)
        if cost_usd is not None:
            self.cost_usd += cost_usd
