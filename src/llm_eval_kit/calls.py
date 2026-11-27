"""One door for every model call: cache, run limits, cost and a ledger.

The runner and the LLM judge both call models through ``ModelCaller`` so
caching, the spend cap and the usage totals treat them the same way.
"""

from dataclasses import dataclass, field
from typing import Any

from llm_eval_kit.cache import ResultCache, make_cache_key
from llm_eval_kit.pricing import estimate_cost
from llm_eval_kit.providers.base import Completion, LLMProvider
from llm_eval_kit.reliability import Budget


@dataclass
class CallRecord:
    role: str  # "model" for the model under test, "judge" for the LLM judge
    completion: Completion
    cost_usd: float | None


@dataclass
class UsageTotals:
    live_calls: int = 0
    live_input_tokens: int = 0
    live_output_tokens: int = 0
    live_cost_usd: float | None = None
    live_unpriced_calls: int = 0
    live_latency_ms_total: float = 0.0
    live_latency_ms_mean: float | None = None
    cached_calls: int = 0
    cached_input_tokens: int = 0
    cached_output_tokens: int = 0
    cached_cost_usd: float | None = None  # what the cached calls cost when they first ran
    by_role: dict[str, dict[str, Any]] = field(default_factory=dict)


def provider_region(provider: LLMProvider) -> str | None:
    region = getattr(provider, "region", None)
    return region if isinstance(region, str) else None


def provider_settings(provider: LLMProvider) -> dict[str, Any]:
    fn = getattr(provider, "cache_settings", None)
    return dict(fn()) if callable(fn) else {}


class ModelCaller:
    def __init__(self, cache: ResultCache | None = None, budget: Budget | None = None):
        self.cache = cache
        self.budget = budget or Budget()
        self.ledger: list[CallRecord] = []

    # What it does: makes one model call through the cache and the run budget, and records its cost.
    # Catch it: "Add a spend cap to a batch job that calls a paid API. Where do you check it, and do cache hits count?"
    # Card: docs/cards/model-call.md
    def call(
        self,
        provider: LLMProvider,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        rubric: str | None = None,
        context: Any = None,
        role: str = "model",
    ) -> tuple[Completion, float | None]:
        """Return (completion, estimated cost). Raises BudgetExceeded before a live call over a limit."""
        model = str(getattr(provider, "model", provider.name))
        region = provider_region(provider)
        key = None
        # DECISION: cache on and an entry exists, so return it without touching the budget
        if self.cache is not None:
            settings = {"max_tokens": max_tokens, "temperature": temperature, **provider_settings(provider)}
            key = make_cache_key(
                provider=provider.name, model=model, settings=settings,
                prompt=prompt, system=system, rubric=rubric, context=context,
            )
            hit = self.cache.get(key)
            if hit is not None:
                cost = estimate_cost(hit.provider, hit.model, hit.input_tokens, hit.output_tokens, region)
                self.ledger.append(CallRecord(role, hit, cost))
                return hit, cost
        # STEP: refuse the call if a run limit is already reached
        self.budget.check()
        # STEP: make the live call
        completion = provider.complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        # STEP: price it and charge it to the budget
        cost = estimate_cost(completion.provider, completion.model, completion.input_tokens,
                             completion.output_tokens, region)
        self.budget.record(completion, cost)
        self.ledger.append(CallRecord(role, completion, cost))
        # STEP: store it for next time
        if self.cache is not None and key is not None:
            self.cache.put(key, completion)
        # RETURN: the completion and its estimated cost
        return completion, cost

    def totals(self) -> UsageTotals:
        t = UsageTotals()
        live_cost = cached_cost = 0.0
        live_priced = cached_priced = 0
        for rec in self.ledger:
            c = rec.completion
            role = t.by_role.setdefault(rec.role, {"live_calls": 0, "cached_calls": 0, "cost_usd": 0.0})
            if c.cached:
                t.cached_calls += 1
                t.cached_input_tokens += c.input_tokens or 0
                t.cached_output_tokens += c.output_tokens or 0
                role["cached_calls"] += 1
                if rec.cost_usd is not None:
                    cached_cost += rec.cost_usd
                    cached_priced += 1
                continue
            t.live_calls += 1
            t.live_input_tokens += c.input_tokens or 0
            t.live_output_tokens += c.output_tokens or 0
            t.live_latency_ms_total += c.latency_ms or 0.0
            role["live_calls"] += 1
            if rec.cost_usd is None:
                t.live_unpriced_calls += 1
            else:
                live_cost += rec.cost_usd
                live_priced += 1
                role["cost_usd"] += rec.cost_usd
        t.live_cost_usd = live_cost if live_priced else None
        t.cached_cost_usd = cached_cost if cached_priced else None
        t.live_latency_ms_mean = t.live_latency_ms_total / t.live_calls if t.live_calls else None
        return t
