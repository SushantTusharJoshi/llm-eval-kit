"""Cache key, file cache, cost table, usage totals and run limits."""

import pytest

from llm_eval_kit.cache import ResultCache, make_cache_key
from llm_eval_kit.calls import ModelCaller
from llm_eval_kit.config import EvalSuite, GoldenExample
from llm_eval_kit.pricing import estimate_cost, lookup_price
from llm_eval_kit.providers.base import Completion, ProviderError
from llm_eval_kit.reliability import Budget, BudgetExceeded, RunLimits, call_with_retries
from llm_eval_kit.runner import run_suite
from tests.conftest import StubProvider

BASE = dict(
    provider="groq",
    model="openai/gpt-oss-120b",
    settings={"max_tokens": 512, "temperature": 0.0, "reasoning_effort": "low"},
    prompt="What is 2+2?",
    system="Answer briefly.",
    rubric="Score 1 if correct.",
    context={"reference": "4"},
)


class TestCacheKey:
    def test_same_parts_same_key(self):
        assert make_cache_key(**BASE) == make_cache_key(**dict(BASE))

    def test_settings_dict_order_does_not_matter(self):
        flipped = dict(BASE, settings={"reasoning_effort": "low", "temperature": 0.0, "max_tokens": 512})
        assert make_cache_key(**flipped) == make_cache_key(**BASE)

    @pytest.mark.parametrize(
        "part,value",
        [
            ("provider", "bedrock"),
            ("model", "openai/gpt-oss-20b"),
            ("settings", {"max_tokens": 256, "temperature": 0.0, "reasoning_effort": "low"}),
            ("settings", {"max_tokens": 512, "temperature": 0.7, "reasoning_effort": "low"}),
            ("settings", {"max_tokens": 512, "temperature": 0.0, "reasoning_effort": "high"}),
            ("prompt", "What is 2+3?"),
            ("system", "Answer at length."),
            ("system", None),
            ("rubric", "Score 1 if polite."),
            ("context", {"reference": "5"}),
        ],
    )
    def test_changing_any_part_misses(self, part, value):
        assert make_cache_key(**dict(BASE, **{part: value})) != make_cache_key(**BASE)


class TestResultCache:
    def test_round_trip_marks_cached(self, tmp_path):
        cache = ResultCache(tmp_path)
        c = Completion("hi", "m", "p", 3, 4, 10.0)
        cache.put("ab" * 32, c)
        got = cache.get("ab" * 32)
        assert got.cached is True and got.text == "hi" and got.input_tokens == 3

    def test_missing_and_corrupt_are_misses(self, tmp_path):
        cache = ResultCache(tmp_path)
        assert cache.get("cd" * 32) is None
        path = tmp_path / "cd" / ("cd" * 32 + ".json")
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        assert cache.get("cd" * 32) is None


class TestModelCaller:
    def test_second_call_hits_cache_and_skips_budget(self, tmp_path):
        p = StubProvider()
        caller = ModelCaller(cache=ResultCache(tmp_path), budget=Budget(RunLimits(max_calls=1)))
        first, _ = caller.call(p, "q")
        second, _ = caller.call(p, "q")  # max_calls=1 would refuse a live call
        assert len(p.calls) == 1
        assert first.cached is False and second.cached is True

    def test_changed_setting_misses_cache(self, tmp_path):
        p = StubProvider()
        caller = ModelCaller(cache=ResultCache(tmp_path))
        caller.call(p, "q", max_tokens=100)
        caller.call(p, "q", max_tokens=200)
        caller.call(p, "q", system="s")
        assert len(p.calls) == 3

    def test_no_cache_always_calls(self):
        p = StubProvider()
        caller = ModelCaller(cache=None)
        caller.call(p, "q")
        caller.call(p, "q")
        assert len(p.calls) == 2

    def test_cached_calls_excluded_from_live_totals(self, tmp_path):
        p = StubProvider(input_tokens=1000, output_tokens=500)
        caller = ModelCaller(cache=ResultCache(tmp_path))
        caller.call(p, "q")
        caller.call(p, "q")
        t = caller.totals()
        assert t.live_calls == 1 and t.cached_calls == 1
        assert t.live_input_tokens == 1000 and t.cached_input_tokens == 1000
        assert t.live_cost_usd == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1e6)
        assert t.live_latency_ms_total == pytest.approx(12.5)


class TestPricing:
    def test_groq_gpt_oss(self):
        assert estimate_cost("groq", "openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)

    def test_bedrock_ids_and_profiles(self):
        assert lookup_price("bedrock", "amazon.nova-lite-v1:0", "us-east-1").input_per_m == 0.06
        assert lookup_price("bedrock", "us.amazon.nova-micro-v1:0", "us-east-1").output_per_m == 0.14
        assert lookup_price("bedrock", "openai.gpt-oss-120b-1:0", "us-east-1").output_per_m == 0.60

    def test_unknown_is_none(self):
        assert estimate_cost("groq", "some-new-model", 10, 10) is None
        assert estimate_cost("bedrock", "amazon.nova-lite-v1:0", 10, 10, region="eu-west-1") is None
        assert estimate_cost("mock", "mock", 10, 10) is None
        assert estimate_cost("groq", "openai/gpt-oss-120b", None, 10) is None


class TestRetriesAndBudget:
    def test_non_retryable_raises_immediately(self):
        calls = []

        def fn():
            calls.append(1)
            raise ProviderError("bad", retryable=False)

        with pytest.raises(ProviderError):
            call_with_retries(fn, retries=5, sleep=lambda s: None)
        assert len(calls) == 1

    def test_backoff_is_capped(self):
        sleeps = []

        def fn():
            raise ProviderError("slow", retryable=True)

        with pytest.raises(ProviderError):
            call_with_retries(fn, retries=5, backoff_s=1.0, max_backoff_s=3.0, sleep=sleeps.append)
        assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0]

    def test_budget_limits(self):
        b = Budget(RunLimits(max_calls=2))
        c = Completion("x", "m", "p", 1, 1)
        b.check()
        b.record(c, None)
        b.check()
        b.record(c, None)
        with pytest.raises(BudgetExceeded, match="max calls"):
            b.check()

    def test_tiny_spend_cap_stops_run_cleanly(self):
        p = StubProvider(input_tokens=1000, output_tokens=500)  # about $0.00045 per call
        suite = EvalSuite(
            name="capped",
            examples=[GoldenExample(input=f"q{i}", expected=f"stub: q{i}") for i in range(5)],
        )
        caller = ModelCaller(budget=Budget(RunLimits(max_cost_usd=0.0001)))
        report = run_suite(suite, p, caller=caller)
        assert len(p.calls) == 1  # first call allowed, then the cap is reached
        assert report.stopped_reason and "spend cap" in report.stopped_reason
        assert report.results[0].passed
        assert [r.error for r in report.results[1:]] == ["not_run"] * 4
        assert not report.gate_passed
        assert report.usage.live_calls == 1

    def test_max_run_tokens(self):
        p = StubProvider(input_tokens=10, output_tokens=10)
        suite = EvalSuite(examples=[GoldenExample(input=f"q{i}", expected="x") for i in range(4)])
        report = run_suite(suite, p, caller=ModelCaller(budget=Budget(RunLimits(max_tokens=30))))
        assert len(p.calls) == 2
        assert "max run tokens" in report.stopped_reason


class TestReportUsage:
    def test_per_case_tokens_cost_latency(self):
        p = StubProvider(responses={"q": "a"})
        report = run_suite(EvalSuite(examples=[GoldenExample(input="q", expected="a")]), p)
        r = report.results[0]
        assert (r.input_tokens, r.output_tokens) == (1000, 500)
        assert r.cost_usd == pytest.approx(0.00045)
        assert r.latency_ms == 12.5 and r.cached is False
        d = report.to_dict()
        assert d["usage"]["live_cost_usd"] == pytest.approx(0.00045)
        assert d["prices_checked_on"] == "2026-11-27"
        assert "est. cost $0.000450" in report.summary()

    def test_unknown_model_cost_is_null(self):
        from llm_eval_kit.providers.mock import MockProvider

        report = run_suite(EvalSuite(examples=[GoldenExample(input="q", expected="x")]), MockProvider())
        assert report.results[0].cost_usd is None
        assert report.to_dict()["usage"]["live_cost_usd"] is None

    def test_provider_error_is_a_failure_not_a_crash(self):
        class Broken(StubProvider):
            def complete(self, prompt, **kw):
                raise ProviderError("groq: HTTP 500: boom")

        report = run_suite(EvalSuite(examples=[GoldenExample(input="q", expected="x")]), Broken())
        assert not report.gate_passed
        assert report.results[0].error == "provider_error"
