"""A throttle that says Retry-After is waited out (capped), not retried after half a second."""
from llm_eval_kit.providers.base import ProviderError
from llm_eval_kit.reliability import MAX_RETRY_AFTER_S, call_with_retries


def _flaky(n_fail, retry_after):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] <= n_fail:
            raise ProviderError("HTTP 429", retryable=True, status=429, retry_after=retry_after)
        return "ok"
    return fn


def test_retry_after_is_honoured():
    waits = []
    assert call_with_retries(_flaky(1, 12.0), retries=2, sleep=waits.append) == "ok"
    assert waits == [12.0]


def test_retry_after_is_capped():
    waits = []
    call_with_retries(_flaky(1, 999.0), retries=2, sleep=waits.append)
    assert waits == [MAX_RETRY_AFTER_S]


def test_without_retry_after_backoff_is_unchanged():
    waits = []
    call_with_retries(_flaky(2, None), retries=2, backoff_s=0.5, sleep=waits.append)
    assert waits == [0.5, 1.0]
