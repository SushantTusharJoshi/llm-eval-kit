"""The nightly drift suite and its committed baseline stay in step (no network)."""

from pathlib import Path

from llm_eval_kit.loader import load_suite
from llm_eval_kit.regression import evaluate_regression, load_baseline
from llm_eval_kit.runner import run_suite

from .conftest import StubProvider

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "examples" / "drift" / "groq_drift.yaml"
BASELINE = ROOT / "examples" / "baselines" / "groq_drift.json"
ANSWERS = {
    "arithmetic-sum": "42",
    "arithmetic-product": "144",
    "single-word-reply": "pong",
    "capital-city": "Paris",
}


def _run(overrides=None):
    suite = load_suite(SUITE)
    answers = {**ANSWERS, **(overrides or {})}
    by_input = {e.input: answers[e.id] for e in suite.examples}
    report = run_suite(suite, StubProvider(by_input))
    return suite, report, evaluate_regression(report, load_baseline(BASELINE), suite.regression)


def test_baseline_covers_every_case_with_the_same_critical_flag():
    suite, report, _ = _run()
    baseline = load_baseline(BASELINE)
    assert set(baseline["cases"]) == {e.id for e in suite.examples}
    assert {r.id: r.critical for r in report.results} == {k: v["critical"] for k, v in baseline["cases"].items()}


def test_expected_answers_match_baseline():
    _, report, reg = _run()
    assert report.gate_passed
    assert reg.passed, reg.reasons


def test_changed_critical_answer_is_drift():
    _, _, reg = _run({"arithmetic-sum": "41"})
    assert not reg.passed
    assert reg.critical_failures == ["arithmetic-sum"]


def test_changed_quality_answer_is_drift_with_zero_tolerance():
    _, _, reg = _run({"capital-city": "Lyon"})
    assert not reg.passed
    assert reg.newly_failing == ["capital-city"]
