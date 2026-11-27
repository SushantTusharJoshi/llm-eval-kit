"""Regression gate: strict critical cases, tolerant quality, CLI exit codes."""

import json

import pytest
import yaml

from llm_eval_kit.cli import main
from llm_eval_kit.config import EvalSuite, GoldenExample, RegressionTolerance, is_critical
from llm_eval_kit.providers.mock import MockProvider
from llm_eval_kit.regression import baseline_from_report, evaluate_regression, load_baseline
from llm_eval_kit.runner import run_suite

ECHO = "Mock response to: "


def _case(cid, ok=True, **kw):
    q = f"q-{cid}"
    return {"id": cid, "input": q, "expected": ECHO + q if ok else "something else", **kw}


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.dump(data))
    return str(path)


def _baseline(tmp_path, cases):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"format": 1, "suite": "s", "cases": cases}))
    return str(path)


class TestCriticalFlag:
    def test_explicit_and_by_type(self):
        suite = EvalSuite()
        assert is_critical(GoldenExample(input="q", expected="a", critical=True), suite)
        assert is_critical(GoldenExample(input="q", expected="a", check_type="security"), suite)
        assert is_critical(GoldenExample(input="q", expected="a", check_type="schema"), suite)
        assert not is_critical(GoldenExample(input="q", expected="a", check_type="quality"), suite)
        assert not is_critical(GoldenExample(input="q", expected="a", check_type="security", critical=False), suite)
        assert not is_critical(GoldenExample(input="q", expected="a"), suite)

    def test_critical_failure_fails_gate_even_with_low_pass_rate(self):
        suite = EvalSuite(
            pass_rate=0.1,
            examples=[GoldenExample(**_case("ok")), GoldenExample(**_case("bad", ok=False, critical=True))],
        )
        report = run_suite(suite, MockProvider())
        assert report.pass_rate == 0.5
        assert not report.gate_passed
        assert report.critical_failures == ["bad"]


class TestEvaluateRegression:
    def _report(self, cases, pass_rate=0.0):
        suite = EvalSuite(pass_rate=pass_rate, examples=[GoldenExample(**c) for c in cases])
        return run_suite(suite, MockProvider())

    def test_drop_inside_tolerance_passes(self):
        base = {f"c{i}": {"passed": True, "score": 1.0, "critical": False} for i in range(10)}
        report = self._report([_case(f"c{i}", ok=(i != 0)) for i in range(10)])  # 90% now
        res = evaluate_regression(report, {"cases": base}, RegressionTolerance(pass_rate_drop=0.1, mean_score_drop=0.1))
        assert res.passed, res.reasons
        assert res.newly_failing == ["c0"]

    def test_drop_beyond_tolerance_fails(self):
        base = {f"c{i}": {"passed": True, "score": 1.0, "critical": False} for i in range(10)}
        report = self._report([_case(f"c{i}", ok=(i > 1)) for i in range(10)])  # 80% now
        res = evaluate_regression(report, {"cases": base}, RegressionTolerance(pass_rate_drop=0.1, mean_score_drop=1.0))
        assert not res.passed
        assert "pass rate dropped" in res.reasons[0]

    def test_critical_failure_is_strict_even_if_baseline_failed_too(self):
        base = {"k": {"passed": False, "score": 0.0, "critical": True}}
        report = self._report([_case("k", ok=False, critical=True)])
        res = evaluate_regression(report, {"cases": base}, RegressionTolerance(pass_rate_drop=1.0, mean_score_drop=1.0))
        assert not res.passed
        assert res.critical_failures == ["k"]

    def test_removing_a_critical_case_fails(self):
        base = {
            "k": {"passed": True, "score": 1.0, "critical": True},
            "q": {"passed": True, "score": 1.0, "critical": False},
        }
        report = self._report([_case("q")])
        res = evaluate_regression(report, {"cases": base}, RegressionTolerance())
        assert not res.passed
        assert res.missing_critical == ["k"]

    def test_new_cases_do_not_affect_quality_metrics(self):
        base = {"a": {"passed": True, "score": 1.0, "critical": False}}
        report = self._report([_case("a"), _case("new", ok=False)])
        res = evaluate_regression(report, {"cases": base}, RegressionTolerance(pass_rate_drop=0.0, mean_score_drop=0.0))
        assert res.passed
        assert res.new_cases == ["new"]

    def test_baseline_round_trip(self, tmp_path):
        report = self._report([_case("a"), _case("b", critical=True)])
        data = baseline_from_report(report)
        path = tmp_path / "b.json"
        path.write_text(json.dumps(data))
        assert load_baseline(path)["cases"]["b"] == {"passed": True, "score": 1.0, "critical": True}

    def test_bad_baseline_file(self, tmp_path):
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"something": 1}))
        with pytest.raises(ValueError, match="not a llm-eval-kit baseline"):
            load_baseline(path)


class TestCLIExitCodes:
    """EK-F4 acceptance: a seeded critical failure exits 1; a drop inside tolerance exits 0."""

    def test_seeded_critical_failure_exits_1(self, tmp_path):
        suite = _write(tmp_path, "s.yaml", {
            "pass_rate": 0.0,
            "examples": [_case("q1"), _case("sec", ok=False, check_type="security")],
        })
        baseline = _baseline(tmp_path, {
            "q1": {"passed": True, "score": 1.0, "critical": False},
            "sec": {"passed": True, "score": 1.0, "critical": True},
        })
        assert main(["--mock", suite, "--baseline", baseline]) == 1

    def test_drop_inside_tolerance_exits_0(self, tmp_path, capsys):
        cases = [_case(f"c{i}", ok=(i != 0)) for i in range(10)] + [_case("crit", critical=True)]
        suite = _write(tmp_path, "s.yaml", {"pass_rate": 0.8, "examples": cases})
        base = {f"c{i}": {"passed": True, "score": 1.0, "critical": False} for i in range(10)}
        base["crit"] = {"passed": True, "score": 1.0, "critical": True}
        baseline = _baseline(tmp_path, base)
        args = ["--mock", suite, "--baseline", baseline, "--tolerance-pass-rate", "0.1", "--tolerance-score", "0.1"]
        assert main(args) == 0
        assert "Regression vs baseline: PASSED" in capsys.readouterr().out

    def test_drop_beyond_tolerance_exits_1(self, tmp_path):
        cases = [_case(f"c{i}", ok=(i > 2)) for i in range(10)]
        suite = _write(tmp_path, "s.yaml", {"pass_rate": 0.0, "examples": cases})
        baseline = _baseline(tmp_path, {f"c{i}": {"passed": True, "score": 1.0, "critical": False} for i in range(10)})
        assert main(["--mock", suite, "--baseline", baseline, "--tolerance-pass-rate", "0.1"]) == 1

    def test_suite_tolerance_used_when_no_flag(self, tmp_path):
        cases = [_case(f"c{i}", ok=(i > 2)) for i in range(10)]  # 30 point drop
        suite = _write(tmp_path, "s.yaml", {
            "pass_rate": 0.0, "regression": {"pass_rate_drop": 0.3, "mean_score_drop": 0.3}, "examples": cases,
        })
        baseline = _baseline(tmp_path, {f"c{i}": {"passed": True, "score": 1.0, "critical": False} for i in range(10)})
        assert main(["--mock", suite, "--baseline", baseline]) == 0

    def test_update_then_compare(self, tmp_path, capsys):
        suite = _write(tmp_path, "s.yaml", {"examples": [_case("a"), _case("b", critical=True)]})
        baseline = str(tmp_path / "out" / "baseline.json")
        assert main(["--mock", suite, "--baseline", baseline, "--update-baseline"]) == 0
        assert "Baseline written" in capsys.readouterr().out
        assert main(["--mock", suite, "--baseline", baseline]) == 0

    def test_update_refused_with_critical_failure(self, tmp_path):
        suite = _write(tmp_path, "s.yaml", {"pass_rate": 0.0, "examples": [_case("b", ok=False, critical=True)]})
        baseline = tmp_path / "baseline.json"
        assert main(["--mock", suite, "--baseline", str(baseline), "--update-baseline"]) == 1
        assert not baseline.exists()

    def test_missing_baseline_exits_1(self, tmp_path, capsys):
        suite = _write(tmp_path, "s.yaml", {"examples": [_case("a")]})
        assert main(["--mock", suite, "--baseline", str(tmp_path / "nope.json")]) == 1
        assert "--update-baseline" in capsys.readouterr().err

    def test_shipped_example_passes_against_its_baseline(self):
        args = ["--mock", "examples/regression_suite.yaml", "--baseline", "examples/baselines/regression_suite.json"]
        assert main(args) == 0

    def test_report_contains_regression(self, tmp_path):
        suite = _write(tmp_path, "s.yaml", {"examples": [_case("a")]})
        baseline = _baseline(tmp_path, {"a": {"passed": True, "score": 1.0, "critical": False}})
        out = tmp_path / "r.json"
        assert main(["--mock", suite, "--baseline", baseline, "--report", str(out)]) == 0
        assert json.loads(out.read_text())["regression"]["passed"] is True
