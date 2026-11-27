"""Regression gate: compare a run with a baseline stored in git.

Two rules:

* Critical cases (``critical: true`` or a critical ``check_type``) must pass on
  every run. No tolerance. A critical case that is in the baseline but missing
  from the run also fails, so a check cannot be dodged by deleting it.
* Non-critical quality may drift inside a tolerance: the pass rate and the
  mean score over cases present in both runs may drop by at most the
  configured amount.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_eval_kit.config import RegressionTolerance
from llm_eval_kit.report import EvalReport

BASELINE_FORMAT = 1


@dataclass
class RegressionResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)
    missing_critical: list[str] = field(default_factory=list)
    newly_failing: list[str] = field(default_factory=list)
    new_cases: list[str] = field(default_factory=list)
    removed_cases: list[str] = field(default_factory=list)
    pass_rate_baseline: float | None = None
    pass_rate_current: float | None = None
    mean_score_baseline: float | None = None
    mean_score_current: float | None = None
    tolerance: dict[str, float] = field(default_factory=dict)


def baseline_from_report(report: EvalReport) -> dict[str, Any]:
    """The slim, diff-friendly form stored in git: per-case pass, score and critical flag."""
    return {
        "format": BASELINE_FORMAT,
        "suite": report.suite_name,
        "provider": report.provider,
        "model": report.model,
        "pass_rate": report.pass_rate,
        "cases": {
            r.id: {"passed": r.passed, "score": round(r.score, 6), "critical": r.critical}
            for r in sorted(report.results, key=lambda r: r.id)
        },
    }


def write_baseline(report: EvalReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline_from_report(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != BASELINE_FORMAT or not isinstance(data.get("cases"), dict):
        raise ValueError(f"{path}: not a llm-eval-kit baseline (format {BASELINE_FORMAT})")
    return data


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# What it does: decides pass or fail for a run against the baseline, strict for critical cases, tolerant for quality.
# Catch it: "Design a CI gate for a non-deterministic system. What fails the build, and what is allowed to wobble?"
# Card: docs/cards/regression-gate.md
def evaluate_regression(
    report: EvalReport,
    baseline: dict[str, Any],
    tolerance: RegressionTolerance,
) -> RegressionResult:
    res = RegressionResult(passed=True, tolerance=tolerance.model_dump())
    base_cases: dict[str, dict[str, Any]] = baseline["cases"]
    current = {r.id: r for r in report.results}

    # STEP: strict rule; any critical case failing now fails the gate, whatever the baseline said
    res.critical_failures = sorted(r.id for r in report.results if r.critical and not r.passed)
    # STEP: strict rule; a critical baseline case missing from this run fails too
    res.missing_critical = sorted(cid for cid, c in base_cases.items() if c.get("critical") and cid not in current)
    res.new_cases = sorted(cid for cid in current if cid not in base_cases)
    res.removed_cases = sorted(cid for cid in base_cases if cid not in current)

    # LOOP: compare non-critical cases present in both runs
    shared = [cid for cid, r in current.items() if cid in base_cases and not r.critical]
    res.newly_failing = sorted(cid for cid in shared if base_cases[cid]["passed"] and not current[cid].passed)
    if shared:
        res.pass_rate_baseline = sum(base_cases[c]["passed"] for c in shared) / len(shared)
        res.pass_rate_current = sum(current[c].passed for c in shared) / len(shared)
        res.mean_score_baseline = _mean([float(base_cases[c]["score"]) for c in shared])
        res.mean_score_current = _mean([current[c].score for c in shared])

    # DECISION: collect every reason to fail; quality drops only count beyond the tolerance
    if res.critical_failures:
        res.reasons.append(f"critical cases failed: {', '.join(res.critical_failures)}")
    if res.missing_critical:
        res.reasons.append(f"critical baseline cases missing from run: {', '.join(res.missing_critical)}")
    eps = 1e-9
    if res.pass_rate_baseline is not None and res.pass_rate_current is not None:
        drop = res.pass_rate_baseline - res.pass_rate_current
        if drop > tolerance.pass_rate_drop + eps:
            res.reasons.append(f"pass rate dropped {drop:.1%} (allowed {tolerance.pass_rate_drop:.1%})")
    if res.mean_score_baseline is not None and res.mean_score_current is not None:
        drop = res.mean_score_baseline - res.mean_score_current
        if drop > tolerance.mean_score_drop + eps:
            res.reasons.append(f"mean score dropped {drop:.3f} (allowed {tolerance.mean_score_drop:.3f})")

    # RETURN: passed only when there is no reason to fail
    res.passed = not res.reasons
    return res


def regression_summary(res: RegressionResult) -> str:
    lines = [f"Regression vs baseline: {'PASSED' if res.passed else 'FAILED'}"]
    if res.pass_rate_baseline is not None:
        lines.append(
            f"  non-critical pass rate {res.pass_rate_baseline:.1%} -> {res.pass_rate_current:.1%}, "
            f"mean score {res.mean_score_baseline:.3f} -> {res.mean_score_current:.3f}"
        )
    for reason in res.reasons:
        lines.append(f"  - {reason}")
    if res.newly_failing:
        lines.append(f"  newly failing (inside tolerance unless listed above): {', '.join(res.newly_failing)}")
    if res.new_cases:
        lines.append(f"  new cases (not in baseline): {len(res.new_cases)}")
    if res.removed_cases:
        lines.append(f"  removed cases: {len(res.removed_cases)}")
    return "\n".join(lines)
