"""Numeric tolerance comparator for LLM outputs containing numbers."""

import re

from llm_eval_kit.comparators.exact import CompareResult

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    if match:
        return float(match.group())
    return None


class NumericComparator:
    """Compare numeric values extracted from text with absolute or relative tolerance."""

    def __init__(self, abs_tol: float | None = None, rel_tol: float | None = None):
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        if abs_tol is None and rel_tol is None:
            self._abs_tol = 0.01

    def compare(self, actual: str, expected: str) -> CompareResult:
        actual_num = _extract_number(actual)
        expected_num = _extract_number(expected)

        if actual_num is None:
            return CompareResult(passed=False, score=0.0, detail=f"No number found in output: '{actual[:80]}'")

        if expected_num is None:
            return CompareResult(passed=False, score=0.0, detail=f"No number found in expected: '{expected[:80]}'")

        diff = abs(actual_num - expected_num)

        if self._abs_tol is not None and diff <= self._abs_tol:
            return CompareResult(
                passed=True,
                score=max(0.0, 1.0 - diff / max(self._abs_tol, 1e-10)),
                detail=f"{actual_num} within abs_tol={self._abs_tol} of {expected_num}",
            )

        if self._rel_tol is not None and expected_num != 0:
            rel_diff = diff / abs(expected_num)
            if rel_diff <= self._rel_tol:
                return CompareResult(
                    passed=True,
                    score=max(0.0, 1.0 - rel_diff / max(self._rel_tol, 1e-10)),
                    detail=f"{actual_num} within rel_tol={self._rel_tol} of {expected_num} (rel_diff={rel_diff:.4f})",
                )

        return CompareResult(
            passed=False,
            score=0.0,
            detail=f"{actual_num} differs from {expected_num} by {diff} (abs_tol={self._abs_tol}, rel_tol={self._rel_tol})",
        )
