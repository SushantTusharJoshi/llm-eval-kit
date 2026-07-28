"""Exact string match comparator."""

import re
from dataclasses import dataclass


@dataclass
class CompareResult:
    passed: bool
    score: float
    detail: str


class ExactComparator:
    """Compare LLM output to expected output via exact string matching."""

    def __init__(self, case_sensitive: bool = True, normalize_whitespace: bool = False):
        self._case_sensitive = case_sensitive
        self._normalize_ws = normalize_whitespace

    def compare(self, actual: str, expected: str) -> CompareResult:
        a = self._normalize(actual)
        e = self._normalize(expected)

        if a == e:
            return CompareResult(passed=True, score=1.0, detail="Exact match")

        return CompareResult(
            passed=False,
            score=0.0,
            detail=f"Mismatch: got '{actual[:80]}', expected '{expected[:80]}'",
        )

    def _normalize(self, text: str) -> str:
        t = text.strip()
        if self._normalize_ws:
            t = re.sub(r"\s+", " ", t)
        if not self._case_sensitive:
            t = t.lower()
        return t
