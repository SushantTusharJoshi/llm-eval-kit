"""Eval run report: structured results with pass/fail gating."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from llm_eval_kit.calls import UsageTotals
from llm_eval_kit.pricing import PRICES_CHECKED_ON
from llm_eval_kit.security import redact, redact_obj


@dataclass
class ExampleResult:
    input: str
    expected: str
    actual: str
    comparator: str
    passed: bool
    score: float
    detail: str
    validator_results: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    id: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    cached: bool = False
    error: str | None = None


@dataclass
class RetrievalExampleResult:
    query: str
    passed: bool
    avg_score: float
    chunk_scores: list[float]
    detail: str


@dataclass
class EvalReport:
    suite_name: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    target_pass_rate: float
    gate_passed: bool
    results: list[ExampleResult] = field(default_factory=list)
    retrieval_results: list[RetrievalExampleResult] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    usage: UsageTotals = field(default_factory=UsageTotals)
    stopped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON, with every string passed through secret redaction."""
        data = asdict(self)
        data["prices_checked_on"] = PRICES_CHECKED_ON
        return redact_obj(data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        status = "PASSED" if self.gate_passed else "FAILED"
        lines = [
            f"Suite: {self.suite_name}",
            f"Status: {status}",
            f"Results: {self.passed}/{self.total} passed ({self.pass_rate:.1%})",
            f"Target: {self.target_pass_rate:.1%}",
        ]
        if self.stopped_reason:
            lines.append(f"Stopped early: {self.stopped_reason}")

        if self.retrieval_results:
            ret_passed = sum(1 for r in self.retrieval_results if r.passed)
            lines.append(f"Retrieval: {ret_passed}/{len(self.retrieval_results)} passed")

        u = self.usage
        if u.live_calls or u.cached_calls:
            cost = "unknown" if u.live_cost_usd is None else f"${u.live_cost_usd:.6f}"
            if u.live_unpriced_calls and u.live_cost_usd is not None:
                cost += f" ({u.live_unpriced_calls} calls unpriced)"
            mean = "n/a" if u.live_latency_ms_mean is None else f"{u.live_latency_ms_mean:.0f} ms mean"
            lines.append(
                f"Live calls: {u.live_calls}, tokens in/out {u.live_input_tokens}/{u.live_output_tokens}, "
                f"est. cost {cost}, latency {mean}"
            )
            if u.cached_calls:
                lines.append(
                    f"Cached calls: {u.cached_calls}, tokens in/out {u.cached_input_tokens}/{u.cached_output_tokens} "
                    "(not counted in live cost or latency)"
                )

        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append("")
            lines.append("Failures:")
            for r in failed[:10]:
                lines.append(f"  - input: {r.input[:60]}")
                lines.append(f"    {r.detail}")

        return redact("\n".join(lines))
