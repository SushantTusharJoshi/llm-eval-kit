"""Eval run report — structured results with pass/fail gating."""

from dataclasses import dataclass, field


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

    def summary(self) -> str:
        status = "PASSED" if self.gate_passed else "FAILED"
        lines = [
            f"Suite: {self.suite_name}",
            f"Status: {status}",
            f"Results: {self.passed}/{self.total} passed ({self.pass_rate:.1%})",
            f"Target: {self.target_pass_rate:.1%}",
        ]

        if self.retrieval_results:
            ret_passed = sum(1 for r in self.retrieval_results if r.passed)
            lines.append(f"Retrieval: {ret_passed}/{len(self.retrieval_results)} passed")

        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append("")
            lines.append("Failures:")
            for r in failed[:10]:
                lines.append(f"  - input: {r.input[:60]}")
                lines.append(f"    {r.detail}")

        return "\n".join(lines)
