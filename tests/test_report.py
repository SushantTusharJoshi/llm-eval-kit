"""Tests for the report module."""

from llm_eval_kit.report import EvalReport, ExampleResult, RetrievalExampleResult


class TestExampleResult:
    def test_construction(self):
        r = ExampleResult(
            input="q",
            expected="a",
            actual="a",
            comparator="exact",
            passed=True,
            score=1.0,
            detail="match",
        )
        assert r.passed is True
        assert r.score == 1.0
        assert r.validator_results == []
        assert r.tags == []

    def test_with_optional_fields(self):
        r = ExampleResult(
            input="q",
            expected="a",
            actual="b",
            comparator="cosine",
            passed=False,
            score=0.5,
            detail="low score",
            validator_results=[{"passed": False, "detail": "too short"}],
            tags=["smoke"],
        )
        assert len(r.validator_results) == 1
        assert r.tags == ["smoke"]


class TestRetrievalExampleResult:
    def test_construction(self):
        r = RetrievalExampleResult(
            query="what is X?",
            passed=True,
            avg_score=0.85,
            chunk_scores=[0.9, 0.8],
            detail="ok",
        )
        assert r.passed is True
        assert r.avg_score == 0.85
        assert len(r.chunk_scores) == 2


class TestEvalReport:
    def _make_result(self, passed: bool, input_text: str = "q") -> ExampleResult:
        return ExampleResult(
            input=input_text,
            expected="a",
            actual="a" if passed else "b",
            comparator="exact",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="match" if passed else "mismatch",
        )

    def test_summary_all_passed(self):
        report = EvalReport(
            suite_name="test",
            total=2,
            passed=2,
            failed=0,
            pass_rate=1.0,
            target_pass_rate=1.0,
            gate_passed=True,
            results=[self._make_result(True), self._make_result(True)],
        )
        text = report.summary()
        assert "PASSED" in text
        assert "2/2" in text
        assert "Failures:" not in text

    def test_summary_with_failures(self):
        report = EvalReport(
            suite_name="test",
            total=2,
            passed=1,
            failed=1,
            pass_rate=0.5,
            target_pass_rate=1.0,
            gate_passed=False,
            results=[self._make_result(True), self._make_result(False)],
        )
        text = report.summary()
        assert "FAILED" in text
        assert "1/2" in text
        assert "Failures:" in text

    def test_summary_with_retrieval(self):
        ret = RetrievalExampleResult(
            query="q",
            passed=True,
            avg_score=0.9,
            chunk_scores=[0.9],
            detail="ok",
        )
        report = EvalReport(
            suite_name="test",
            total=1,
            passed=1,
            failed=0,
            pass_rate=1.0,
            target_pass_rate=1.0,
            gate_passed=True,
            results=[self._make_result(True)],
            retrieval_results=[ret],
        )
        text = report.summary()
        assert "Retrieval: 1/1" in text

    def test_summary_truncates_failures(self):
        """Failures list is capped at 10 items."""
        results = [self._make_result(False, input_text=f"q{i}") for i in range(15)]
        report = EvalReport(
            suite_name="test",
            total=15,
            passed=0,
            failed=15,
            pass_rate=0.0,
            target_pass_rate=1.0,
            gate_passed=False,
            results=results,
        )
        text = report.summary()
        # Only first 10 failures should appear
        assert "q9" in text
        assert "q10" not in text

    def test_gate_logic(self):
        report = EvalReport(
            suite_name="test",
            total=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            target_pass_rate=0.8,
            gate_passed=True,
        )
        assert report.gate_passed is True
        assert "PASSED" in report.summary()
