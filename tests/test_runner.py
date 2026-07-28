"""Tests for the eval runner."""

from llm_eval_kit.config import EvalSuite, GoldenExample, RetrievalExample, ValidatorConfig
from llm_eval_kit.providers.mock import MockEmbedder, MockProvider
from llm_eval_kit.runner import run_suite


class TestRunSuite:
    def test_all_pass(self):
        provider = MockProvider(responses={"What is 2+2?": "4", "Capital of France?": "Paris"})
        suite = EvalSuite(
            name="basic",
            examples=[
                GoldenExample(input="What is 2+2?", expected="4"),
                GoldenExample(input="Capital of France?", expected="Paris"),
            ],
        )
        report = run_suite(suite, provider)
        assert report.gate_passed
        assert report.passed == 2
        assert report.failed == 0
        assert report.pass_rate == 1.0

    def test_partial_pass(self):
        provider = MockProvider(responses={"q1": "correct"})
        suite = EvalSuite(
            name="partial",
            examples=[
                GoldenExample(input="q1", expected="correct"),
                GoldenExample(input="q2", expected="right answer"),
            ],
            pass_rate=0.5,
        )
        report = run_suite(suite, provider)
        assert report.gate_passed
        assert report.passed == 1
        assert report.failed == 1

    def test_gate_fails_below_threshold(self):
        provider = MockProvider()
        suite = EvalSuite(
            name="strict",
            examples=[
                GoldenExample(input="q1", expected="answer"),
            ],
            pass_rate=1.0,
        )
        report = run_suite(suite, provider)
        assert not report.gate_passed

    def test_with_validators(self):
        provider = MockProvider(responses={"gen json": '{"name": "test"}'})
        suite = EvalSuite(
            name="validated",
            examples=[
                GoldenExample(
                    input="gen json",
                    expected='{"name": "test"}',
                    validators=[
                        ValidatorConfig(type="json"),
                        ValidatorConfig(type="json_keys", keys=["name"]),
                    ],
                ),
            ],
        )
        report = run_suite(suite, provider)
        assert report.gate_passed
        assert len(report.results[0].validator_results) == 2

    def test_validator_failure_overrides_comparator(self):
        provider = MockProvider(responses={"q": "answer"})
        suite = EvalSuite(
            name="val-fail",
            examples=[
                GoldenExample(
                    input="q",
                    expected="answer",
                    validators=[ValidatorConfig(type="length", max_length=1)],
                ),
            ],
        )
        report = run_suite(suite, provider)
        assert not report.gate_passed
        assert not report.results[0].passed

    def test_numeric_comparator(self):
        provider = MockProvider(responses={"estimate pi": "3.14159"})
        suite = EvalSuite(
            name="numeric",
            examples=[
                GoldenExample(
                    input="estimate pi",
                    expected="3.14",
                    comparator="numeric",
                    abs_tol=0.01,
                ),
            ],
        )
        report = run_suite(suite, provider)
        assert report.gate_passed

    def test_retrieval_examples(self):
        provider = MockProvider()
        embedder = MockEmbedder(dim=32)
        suite = EvalSuite(
            name="retrieval",
            retrieval_examples=[
                RetrievalExample(
                    query="test query",
                    context_chunks=["test query"],
                    threshold=0.9,
                ),
            ],
        )
        report = run_suite(suite, provider, embedder)
        assert len(report.retrieval_results) == 1
        assert report.retrieval_results[0].passed

    def test_retrieval_without_embedder(self):
        provider = MockProvider()
        suite = EvalSuite(
            name="no-embedder",
            retrieval_examples=[
                RetrievalExample(query="q", context_chunks=["chunk"], threshold=0.5),
            ],
        )
        report = run_suite(suite, provider)
        assert not report.retrieval_results[0].passed
        assert "No embedder" in report.retrieval_results[0].detail

    def test_empty_suite(self):
        provider = MockProvider()
        suite = EvalSuite(name="empty")
        report = run_suite(suite, provider)
        assert report.total == 0
        assert report.gate_passed

    def test_report_summary(self):
        provider = MockProvider(responses={"q": "a"})
        suite = EvalSuite(
            name="summary-test",
            examples=[GoldenExample(input="q", expected="a")],
        )
        report = run_suite(suite, provider)
        summary = report.summary()
        assert "summary-test" in summary
        assert "PASSED" in summary
