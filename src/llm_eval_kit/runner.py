"""Eval runner — orchestrates provider calls, comparisons, and validation."""

from llm_eval_kit.comparators import get_comparator
from llm_eval_kit.config import EvalSuite
from llm_eval_kit.providers.base import Embedder, LLMProvider
from llm_eval_kit.report import EvalReport, ExampleResult, RetrievalExampleResult
from llm_eval_kit.retrieval import check_retrieval
from llm_eval_kit.validators import run_validators


def run_suite(
    suite: EvalSuite,
    provider: LLMProvider,
    embedder: Embedder | None = None,
) -> EvalReport:
    results: list[ExampleResult] = []

    for example in suite.examples:
        actual = provider.generate(example.input)

        comparator = get_comparator(example, embedder)
        cmp_result = comparator.compare(actual, example.expected)

        val_results = []
        all_validators_passed = True
        if example.validators:
            v_results = run_validators(actual, example.validators)
            for vr in v_results:
                val_results.append({"passed": vr.passed, "detail": vr.detail})
                if not vr.passed:
                    all_validators_passed = False

        passed = cmp_result.passed and all_validators_passed

        results.append(
            ExampleResult(
                input=example.input,
                expected=example.expected,
                actual=actual,
                comparator=example.comparator,
                passed=passed,
                score=cmp_result.score,
                detail=cmp_result.detail,
                validator_results=val_results,
                tags=example.tags,
            )
        )

    retrieval_results: list[RetrievalExampleResult] = []
    for ret_example in suite.retrieval_examples:
        if embedder is None:
            retrieval_results.append(
                RetrievalExampleResult(
                    query=ret_example.query,
                    passed=False,
                    avg_score=0.0,
                    chunk_scores=[],
                    detail="No embedder provided for retrieval check",
                )
            )
            continue

        ret_result = check_retrieval(
            ret_example.query,
            ret_example.context_chunks,
            embedder,
            ret_example.threshold,
        )
        retrieval_results.append(
            RetrievalExampleResult(
                query=ret_example.query,
                passed=ret_result.passed,
                avg_score=ret_result.avg_score,
                chunk_scores=ret_result.chunk_scores,
                detail=ret_result.detail,
            )
        )

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    rate = passed_count / total if total > 0 else 1.0
    gate_passed = rate >= suite.pass_rate

    return EvalReport(
        suite_name=suite.name,
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=rate,
        target_pass_rate=suite.pass_rate,
        gate_passed=gate_passed,
        results=results,
        retrieval_results=retrieval_results,
    )
