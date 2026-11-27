"""Eval runner: orchestrates provider calls, comparisons, and validation."""

from llm_eval_kit.calls import ModelCaller
from llm_eval_kit.comparators import JudgeContext, get_comparator
from llm_eval_kit.config import EvalSuite, GoldenExample, example_id, is_critical
from llm_eval_kit.guardrails import BedrockGuardrail, run_guardrail_checks
from llm_eval_kit.providers.base import Embedder, LLMProvider, ProviderError
from llm_eval_kit.reliability import BudgetExceeded
from llm_eval_kit.report import EvalReport, ExampleResult, RetrievalExampleResult
from llm_eval_kit.retrieval import check_retrieval
from llm_eval_kit.security import redact
from llm_eval_kit.validators import run_validators


def _error_result(example: GoldenExample, suite: EvalSuite, detail: str, error: str) -> ExampleResult:
    return ExampleResult(
        input=example.input,
        expected=example.expected,
        actual="",
        comparator=example.comparator,
        passed=False,
        score=0.0,
        detail=detail,
        tags=example.tags,
        id=example_id(example),
        error=error,
        critical=is_critical(example, suite),
        check_type=example.check_type,
    )


def run_suite(
    suite: EvalSuite,
    provider: LLMProvider,
    embedder: Embedder | None = None,
    *,
    caller: ModelCaller | None = None,
    judge_provider: LLMProvider | None = None,
    guardrail: BedrockGuardrail | None = None,
    guardrail_skip_reason: str = "no guardrail configured",
) -> EvalReport:
    """Run every example. Errors and unrun cases count as failures, never passes.

    ``caller`` carries the cache and run limits; without one, calls are uncached
    and unlimited, which matches the old behaviour. ``judge_provider`` is the
    model used by ``comparator: judge`` cases, configured by the suite's
    ``judge:`` block. The gate fails if any critical case fails.
    """
    caller = caller or ModelCaller()
    results: list[ExampleResult] = []
    stopped_reason: str | None = None
    warnings: list[str] = []
    model_name = str(getattr(provider, "model", provider.name))

    judge_ctx = None
    if judge_provider is not None and suite.judge is not None:
        judge_ctx = JudgeContext(caller=caller, provider=judge_provider, config=suite.judge)
        judge_model = str(getattr(judge_provider, "model", judge_provider.name))
        if (judge_provider.name, judge_model) == (provider.name, model_name):
            warnings.append(f"judge is the same model as the model under test ({provider.name}/{model_name})")

    for example in suite.examples:
        # Apply suite-level defaults for fields not explicitly set on the example
        if "comparator" not in example.model_fields_set:
            example.comparator = suite.default_comparator
        if "threshold" not in example.model_fields_set:
            example.threshold = suite.default_threshold

        if stopped_reason:
            results.append(_error_result(example, suite, f"Not run: {stopped_reason}", "not_run"))
            continue

        try:
            completion, cost = caller.call(
                provider,
                example.input,
                system=suite.system,
                max_tokens=suite.max_tokens,
                temperature=suite.temperature,
            )
        except BudgetExceeded as e:
            stopped_reason = str(e)
            results.append(_error_result(example, suite, f"Not run: {stopped_reason}", "not_run"))
            continue
        except ProviderError as e:
            msg = redact(str(e))
            results.append(_error_result(example, suite, f"Provider error: {msg}", "provider_error"))
            continue

        actual = completion.text
        try:
            comparator = get_comparator(example, embedder, judge_ctx)
            cmp_result = comparator.compare(actual, example.expected)
        except BudgetExceeded as e:  # the judge call hit a run limit
            stopped_reason = str(e)
            res = _error_result(example, suite, f"Not judged: {stopped_reason}", "not_run")
            res.actual = actual
            results.append(res)
            continue
        except ValueError as e:
            res = _error_result(example, suite, f"Comparator error: {redact(str(e))}", "comparator_error")
            res.actual = actual
            results.append(res)
            continue

        val_results = []
        all_validators_passed = True
        if example.validators:
            v_results = run_validators(actual, example.validators)
            for vr in v_results:
                val_results.append({"passed": vr.passed, "detail": vr.detail})
                if not vr.passed:
                    all_validators_passed = False

        passed = cmp_result.passed and all_validators_passed and cmp_result.error is None

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
                id=example_id(example),
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                latency_ms=completion.latency_ms,
                cost_usd=cost,
                cached=completion.cached,
                error=cmp_result.error,
                critical=is_critical(example, suite),
                check_type=example.check_type,
                judge=cmp_result.extra or None,
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

    guardrail_results = run_guardrail_checks(suite.guardrail_examples, guardrail, guardrail_skip_reason)

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    rate = passed_count / total if total > 0 else 1.0

    report = EvalReport(
        suite_name=suite.name,
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=rate,
        target_pass_rate=suite.pass_rate,
        gate_passed=False,
        results=results,
        retrieval_results=retrieval_results,
        provider=provider.name,
        model=model_name,
        usage=caller.totals(),
        stopped_reason=stopped_reason,
        guardrail_results=guardrail_results,
        warnings=warnings,
    )
    report.gate_passed = rate >= suite.pass_rate and stopped_reason is None and not report.critical_failures
    return report
