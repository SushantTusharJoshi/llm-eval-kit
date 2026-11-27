"""LLM judge comparator with a stub judge provider."""

import json

import pytest
import yaml

from llm_eval_kit import cli
from llm_eval_kit.cache import ResultCache
from llm_eval_kit.calls import ModelCaller
from llm_eval_kit.comparators.judge import JUDGE_SYSTEM, JudgeError, parse_verdict
from llm_eval_kit.config import EvalSuite, GoldenExample, JudgeConfig
from llm_eval_kit.providers.base import Completion, LLMProvider, ProviderError
from llm_eval_kit.providers.mock import MockProvider
from llm_eval_kit.reliability import Budget, RunLimits
from llm_eval_kit.runner import run_suite


class StubJudge(LLMProvider):
    name = "stubjudge"
    model = "judge-model-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt):
        return self.complete(prompt).text

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.0):
        self.calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens, "temperature": temperature})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return Completion(reply, self.model, self.name, input_tokens=50, output_tokens=20, latency_ms=5.0)


def _suite(rubric="Score 1 if correct.", threshold=0.7, n=1):
    return EvalSuite(
        judge=JudgeConfig(provider="stubjudge", max_tokens=300, pass_threshold=threshold),
        examples=[
            GoldenExample(input=f"q{i}", expected="reference answer", comparator="judge", rubric=rubric)
            for i in range(n)
        ],
    )


class TestParseVerdict:
    @pytest.mark.parametrize(
        "text,score",
        [
            ('{"score": 0.9, "reason": "good"}', 0.9),
            ('  {"score": 1, "reason": "int is fine"}\n', 1.0),
            ('```json\n{"score": 0.2, "reason": "fenced"}\n```', 0.2),
        ],
    )
    def test_valid(self, text, score):
        assert parse_verdict(text).score == score

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "The answer looks correct. Score: 0.9",
            'Sure! {"score": 0.9, "reason": "x"}',
            '{"score": "0.9", "reason": "string score"}',
            '{"score": true, "reason": "bool score"}',
            '{"score": 1.5, "reason": "out of range"}',
            '{"score": -0.1, "reason": "negative"}',
            '{"score": 0.9}',
            '{"score": 0.9, "reason": ""}',
            '{"score": 0.9, "reason": "x", "verdict": "pass"}',
            "[0.9]",
            '{"score": 0.9, "reason": "x"',
        ],
    )
    def test_malformed_raises(self, text):
        with pytest.raises(JudgeError):
            parse_verdict(text)

    def test_oversized_reply_rejected(self):
        with pytest.raises(JudgeError, match="too long"):
            parse_verdict('{"score": 1, "reason": "' + "x" * 30000 + '"}')


class TestJudgeComparator:
    def test_pass_and_fail_by_threshold(self):
        judge = StubJudge(['{"score": 0.9, "reason": "matches"}', '{"score": 0.3, "reason": "wrong city"}'])
        report = run_suite(_suite(n=2), MockProvider(), judge_provider=judge)
        assert [r.passed for r in report.results] == [True, False]
        assert report.results[0].score == 0.9
        assert "wrong city" in report.results[1].detail
        assert report.results[0].judge["judge_model"] == "judge-model-1"

    def test_malformed_output_is_error_never_pass(self):
        judge = StubJudge(["Looks great, 10/10!"])
        report = run_suite(_suite(threshold=0.0), MockProvider(), judge_provider=judge)
        r = report.results[0]
        assert not r.passed
        assert r.error == "judge_malformed"
        assert not report.gate_passed

    def test_judge_provider_error_is_error(self):
        judge = StubJudge([ProviderError("stubjudge: HTTP 500")])
        report = run_suite(_suite(), MockProvider(), judge_provider=judge)
        assert report.results[0].error == "judge_provider_error"
        assert not report.results[0].passed

    def test_judge_uses_its_own_settings_and_sees_rubric(self):
        judge = StubJudge(['{"score": 1, "reason": "ok"}'])
        suite = _suite(rubric="RUBRIC-XYZ")
        suite.max_tokens = 64  # model-under-test setting, must not leak into the judge
        run_suite(suite, MockProvider(), judge_provider=judge)
        call = judge.calls[0]
        assert call["max_tokens"] == 300
        assert call["system"] == JUDGE_SYSTEM
        assert "RUBRIC-XYZ" in call["prompt"]
        assert "<candidate>\nMock response to: q0\n</candidate>" in call["prompt"]
        assert "reference answer" in call["prompt"]

    def test_suite_level_rubric_default(self):
        judge = StubJudge(['{"score": 1, "reason": "ok"}'])
        suite = _suite(rubric=None)
        suite.judge.rubric = "DEFAULT-RUBRIC"
        run_suite(suite, MockProvider(), judge_provider=judge)
        assert "DEFAULT-RUBRIC" in judge.calls[0]["prompt"]

    def test_no_rubric_is_error(self):
        report = run_suite(_suite(rubric=None), MockProvider(), judge_provider=StubJudge(['{"score":1,"reason":"x"}']))
        assert report.results[0].error == "comparator_error"

    def test_no_judge_provider_is_error(self):
        report = run_suite(_suite(), MockProvider())
        assert report.results[0].error == "comparator_error"
        assert "judge" in report.results[0].detail

    def test_judge_calls_counted_in_usage_and_budget(self):
        judge = StubJudge(['{"score": 1, "reason": "ok"}'])
        caller = ModelCaller(budget=Budget(RunLimits(max_calls=3)))
        report = run_suite(_suite(n=3), MockProvider(), caller=caller, judge_provider=judge)
        # 3 model calls + 3 judge calls wanted; the cap of 3 stops the run.
        assert report.stopped_reason
        assert report.usage.by_role["judge"]["live_calls"] >= 1

    def test_rubric_change_misses_judge_cache(self, tmp_path):
        cache = ResultCache(tmp_path)
        judge = StubJudge(['{"score": 1, "reason": "ok"}'])
        run_suite(_suite(rubric="A"), MockProvider(), caller=ModelCaller(cache=cache), judge_provider=judge)
        run_suite(_suite(rubric="A"), MockProvider(), caller=ModelCaller(cache=cache), judge_provider=judge)
        assert len(judge.calls) == 1
        run_suite(_suite(rubric="B"), MockProvider(), caller=ModelCaller(cache=cache), judge_provider=judge)
        assert len(judge.calls) == 2

    def test_same_model_as_judge_warns(self):
        class SameAsMock(StubJudge):
            name = "mock"
            model = "mock"

        report = run_suite(_suite(), MockProvider(), judge_provider=SameAsMock(['{"score": 1, "reason": "ok"}']))
        assert any("same model" in w for w in report.warnings)


class TestJudgeCLI:
    def test_cli_builds_separate_judge(self, tmp_path, monkeypatch):
        built = []
        judge = StubJudge(['{"score": 0.95, "reason": "fine"}'])

        def fake_build(name, **kw):
            built.append((name, kw.get("model")))
            return MockProvider() if name == "mock" else judge

        monkeypatch.setattr(cli, "build_provider", fake_build)
        data = {
            "judge": {"provider": "groq", "model": "openai/gpt-oss-120b"},
            "examples": [{"input": "q", "expected": "ref", "comparator": "judge", "rubric": "r"}],
        }
        path = tmp_path / "s.yaml"
        path.write_text(yaml.dump(data))
        out = tmp_path / "r.json"
        assert cli.main(["--mock", str(path), "--report", str(out), "--cache-dir", str(tmp_path / "c")]) == 0
        assert ("groq", "openai/gpt-oss-120b") in built
        assert json.loads(out.read_text())["results"][0]["judge"]["judge_reason"] == "fine"
