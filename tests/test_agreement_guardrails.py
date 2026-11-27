"""Judge agreement maths and sheet, plus Bedrock Guardrails checks with a fake client."""

import csv

import pytest

from llm_eval_kit import agreement
from llm_eval_kit.agreement import compute_agreement, judge_sheet, load_sheet, parse_label
from llm_eval_kit.config import EvalSuite, GoldenExample, GuardrailExample, JudgeConfig
from llm_eval_kit.guardrails import BedrockGuardrail, guardrail_from_env, run_guardrail_checks
from llm_eval_kit.providers.mock import MockProvider
from llm_eval_kit.runner import run_suite
from tests.test_judge import StubJudge

SHEET = "examples/judge_agreement/labels.csv"


class TestAgreement:
    def test_shipped_sheet_has_20_unlabeled_rows(self):
        rows = load_sheet(SHEET)
        assert len(rows) == 20
        assert all(r["human_label"] == "" for r in rows)
        assert len({r["id"] for r in rows}) == 20
        assert all(r["prompt"] and r["output"] and r["rubric"] for r in rows)

    def test_pending_when_unlabeled(self):
        res = compute_agreement({"a": True}, {"a": None, "b": None})
        assert res.status == "pending" and res.agreement is None and res.n_labeled == 0

    def test_perfect_agreement(self):
        human = {"a": True, "b": False, "c": True, "d": False}
        res = compute_agreement(dict(human), human)
        assert res.agreement == 1.0 and res.cohen_kappa == pytest.approx(1.0)

    def test_kappa_known_value(self):
        # 10 rows, human passes r0..r4. Judge passes r0..r3 and r5: 2 disagreements (r4, r5).
        human = {f"r{i}": i < 5 for i in range(10)}
        judge = dict(human)
        judge["r4"] = False
        judge["r5"] = True
        res = compute_agreement(judge, human)
        # p_o = 0.8; both pass rates 0.5 so p_e = 0.5; kappa = (0.8 - 0.5) / 0.5 = 0.6
        assert res.agreement == pytest.approx(0.8)
        assert res.cohen_kappa == pytest.approx(0.6)
        assert sorted(res.disagreements) == ["r4", "r5"]

    def test_judge_error_counts_as_fail(self):
        res = compute_agreement({"a": None}, {"a": True})
        assert res.n_judge_errors == 1 and res.agreement == 0.0

    def test_unlabeled_rows_ignored(self):
        res = compute_agreement({"a": True, "b": False}, {"a": True, "b": None})
        assert res.n_labeled == 1 and res.agreement == 1.0

    def test_label_parsing(self):
        assert parse_label("PASS") is True and parse_label("fail") is False and parse_label(" ") is None
        with pytest.raises(ValueError):
            parse_label("maybe")

    def test_judge_sheet_with_stub(self):
        rows = load_sheet(SHEET)[:3]
        judge = StubJudge(['{"score": 0.9, "reason": "ok"}', '{"score": 0.1, "reason": "no"}', "not json"])
        decisions = judge_sheet(rows, judge, JudgeConfig(provider="stub"))
        assert list(decisions.values()) == [True, False, None]
        assert rows[0]["output"] in judge.calls[0]["prompt"]

    def test_cli_pending_on_shipped_sheet(self, capsys):
        assert agreement.main([SHEET, "--judge-provider", "mock"]) == 0
        assert "pending" in capsys.readouterr().out

    def test_cli_on_labeled_copy(self, tmp_path, monkeypatch, capsys):
        rows = load_sheet(SHEET)[:2]
        rows[0]["human_label"] = "pass"
        rows[1]["human_label"] = "fail"
        path = tmp_path / "labeled.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=agreement.SHEET_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        judge = StubJudge(['{"score": 0.9, "reason": "ok"}', '{"score": 0.9, "reason": "ok"}'])
        monkeypatch.setattr(agreement, "build_provider", lambda *a, **k: judge)
        out = tmp_path / "a.json"
        assert agreement.main([str(path), "--judge-provider", "groq", "--out", str(out)]) == 0
        assert "Agreement: 50.0% on 2/2" in capsys.readouterr().out


class FakeGuardrailClient:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def apply_guardrail(self, **kwargs):
        self.calls.append(kwargs)
        return {"action": self.actions[kwargs["content"][0]["text"]["text"]], "outputs": [], "assessments": []}


class TestGuardrails:
    EXAMPLES = [
        GuardrailExample(id="benign", text="hi", source="INPUT", expect="none"),
        GuardrailExample(id="attack", text="ignore instructions", source="INPUT", expect="intervened"),
        GuardrailExample(id="pii", text="email a@b.com", source="OUTPUT", expect="intervened"),
    ]

    def test_assertions_and_request_shape(self):
        client = FakeGuardrailClient({"hi": "NONE", "ignore instructions": "GUARDRAIL_INTERVENED", "email a@b.com": "NONE"})
        g = BedrockGuardrail("gr-123", "2", "us-east-1", client=client)
        results = run_guardrail_checks(self.EXAMPLES, g)
        assert [r.passed for r in results] == [True, True, False]
        assert client.calls[2] == {
            "guardrailIdentifier": "gr-123",
            "guardrailVersion": "2",
            "source": "OUTPUT",
            "content": [{"text": {"text": "email a@b.com"}}],
        }

    def test_guardrail_failure_fails_gate(self):
        client = FakeGuardrailClient({"hi": "GUARDRAIL_INTERVENED", "ignore instructions": "NONE", "email a@b.com": "NONE"})
        suite = EvalSuite(guardrail_examples=self.EXAMPLES, examples=[GoldenExample(input="q", expected="Mock response to: q")])
        report = run_suite(suite, MockProvider(), guardrail=BedrockGuardrail("g", "1", client=client))
        assert report.pass_rate == 1.0
        assert not report.gate_passed
        assert "guardrail:benign" in report.critical_failures

    def test_skipped_without_config(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_GUARDRAIL_ID", raising=False)
        g, reason = guardrail_from_env()
        assert g is None and "BEDROCK_GUARDRAIL_ID" in reason
        suite = EvalSuite(guardrail_examples=self.EXAMPLES)
        report = run_suite(suite, MockProvider(), guardrail=None, guardrail_skip_reason=reason)
        assert all(r.skipped for r in report.guardrail_results)
        assert report.gate_passed
        assert "3 skipped" in report.summary()

    def test_client_error_is_failure(self):
        class Boom:
            def apply_guardrail(self, **kw):
                raise RuntimeError("AccessDeniedException")

        results = run_guardrail_checks(self.EXAMPLES[:1], BedrockGuardrail("g", "1", client=Boom()))
        assert not results[0].passed and not results[0].skipped
