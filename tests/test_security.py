"""Security: secret redaction (planted fake key), safe YAML, regex limits."""

import json

import httpx
import pytest
import yaml
from pydantic import ValidationError

from llm_eval_kit import cli
from llm_eval_kit.config import EvalSuite, GoldenExample, ValidatorConfig
from llm_eval_kit.loader import load_suite
from llm_eval_kit.providers.openai_compat import groq_provider
from llm_eval_kit.runner import run_suite
from llm_eval_kit.security import redact, redact_obj, secret_values
from llm_eval_kit.validators.structural import MAX_REGEX_INPUT_CHARS, validate_regex
from tests.conftest import StubProvider

PLANTED = "gsk_PLANTED_FAKE_KEY_do_not_leak_42"


class TestRedact:
    def test_env_secret_is_masked(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", PLANTED)
        assert PLANTED in secret_values()
        assert redact(f"key={PLANTED}!") == "key=***!"

    def test_non_secret_env_names_ignored(self):
        assert secret_values({"HOME": "/Users/someone/long/path"}) == []

    def test_short_values_ignored(self):
        assert secret_values({"API_KEY": "abc"}) == []

    def test_nested(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "tok_123456789")
        assert redact_obj({"a": ["x tok_123456789", 3]}) == {"a": ["x ***", 3]}


class TestPlantedKeyNeverWritten:
    def test_report_and_cli_output_have_no_key(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("GROQ_API_KEY", PLANTED)

        def handler(request):
            # Server echoes the key back in an error body and in a model answer.
            if b"leak-error" in request.content:
                return httpx.Response(401, text=f"bad key {request.headers['authorization']}")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": f"the key is {PLANTED}"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        provider = groq_provider(transport=httpx.MockTransport(handler), retries=0)
        monkeypatch.setattr(cli, "build_provider", lambda *a, **k: provider)

        suite = tmp_path / "s.yaml"
        suite.write_text(
            yaml.dump({"examples": [{"input": "leak-error", "expected": "x"}, {"input": "echo", "expected": "x"}]})
        )
        report_path = tmp_path / "report.json"
        code = cli.main([str(suite), "--provider", "groq", "--no-cache", "-v", "--report", str(report_path)])
        out = capsys.readouterr()

        assert code == 1
        text = report_path.read_text()
        assert PLANTED not in text
        assert PLANTED not in out.out and PLANTED not in out.err
        data = json.loads(text)
        assert data["results"][0]["error"] == "provider_error"
        assert "***" in data["results"][1]["actual"]

    def test_report_to_json_redacts(self, monkeypatch):
        monkeypatch.setenv("SERVICE_SECRET", PLANTED)
        p = StubProvider(responses={"q": f"leaked {PLANTED}"})
        report = run_suite(EvalSuite(examples=[GoldenExample(input="q", expected="x")]), p)
        assert PLANTED not in report.to_json()
        assert PLANTED not in report.summary()


class TestSafeYaml:
    def test_python_tags_rejected(self, tmp_path):
        path = tmp_path / "evil.yaml"
        path.write_text("!!python/object/apply:os.system ['echo pwned']\n")
        with pytest.raises(yaml.YAMLError):
            load_suite(path)


class TestRegexLimits:
    def test_long_pattern_rejected(self):
        with pytest.raises(ValidationError, match="longer than"):
            ValidatorConfig(type="regex", pattern="a" * 501)

    def test_invalid_pattern_rejected_at_load(self):
        with pytest.raises(ValidationError, match="invalid regex"):
            ValidatorConfig(type="regex", pattern="(unclosed")

    def test_oversized_output_fails_without_running_regex(self):
        cfg = ValidatorConfig(type="regex", pattern="x", invert=True)
        result = validate_regex("a" * (MAX_REGEX_INPUT_CHARS + 1), cfg)
        assert not result.passed
        assert "too long" in result.detail
