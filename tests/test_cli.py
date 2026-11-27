"""Tests for the CLI entry point."""

import json

import pytest
import yaml

from llm_eval_kit.cli import main


@pytest.fixture()
def passing_suite(tmp_path):
    data = {
        "name": "cli-pass",
        "examples": [
            {"input": "hello", "expected": "Mock response to: hello"},
        ],
    }
    path = tmp_path / "pass.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture()
def failing_suite(tmp_path):
    data = {
        "name": "cli-fail",
        "examples": [
            {"input": "hello", "expected": "impossible answer"},
        ],
    }
    path = tmp_path / "fail.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture()
def json_suite(tmp_path):
    data = {
        "name": "cli-json",
        "examples": [
            {"input": "test", "expected": "Mock response to: test"},
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestCLI:
    def test_passing_suite_exits_zero(self, passing_suite):
        assert main(["--mock", passing_suite]) == 0

    def test_failing_suite_exits_one(self, failing_suite):
        assert main(["--mock", failing_suite]) == 1

    def test_verbose_flag(self, passing_suite, capsys):
        main(["--mock", "--verbose", passing_suite])
        captured = capsys.readouterr()
        assert "[PASS]" in captured.out

    def test_json_suite_file(self, json_suite):
        assert main(["--mock", json_suite]) == 0

    def test_missing_file(self, capsys):
        result = main(["--mock", "/nonexistent/path.yaml"])
        assert result == 1
        assert "Error" in capsys.readouterr().err

    def test_no_mock_flag_errors(self, passing_suite, capsys):
        result = main([passing_suite])
        assert result == 1
        assert "only --mock" in capsys.readouterr().err

    def test_embed_dim_flag(self, passing_suite):
        assert main(["--mock", "--embed-dim", "32", passing_suite]) == 0


class TestCLIStage2:
    def test_report_file_written(self, passing_suite, tmp_path):
        out = tmp_path / "out" / "report.json"
        assert main(["--mock", passing_suite, "--report", str(out)]) == 0
        data = json.loads(out.read_text())
        assert data["gate_passed"] is True
        assert data["provider"] == "mock"

    def test_provider_flag_mock(self, passing_suite):
        assert main(["--provider", "mock", passing_suite]) == 0

    def test_tiny_max_cost_exits_3(self, tmp_path, monkeypatch, capsys):
        from llm_eval_kit import cli
        from tests.conftest import StubProvider

        stub = StubProvider()
        monkeypatch.setattr(cli, "build_provider", lambda *a, **k: stub)
        data = {"examples": [{"input": f"q{i}", "expected": f"stub: q{i}"} for i in range(3)]}
        path = tmp_path / "s.yaml"
        path.write_text(yaml.dump(data))
        code = main([str(path), "--provider", "groq", "--max-cost", "0.000001", "--no-cache"])
        assert code == 3
        assert len(stub.calls) == 1
        assert "spend cap" in capsys.readouterr().out

    def test_cache_used_on_second_run(self, tmp_path, monkeypatch, capsys):
        from llm_eval_kit import cli
        from tests.conftest import StubProvider

        stub = StubProvider(responses={"q": "a"})
        monkeypatch.setattr(cli, "build_provider", lambda *a, **k: stub)
        path = tmp_path / "s.yaml"
        path.write_text(yaml.dump({"examples": [{"input": "q", "expected": "a"}]}))
        args = [str(path), "--provider", "groq", "--cache-dir", str(tmp_path / "c")]
        assert main(args) == 0
        assert main(args) == 0
        assert len(stub.calls) == 1
        assert "Cached calls: 1" in capsys.readouterr().out
        assert main(args + ["--no-cache"]) == 0
        assert len(stub.calls) == 2

    def test_bad_provider_config_exits_1(self, passing_suite, monkeypatch, capsys):
        monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
        assert main([passing_suite, "--provider", "bedrock"]) == 1
        assert "BEDROCK_MODEL_ID" in capsys.readouterr().err
