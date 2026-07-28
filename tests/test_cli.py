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
