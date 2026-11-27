"""S3 report upload with a fake client; skip paths without boto3 or credentials."""

import builtins
from datetime import datetime, timezone

import pytest
import yaml

from llm_eval_kit import cli
from llm_eval_kit.s3upload import parse_s3_uri, report_key, upload_report

NOW = datetime(2026, 11, 27, 12, 30, 5, tzinfo=timezone.utc)


class FakeS3:
    def __init__(self):
        self.calls = []

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        return {}


def test_parse_uri():
    assert parse_s3_uri("s3://bucket/a/b/") == ("bucket", "a/b")
    assert parse_s3_uri("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        parse_s3_uri("https://bucket/x")


def test_versioned_key():
    assert report_key("evals", "my suite!", NOW, "abcdef1234567890") == "evals/my-suite-/20261127T123005Z-abcdef123456.json"
    assert report_key("", "s", NOW) == "s/20261127T123005Z.json"


def test_upload_redacts_body(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_PLANTED_FAKE_KEY_s3_000000")
    s3 = FakeS3()
    ok, msg = upload_report('{"actual": "gsk_PLANTED_FAKE_KEY_s3_000000"}', "s3://b/p", "suite", client=s3, now=NOW)
    assert ok and "s3://b/p/suite/20261127T123005Z.json" in msg
    call = s3.calls[0]
    assert call["Bucket"] == "b" and call["ContentType"] == "application/json"
    assert b"PLANTED" not in call["Body"]


def test_skipped_without_boto3(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, msg = upload_report("{}", "s3://b/p", "s")
    assert not ok and "boto3 not installed" in msg


def test_skipped_without_credentials(monkeypatch):
    boto3 = pytest.importorskip("boto3")

    class NoCreds:
        def __init__(self, **kw):
            pass

        def get_credentials(self):
            return None

    monkeypatch.setattr(boto3, "Session", NoCreds)
    ok, msg = upload_report("{}", "s3://b/p", "s")
    assert not ok and "no AWS credentials" in msg


def test_client_error_reported_not_raised():
    class Broken:
        def put_object(self, **kw):
            raise RuntimeError("AccessDenied")

    ok, msg = upload_report("{}", "s3://b/p", "s", client=Broken(), now=NOW)
    assert not ok and "AccessDenied" in msg


def test_cli_upload_does_not_change_exit_code(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "upload_report", lambda *a, **k: (calls.append(a) or (False, "S3 upload skipped: test")))
    path = tmp_path / "s.yaml"
    path.write_text(yaml.dump({"examples": [{"input": "q", "expected": "Mock response to: q"}]}))
    assert cli.main(["--mock", str(path), "--upload-s3", "s3://bucket/prefix"]) == 0
    assert calls and calls[0][1] == "s3://bucket/prefix"
    assert "S3 upload skipped" in capsys.readouterr().err
