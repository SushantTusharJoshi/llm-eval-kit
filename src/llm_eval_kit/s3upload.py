"""Optional upload of versioned reports to S3 (``--upload-s3 s3://bucket/prefix``).

The baseline stays in git; S3 keeps the history of full reports. Upload
is best effort: without boto3 or AWS credentials it is skipped with a
message and never changes the exit code. The body is the redacted report.
"""

import re
from datetime import datetime, timezone
from typing import Any

from llm_eval_kit.security import redact


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri}")
    return bucket, prefix.strip("/")


def report_key(prefix: str, suite_name: str, now: datetime, sha: str | None = None) -> str:
    safe_suite = re.sub(r"[^A-Za-z0-9._-]", "-", suite_name) or "suite"
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{sha[:12]}.json" if sha else f"{stamp}.json"
    return "/".join(p for p in (prefix, safe_suite, name) if p)


def upload_report(
    report_json: str,
    uri: str,
    suite_name: str,
    *,
    client: Any = None,
    region: str | None = None,
    now: datetime | None = None,
    sha: str | None = None,
) -> tuple[bool, str]:
    """Return (uploaded, message). Never raises for missing boto3 or credentials."""
    bucket, prefix = parse_s3_uri(uri)
    if client is None:
        try:
            import boto3
        except ImportError:
            return False, "S3 upload skipped: boto3 not installed (pip install 'llm-eval-kit[bedrock]')"
        session = boto3.Session(region_name=region)
        if session.get_credentials() is None:
            return False, "S3 upload skipped: no AWS credentials"
        client = session.client("s3")
    key = report_key(prefix, suite_name, now or datetime.now(timezone.utc), sha)
    try:
        client.put_object(
            Bucket=bucket, Key=key, Body=redact(report_json).encode("utf-8"), ContentType="application/json"
        )
    except Exception as e:  # botocore error types vary; report, do not crash the run
        return False, f"S3 upload failed: {type(e).__name__}: {redact(str(e))[:200]}"
    return True, f"Report uploaded to s3://{bucket}/{key}"
