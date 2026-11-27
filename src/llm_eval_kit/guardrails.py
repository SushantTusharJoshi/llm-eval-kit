"""Optional Bedrock Guardrails checks (``guardrail_examples`` in a suite).

Each case sends fixed text to ApplyGuardrail with source INPUT or OUTPUT
and asserts the guardrail intervened or did not. Guardrail cases are
security checks, so a failure always fails the gate. Without a guardrail
ID (env BEDROCK_GUARDRAIL_ID) or without boto3 they are reported as skipped.
"""

import os
from dataclasses import dataclass
from typing import Any

from llm_eval_kit.config import GuardrailExample
from llm_eval_kit.providers.base import ProviderError
from llm_eval_kit.providers.bedrock import default_region, make_aws_client
from llm_eval_kit.security import redact


@dataclass
class GuardrailResult:
    id: str
    source: str
    expect: str
    action: str | None
    passed: bool
    skipped: bool
    detail: str
    critical: bool = True


class BedrockGuardrail:
    def __init__(
        self,
        guardrail_id: str,
        version: str,
        region: str | None = None,
        *,
        client: Any = None,
    ):
        self.guardrail_id = guardrail_id
        self.version = version
        self.region = region or default_region()
        self._client = client or make_aws_client("bedrock-runtime", self.region)

    def apply(self, text: str, source: str) -> str:
        """Return the action: GUARDRAIL_INTERVENED or NONE."""
        try:
            resp = self._client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.version,
                source=source,
                content=[{"text": {"text": text}}],
            )
        except Exception as e:  # botocore raises many types; report uniformly
            raise ProviderError(f"bedrock: apply_guardrail failed: {type(e).__name__}: {redact(str(e))[:300]}") from None
        return str(resp.get("action", ""))


def guardrail_from_env(region: str | None = None) -> tuple[BedrockGuardrail | None, str]:
    """Build a guardrail from BEDROCK_GUARDRAIL_ID / BEDROCK_GUARDRAIL_VERSION, or explain why not."""
    gid = os.environ.get("BEDROCK_GUARDRAIL_ID")
    version = os.environ.get("BEDROCK_GUARDRAIL_VERSION")
    if not gid or not version:
        return None, "BEDROCK_GUARDRAIL_ID and BEDROCK_GUARDRAIL_VERSION not set"
    try:
        return BedrockGuardrail(gid, version, region), ""
    except ProviderError as e:
        return None, str(e)


def run_guardrail_checks(
    examples: list[GuardrailExample],
    guardrail: BedrockGuardrail | None,
    skip_reason: str = "no guardrail configured",
) -> list[GuardrailResult]:
    results = []
    for ex in examples:
        if guardrail is None:
            results.append(GuardrailResult(ex.id, ex.source, ex.expect, None, False, True, f"Skipped: {skip_reason}"))
            continue
        try:
            action = guardrail.apply(ex.text, ex.source)
        except ProviderError as e:
            results.append(GuardrailResult(ex.id, ex.source, ex.expect, None, False, False, f"Error: {e}"))
            continue
        intervened = action == "GUARDRAIL_INTERVENED"
        want = ex.expect == "intervened"
        passed = intervened == want
        detail = f"action={action}, expected {'GUARDRAIL_INTERVENED' if want else 'NONE'}"
        results.append(GuardrailResult(ex.id, ex.source, ex.expect, action, passed, False, detail))
    return results
