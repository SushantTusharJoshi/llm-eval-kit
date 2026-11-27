"""Keep secrets out of reports, cache files and error messages.

Anything the kit writes (report JSON, summaries, S3 uploads, cache entries,
error text) passes through ``redact`` first. The secret list is built from
environment variables whose names look like credentials, plus any value a
provider passes in explicitly (its API key).
"""

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|SESSION)", re.IGNORECASE)
_MIN_SECRET_LEN = 8
MASK = "***"


def secret_values(environ: Mapping[str, str] | None = None) -> list[str]:
    """Values of env vars whose names look like secrets, longest first."""
    env = os.environ if environ is None else environ
    values = {v for k, v in env.items() if _SECRET_NAME.search(k) and len(v) >= _MIN_SECRET_LEN}
    return sorted(values, key=len, reverse=True)


# What it does: masks every known secret value inside a piece of text before the kit writes or prints it.
# Catch it: "How do you guarantee an API key never ends up in a CI log or an uploaded report?"
# Card: docs/cards/redact-secrets.md
def redact(text: str, extra: Iterable[str] = ()) -> str:
    """Replace every known secret value in ``text`` with ``***``."""
    # STEP: collect secrets from the environment plus the caller's explicit values
    secrets = set(secret_values())
    secrets.update(s for s in extra if s and len(s) >= _MIN_SECRET_LEN)
    # DECISION: nothing to hide or nothing to scan, so return the text untouched
    if not secrets or not text:
        return text
    # LOOP: longest secret first, so a key that contains another key is masked whole
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, MASK)
    # RETURN: the masked text
    return text


def redact_obj(obj: Any, extra: Iterable[str] = ()) -> Any:
    """Apply ``redact`` to every string inside nested dicts and lists."""
    extra = list(extra)
    if isinstance(obj, str):
        return redact(obj, extra)
    if isinstance(obj, dict):
        return {k: redact_obj(v, extra) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [redact_obj(v, extra) for v in obj]
    return obj
