"""Local file cache for model calls.

One JSON file per key under ``<dir>/<key[:2]>/<key>.json``. A cached
completion comes back with ``cached=True`` so the report can keep it out
of live latency and cost totals.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_eval_kit.providers.base import Completion
from llm_eval_kit.security import redact

CACHE_KEY_VERSION = 1
DEFAULT_CACHE_DIR = ".llm-eval-cache"


# What it does: hashes every input that can change a model's answer into one stable cache key.
# Catch it: "Design a cache for LLM calls. What goes in the key so you never serve a stale answer?"
# Card: docs/cards/cache-key.md
def make_cache_key(
    *,
    provider: str,
    model: str,
    settings: dict[str, Any],
    prompt: str,
    system: str | None = None,
    rubric: str | None = None,
    context: Any = None,
) -> str:
    """SHA-256 over provider, model, all settings, prompt, system, rubric and context."""
    # STEP: gather every part that can change the output, plus a version to invalidate old entries
    parts = {
        "v": CACHE_KEY_VERSION,
        "provider": provider,
        "model": model,
        "settings": settings,
        "prompt": prompt,
        "system": system,
        "rubric": rubric,
        "context": context,
    }
    # STEP: serialise with sorted keys so dict order never changes the key
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    # STEP: hash the UTF-8 bytes; the digest is fixed length and holds no prompt text
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    # RETURN: hex digest, safe to use as a file name
    return digest


class ResultCache:
    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["cached"] = True
            return Completion(**data)
        except (OSError, ValueError, TypeError):
            return None  # missing or corrupt entry is a miss, never an error

    def put(self, key: str, completion: Completion) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(completion)
        data["text"] = redact(data["text"])
        data["cached"] = False
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # atomic, so a crash never leaves half a file
