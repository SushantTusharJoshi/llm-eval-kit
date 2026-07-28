"""Load eval suites from YAML or JSON files."""

import json
from pathlib import Path

import yaml

from llm_eval_kit.config import EvalSuite


def load_suite(path: str | Path) -> EvalSuite:
    """Load and validate an eval suite from a YAML or JSON file.

    Raises FileNotFoundError if the path doesn't exist.
    Raises ValueError for unsupported file extensions or malformed content.
    Raises pydantic.ValidationError for schema violations.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(path) as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use .yaml, .yml, or .json")

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping at top level, got {type(raw).__name__}")

    return EvalSuite(**raw)
