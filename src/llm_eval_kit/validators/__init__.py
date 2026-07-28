"""Validator registry — resolve validator type names to functions."""

from llm_eval_kit.config import ValidatorConfig
from llm_eval_kit.validators.structural import (
    ValidationResult,
    validate_json,
    validate_json_keys,
    validate_length,
    validate_regex,
)

_REGISTRY = {
    "json": validate_json,
    "json_keys": validate_json_keys,
    "length": validate_length,
    "regex": validate_regex,
}


def run_validators(text: str, configs: list[ValidatorConfig]) -> list[ValidationResult]:
    results = []
    for cfg in configs:
        fn = _REGISTRY.get(cfg.type)
        if fn is None:
            results.append(ValidationResult(passed=False, detail=f"Unknown validator type: '{cfg.type}'"))
            continue
        results.append(fn(text, cfg))
    return results


__all__ = [
    "ValidationResult",
    "run_validators",
]
