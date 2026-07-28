"""Structural validators for LLM outputs."""

import json
import re
from dataclasses import dataclass

from llm_eval_kit.config import ValidatorConfig


@dataclass
class ValidationResult:
    passed: bool
    detail: str


def validate_json(text: str, config: ValidatorConfig) -> ValidationResult:
    try:
        json.loads(text)
        passed = True
        detail = "Valid JSON"
    except (json.JSONDecodeError, ValueError) as e:
        passed = False
        detail = f"Invalid JSON: {e}"

    if config.invert:
        passed = not passed
        detail = f"(inverted) {detail}"

    return ValidationResult(passed=passed, detail=detail)


def validate_json_keys(text: str, config: ValidatorConfig) -> ValidationResult:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ValidationResult(passed=False, detail="Cannot check keys: invalid JSON")

    if not isinstance(data, dict):
        return ValidationResult(passed=False, detail=f"Expected JSON object, got {type(data).__name__}")

    missing = [k for k in config.keys if k not in data]
    if missing:
        passed = False
        detail = f"Missing required keys: {missing}"
    else:
        passed = True
        detail = f"All required keys present: {config.keys}"

    if config.invert:
        passed = not passed
        detail = f"(inverted) {detail}"

    return ValidationResult(passed=passed, detail=detail)


def validate_length(text: str, config: ValidatorConfig) -> ValidationResult:
    length = len(text)
    issues = []

    if config.min_length is not None and length < config.min_length:
        issues.append(f"length {length} < min {config.min_length}")

    if config.max_length is not None and length > config.max_length:
        issues.append(f"length {length} > max {config.max_length}")

    if issues:
        passed = False
        detail = f"Length check failed: {'; '.join(issues)}"
    else:
        passed = True
        detail = f"Length {length} within bounds"

    if config.invert:
        passed = not passed
        detail = f"(inverted) {detail}"

    return ValidationResult(passed=passed, detail=detail)


def validate_regex(text: str, config: ValidatorConfig) -> ValidationResult:
    if config.pattern is None:
        return ValidationResult(passed=True, detail="No pattern specified, skipping")

    match = re.search(config.pattern, text)
    if match:
        passed = True
        detail = f"Pattern '{config.pattern}' matched"
    else:
        passed = False
        detail = f"Pattern '{config.pattern}' not found in output"

    if config.invert:
        passed = not passed
        detail = f"(inverted) {detail}"

    return ValidationResult(passed=passed, detail=detail)
