# llm-eval-kit

Evaluate LLM outputs against golden examples with CI-friendly pass/fail gating.

## Features

- **Golden example testing** — define input/expected pairs, compare outputs with exact match, numeric tolerance, or cosine similarity
- **Structural validators** — verify JSON validity, required keys, string length bounds, and regex patterns
- **Retrieval confidence** — check RAG pipeline quality by scoring context chunk relevance
- **CI gating** — set a pass rate threshold, get exit code 0 (pass) or 1 (fail) for pipeline integration
- **Pluggable providers** — implement `LLMProvider` and `Embedder` interfaces to wire any model

## Install

```bash
pip install llm-eval-kit
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Define a suite

```yaml
# suite.yaml
name: my-eval
examples:
  - input: "What is 2+2?"
    expected: "4"
    comparator: exact

  - input: "Estimate pi"
    expected: "3.14159"
    comparator: numeric
    abs_tol: 0.01

pass_rate: 0.9
```

### 2. Run with the CLI

```bash
llm-eval-kit --mock suite.yaml
```

### 3. Use the Python API

```python
from llm_eval_kit import load_suite, run_suite
from llm_eval_kit.providers.mock import MockProvider, MockEmbedder

suite = load_suite("suite.yaml")
provider = MockProvider(responses={"What is 2+2?": "4"})
report = run_suite(suite, provider)

print(report.summary())
assert report.gate_passed
```

## Comparators

| Name | Description | Config |
|------|-------------|--------|
| `exact` | String equality | `case_sensitive`, `normalize_whitespace` |
| `numeric` | Number extraction + tolerance | `abs_tol`, `rel_tol` |
| `cosine` | Embedding similarity | `threshold`, requires `Embedder` |

## Validators

| Type | Description | Config |
|------|-------------|--------|
| `json` | Valid JSON check | `invert` |
| `json_keys` | Required keys in JSON object | `keys`, `invert` |
| `length` | Character count bounds | `min_length`, `max_length`, `invert` |
| `regex` | Pattern matching | `pattern`, `invert` |

## Implementing a Provider

```python
from llm_eval_kit.providers.base import LLMProvider, Embedder

class MyProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        # call your LLM here
        return response

class MyEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # call your embedding model
        return vectors
```

## Development

```bash
pytest                    # run tests
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

## License

MIT
