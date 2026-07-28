"""llm-eval-kit: Evaluate LLM outputs against golden examples."""

__version__ = "0.1.0"

from llm_eval_kit.config import EvalSuite, GoldenExample, RetrievalExample, ValidatorConfig
from llm_eval_kit.loader import load_suite
from llm_eval_kit.report import EvalReport
from llm_eval_kit.runner import run_suite

__all__ = [
    "EvalReport",
    "EvalSuite",
    "GoldenExample",
    "RetrievalExample",
    "ValidatorConfig",
    "__version__",
    "load_suite",
    "run_suite",
]
