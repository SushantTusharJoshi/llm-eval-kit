"""Judge agreement: how often the LLM judge agrees with a human label.

The labeling sheet is a CSV with columns
``id, prompt, output, reference, rubric, human_label``. A human fills
``human_label`` with ``pass`` or ``fail``; blank rows are unlabeled and
ignored. Until at least one row is labeled the status is ``pending``.

Usage::

    python -m llm_eval_kit.agreement examples/judge_agreement/labels.csv \\
        --judge-provider groq [--judge-model MODEL] [--out agreement.json]
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_eval_kit.calls import ModelCaller
from llm_eval_kit.comparators.judge import JudgeComparator, JudgeContext
from llm_eval_kit.config import JudgeConfig
from llm_eval_kit.providers.base import LLMProvider, ProviderError
from llm_eval_kit.providers.factory import PROVIDER_NAMES, build_provider
from llm_eval_kit.reliability import Budget, BudgetExceeded, RunLimits
from llm_eval_kit.security import redact

SHEET_COLUMNS = ["id", "prompt", "output", "reference", "rubric", "human_label"]
_PASS = {"pass", "p", "1", "yes", "y", "true"}
_FAIL = {"fail", "f", "0", "no", "n", "false"}


@dataclass
class AgreementResult:
    status: str  # "pending" until a human labels at least one row, then "computed"
    n_rows: int
    n_labeled: int
    n_judge_errors: int = 0
    agreement: float | None = None
    cohen_kappa: float | None = None
    disagreements: list[str] = field(default_factory=list)


def load_sheet(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in SHEET_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return [dict(row) for row in reader]


def parse_label(value: str | None) -> bool | None:
    v = (value or "").strip().lower()
    if v in _PASS:
        return True
    if v in _FAIL:
        return False
    if v == "":
        return None
    raise ValueError(f"unrecognised label {value!r}; use pass or fail")


# What it does: measures how often the judge agrees with human labels, corrected for chance, on labeled rows only.
# Catch it: "How do you know your LLM judge is any good?"
# Card: docs/cards/judge-agreement.md
def compute_agreement(judge: dict[str, bool | None], human: dict[str, bool | None]) -> AgreementResult:
    """Percent agreement and Cohen's kappa. A judge error (None) counts as a fail, never a pass."""
    # STEP: keep only rows a human has labeled
    labeled = [rid for rid, label in human.items() if label is not None]
    result = AgreementResult(status="pending", n_rows=len(human), n_labeled=len(labeled))
    # DECISION: nothing labeled yet, so report pending instead of a made-up number
    if not labeled:
        return result
    # LOOP: compare judge decision with human label per row; judge errors count as fail
    pairs = []
    for rid in labeled:
        j = judge.get(rid)
        if j is None:
            result.n_judge_errors += 1
        pairs.append((bool(j), bool(human[rid])))
        if bool(j) != human[rid]:
            result.disagreements.append(rid)
    # STEP: observed agreement and chance agreement from each side's pass rate
    n = len(pairs)
    p_o = sum(1 for j, h in pairs if j == h) / n
    j_pass = sum(1 for j, _ in pairs if j) / n
    h_pass = sum(1 for _, h in pairs if h) / n
    p_e = j_pass * h_pass + (1 - j_pass) * (1 - h_pass)
    result.status = "computed"
    result.agreement = p_o
    result.cohen_kappa = None if p_e >= 1.0 else (p_o - p_e) / (1 - p_e)
    # RETURN: the result; kappa is None when both sides gave one answer only
    return result


def judge_sheet(
    rows: list[dict[str, str]], provider: LLMProvider, config: JudgeConfig, caller: ModelCaller | None = None
) -> dict[str, bool | None]:
    """Run the judge on each row's fixed output. None means the judge errored."""
    ctx = JudgeContext(caller=caller or ModelCaller(), provider=provider, config=config)
    decisions: dict[str, bool | None] = {}
    for row in rows:
        rubric = row.get("rubric") or config.rubric
        if not rubric:
            raise ValueError(f"row {row['id']}: no rubric")
        res = JudgeComparator(ctx, rubric=rubric, question=row["prompt"]).compare(row["output"], row.get("reference", ""))
        decisions[row["id"]] = None if res.error else res.passed
    return decisions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm_eval_kit.agreement", description=__doc__.splitlines()[0])
    ap.add_argument("sheet", help="Labeling CSV")
    ap.add_argument("--judge-provider", choices=PROVIDER_NAMES, required=True)
    ap.add_argument("--judge-model")
    ap.add_argument("--region")
    ap.add_argument("--pass-threshold", type=float, default=0.7)
    ap.add_argument("--max-calls", type=int, default=50, help="Run limit for judge calls (default: 50)")
    ap.add_argument("--out", help="Write the agreement JSON here")
    args = ap.parse_args(argv)

    try:
        rows = load_sheet(args.sheet)
        human = {r["id"]: parse_label(r["human_label"]) for r in rows}
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not any(v is not None for v in human.values()):
        result = AgreementResult(status="pending", n_rows=len(rows), n_labeled=0)
        print(f"Agreement: pending (0/{len(rows)} rows labeled). Fill the human_label column first.")
    else:
        try:
            provider = build_provider(args.judge_provider, model=args.judge_model, region=args.region)
        except (ValueError, ProviderError) as e:
            print(f"Error: {redact(str(e))}", file=sys.stderr)
            return 1
        config = JudgeConfig(provider=args.judge_provider, model=args.judge_model, pass_threshold=args.pass_threshold)
        caller = ModelCaller(budget=Budget(RunLimits(max_calls=args.max_calls)))
        labeled_rows = [r for r in rows if human[r["id"]] is not None]
        try:
            judge = judge_sheet(labeled_rows, provider, config, caller)
        except (BudgetExceeded, ValueError) as e:
            print(f"Error: {redact(str(e))}", file=sys.stderr)
            return 1
        result = compute_agreement(judge, human)
        kappa = "n/a" if result.cohen_kappa is None else f"{result.cohen_kappa:.2f}"
        print(
            f"Agreement: {result.agreement:.1%} on {result.n_labeled}/{result.n_rows} labeled rows, "
            f"Cohen's kappa {kappa}, judge errors {result.n_judge_errors}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
