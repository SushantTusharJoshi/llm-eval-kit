# Architecture

How one `llm-eval-kit` run travels through the code on `main`, and what the workflows run. Every box names the file that does the work (paths under `src/llm_eval_kit/` unless shown).

```mermaid
flowchart TB
  subgraph runtime["Runtime: one llm-eval-kit run"]
    direction TB
    cli["CLI, flags and exit codes 0 / 1 / 3<br/>cli.py main()"]
    loader["Suite loader, yaml.safe_load or JSON<br/>loader.py load_suite()"]
    config["Suite schema, regex length cap<br/>config.py EvalSuite (pydantic)"]
    factory["Provider factory<br/>providers/factory.py build_provider()"]
    runner["Runner, one case at a time<br/>runner.py run_suite()"]

    cli --> loader --> config
    cli --> factory
    config --> runner
    factory --> runner

    caller["Model call: cache, then budget, then call, then price<br/>calls.py ModelCaller.call()"]
    cache[("File cache, SHA-256 key<br/>cache.py ResultCache, make_cache_key()")]
    budget["Run limits: --max-cost, --max-calls, --max-run-tokens<br/>reliability.py Budget"]
    pricing["Dated price table<br/>pricing.py estimate_cost()"]
    runner --> caller
    caller <--> cache
    caller --> budget
    caller --> pricing

    mock["Mock, offline echo<br/>providers/mock.py MockProvider"]
    oai["Groq and OpenAI-compatible, httpx<br/>providers/openai_compat.py"]
    retry["Timeout and capped exponential backoff<br/>reliability.py call_with_retries()"]
    bedrock["Bedrock Converse, boto3<br/>providers/bedrock.py"]
    caller --> mock
    caller --> oai --> retry
    caller --> bedrock

    cmp["Comparators<br/>comparators/exact.py, numeric.py, cosine.py"]
    judge["LLM judge, strict JSON verdict<br/>comparators/judge.py parse_verdict()"]
    val["Structural validators: json, json_keys, length, regex<br/>validators/structural.py"]
    retr["Retrieval scoring, mock embedder from the CLI<br/>retrieval.py check_retrieval()"]
    guard["Bedrock ApplyGuardrail checks, skipped without<br/>BEDROCK_GUARDRAIL_ID / _VERSION<br/>guardrails.py"]
    runner --> cmp
    runner --> judge
    judge -- "judge model call" --> caller
    runner --> val
    runner --> retr
    runner --> guard

    report["Report: per-case tokens, latency, cost, cached<br/>report.py EvalReport"]
    runner --> report
    gate{"Regression gate vs baseline JSON in git<br/>critical cases strict, quality within tolerance<br/>regression.py evaluate_regression()"}
    report --> gate
    redact["Secret redaction on every writer<br/>security.py redact()"]
    report --> redact
    redact --> out["Summary on stdout, --report JSON file"]
    redact --> s3["Optional --upload-s3, never changes the exit code<br/>s3upload.py upload_report()"]
    gate --> exit["Exit code"]

    agree["Separate CLI: judge agreement, percent and Cohen's kappa<br/>agreement.py, examples/judge_agreement/labels.csv"]
    agree --> judge
  end

  subgraph ci["CI and automation: .github/workflows/"]
    direction TB
    action["Composite action: setup-python, optional OIDC role,<br/>pip install the kit, run the CLI<br/>action.yml"]
    ciyml["ci.yml, push to main and pull requests<br/>test: ruff + pytest with coverage on Python 3.10, 3.11, 3.12<br/>action-smoke: runs action.yml from the checkout, mock provider,<br/>examples/regression_suite.yaml + baseline"]
    live["live.yml, push to main and manual only<br/>groq: pytest -m live -k groq with secrets.GROQ_API_KEY<br/>bedrock: only if vars.AWS_ROLE_ARN is set, OIDC role"]
    grd["guardrails.yml, nightly and manual<br/>only if vars.AWS_ROLE_ARN and vars.BEDROCK_GUARDRAIL_ID are set<br/>examples/guardrail_suite.yaml"]
    ciyml --> action
  end
```

Notes, each checked against the code:

- The runner and the judge both call models through `ModelCaller`, so cache hits, the spend cap and the usage totals treat them the same way ([calls.py](../src/llm_eval_kit/calls.py)). The budget is checked before each live call; cache hits do not touch it.
- All-mock runs are never cached ([cli.py](../src/llm_eval_kit/cli.py)). The CLI passes an embedder only for the mock provider, so `cosine` cases and retrieval checks need `--provider mock` from the command line ([cli.py](../src/llm_eval_kit/cli.py), [comparators/__init__.py](../src/llm_eval_kit/comparators/__init__.py)).
- The Bedrock provider relies on botocore's `standard` retry mode instead of `call_with_retries` ([providers/bedrock.py](../src/llm_eval_kit/providers/bedrock.py)).
- `--update-baseline` writes the baseline only when no critical case fails ([cli.py](../src/llm_eval_kit/cli.py), [regression.py](../src/llm_eval_kit/regression.py)).
- In the workflows, only the `bedrock` job in `live.yml` and the `guardrails` job get `id-token: write`; everything else runs with `contents: read`.
