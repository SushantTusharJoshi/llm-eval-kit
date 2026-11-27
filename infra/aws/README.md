# AWS setup (policy files only)

Nothing here is created automatically. These files are templates for a least-privilege role that GitHub Actions
assumes with OIDC, so no long-lived AWS keys are stored in the repo. Replace the placeholders
(`ACCOUNT_ID`, `MODEL_ID_*`, `INFERENCE_PROFILE_ID`, `GUARDRAIL_ID`, `REPORT_BUCKET`) before use.

| File | What it allows |
|---|---|
| [permissions-policy.json](permissions-policy.json) | `bedrock:InvokeModel` on listed model ARNs only, `bedrock:ApplyGuardrail` on one guardrail, `s3:PutObject` under one bucket prefix |
| [github-oidc-trust-policy.json](github-oidc-trust-policy.json) | `sts:AssumeRoleWithWebIdentity` from `token.actions.githubusercontent.com`, audience `sts.amazonaws.com`, only for this repo's `main` branch |

Notes:

- Converse is authorised by `bedrock:InvokeModel`; there is no separate Converse action. AWS docs: "To call
  `Converse`, you require permission for the `bedrock:InvokeModel` operation"
  ([Inference using Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html), checked 2026-11-27).
- A cross-region inference profile (IDs like `us.amazon.nova-lite-v1:0`) also needs `bedrock:InvokeModel` on the
  foundation-model ARN in every region the profile can route to. Add those ARNs if you use a profile.
- Check exact model IDs in the Bedrock console for your account and region; this repo does not assert that any ID is
  available to you.
- The trust policy `sub` is `repo:SushantTusharJoshi/llm-eval-kit:ref:refs/heads/main`. That covers push, schedule
  and `workflow_dispatch` runs on `main`. Pull requests get a different `sub` (`...:pull_request`) and cannot assume
  the role. If you use a GitHub environment, the `sub` becomes `repo:OWNER/REPO:environment:NAME`.
- Workflows request `id-token: write` only on the job that assumes the role.

## Console steps

1. IAM, Identity providers, Add provider: OpenID Connect, provider URL `https://token.actions.githubusercontent.com`,
   audience `sts.amazonaws.com`.
2. IAM, Roles, Create role, Web identity: pick that provider and audience. After creation, replace the trust policy
   with `github-oidc-trust-policy.json` (with your account ID).
3. Attach a policy created from `permissions-policy.json` (placeholders replaced). Remove any statement you do not use,
   for example the S3 one if you skip `--upload-s3`.
4. Bedrock console: make sure the models you list are available to your account in that region.
5. Optional guardrail: Bedrock, Guardrails, create one, then create a version. Note the guardrail ID and version.
6. Optional report bucket: S3, create bucket with Block Public Access on and default encryption. Versioning is optional;
   report keys already include a timestamp and commit SHA.
7. GitHub repo, Settings, Secrets and variables, Actions:
   - secret `GROQ_API_KEY` (for the Groq live job)
   - variables `AWS_ROLE_ARN`, `AWS_REGION`, `BEDROCK_MODEL_ID`, and for guardrails `BEDROCK_GUARDRAIL_ID`,
     `BEDROCK_GUARDRAIL_VERSION`

Jobs that need these skip themselves when the variables are unset.
