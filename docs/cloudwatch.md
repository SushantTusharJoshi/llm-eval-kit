# Watching Bedrock in CloudWatch

llm-eval-kit writes no custom metrics. Bedrock already publishes runtime metrics to CloudWatch under the
`AWS/Bedrock` namespace, with and without a `ModelId` dimension, for Converse, ConverseStream, InvokeModel and
InvokeModelWithResponseStream calls
([Monitor bedrock-runtime inference using CloudWatch metrics](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-runtime-metrics.html), checked 2026-11-27).

The ones that matter for an eval job:

| Metric | Unit | Use |
|---|---|---|
| `Invocations` | Count | How many calls the eval run made |
| `InvocationLatency` | Milliseconds | Compare with the `latency_ms` in the kit's report |
| `InputTokenCount`, `OutputTokenCount` | Count | Cross-check the token totals the kit reports |
| `InvocationClientErrors` | Count | Bad requests, access denied, quota rejections |
| `InvocationThrottles` | Count | Throttled calls (not counted in `Invocations`) |

The AWS page notes that a request failing before Bedrock knows the model is published without `ModelId`, so filter
by model only when you accept undercounting errors.

## One suggested alarm

Alarm when the eval role starts failing or getting throttled:

- Metric math: `InvocationClientErrors + InvocationThrottles`, no dimension, statistic Sum, period 5 minutes.
- Threshold: greater than or equal to 5 for 1 datapoint.
- Action: an SNS topic that emails you.

Create it in the CloudWatch console (Alarms, Create alarm, Select metric, Bedrock). This repo does not create it.
