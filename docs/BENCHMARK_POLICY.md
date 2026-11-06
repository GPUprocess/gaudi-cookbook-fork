# Benchmark and Baseline Acceptance Policy

## Purpose

This policy defines when a performance result may be described as an improvement and when it may be promoted into the project's verified baseline.

It applies to:

- Token generation/decode (TG)
- Prefill (PP)
- Time to first token (TTFT)
- MTP
- Concurrency
- Memory use
- HPU scaling
- Kernel performance
- Other performance measurements

## No improvement claim without a verified comparison

A percentage improvement is not accepted unless it is compared against a verified, reproducible baseline measured under equivalent conditions.

The reported approximately 21% batch-1 decode uplift from MTP=1 is a PROVISIONAL REPORTED RESULT, not an established project baseline, until the agent reproduces and documents both control and treatment measurements.

Do not erase, dismiss, or misrepresent the reported result.

Preserve it as a prior observation with its original provenance and mark any missing benchmark details as unknown.

## Establishing the baseline

Before accepting a performance improvement:

1. Identify the exact model/checkpoint and artifact revision, including precision or quantization.
2. Identify code commit, engine/fork revision, configuration, launch arguments, environment variables, and relevant dependency versions.
3. Record HPU count and device allocation, parallelism strategy, batch size, concurrency, input/prompt length, output length, context/KV settings, and any cache state that affects the result.
4. Record host and HPU state relevant to performance, including other workloads and resource contention.
5. Use the same workload, prompt set, generation settings, warmup procedure, measurement boundaries, and stopping conditions for control and treatment.
6. Run a control configuration without the optimization and a treatment configuration with it. Keep all other relevant variables fixed.
7. Perform enough repetitions to estimate variability. Target at least five measured repetitions per condition where practical; use more when results are noisy.
8. Report median, mean, standard deviation or another suitable variability measure, and P95 when meaningful. Preserve individual run results, not just the aggregate.
9. Where practical, alternate or randomize control/treatment order to reduce drift and thermal or system-state bias.
10. Repeat the experiment on a separate run/session when practical before promoting a major result to verified status.

If a historical control cannot be reconstructed, do not invent one.

Establish a new documented baseline and compare the treatment against it.

## Measurement definitions

Every benchmark report must define exactly what is being measured.

At minimum:

### TG/decode
Generated output tokens per second, with the token counting method and measurement interval.

### PP/prefill
Prompt tokens processed per second, with prompt length and timing boundary.

### TTFT
Time from the defined request start point to the first generated token. Specify whether queueing and network time are included.

### End-to-end latency
Request start to completion, with input/output lengths.

### Throughput
Aggregate tokens/sec across the stated number of concurrent requests. Distinguish aggregate throughput from per-request speed.

### MTP
Draft/prediction configuration, verification behavior, proposed tokens, accepted tokens, acceptance rate, verifier overhead, and net end-to-end benefit.

Do not compare measurements that use different definitions as if they were directly comparable.

## Correctness and accuracy gates

Performance is not an improvement if it silently degrades correctness.

- Use a fixed, versioned evaluation suite and preserve its exact prompts, expected outputs or scoring method, decoding settings, and evaluation code.
- Compare control and treatment on the same evaluation set.
- Default maximum allowed regression is 0.5 ABSOLUTE PERCENTAGE POINTS on the agreed accuracy metric, unless the user explicitly approves a different threshold.
- Report per-category results as well as aggregate results. A good aggregate must not conceal a serious regression in an important category.
- Report sample size and uncertainty where possible.
- If the evidence is too weak to determine whether the regression threshold is met, mark the result INCONCLUSIVE, not PASS.
- For speculative decoding, separately verify that the algorithm preserves the intended target-model decoding behavior.
- For greedy decoding, compare against the target-only reference behavior.
- For sampled decoding, verify preservation of the intended sampling distribution; identical sampled sequences are not generally required.
- Test edge cases, stop conditions, EOS handling, partial batches, and relevant context lengths.

## MTP-specific reporting

For every MTP experiment, report:

- MTP disabled / MTP=1 / higher depth as applicable.
- Whether the checkpoint actually contains the required prediction heads or other architecture support.
- Draft-token generation method and target-model verification method.
- Proposed tokens, accepted tokens, acceptance rate, and acceptance distribution by position.
- Draft and verification compute/latency overhead.
- Net decode throughput and end-to-end latency.
- Batch size, concurrency, prompt length, output length, HPU count, and parallelism.
- Correctness comparison against target-only decoding.

Do not assume MTP=2 or deeper is supported merely because MTP=1 works.

Inspect the actual model architecture and checkpoint.

Do not fabricate missing prediction heads or claim native support without evidence.

## Status definitions

Use these labels consistently:

- PROVISIONAL: observed but not yet reproduced or not fully documented.
- BASELINE CANDIDATE: controlled comparison completed, but further validation or review remains.
- VERIFIED: reproducible comparison, complete metadata, and correctness/accuracy gates passed.
- INCONCLUSIVE: insufficient or noisy evidence.
- FAILED: the optimization did not meet the required performance or correctness criteria.
- REGRESSION: performance or correctness materially worsened.

Only VERIFIED results may update the authoritative verified figures in BASELINE.md.

Keep provisional, failed, and inconclusive results in EXPERIMENTS.md and their full experiment reports.

## Required benchmark report fields

Each report must include:

- Experiment ID and date/time.
- Hypothesis and the single principal variable changed.
- Control and treatment commit/configuration.
- Hardware and HPU allocation.
- Model/checkpoint and engine versions.
- Workload and measurement definitions.
- Warmup and repetition counts.
- Per-run raw measurements.
- Median, mean, variability, and P95 where meaningful.
- Percentage change and calculation method.
- Accuracy/correctness results and evaluation-set version.
- Resource utilization and relevant bottleneck evidence.
- Result status and rationale.
- Reproduction instructions and raw output locations.
- Known limitations, confounders, and next action.

## Reporting percentage changes

State whether higher or lower is better.

For a metric where higher is better:

improvement_percent = ((treatment - control) / control) * 100

For a metric where lower is better:

improvement_percent = ((control - treatment) / control) * 100

Always include the underlying control and treatment values.

Never report only a percentage.
