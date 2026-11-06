# Verified Performance Baseline

This file is the authoritative record of verified performance.

Do not populate it with guessed values or unverified historical claims.

Keep provisional and failed results in EXPERIMENTS.md and the corresponding experiment reports.

## Current status

- Baseline status: NOT YET VERIFIED
- Model: GLM-5.3-Flash FP8 (revalidate exact checkpoint/artifact).
- Hardware: 8× Intel Gaudi 2 (HL-225); actual HPU allocation must be recorded per benchmark.
- Engine/runtime: Revalidate exact repository revision and software versions.
- Current MTP observation: MTP=1 reportedly produced approximately 21% decode throughput uplift at batch size 1.

**Status: PROVISIONAL — controlled baseline and full run metadata still required.**

## Verified results

No verified performance figures have been entered yet.

| Metric | Configuration | Verified value | Workload/conditions | Experiment report |
|---|---|---|---|---|
| TG/decode | TBD | TBD | TBD | TBD |
| PP/prefill | TBD | TBD | TBD | TBD |
| TTFT | TBD | TBD | TBD | TBD |
| Aggregate throughput | TBD | TBD | TBD | TBD |
| Context/memory | TBD | TBD | TBD | TBD |

## Baseline establishment checklist

- [ ] Exact model/checkpoint and precision recorded.
- [ ] Engine, code commit, dependencies, and launch configuration recorded.
- [ ] HPU count, allocation, and parallelism recorded.
- [ ] Workload, prompt/output lengths, batch size, concurrency, and decoding settings recorded.
- [ ] Measurement boundaries and token counting method defined.
- [ ] Control and treatment use equivalent conditions.
- [ ] Warmups and repeated runs completed; raw per-run data preserved.
- [ ] Summary statistics and variability reported.
- [ ] Correctness and accuracy evaluation completed.
- [ ] Accuracy regression is within the agreed maximum of 0.5 absolute percentage points, or the result is explicitly marked inconclusive/failing.
- [ ] Reproduction instructions and experiment report linked.
- [ ] Result reviewed and marked VERIFIED.

## Update rules

Only update the verified figures after all applicable requirements in docs/BENCHMARK_POLICY.md have passed.

Preserve prior baseline entries in Git history and document any replacement baseline with a new experiment report and rationale.
