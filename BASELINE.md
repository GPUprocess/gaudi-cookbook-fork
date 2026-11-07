# Verified Performance Baseline

This file is the authoritative record of verified performance.

Do not populate it with guessed values or unverified historical claims.

Keep provisional and failed results in EXPERIMENTS.md and the corresponding experiment reports.

## Current status

- Baseline status: STAGE-1 VERIFIED (decode throughput + accuracy gate)
- Model: GLM-5.3-Flash FP8 (revalidate exact checkpoint/artifact).
- Hardware: 8× Intel Gaudi 2 (HL-225); actual HPU allocation must be recorded per benchmark.
- Engine/runtime: Revalidate exact repository revision and software versions.
- Current MTP observation: MTP=1 reportedly produced approximately 21% decode throughput uplift at batch size 1.

**Status: PROVISIONAL — controlled baseline and full run metadata still required.**

## Verified results

No verified performance figures have been entered yet.

| Metric | Configuration | Verified value | Workload/conditions | Experiment report |
|---|---|---|---|---|
| TG/decode bs=1 (wall) | MTP k=1, TP8+EP, 32K/8seq, bucket-pinned | **36.83 mean / 36.91 median** (control 32.16/32.05, +15.1%) | 256-token generations, greedy, natural prompt, 5 reps each | experiments/MTP-001, benchmarks/stage1_results/ |
| PP/prefill @128K | non-MTP, TP8+EP, production window | 4,928 tok/s | 104,867 prompt tokens, chunked prefill 8192 | EXP-OPT-001 |
| TTFT bs=1 (32K window) | MTP k=1 | 0.14 s | 256-token generation, bucket-pinned | benchmarks/stage1_results/ |
| Aggregate @16 streams (no MTP) | non-MTP, TP8+EP, production window | 213.2-217.2 tok/s | 128-token generations, 2 sessions | EXP-OPT-001, EXP-OPT-004 |
| HBM under load (MTP k=1) | TP8+EP, 32K/8seq | 64.1 GiB/card | weights + KV + candidate columns + graphs | EXP-MTP-001 |

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
