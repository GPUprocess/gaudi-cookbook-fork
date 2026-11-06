# OPT-001 — Production TP8+EP baseline capture (pre-MTP)

**Category:** baseline / decode / prefill / memory
**Date:** pre-MTP phase (deployment engagement)
**Author/agent:** project agent
**Git commit:** pre-fork working tree (cookbook port at vllm-gaudi v0.26.0, b059f9c)
**Status:** BASELINE CANDIDATE (measured, replicated across two sessions; pre-dates the
current benchmark policy — no formal accuracy gate was run)

## 1. Problem

Establish the verified performance baseline of the production GLM-5.3-Flash deployment before
any optimization.

## 2. Hypothesis

The cookbook production configuration meets the aggregate-throughput and prefill targets on
8× Gaudi 2; single-stream decode is latency-bound and will not meet a 200 tok/s per-request
target without speculative decoding.

## 3. Previous research

Cookbook port docs and gates; the deployment runbooks; GaudiGrimoire MCP for lazy-mode graph
fill behavior (13–15 s per new shape with skip-warmup) and utilization interpretation
(AIP % during latency-bound decode).

## 4. Previous attempts

None (first measurement campaign for this deployment).

## 5. Experimental design

- Production config: TP8+EP, GMU 0.82, KV override 1280 blocks/card (163,840-token pool),
  graph capture 2048, chunked prefill 8192, prefix caching off, 262K context, 32 seqs.
- Workloads: 128-token generations at 1/8/16 streams (with a bs=1 repeat); TTFT ladder
  ~4K/~32K/~128K prompts; hl-smi sampling for HBM/power/utilization.

## 6. Implementation

No code change — measurement campaign against the deployed service.

## 7. Results

- Decode: 30.1 tok/s bs=1 (repeat 29.8); 125.1 agg @8 streams; 213.2 agg @16 streams
  (13.6/stream).
- Prefill: 4,928 tok/s @128K (104,867 prompt tokens); 1,807 @32K; ~185 @4K cold (graph
  compile), warm small-prompt ~770+ tok/s.
- HBM: 65.5 GiB/card steady, 68.2 peak; composition ≈ weights 38.2 + KV ~1.8 + graphs/
  workspace ~25.5.
- Power 81–152 W decode; AIP 0–24% during decode; temp 33–41 °C.

## 8. Correctness and accuracy

Port gates passed (the deployment was gate-verified end to end). Formal accuracy suite: not
run.

## 9. Interpretation

Aggregate (200 tok/s) and prefill (3–6K tok/s) targets MET on 8 HPUs without MTP.
Single-stream 200 tok/s NOT met — decode is latency/sync-bound, motivating the MTP work.

## 10. Decision

Adopted as the pre-MTP baseline. Post-rollback re-verification (EXP-OPT-004) replicated it.

## 11. Reproducibility

Production launch per the deployment runbook; workload scripts preserved in the operator's
local working notes (raw outputs not published). Stage-1 reproduction will re-run this control
under the current benchmark policy with published raw data.
