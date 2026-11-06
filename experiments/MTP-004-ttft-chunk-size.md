# MTP-004 — TTFT under MTP and the KDA prefill chunk-size experiment

**Category:** TTFT / prefill knobs
**Date:** MTP phase
**Author/agent:** project agent
**Git commit:** `e3c6f52` branch state
**Status:** VERIFIED (measurement)

## 1. Problem

The draft head needs its KV filled during prefill (a `prefill_cache_only` pass), which may
hurt time-to-first-token. Separately, the KDA prefill scan chunk size is tunable and the
optimal value for this model/hardware was unknown.

## 2. Hypothesis

(a) The draft prefill pass adds prompt-proportional overhead but totals stay practical.
(b) A larger KDA chunk (128 vs 64) halves the chunk-loop count and should reduce TTFT.

## 3. Previous research

MTP-001; base port KDA implementation (chunked scan prefill, recurrent decode).

## 4. Previous attempts

None.

## 5. Experimental design

- Warm server (graphs compiled), 64 generated tokens, single stream.
- Depths: 2K / 8K / 16K prompt tokens.
- KDA chunk: default 64 vs 128 (via the `GLM5_NEXT_KDA_CHUNK` env knob added for this test).

## 6. Implementation

Added the `GLM5_NEXT_KDA_CHUNK` environment knob to the KDA layer (chunk size read at module
load; const shapes rebuilt accordingly).

## 7. Results

- TTFT @k=1 (warm): 2.1 s @2K, 3.1 s @8K, 4.2 s @16K.
- Chunk=128 was WORSE: 8K TTFT 1.85 s vs 1.29 s; 16K 3.59 s vs 2.63 s (+43%/+36%).

## 8. Correctness and accuracy

Correctness gates unaffected (chunk size is a tiling parameter; gates re-run at 64).

## 9. Interpretation

Larger chunks double the intermediate tensors in the blocked scan kernel; that cost exceeds
the halved loop count. 64 is optimal for this model/hardware combination.

## 10. Decision

Keep chunk=64 as the default; retain `GLM5_NEXT_KDA_CHUNK` for future tuning.

## 11. Reproducibility

`tests/hpu/mtp/check_ttft.py` (phase-timed TTFT ladder) with `GLM5_NEXT_KDA_CHUNK` varied;
server per `serving/serve_mtp.sh`.
