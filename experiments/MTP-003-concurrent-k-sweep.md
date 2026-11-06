# MTP-003 — 16-stream concurrent k-sweep (like-for-like)

**Category:** concurrency / MTP interaction
**Date:** MTP phase
**Author/agent:** project agent
**Git commit:** `e3c6f52` branch state
**Status:** VERIFIED (measurement)

## 1. Problem

The single-stream MTP gain may not survive concurrent load: verify rows scale with batch
(seq × (k+1)) while the eager draft cost is per-step.

## 2. Hypothesis

Under 16-stream concurrent load, MTP k=1 aggregate throughput drops below the non-MTP control
because the eager draft cost is paid per step regardless of acceptance.

## 3. Previous research

MTP-001/MTP-002; upstream notes that speculative benefits shrink at saturation.

## 4. Previous attempts

None for this model.

## 5. Experimental design

- Like-for-like 32K/8seq window for ALL conditions (k=0, k=1, k=2) — isolates the MTP
  variable from the window variable.
- 16 concurrent streams, staggered start; natural prompts; ~170-token completions.
- Metrics: aggregate tok/s, per-stream mean/min/max, acceptance length, errors.

## 6. Implementation

No code change; three server boots (k=0/1/2) with identical window flags.

## 7. Results

- Aggregate: k=0 **59.4**, k=1 **42.7**, k=2 **41.8** tok/s.
- Single-stream decode within the same config: k=0 31.4, k=1 35.7, k=2 30.4 tok/s.
- Per-stream mean @16: 8.0 / 5.6 / 5.7 tok/s.
- Acceptance under load: k=1 1.72–1.87; k=2 2.34–2.46 (draft quality holds).
- Golden parity in this window: k=0 6/6, k=1 2/12, k=2 1/12 (near-tie class).
- Errors: 0 across all runs.

## 8. Correctness and accuracy

Correctness gates passed in all three configurations. Accuracy gate pending (suite not built).

## 9. Interpretation

MTP is net-positive for single-stream/interactive workloads and net-negative by 16 streams in
the eager-draft configuration — the crossover is measured, not assumed. Acceptance length does
not degrade with batch pressure, so the loss is draft/verify overhead, not draft quality.

## 10. Decision

Enable MTP for single-stream/interactive serving; disable for high-concurrency batch serving.
Documented in the fork README and docs page. Revisit with the draft HPU graph.

## 11. Reproducibility

`tests/hpu/mtp/check_concurrent_16.py` (staggered 16-stream harness) against the server in
each configuration; window flags per `serving/serve_mtp.sh` with k varied.
