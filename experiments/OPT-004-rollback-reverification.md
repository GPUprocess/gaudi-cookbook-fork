# OPT-004 — Post-rollback production re-verification

**Category:** stability / baseline reproduction
**Date:** pre-MTP phase (after the TP4/co-residency experiments)
**Author/agent:** project agent
**Git commit:** pre-fork working tree
**Status:** VERIFIED (reproduction)

## 1. Problem

After a series of configuration experiments (TP4, co-residency with scout instances), confirm
the production TP8 layout still delivers baseline performance.

## 2. Hypothesis

A fresh production boot matches OPT-001 within noise.

## 3. Previous research

EXP-OPT-001.

## 4. Previous attempts

None.

## 5. Experimental design

Fresh TP8 boot, production config, identical workloads to OPT-001.

## 6. Implementation

None (measurement).

## 7. Results

- Decode bs=1: 30.3 / 30.1 tok/s (replicated).
- 8 streams: 136.8 agg (17.1/stream); 16 streams: 217.2 agg.
- TTFT @128K: 25.4 s (4,133 tok/s prefill).

## 8. Correctness and accuracy

Gates passed on the fresh boot.

## 9. Interpretation

Matches OPT-001 within noise; the production layout is stable across configuration churn and
restarts.

## 10. Decision

Confirmed as the reproducible production baseline; adopted as the Stage-1 non-MTP control
configuration (with the window-equivalence caveat documented in EXP-MTP-001).

## 11. Reproducibility

Production launch + the same workload scripts as OPT-001.
