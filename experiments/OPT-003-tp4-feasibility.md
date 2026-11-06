# OPT-003 — TP4+EP feasibility for a 4-HPU footprint

**Category:** HPU scaling
**Date:** pre-MTP phase
**Author/agent:** project agent
**Git commit:** pre-fork working tree
**Status:** VERIFIED (feasibility)

## 1. Problem

A later directive capped GLM at fewer cards (prefer 4). Determine the minimum viable
tensor-parallel footprint.

## 2. Hypothesis

TP4+EP fits the model in 4× 96 GiB cards with reduced graph capture, at a measurable
per-card memory and performance cost.

## 3. Previous research

Head-count arithmetic (64 attention heads); HBM decomposition (EXP-OPT-002).

## 4. Previous attempts

None.

## 5. Experimental design

- TP4+EP on 4 cards; GMU 0.95; graph capture 128 (the 2048-size ladder OOMs with only
  ~14.7 GiB/card non-weight budget); same per-card KV override.
- Compare memory, boot, and functional behavior against TP8.

## 6. Implementation

Config change: `HABANA_VISIBLE_DEVICES` to 4 cards, TP4, GMU 0.95, capture 128.

## 7. Results

- Weights 76.5 GiB/card (TP4 doubles per-card weight residency); viable at GMU 0.95 with
  capture 128.
- Functional boot and serving verified.
- Layout arithmetic: 64 attention heads % {6,5,3} = 0 → 6/5/3-HPU TP invalid; 1/2-HPU
  impossible on memory (328 GiB weights).

## 8. Correctness and accuracy

Functional gates passed at TP4 (pre-MTP). Numerical cross-config comparison: not run.

## 9. Interpretation

A 4-HPU footprint exists and is the Phase-4 target per `docs/HARDWARE_ROADMAP.md`. The
performance/memory tradeoff of dropping from TP8 to TP4 was not fully benchmarked in this
experiment (context/concurrency tradeoffs pending in Phase 4).

## 10. Decision

TP4+EP adopted as the future 4-HPU deployment direction; full Phase-4 configuration,
benchmarking, and accuracy work deferred until the best 8-HPU implementation is established.

## 11. Reproducibility

Device selection + TP/GMU/capture flags per the deployment runbook; no code changes.
