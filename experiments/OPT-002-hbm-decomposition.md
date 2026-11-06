# OPT-002 — HBM decomposition: the graph-capture ladder

**Category:** memory
**Date:** pre-MTP phase
**Author/agent:** project agent
**Git commit:** pre-fork working tree
**Status:** VERIFIED (measurement)

## 1. Problem

At GMU 0.82 the engine leaves ~17% HBM headroom unused. Where does the memory go, and what
can be reclaimed?

## 2. Hypothesis

Weights are known (38.2 GiB/card at TP8+EP); the gap between steady-state HBM and weights is
graph-capture ladder + activation workspace, and it is the first reclaimable block.

## 3. Previous research

GaudiGrimoire MCP on HPU graph memory behavior (captured graphs pin their buffers).

## 4. Previous attempts

None.

## 5. Experimental design

Steady-state and peak hl-smi sampling under load; composition derived from known component
sizes.

## 6. Implementation

None (measurement).

## 7. Results

- Steady 65.5 GiB/card, peak 68.2.
- Weights 38.2; KV pool ~1.8; graphs/workspace ~25.5 GiB/card (the 2048-size capture ladder).

## 8. Correctness and accuracy

n/a (measurement).

## 9. Interpretation

GMU 0.95 is unreachable at capture 2048. Conversely, shrinking the ladder is the first lever
when HBM must fund a bigger KV pool or a smaller-TP layout (used later by the TP4 experiment).

## 10. Decision

Keep capture 2048 in production; document the ladder as the reclaimable block.

## 11. Reproducibility

hl-smi sampling against the production launch; composition arithmetic in the report.
