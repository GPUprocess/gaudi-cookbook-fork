# MTP-002 — MTP depth 2 sweep (chained single-head draft)

**Category:** MTP depth
**Date:** MTP phase (second validation window; see Git history)
**Author/agent:** project agent
**Git commit:** `e3c6f52` branch state (k=2 exercised via the same launch script with
`num_speculative_tokens: 2`)
**Status:** VERIFIED (functionality); REGRESSION-neutral (performance) — k=1 remains default

## 1. Problem

If depth 2 doubles accepted tokens per step (acceptance ~2.5 of 3.0), decode could be
substantially faster — provided the second draft step and the 3-row verify cost less than they
save.

## 2. Hypothesis

Depth 2 is architecturally supported by chaining the single MTP head (no second prediction
head exists — the draft re-applies layer 45 to its own output). If the chained draft and the
S=4 candidate columns are correct, throughput should scale with acceptance minus the extra
eager draft step.

## 3. Previous research

MTP-001 (k=1 validation); vllm-gaudi speculative machinery (Eagle-style proposer reused for
the chained draft); GaudiGrimoire MCP consultation on graph capture limits (shared storage).

## 4. Previous attempts

None (first depth-2 exercise for this model on HPU).

## 5. Experimental design

- Control: k=1 (MTP-001 configuration).
- Treatment: k=2 — same everything else (window, seqs, TP8+EP, prompts, greedy decoding).
- New paths exercised: per-step draft chaining metadata, S=4 candidate columns, 3-row verify.
- Metrics: acceptance length, decode tok/s, correctness gates, golden comparison.

## 6. Implementation

No new code — the k=1 implementation already generalized (candidate-slot algebra parameterized
by S; verify rows parameterized by k+1). Configuration change only (`num_speculative_tokens:
2`), plus re-validation that the 3-row graph shapes compile.

## 7. Results

- Acceptance length 2.04–2.52 of 3.0.
- Decode bs=1: 30.2–34.1 tok/s (vs 36.8 k=1) — net-neutral to worse.
- 16-stream aggregate: 41.8 tok/s (vs 42.7 k=1, 59.4 k=0).
- Golden comparison: 1/12 token-identical; divergences at the SAME text positions as k=1
  (model-inherent near-ties, depth-stable).
- Long generation: 1918 tokens, zero degeneration; follow-up and second long generation
  coherent (candidate-slot chaining held across ~1900 sequential verify steps).

## 8. Correctness and accuracy

- Correctness: PASS (full harness re-run at k=2, block boundaries, needles, churn).
- Accuracy gate: not yet runnable (suite pending); correctness evidence as above.

## 9. Interpretation

The chained single-head design is correct and the state machinery scales with depth, but the
second draft step runs eagerly and its per-step CPU/GPU cost exceeds the extra accepted-token
yield in this configuration. This matches the documented upstream behavior where draft
execution cost dominates at constrained draft paths.

## 10. Decision

Retain k=1 as the default; keep k=2 as a supported, correct capability. Re-test depth after
the draft HPU graph (A014/A018) lands.

## 11. Reproducibility

Set `num_speculative_tokens: 2` in `ports/glm-5-3-flash/serving/serve_mtp.sh` (verify rows =
seqs × 3; re-validate graph shapes) and run the same harness. Requires the same environment as
MTP-001.
