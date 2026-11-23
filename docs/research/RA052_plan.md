# RA-052 Implementation Plan — MoE Node Collapse via the 1Cat FX-Pass

Status: **PLAN WRITTEN, CENSUS-SEQUENCED** (2026-11-23; the weeks-class
kernel/FX work — the largest structural win in the program)

## Problem
The MoE region dominates the recipe graph: 20.5-26.4k nodes (the
census re-targeted: `moe_multiplexer` 652/graph at the biggest dump +
`batch_gemm` + `gemm` classes across the 42 MoE layers). The host
replay floor 25-45 ms/step tracks the node count. The target: ~6-10k
nodes → the floor 8-15 ms.

## The 1Cat FX-pass pattern (the corpus-verified reference)
The 1Cat `block_fp8_dequant` core proves the pattern: the FX pass
rewrites the module region into one fused op; the TPC kernel executes
the fused op; the registrar collision is bypassed via the PT2
custom-op API (the DSV4.1-proven loading mechanism, the adaptation
staged per RA-002 rev 10).

## The census-sequenced steps
1. **The census input** (banked): the biggest dump
   (HabanaFusedOpLazy_340_350, 3,352 nodes): moe_multiplexer 652,
   logical_slice_fwd 481, reshape 417, expand_dims 336, MME gemm 351,
   memcpy_dma 198, the fused TPC kernels 163x3, static_reshape 65.
2. **The pass 1 — the logical-class elision** (the lowest risk): the
   slice/reshape/expand_dims classes (1,234 nodes = 37% of the
   biggest graph) are the graph-construction plumbing; the FX pass
   folds them into the producers. The estimated collapse: the recipe
   26.4k → ~17k. The floor: the proportional 25-45 → ~16-29 ms.
3. **The pass 2 — the multiplexer fusion**: the 652 moe_multiplexer
   nodes → the fused router-gather op (the Q21 kernel's routing
   half). The estimated: → ~12k nodes; the floor → ~12-21 ms.
4. **The pass 3 — the batch_gemm fusion** (the Q21 kernel proper):
   the routed-union read as one kernel per layer. The estimated: →
   6-10k; the floor → 8-15 ms (the doc's target).
5. **The gate per pass**: the R003 suite (the knife-edge applies to
   every structural change of the decode residual stream) + the
   [LAUNCH]-stamp measurement (the T2 tool now operational).

## The effort
- The pass 1: the FX-pass scaffold (the 1Cat pattern) + the elision
  rules: ~2-3 days.
- The pass 2: the router-gather fused op + the PT2 custom-op
  registration: ~3-5 days.
- The pass 3: the Q21 kernel integration (the RA-002 rev-10 core
  exists, the calibration step pending): ~1-2 weeks.
- The total: ~3-4 weeks to the full target; the pass-1-only win
  (~16-29 ms floor) lands in the first week.

## The risk
The knife-edge (the decode residual stream): every pass gated by the
R003 parity suite; the pass 3 touches the weights (the block-FP8
dequant in-kernel — the poison-byte contract from RA-044).
