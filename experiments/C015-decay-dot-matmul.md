# C015 — KDA diagonal decay-dot: anchored MME matmul

**Category:** prefill kernel (H-series/C-series)
**Date:** knowledge-base session
**Author/agent:** project agent
**Git commit:** this branch, kda.py `_gauged_scores`
**Status:** WORKING — landed with a CPU oracle; HPU measurement pending a window

## 1. Problem

`_gauged_scores` computes `S[i,j] = sum_d a[i,d] b[j,d] exp(g[i,d] - g[j,d])`. The
off-diagonal sub-blocks already use an anchored matmul; the diagonal sub-block materialized a
`[sub, sub, d]` elementwise product and reduced it on the TPCs — the measured bottleneck of
KDA prefill upstream (57.6 ms/layer at 512 tokens; the 2048-token TTFT cliff).

## 2. Hypothesis

Factoring the diagonal block's exponent around its own first-row gate turns the elementwise
product into two bounded matrices whose product is an MME matmul, cutting kernel time
multiplicatively (upstream reference: 15.6x per layer, TTFT up to 3.23x).

## 3. Previous research

abotsis commit c860b8ea (found via GaudiGrimoire; RESEARCH-001 addendum 4): per 16-row block,
split the exponent around the block's first-row gate; the -5/token gate bound keeps every
factor inside fp32 (e^75 max within a block); oracle transition tests 12/12 CPU,
test_hpu_kda 14/14 HPU; serving gates unchanged.

## 4. Previous attempts

None in this port (our chunk-size experiment, EXP-MTP-004, tuned around the unfixed kernel).

## 5. Experimental design

- Oracle: random inputs across 6 shapes x {strict, inclusive} keeps; compare the landed
  anchored-matmul form against the pre-C015 elementwise form (max relative error).
- HPU plan (pending window): prefill ladder + TTFT sweep (123-3099 tokens), serving gates
  (rowdep, parity, decode bench), then re-test the chunk-size tradeoff (EXP-MTP-004 may shift).

## 6. Implementation

`kda.py` `_gauged_scores`: the diagonal `mid` block is now
`row_f = a * exp(g_i - g_ref)` (<= 1), `col_f = b * exp(g_ref - g_j)` (exceeds 1 only in the
masked upper triangle, inside fp32 at <= e^75 with the -5/token gate bound),
`mid = (row_f @ col_f.T) * keep`. Off-diagonal blocks unchanged (already anchored matmuls).

## 7. Results

- Oracle (CPU): 12/12 shapes match, max relative error 5.7e-7.
- CPU microbench: ~1.0x (expected — the win is TPC-elementwise vs MME-matmul on HPU).
- HPU measurement: pending.

## 8. Correctness and accuracy

Oracle-equivalent on CPU within fp32 rounding. HPU serving gates pending.

## 9. Interpretation

Numerically equivalent by construction (the factorization is exact in real arithmetic); the
upstream reference measured the expected large kernel-time reduction on HPU.

## 10. Decision

Landed as WORKING; promote to VERIFIED after the HPU prefill ladder + serving gates.

## 11. Reproducibility

`ports/glm-5-3-flash/glm5_next_hpu/glm5_next_hpu/check_c015_gauged_scores.py` (CPU oracle).
HPU: prefill gate with the patched plugin; compare against the pre-change capture.
