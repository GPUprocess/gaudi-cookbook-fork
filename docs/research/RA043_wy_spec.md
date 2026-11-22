# RA-043 Implementation Plan — WY-form KDA Kernel (GaudiInfer Reference)

Status: DESIGN COMPLETE, IMPLEMENTATION STAGED (the TPC kernel track; weeks-class)

## Problem
The KDA chunked scan runs as the loop implementation (the fast reference).
The WY (Wyman-Young) form rewrites the chunked scan as a fixed set of
per-chunk matrix products, enabling one fused TPC kernel per chunk instead
of the sequential recurrence chain.

## GaudiInfer reference algorithm (the WY form)
For a chunk of size B with decay d:
1. `A = tril(diag(d^-1) * K^T K * mask)` — the intra-chunk coefficient
2. `M = d^(i-j)` inter-chunk decay matrix, precomputed per (B, position)
3. `V' = A @ V` intra-chunk mix; `S' = K^T @ V'` new state
4. `out = (Q @ S_prev) * M_row + Q @ V'` — the two-term output
All four are standard GEMM/elementwise: mappable to MME (GEMM) + TPC
(elementwise/mask), no recurrence chain.

## Gaudi mapping
- Per chunk B=128: A is 128x128 (TPC), the two GEMMs are MME (128x128x64)
- The M matrix is a constant tensor per shape (materialized once, cached)
- State update S' is one MME GEMM; the output term is one MME GEMM + one
  elementwise add
- The graph: 4 MME + 2 TPC nodes per layer per chunk vs the ~20-node
  recurrence loop

## Acceptance
- Bit-parity vs the loop reference at bf16 (the T1 gates) on 128-token
  chunks; the max-abs-diff gate at the T1 threshold
- The device time per layer per chunk must beat the loop reference's
  measured span (the D9 fold-in numbers are the baseline)

## Risk
The MME/TPC alternation must overlap (the RA-044 dual-metric discipline:
win node count AND device time); the chunk-boundary state handoff is the
correctness-critical seam.
