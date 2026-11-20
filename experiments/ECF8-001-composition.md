# ECF8-001 — Checkpoint composition + exponent analysis (2026-11-19)

Scope: the separate ECF8 workstream (RA-017). Q21 is untouched by this analysis.

## Checkpoint composition (model-00001-of-00062 + the index)

| Class | Dtype | Count (shard 1) | Notes |
|---|---|---|---|
| Expert weights | F8_E4M3 | 329/shard | gate/up [2048, 4096], down [4096, 2048] per expert; 74,562 expert tensors total, 62 shards |
| Expert scales | F32 | 329/shard | weight_scale_inv [16, 32] / [32, 16] — the 128×128 block scales, f32 |
| Dense/embed/head | BF16 | 6/shard | lm_head [154880, 4096], embed_tokens, eh_proj, the shared expert... |
| Norms/misc | F32 | (small) | |

The whole MoE expert stack is FP8 with f32 block scales — exactly the tensors the Q21
kernels consume. The dense path (the attention, the shared expert, the embeddings) is BF16.

## FP8 exponent distribution (layer 45, expert 0, gate_proj, 8 MB sample)

| Statistic | Value |
|---|---|
| Exponent histogram (E=0..15) | [1299, 1374, 2542, 5309, 10430, 20917, 41885, 83625, 167291, 334272, 659228, 1267687, 2159935, 2535935, 1065077, 31802] |
| Exponent entropy | **2.58 bits** of the stored 4 |
| Mantissa entropy | ≈ 2.98 of 3 (uniform — not compressible) |
| Sign balance | 50.0% |
| E=15 (the reserved bin) codes | 31,802 = **0.38%** — present in the weights; production serves correctly, so the HPU decode treats them as finite (the corpus bias-selectable {3,7,11,15} note; do not assume the strict-fn NaN semantics) |
| Per-128-block exponent spread | mean 7.76, p95 11, max 15 — the per-block base+fixed-offset coding does NOT compress |

## Achievable lossless ratio

The total e4m3 entropy ≈ 2.58 (E) + 2.98 (M) + ~0.00 (S) ≈ **5.56 bits vs 8 → the ideal
lossless bound ≈ 1.44×**. A practical per-block fixed-Huffman (the code lengths from the
global histogram, the table in TPC LUT `v_i16_lookup_1c` — the mxk8 table-457 pattern)
should land ≈ 1.35–1.40×.

## HBM impact (TP8 / TP4)

| Config | Weights/card | At 1.37× | Saved/card |
|---|---|---|---|
| TP8+EP (38.2 GiB weights) | 38.2 GiB | ~27.9 GiB | ~10.3 GiB |
| TP4+EP (76.5 GiB weights) | 76.5 GiB | ~55.8 GiB | ~20.7 GiB |

The primary value: the 4-HPU deployment headroom (the roadmap Phase 4), not the TP8 decode
speed (the KV pool is only ~1.8 GiB; the graphs/workspace 25.5 GiB is the bigger TP8 reclaim).

## Decode architecture requirements (the design constraints)

1. The compressed representation stays RESIDENT in HBM; the decode happens per tile in the
   TPC registers — no materialized BF16/FP8 copy (that would negate the saving).
2. The bit-level unpacking (the shifts/ands) + a fixed LUT (the code → the e4m3 byte) is
   TPC-feasible; the arithmetic/range coding is NOT (the serial state).
3. The decode must fuse into the consumer (the Q21-style GEMV load path or the MME operand
   prep) — a standalone decode pass would double the traffic it saves.
4. The scales stay f32 (0.2% of the expert bytes — no compression value).

## Verdict

FEASIBLE at ~1.35–1.40× lossless with the per-block fixed-Huffman + the TPC LUT decode
fused into the weight-load path. A substantial kernel project (the bitstream alignment per
tile, the LUT plumbing, the correctness gates). Recommend: prototype the decode kernel on
ONE expert tensor with the Q21 w13 as the consumer after the Q21 serving integration lands.
