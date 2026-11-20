# Q21 vs mxk8 — structural comparison (2026-11-19)

Reference: `kernels/mxfp4-moe/src/mxk8_w13.h` + `mxk8_core.h` (the production-validated
MXFP4 MoE decode kernel, same stack). Subject: the Q21 w13 v12 (`extra/block_fp8_moe_w13_glm/`).
Purpose: the instruction/memory-operation-level comparison per the anti-looping protocol, and
the design input for a performance restructure if Layer H shows the correctness-first design
regressing.

## The two designs

| Aspect | mxk8 w13 | Q21 w13 v12 |
|---|---|---|
| Weight format | MXFP4 (u8, 2 values/byte, nibbles) + u8 E8M0 scales | block-FP8 e4m3 (1 byte/value) + f32 128×128 scales |
| Index space | 1-D over output column pairs P (one point per TPC, `pb = is[0]*P/npts`) | 1-D over (t,k,col) work items with the SAME slice idiom (`ip*W/npts`) — adopted from mxk8 |
| Walk order | route-outer, row-pairs inner, software-pipelined (`mxk8_first`/`mxk8_pass_h4`) | work-item outer, h-spans inner; no pipelining |
| x reuse | staged ONCE per route in VLM (`MXK8_STAGE_X` → `v8_x[4*12]`) | re-loaded from HBM per span per item (L1/L2-resident in practice; the measured cost pending) |
| Weight loads | 256-byte u8 loads per 4 rows per chunk, nibble-split, LUT 457 (E2M1→bf16) | 256-value f8 loads per span (`SW_UNPACK|SW_UNPCK_8_TO_16`), `v_convert_f8_to_bf16_all_b` |
| MACs | 4× `v_bf16_mac_acc32_b` per 512-chunk (4 nibble planes × the staged x vectors) | 2× per 256-span (the `.v1`/`.v2` convert pair × the two x loads) |
| Result extraction | lane-packed float64 h[4] + the lane-id predicate moves (`lid4 == m`) into the batch buffers, ONE dual-group move at flush | scalar `v_f32_reduce_add(acc.v1+acc.v2)` per column |
| Store | `v_bf16_st_tnsr_partial(co, ap, convert_all(r), 2m-1, 0)` per run (the count = the last index) | `v_bf16_st_tnsr_partial(oc, raw, tb, 0, 0)` = 1 value per column |
| Reduction | `MXK8_HSUM`: `v_f32_reduce_add(v1+v2)` — the SAME intrinsic | same |
| Routing | `s8_sorted`/`s8_ge` local arrays (the pre-pass builds the per-expert routes); `experts_min` handled host-side | the (t,k) grid with the -1 sentinel for the non-local EP pairs |
| Activation | fused in-kernel (the gated activation, the batched epilogue) | NOT fused (the PyTorch between-math: SiLU·up·weights — the deliberate split, the TPC has no scalar exp) |

## The verdicts

1. **The MAC/reduce/store intrinsics are IDENTICAL** — the same `v_bf16_mac_acc32_b` lane
   semantics (the element-wise lane j accumulates the operand lane j), the same horizontal
   reduce, the same partial-store convention. The historical "MAC layout" suspicion is
   fully retired: the probes + the passing gate settle it.
2. **The load addressing is the only structural difference that ever mattered**: mxk8's
   staged-VLM x + the fixed row loop vs Q21's direct HBM loads with runtime indices. Both
   proven exact on this stack once the index space is right.
3. **The performance gap (expected)**: the per-element scalar reduce + the 1-value stores +
   the per-item weight-row re-reads are ~O(element) overheads vs mxk8's O(128-lane) batching.
   The real-dim estimate: pass-1 ≈ 2–4 ms at T=32 vs the 0.9–6.9 ms routed floor — likely a
   regression vs the stock step time UNLESS the node-count collapse dominates (the stock MoE
   region is ~18,200 recipe nodes ≈ 24 ms device + the replay share). The Layer H measurement
   decides.
4. **If the restructure is needed**: adopt mxk8's column-pair-per-point walk (the 128-col
   results lane-packed, one reduce per 128, one partial store per run), keep the f8-load +
   ALL_LANES convert in place of the nibble unpack, stage x in VLM per (t). The estimate:
   pass-1 → ~0.5–1 ms at T=32. This is a bounded change to the CS3-verified skeleton (the
   coverage + the dim fixes stay).
