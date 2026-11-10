# RA-017 — ECFP8 (ECF8): lossless exponent-concentrated FP8 weights for GLM-5.3-Flash on Gaudi 2

**Status:** PROPOSED (research architect, 2026-11-09) · **Priority:** 4b — the Phase-0 measurement (1 day, CPU-only) starts NOW; the kernel work stacks on the Q21/RA-012 fused-dequant-GEMM skeleton · **Complexity:** Phase 0 Low / Phase 1-2 Medium-High (kernel) / Phase 3 Medium (the 4-HPU loader)
**Parent:** `BOTTLENECK_RESEARCH.md` (Section 3, RA-017) · **Targets:** BG-7 (the 4-HPU 1M-context KV fit — the FINAL deployment), B1 (the weight-streaming floor, secondary at 8 HPU)
**User request (2026-11-09):** "a long dive into getting a version of GLM 5.3 flash that is quantized using the new and novel ECFP8. Estimate the impact of performance and how it could be implemented."
**Naming:** the paper/repo call it **ECF8** (Exponent-Concentrated FP8); "ECFP8" is the shorthand. This note uses both.
**ID note:** RA-016 went to the DSpine drafter (earlier user request in the same session); this proposal is RA-017.

---

## 1. Executive summary

ECF8 (arXiv 2510.02676, ICLR 2026, Rice/Lambda — Yang, Zhang, Xie, Li, Xu, Shrivastava; code at
github.com/zeyuyang8/ecf8) is a **lossless** compression framework for native-FP8 (E4M3) weights, built
on a proven statistical law: the 4-bit exponent fields of trained weights carry only ~2-3 bits of
entropy (α-stable distributions induced by SGD → tight entropy bounds; theoretical compression limit
≈ FP4.67). ECF8 Huffman-codes the exponents (variable-length, ≤16-bit codes, per-tensor tree) and stores
the sign+mantissa as a packed 4-bit nibble stream (0.5 bytes/element). Measured on released FP8 LLMs:
**9.8-14.8% weight-memory reduction, bit-exact** (DeepSeek-R1-0528 671B MoE: 14.8%; Qwen3-235B-A22B
MoE: 14.4%; Llama-3.3-70B dense: 13.4%; Qwen3-8B: 9.8%). The paper's "up to 26.9% / 177.1%" headline
numbers are **DiT models (Wan2.2, FLUX.1)**; the LLM-class number for our planning is **~13-15% on the
FP8 mass** (≈12% of the full 328.3 GB checkpoint including the BF16 parts).

**The decisive correction to the paper's framing for our project:** the paper's throughput gains
(11.3-177.1%) are **memory-capacity effects** — smaller weights → larger batches under a fixed memory
budget (their Table 2: batch 2→16, 32→64, 32→48…) — NOT per-token decode-bandwidth gains. Their
inference mode (JIT layer-wise decompression via forward hooks: compressed weights on device/host →
decompress each layer into a scratch buffer before its GEMM) would **multiply the per-step HBM weight
traffic ~2.86×** on a system where the weights are HBM-resident (0.855× read compressed + 1.0× write
scratch + 1.0× read scratch = 2.855× vs the baseline 1.0× read) — **catastrophic for our decode**
(bs=1 floor 15.5 ms → ~44 ms). That mode is rejected here (RJ-20).

**The value of ECFP8 for this project, precisely:**
1. **4-HPU final deployment (the binding value — closes BG-7):** weights stored compressed in HBM
   (328.3 → ~282 GB) + **fused decompress-in-register GEMM** (the RA-012/Q21 TPC pattern with the
   Huffman decode in the load path) → 1M-context KV (≈13 GB) + graphs (≈25 GB) fit with ~64 GB margin
   instead of the current ~8-18 GB (BG-7). This is the cleanest 4-HPU fit solution that ALSO cuts the
   per-card weight bandwidth 14.5% — better than FP8-KV (the RJ-10 needle risk, no bandwidth help) or
   a window reduction (product trade-off).
2. **8-HPU dev (secondary):** the same fused kernel cuts the weight-streaming floor 14.5% (bs=1:
   15.5 → 13.3 ms ⇒ 119 → 139 t/s ceiling; Q21 routed-only floors scale by the same factor). Modest
   alone; the 8-HPU system is not memory-constrained, so there is no capacity effect.
3. **Zero accuracy risk by construction:** lossless = the E4M3 byte stream into the GEMM is
   bit-identical to today's → the 0.5 pp gate is trivially met (verified on our checkpoint in Phase 1).

**The make-or-break question** (a microbenchmark, not a paper argument): the Huffman decode is a
sequential per-element ALU chain (~8-15 ops/element: 1-2 LUT lookups + 64-bit window shifts + nibble
assembly). Gaudi2 TPC decode GEMVs are already latency/issue-bound at small M (Strata: 5-tile expert
GEMV = 237 GB/s = 10% of HBM peak). At bs=1 (M=2 verify rows) the decode ALU can rival or exceed the
FMA work (2×M FLOPs/element) → the net gain at bs=1 may be a fraction of the 14.5% byte saving. At
@16 (M=32-48) the FMA work (64-96/element) dominates the decode (~10) → the full byte saving
materializes. **Go/no-go gate: a standalone TPC decode microbenchmark (Phase 2a) measuring effective
compressed-read GB/s; the fused kernel must beat the baseline kernel's effective GB/s at the target M.**

## 2. What ECF8 is (verified from the paper + repo, 2026-11-09)

**The compression (Section 2 of the paper, Theorem 2.1):** for X ~ symmetric α-stable (the SGD-induced
weight law), E = ⌊log2|X|⌋ follows a two-sided geometric with q = 2^(-α); H(E) is bounded
(α=2 → 1.6 ≤ H(E) ≤ 2.67 bits). Empirically, FP8 exponent entropy in trained LLMs/DiTs measures
**~2-3 bits per layer** vs the 4 bits allocated. Total floor: 2.67 (exponent) + 1 (sign) + 1 (mantissa)
≈ **4.67 bits/element** ("FP4.67") — the theoretical limit; ECF8 is the practical engineering point
(4 bits for sign+mantissa, Huffman for the exponent).

**The format (Section 3 + Appendix E/F, verified from the pseudocode):** per FP8 tensor,
| Component | Size | Content |
|---|---|---|
| `LUT` | nluts × 256 bytes (nluts ≤ ~17; typically ≤4 for ≤16-bit codes → **≤ ~1-4 KB/tensor**) | cascaded byte-aligned decode tables: entry < 240 = the decoded exponent (0-15); entry ≥ 240 = pointer to a 2nd-level subtable (256−entry); the LAST row is the code-length table bℓ[x] |
| `encoded` | nbytes (≈ 0.35-0.42 × nelem) | the Huffman bitstream of the exponent fields (≤16-bit codes; max-length constraint via rare-symbol frequency adjustment) |
| `packed` | ⌈nelem/2⌉ bytes (0.5 B/element) | sign+mantissa as 4-bit nibbles (bit7 = sign, bits4-6 = mantissa) — stored uncompressed |
| `outpos` | (nblocks+1) × int64 | per-thread-block output positions (the separate-decode mode's placement metadata) |
| `gaps` | ⌈nthreads/2⌉ bytes (4-bit each) | per-thread bit-alignment gaps (same) |

**Bit-exactness:** decode reassembles byte = (x<<3) | (q&0x80) | ((q>>4)&0x07) — the original E4M3
byte (sign, exponent field, mantissa). The GEMM math is unchanged (E4M3 × FP32 block scale, exactly as
today). Losslessness is verified in the repo (`compress.py --validate_cuda`; boolq 0.82/0.82 identical
on Qwen3-8B in their table).

**The decode kernel (Algorithm 1, CUDA):** each thread owns B=8 bytes (+2 lookahead) of the bitstream;
a 64-bit sliding window L + 16-bit tail S in registers; Phase 1 counts decodable symbols; a block-level
prefix-sum (shared memory) assigns output positions; Phase 2 re-decodes, fetches the packed nibble,
assembles the FP8 bytes into a shared write buffer; coalesced write-back. **For the FUSED GEMM design
(ours, not theirs): the counting phase, the prefix sum, and the write-back are all eliminated** — the
decode is interleaved with the FMA (decode element i's exponent → nibble → e4m3 byte → dequant → FMA),
and each lane's byte span is made byte-aligned at ENCODE time (flush every span), so no `gaps` metadata
is needed — only a per-span byte-offset table (precomputed, ~1.5% overhead).

**The paper's inference mode (Section 3.3 — the one we do NOT adopt):** PyTorch forward hooks
JIT-decompress each layer's weights into a single pre-allocated buffer (sized to the largest layer)
right before the layer's forward; the buffer is reused next layer. This is an OFFLOAD pattern (weights
compressed on host/low-memory device). On our HBM-resident system it costs the 2.86× traffic
(Section 1) and buys nothing — rejected (RJ-20).

## 3. Gaudi 2 specifics (from GaudiGrimoire + the deployed stack)

1. **Gaudi2's native E4M3 read caps at 240** (corpus, gaudi-cookbook minimax-m3/kimi-k2-7 codecs):
   "Gaudi 2 reads e4m3fn bits with a maximum of 240: exponent field 15 is not finite there" — the
   ports' quantizers keep values in the top-2^7 binade (exponent fields 0-7) with the FP32 block scale
   supplying the rest. **Consequence for Phase 0:** our checkpoint's exponent histogram likely lives in
   {0..7} (8 of 16 values) — a SMALLER Huffman space than the paper's 16-value assumption → our
   compression may be slightly BETTER than the paper's LLM numbers. Phase 0 measures it. (The lossless
   property is unaffected either way: ECF8 reproduces whatever bytes the hardware reads today, exactly.)
2. **The lossless-format lesson (corpus, gaudi-cookbook minimax-m3.md):** "A lossless format change can
   be the FASTEST kernel change. No new kernel was written: the same numbers in a format the matrix
   engine reads natively removed the per-step unpack." — the shortcut (native format, no decode) does
   NOT apply to ECF8 (the compressed form is a bitstream, not a native type) — a decode stage is
   required; but the lesson validates the class of change (lossless re-encoding is production-
   demonstrable on Gaudi2) and its verification discipline (float64 re-decode, byte-exact checks,
   validate.py re-readers — exactly our Phase 1 protocol).
3. **The fused-dequant-GEMM skeleton exists in the project's plan** (RA-012A/Q21): dequant-in-register
   E4M3 GEMV on TPC (the 1Cat `v_f8_ld_tnsr_b` SW_UNPACK + `v_convert_f8_to_bf16_b` pattern, tier D),
   FP32 block-scale multiply, FP32 accumulation — production-qualified at the dequant level (AB-DEQ),
   the B7 load recipe in production. ECF8's fused op is that skeleton **+ the Huffman decode stage** in
   the load path (+ the LUT in TPC local memory, ≤4.4 KB — within the TPC local-store budget; the
   per-layer LUT DMA is KB-scale, negligible).
4. **TPC issue-boundness (corpus, Strata single-card audit):** "5-tile TPC expert GEMV = 237 GB/s = 10%
   of HBM peak → latency- and issue-bound, not bandwidth-bound" — the decode ALU chain competes for
   exactly the issue slots the GEMV already uses; the microbenchmark (Phase 2a) is the decision.
   MME alternative: the matrix engine has more ALU headroom; the tile sweep (M=1/2/8/16/32/48, TPC vs
   MME) picks per M, the Q21 D4 discipline.

## 4. Proposed architecture (the Gaudi 2 ECFP8 stack)

```
OFFLINE (Phase 0-1, CPU/GPU-once):
  GLM-5.3-Flash block-FP8 checkpoint (328.3 GB, E4M3 + FP32 128×128 block scales)
    → ECF8 encoder (per-tensor: exponent histogram → Huffman tree (max 16-bit codes)
       → cascaded LUT (≤4 KB) → bitstream + packed nibbles; SPAN-ALIGNED FLUSH: byte-align every
       256-element span (the TPC lane span) → per-span offset table int32)
    → bit-exact verification: decode all tensors → compare byte-identical to the original E4M3
       (float64 re-decode discipline, the minimax-m3 protocol) → ECFP8 archive per EP shard
    → measure: compression ratio per tensor class (MoE experts / attention / dense), the exponent
       histogram (the {0..7} vs {0..15} question), LUT sizes, code-length distribution
ON-DEVICE (Phase 2, the kernel):
  ecf8_dequant_gemm TPC op (the RA-012A/Q21 skeleton + decode stage):
    inputs: compressed span (bitstream slice) + packed nibble slice + per-span offset table
            + LUT (≤4 KB, DMA'd to TPC local memory once per op) + FP32 block scales
            + BF16 activation rows [M,K] (M ≤ 48 decode rows)
    per TPC lane (owns a byte span of the element-major stream):
      sliding window L (64-bit) + tail S (16-bit) in registers; cursor f
      for each element in the lane's span:
        x = LUT[L>>56]; if x ≥ 240: x = LUT2[(L>>48)&255]     (1-2 local-mem lookups)
        q = packed_nibble (sequential 4-bit stream read)
        e4m3 = (x<<3) | (q&0x80) | ((q>>4)&7)                 (the byte, bit-exact)
        v = e4m3_decode(e4m3) * block_scale                   (the existing 1Cat pattern)
        acc += v * act_row                                    (FMA, FP32 acc)
    (no counting phase, no prefix sum, no write-back — decode fused into the GEMV)
ON-DEVICE (Phase 3, the 4-HPU system):
  weight loader: reads the ECFP8 archive (the compressed shards) instead of the safetensors;
  HBM holds the COMPRESSED weights (282 GB) — no expanded copy, no scratch buffer;
  the FP32 block scales stay uncompressed (1.2 GB, negligible); KV + graphs fit (BG-7 closed).
```

**What is NOT changed:** the model math (bit-exact by construction), the GEMM precision (E4M3 × FP32
scale), the graph structure (the op is a drop-in replacement for the weight-read half of the existing
ops — it captures like the dequant op did in AB-DEQ), the KV, the MTP, the scheduler.

## 5. Expected performance impact (all ESTIMATED until Phase 0/2a measure)

**Compression on our checkpoint (ESTIMATED from the paper's LLM-class data):** 13-15% on the FP8 mass
(320.2 GB → ~274-279 GB), ≈12% on the full checkpoint (328.3 GB → ~289-290 GB; the 8.1 GB BF16 mass
is untouched). Planning number: **0.855 bytes/FP8-byte (14.5% on the FP8 mass)**, pending Phase 0.
(If Phase 0 shows the {0..7} exponent restriction, expect toward 15-16%; if the block-scaled
distribution is flatter, as low as 10-12% — the gate: <8% kills the 4-HPU value.)

| Metric | Current | ECFP8-fused (ESTIMATED) | Notes |
|---|---|---|---|
| 8-HPU bs=1 all-experts floor (B1) | 15.5 ms (38.1 GB/rank) | 13.3 ms (32.6 GB) if the decode is free; 14.0-15.5 ms if issue-bound at M=2 | +0 to +17% on that term; the Q21 kernel makes this term moot at bs=1 anyway (routed-only 0.86-1.26 ms → 0.74-1.08 ms) |
| 8-HPU @16 routed floor (C1) | 9.1 ms | 7.8 ms (M=32-48: decode ≈ 12% of FMA work) | the batch-size regime where the byte saving materializes |
| 4-HPU bs=1 floor | 31 ms (2× per-card) | 26.6 ms | same 14.5% factor |
| **4-HPU 1M-context fit (BG-7)** | ~8-18 GB margin (TIGHT) | **~64 GB margin** (282 + 13 + 25 ≈ 320 vs 384 GB) | the decisive value: 1M context at 4 HPU fits with headroom; the FP8-KV (RJ-10) and window-reduction alternatives become unnecessary |
| 4-HPU @16 (800-target physics, RA-005) | the weight term is 2× the 8-HPU per-card bytes | −14.5% on the weight term on top of Q21/RA-011/RA-012 | composes with everything |
| Load time | 328.3 GB host→HBM | ~282 GB (−14%) | minor |
| Accuracy | — | **bit-exact** (the 0.5 pp gate is trivially met) | Phase 1 byte-identity verification on our checkpoint |

**Conservative (decode issue-bound at small M):** 8-HPU value ≈ the 4-HPU fit only (the kernel at
M≤8 recovers <50% of the byte saving); the 4-HPU fit value is unaffected by the decode cost (the
weights sit compressed in HBM either way — even a SLOW fused decode still fits; the decode cost then
shows up as a step-time tax, bounded by the 2.86× worst case if the decode is pathologically slow —
which the microbenchmark rules out or in).
**Optimistic (decode ≤12% of FMA work at M≥8):** the full 14.5% weight-term saving at every batch
size + the 4-HPU fit.
**Worst case:** Phase 0 compression <8% on our checkpoint (the block-scaled distribution is flatter
than the paper's) → the 4-HPU margin improvement shrinks to ~30 GB (still useful, still better than
nothing) and the decode kernel is not worth it → close after Phase 0 (1 day, no HPU time).

## 6. Risks

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | TPC decode ALU cost exceeds the byte saving at M ≤ 8 (issue-bound GEMV + the sequential decode chain) | performance | Phase 2a microbenchmark FIRST: standalone TPC decode (compressed GB/s) + the fused op vs the RA-012A/Q21 skeleton at M=1/2/8/16/32/48; go/no-go per M (the MME fallback has more ALU headroom — the tile sweep decides); at bs=1 the Q21 routed-only kernel already removes most of the weight term, so a decode-limited fused op degrades gracefully to "the 4-HPU fit value only" |
| R2 | Compression on our block-FP8 checkpoint < the paper's LLM numbers | scope | Phase 0 measures it (1 day, CPU); the 8% gate (Section 5 worst case); the {0..7} exponent restriction (Section 3.1) may make us BETTER than the paper |
| R3 | The repo code has no visible LICENSE (the algorithm spec is fully in the paper) | IP | Re-implement the encoder in Python from the paper's spec (Huffman + cascaded LUT is standard; our bitstream layout is ours — we add the span-flush anyway); the repo is referenced for cross-validation only; if the authors grant a license, use their encoder as a second opinion |
| R4 | The span-flush byte alignment adds size overhead | format | ≤7 bits per 256-element span ≈ ≤0.35% — negligible; measured in Phase 0 |
| R5 | LUT local-memory pressure on TPC (≤4.4 KB/tensor) | kernel | The TPC local store holds one layer's LUT per op (KB-scale vs the 16 KB-class local memory); the LUT is per-tensor (changes per op) — DMA'd at op entry, 4.4 KB × 135 tensors × 8 ranks/step = ~4.8 MB total — 0.02% of the step's HBM traffic |
| R6 | The 4-HPU loader change (reading the ECFP8 archive) touches the weight-load path (B7-adjacent) | integration | Phase 3 is isolated to the 4-HPU config (the 8-HPU dev stack keeps the stock loader until the 4-HPU freeze); the compressed archive is a separate artifact (the stock checkpoint stays as-is — rollback is trivial) |
| R7 | Numerical | — | **None by construction** (lossless — the bytes into the GEMM are bit-identical); the only failure mode is a decode BUG (a wrong byte), caught by Phase 1's byte-identity verification + the A/B parity gate (greedy bit-identical vs the stock loader — a STRONGER gate than RA-012A's, because no re-accumulation-order change exists) |

## 7. Dependencies

- **Q21/RA-012A fused-dequant-GEMM skeleton** (the kernel ECF8 extends — build that first; ECF8 adds
  the decode stage to it). The Phase 0-1 (offline) work has NO dependencies — start immediately.
- **BG-7** (the 4-HPU fit math) — RA-017 is the leading solution; Phase 0's compression number updates
  BG-7's headroom arithmetic directly.
- Composes orthogonally with: RA-011 (host — unchanged), Q21 (routed-only read — ECF8 makes each
  routed expert's bytes 14.5% smaller: the factors multiply), RA-013 (placement — unchanged),
  RA-004/RA-016 (the drafters — the DRAFT model's weights could also be ECF8-compressed if it's FP8;
  the MTP head is BF16 — out of scope).

## 8. Implementation instructions

**Phase 0 — the measurement (1 day, CPU-only, NO HPU, start immediately):**
1. Get the ECF8 encoder (re-implement from the paper's Section 3 spec, or license-check the repo's
   `scripts/compress.py` + `dfloat/` and use it); adapt to our checkpoint: per-tensor (per-layer
   expert weight, per attention weight) encoding of the E4M3 bytes — the exponent field is bits 3-6 of
   each byte; the packed stream is bits {7, 4-6}.
2. Add the SPAN-FLUSH extension: byte-align the bitstream every 256 elements (the planned TPC lane
   span); emit the per-span int32 offset table (the `gaps`/`outpos` metadata is NOT needed for our
   fused design — skip it in our variant; keep it only if we also want the separate-decode fallback).
3. Run on the full checkpoint (the repo's `--n_processes 16` pattern; ~hours on 328 GB) and record:
   (a) compression ratio per tensor class (MoE routed experts 305.5 GB / attention+KDA 14.2 GB / dense
   0.45 GB / shared experts 1.06 GB) → the planning number for BG-7; (b) the per-tensor exponent
   histogram → the {0..7} vs {0..15} question (Section 3.1); (c) LUT sizes; (d) the max-code-length
   distribution (the 16-bit constraint violations — the paper says "rarely violated"); (e) the
   flush overhead (R4).
4. **Gate:** ≥8% on the MoE mass → proceed; <8% → close RA-017 with the measurement (still useful:
   BG-7 keeps its current options).

**Phase 1 — bit-exact verification (days, CPU):**
5. Decode every tensor (CPU re-implementation, independent of the encoder — the minimax-m3
   "validate.py re-reads with its own decoders" discipline); compare byte-identical to the original
   E4M3 (−0 == +0 per the corpus codec convention); any mismatch = stop.
6. Produce the ECFP8 archive layout (per-EP-shard, the safetensors-analog: tensor → {LUT, encoded,
   packed, span_offsets} + the FP32 scales uncompressed + metadata).

**Phase 2 — the kernel (weeks; AFTER the Q21/RA-012A skeleton exists):**
7. **Phase 2a microbenchmark FIRST (the go/no-go, R1):** a standalone TPC kernel that reads the
   compressed span + packed stream and emits the decoded E4M3 bytes (no GEMM) — measure effective
   compressed-read GB/s at the target spans; then the fused `ecf8_dequant_gemm` op (the skeleton +
   the decode stage) vs the stock skeleton at M=1/2/8/16/32/48, TPC vs MME variants. **Go/no-go per M:
   the fused op's effective GB/s must beat the stock op's.** (The Strata data point — 237 GB/s
   issue-bound at M=1 — sets the pessimistic prior; the MME variant is the rescue if TPC decode
   saturates issue.)
8. If go: implement `custom_op::ecf8_dequant_gemm` (the op contract = RA-012A's + the LUT/span-offset
   inputs); the B7 load recipe covers the op lib (already in production); wire behind an env gate
   (`GLM_ECF8_GEMM=1`) with fail-closed fallback to the stock path.
9. Qualification: arithmetic sanity vs the CPU reference (the AB-DEQ protocol), capture test (the
   op in a captured graph — the AB-DEQ precedent), stable serving soak.

**Phase 3 — the 4-HPU system (after the 4-HPU config work, before the 4-HPU freeze):**
10. The weight-loader variant: read the ECFP8 archive at boot (the compressed shards land directly in
    HBM — no expanded copy); update BG-7's arithmetic with the Phase 0 number; the 4-HPU 1M-context
    A/B (the needle + the quality slice + the decode t/s vs the stock-loader 4-HPU arm).

**Diagnostics to capture:** Phase 0's per-class compression + histograms (committed to the experiment
record — they are BG-7's input); Phase 2a's GB/s table per M per engine (the go/no-go record); the
4-HPU pool headroom before/after (the BG-7 closure evidence).

## 9. Validation criteria

1. **Phase 1 (losslessness):** every tensor byte-identical after decode (the hard gate — any
   mismatch aborts the proposal, not the deployment).
2. **Phase 2a (the kernel economics):** the fused op beats the stock op's effective GB/s at the target
   M (the go/no-go; recorded per M).
3. **4-HPU (the product gate):** 1M context serves at TP4 (the needle 10/10 + the quality slice
   within the 0.5 pp budget — trivially expected, lossless) with the pool headroom ≥ the Phase 0
   prediction − 20%; decode t/s within 10% of the stock-loader 4-HPU arm at M=1 (the R1 degradation
   bound) and within the byte-saving prediction at M≥16.
4. **8-HPU (secondary, if the kernel qualifies at M=1-2):** the bs=1 weight term improves ≥5% in the
   per-layer trace (the linear/MoE phase HBM traffic) with greedy parity (bit-identical — R7's strong
   gate).

## 10. Novelty assessment

- **Not attempted** in the project (no ECF8/DFloat entry in TASKS/EXPERIMENTS/DECISIONS; GaudiGrimoire
  search 2026-11-09: no ECF8 content — the corpus's nearest neighbors are the minimax-m3/kimi-k2-7
  lossless MXFP4→FP8 re-encodings (different: format re-encoding, no entropy coding) and the DFloat11
  lineage in the paper's related work (BF16-only, pre-FP8)).
- **Different from the adjacent items:**
  - Q21/RA-002 (routed-only MoE read): orthogonal — Q21 reads FEWER experts; ECF8 makes each expert's
    bytes 14.5% smaller; the factors multiply (Q21×ECF8 bs=1 floor: 0.86-1.26 ms → 0.74-1.08 ms).
  - RA-012A (fused dequant-GEMM): the SKELETON ECF8 extends (same TPC pattern; ECF8 adds the decode
    stage to the load path).
  - RJ-15 (expert tiering RAM/SSD): still rejected on 8 HPU (weights fit); at 4 HPU ECF8-compressed
    HBM-resident (RA-017) is the superior tier (HBM bandwidth, not PCIe/NVMe).
  - RJ-16 (MXFP4 lossy): different class — ECF8 is LOSSLESS (no accuracy study needed; the 0.5 pp
    gate is a non-event); RJ-16 remains the stretch option for byte-count, RA-017 the stretch option
    for the 4-HPU fit.
  - The paper's JIT mode (RJ-20, new rejection): the 2.86× traffic math makes it unusable on
    HBM-resident systems (Section 1); it would apply only to a sub-HBM deployment (not our case).
- **Why it deserves investment:** (a) the user asked; (b) it is the cleanest answer to BG-7 — the
  FINAL deployment's binding constraint (4 HPU × 1M context) — with a side benefit (−14.5% weight
  bandwidth) that composes with every other lever; (c) the risk is front-loaded and CHEAP: Phase 0 is
  1 day of CPU time with a hard gate, and Phase 2a is a microbenchmark with a go/no-go per M — the
  expensive work starts only if the measurement says the physics work; (d) zero accuracy risk by
  construction (lossless) — the rare proposal in this document that cannot regress the 0.5 pp gate.

## 11. Sources (all accessed 2026-11-09)

- arXiv 2510.02676 (full text read, v1): Theorem 2.1 (exponent entropy bounds), Corollary 2.2 (the
  FP4.67 limit), Section 3 (Huffman encoding, cascaded LUT, gap/outpos metadata, the 5-phase CUDA
  decode, the JIT tensor management), Section 4 (Tables 1-3: per-model memory savings — LLMs 9.8-14.8%,
  DiTs to 26.9%; the throughput tables — batch-capacity effects; the pixel-identical Qwen-Image
  losslessness evidence), Appendix E/F (the notation table + Algorithm 1 pseudocode — the byte layout
  and the decode arithmetic verified from the pseudocode).
- github.com/zeyuyang8/ecf8 (ICLR 2026 official implementation): the repo layout (`dfloat/run/
  decode.cu`, `scripts/compress.py`, `inference_llm.py`), the Qwen3-8B reproducibility flow
  (`--validate_cuda`), the boolq 0.82/0.82 identity, NO LICENSE file visible (R3).
- huggingface.co/collections + the OpenReview page (ICLR 2026 acceptance — the user's "new and novel"
  framing confirmed: the paper is ICLR'26).
- GaudiGrimoire (search 2026-11-09): no ECF8/DFloat content; the corpus neighbors — gaudi-cookbook
  minimax-m3 (`fp8_codec.py`: the Gaudi2 e4m3 max-240 constraint, the lossless re-encoding discipline,
  "A lossless format change can be the fastest kernel change"); kimi-k2-7 (the same codec family);
  gaudi2-practical-inference (the Gaudi2 exponent-bias constraint, 4 bias values {3,7,11,15}); Strata
  single-card (the TPC issue-bound GEMV data point — 237 GB/s).
- Project data: B1/C1 (the weight floors), BG-7 (the 4-HPU fit), RA-012A/Q21 (the skeleton), AB-DEQ
  (the op qualification precedent), RJ-10/RJ-15/RJ-16 (the adjacent rejections).
