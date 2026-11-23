# RA-053..RA-057 — Round 8 (2026-11-23): the execution-model round

Companion detail to `EXTERNAL_RESEARCH_PROPOSALS.md` revision 17. The owner's directive of
the day: *"the hardware is clearly capable of delivering my target performance... look at
real out-of-the-box optimizations. If need be the whole engine will be rewritten, so will
the drivers, etc."* Constraint re-confirmed: **the model stays FP8 (block [128,128]), the
KV cache stays BF16** — the levers are read counts, launch counts, per-route overhead, and
parallelism, never bytes-per-element.

**The round's headline (DERIVED, shown in §0):** the measured step is NOT host-replay-bound
and NOT HBM-bound. It is **per-route expert-overhead bound**: the expert term is 40-100× its
HBM weight-stream floor (19.5 ms measured vs 0.2-0.5 ms floor at bs1; ~150 ms vs 15.6 ms at
@16). Closing that gap — an **expert-streaming kernel** (one contiguous weight stream per
active expert, tokens pre-sorted into per-expert batches, per-expert GEMM setup instead of
per-route staging) — plus the attention consolidation, the collective reduction, and the
device-side step chain, delivers in arithmetic: **188-288 t/s bs1 and 970-1250 t/s @16 at
8 HPUs WITHOUT a drafter; 198-288 bs1 / 560-770 @16 at 4 HPUs**. Both 8-HPU targets are met
without a drafter; the drafter (RA-049) becomes the 4-HPU @16 answer and the margin, not the
critical path.

Evidence labels: MEASURED (project or cited third-party run), VENDOR (Tier B corpus),
CORPUS (Tier C/D corpus), DERIVED (theoretical computation — shown). GaudiGrimoire
citations in [GG: ...]. All numbers theoretical until measured.

---

## 0. The corrected step model (supersedes the rev-16 host-floor framing)

### 0.1 Why rev-16's wall was wrong

Rev-16 carried the rev-14 model: the 20.5-26.4k-node decode recipe × ~1.4 µs/node host
replay = a 25-45 ms host floor, treated as the critical path. The cookbook's 8-rank trace
(GLM-5.3, TP8+EP8, MTP k=1, same model family and topology, `docs/results/moe-decode-step.md`,
CORPUS Tier D but traced, not estimated) says otherwise, and our own Q17/Q18 measurements are
consistent with it:

- *"Not node-bound. Compute engines are busy 15.4 ms of a 37.4 ms device span; the idle time
  is waiting at all-reduces, not gaps between nodes."*
- *"The host runs ahead during the forward and is on the critical path only at the step
  boundary."* — the 322 graph launches × ~74 µs host each are PREFETCHED; only ~5.5 ms of
  host work sits on the critical path (the step boundary + draft hand-off).

So the critical path = the **device span** + the boundary. The 1.4 µs/node is a real host
cost but it is overlapped, not additive.

### 0.2 The measured decomposition (cookbook trace, 1 stream, 2.8K context, 39.0-39.4 ms step)

| Term | ms/step | How measured |
|---|---:|---|
| Expert kernels, slowest EP rank/layer, summed | **19.5** | max over ranks per layer; **mean rank 9.6 → 8.9 ms is EP imbalance** |
| Attention (MLA decode), all layers | 6.9 | same on every rank |
| MoE outside kernels (gate, shared expert, router, combine) + dense | 3.8 | slowest rank |
| All-reduce latency, 162/step × ~20 µs (24 KB) | 3.4 | last rank ready → first resumed |
| Host work with device idle (step boundary, draft hand-off) | ~5.5 | host vs device timeline |

Our production (bs1 34.9 ms, k=1+T1+Q01v2, FP8 stock op) differs from that trace by exactly
the banked levers: Q18's router fast path (−1.7 ms) + early draft + device-side rejection
(−~2 ms) → boundary ~5.5 → ~2.5-3.5 ms. So: **34.9 ≈ 19 (expert, FP8 stock op, EP8) + 6.9
(attention) + 3.8 (MoE-adjacent) + 3.4 (AR) + 2.5 (boundary) ≈ 35.6 ✓.**

At 4 streams the trace shows the expert term's batch scaling: slowest rank 50.7 ms vs mean
30.7 (20 ms imbalance). At 16 streams (our @16 bench shape: 32 verify rows × 8 routes =
256 routes) the expert term is ~150 ms of the 166 ms measured step (DERIVED: 166 − 6.9
attention × ~2 (row scaling) − 3.8 − 3.4 − 3 ≈ 130-150; the agent's D24 @16 8-rank trace,
§10, is the confirmation task).

### 0.3 The HBM floor (DERIVED from the real checkpoint config, server-verified
`/data/models/GLM-5.3-Flash/config.json`, 2026-11-23)

Config: hidden 4096; 45 layers (34 KDA + 11 DSA, DSA at every 4th from layer 3); 42 MoE
layers (3 dense first); **288 routed experts, top-8, moe_intermediate 2048, 1 shared
expert**; KDA: 64 heads × 128 head_dim, conv-4; DSA: kv_lora_rank 512, 64 heads × 256
NoPE, **top-2048 indexer** (index 32×128, kpool-4 compressed, tail-always-selected);
block-FP8 [128,128] experts (e4m3); BF16 attention/KDA/indexer/router/dense/embedding/lm_head;
vocab 154,880; max_position 1,048,576.

Per-expert weight (FP8, 1 B/param): 3 × 4096 × 2048 = 25.17 MB. 288 × 25.17 MB = 7.20 GB per
MoE layer; × 42 = 302 GB (the bulk of the 328 GiB checkpoint ✓).

**bs1, k=1 (verify + draft = 16 routes/layer), TP8+EP8 (36 local experts/rank):**
- Expert weight stream (routed + shared): ~12-14 local distinct × 3.15 MB (25.17/8) ≈
  40-44 MB/layer → **1.7-1.85 GB/step/HPU → 0.87-0.95 ms at 1.945 GB/s** (the compiler
  cost-model HBM number [GG: libSynapse 1.24.1 `getHbmBwGBps()` = 1945; spec 2.45 TB/s
  Tier B; measured envelope 1.696-2.147 project]).
- Attention weights (BF16, head-sharded /8): ~2.4 GB → 1.2 ms. KDA state read+write:
  8.5 MB (D18 MEASURED) → negligible. DSA KV (top-2048 × 512 × 2B × 11 layers) = 22 MB →
  negligible. **The DSA read is bounded by top-2048, NOT context length** — the 1M-context
  KV term is the indexer read, §0.5.
- Dense (3 layers, BF16) + lm_head (1.27 GB /8) + hc/norms: ~0.5 GB → 0.25 ms.
- **bs1 weight-stream floor ≈ 3.2-3.5 GB/HPU/step → 1.6-1.8 ms** (+ MTP draft pass ~1-1.5
  ms of the same weights) → **expert+attention HBM floor ≈ 3-3.5 ms total.**

**@16 (32 verify rows, 256 routes/layer, ~230-250 distinct experts/layer):**
- Expert: 288 distinct (worst) × 3.15 MB = 907 MB/layer × 42 = **38 GB/HPU/step → 19.6 ms
  at 1.945** (at 70% HBM efficiency: 28 ms).
- Attention weights: 2.4 GB (read once, batched) → 1.2 ms; DSA KV × 16 seqs: 16 × 22 MB →
  0.2 ms; KDA state × 16: 136 MB → 0.07 ms; indexer (2.8K ctx × 16 × 11 × 1 KB): 0.5 ms.
- **@16 weight-stream floor ≈ 41-43 GB/HPU/step → 21-22 ms** (at 100%; 30 ms at 70%).

### 0.4 The 40-100× gap (the reservoir this round attacks)

| Shape | Expert term MEASURED | Expert HBM floor DERIVED | Gap |
|---|---:|---:|---:|
| bs1 (1 stream) | 19.5 ms (slowest rank; mean 9.6) | 0.9-1.9 ms (incl. draft) | **10-20×** |
| @16 (32 rows) | ~130-150 ms (of 166; D24 to confirm) | 19.6-28 ms | **5-7×** |

The gap is per-route overhead: each of the 256 routes pays staging (input-row copy, scale
loads, routing-table processing, GEMM dispatch) inside the stock `mixture_of_experts`
fused op. The cookbook's custom-kernel data quantifies the per-route cost: 47 µs/route at
full width (v7 MXFP4 kernel, measured single-card), 25 µs half, 17.7 µs quarter; the stock
FP8 op at 32 rows implies ~300-350 µs/route/layer (DERIVED from the 166 ms step, §0.2).
**The expert term is a kernel-design problem, not a bandwidth problem, at bs1 — and at @16
it is a batching problem (256 separate routes vs ~230 streamed expert batches).**

The cookbook's own levers bound what layout+placement alone buy: EP4×TP2 + static
placement took the expert term 19.5 → 14.0-14.4 ms on THEIR kernels (+11% step) — and our
Q17 A/B REFUTED it on the stock FP8 op (bs1 −22%, @16 −21%, TASKS.md Q17: "the study's
+11% was specific to their custom MXFP4 kernels' per-route staging cost"). **Order
matters: the kernel first, the layout second** — the Q17 patch stays in the repo inert,
relevant again under the Q21/RA-053 kernels (its recorded next action).

### 0.5 The 1M-context term (correcting the round-7 RA-047 premise)

The DSA indexer scores EVERY position (the top-2048 selection needs all scores), so the
indexer read scales with context while the latent read (2048 slots) does not. Index keys:
index_kpool_compress=true, kpool 4 → ~1 KB/position (BF16, the indexer is unconverted).
Per DSA layer per token: C × 1 KB. At 1M: 1 GB × 11 layers = 11 GB/HPU/token → **5.6 ms at
1.945** (replicated across TP ranks — the index keys are per-sequence, not head-sharded).
At 32K: 0.35 GB → 0.18 ms. So: **1M decode bs1 = the 32K step + ~5.4 ms** → the 1M decode
rate is the 32K rate × 0.85-0.90 (NOT the "never measured / software-blocked" framing of
round 7 — the needle at 1M already PASSES: `needle_1m_result.json`, ZEPHYR-7492 retrieved;
the decode RATE at 1M is the unmeasured number, and its design is now known). The KV
sequence-sharding option (RA-054) cuts the indexer read 4-8× for the 4-HPU 1M fit.

### 0.6 Hardware facts (corpus-verified this round)

- **Per chip: 24 TPCs + 2 MMEs + 48 MB shared SRAM** (Databricks + arXiv 2501.00210 + 1Cat
  notes, three concordant CORPUS-D; vendor Tier B confirms the 48 MB SRAM and MME/TPC
  parallel operation; the TPC count is not stated in the Tier-B text found — a 10-minute
  `hl-smi`/driver probe is the confirmation task, not a blocker).
- **The MME is not directly programmable** — "access to MME units is limited to the
  PyTorch level" (arXiv 2501.00210, CORPUS-D); the 1Cat REFERENCE_NOTES: "A normal Gaudi
  TPC kernel cannot simply invoke MME matrix multiplication" (CORPUS-D, their production
  fork). Consequence: the megakernel is NOT one CUDA-style kernel. It is **fat TPC nodes +
  MME GEMM nodes in a small hand-built graph** — the TPC does everything vector/recurrence/
  gather/sort; the MME does the GEMMs as standard nodes the compiler schedules.
- MME: 432 TFLOPS bf16 / 865 fp8 peak; 99.3% of peak measured at 8192³ (ISCA'25, CORPUS-D
  web); thin GEMM (decode shapes) M=16, K=N=4096: 37.3 TFLOPS bf16 / 67.8 fp8 (arXiv table,
  CORPUS-D) — **decode GEMMs are bandwidth-bound, not MME-bound, at M ≤ 32**.
- TPC: 2048-bit SIMD, 64 f32 ops/cycle, VLIW4, 11 TFLOPS bf16 aggregate; fully
  programmable (gdc/tpc-clang, the TPC-C language, the simulator + VSCode debugger —
  VENDOR Tier B). The in-tree Q21/AB-DEQ kernels are the production-qualified precedent.
- Inter-chip: 21 × 100 Gbps RoCEv2 on the baseboard, 3 ports per peer, non-blocking
  all-to-all (NVSwitch-like; VENDOR Tier B). MEASURED (gaudi-benchmark, CORPUS-D):
  all_reduce busbw 250.5 GB/s @ 2 GB but 91 @ 16 MB; all_to_all 218 @ 2 GB but **9.1 @
  16 MB** — small messages are 20-30× worse (the per-collective LATENCY, not the
  bandwidth, is the decode cost: 162 ARs × ~20 µs = 3.4 ms MEASURED).
- HBM: 2.45 TB/s spec (VENDOR); 1945 GB/s compiler cost model (CORPUS-C, libSynapse
  1.24.1); project envelope 1.696-2.147 TB/s (MEASURED).

---

## RA-053 — The expert-streaming kernel: the expert term to its HBM floor
### (the 40-100× reservoir; the #1 lever)

**Status:** PROPOSED (architectural, authorized by the owner's engine-rewrite directive) ·
**Complexity:** High (kernel + runner rewiring; months-class to full bank) · **Effort:**
2-4 weeks to the first measured A/B; the program continues to the floor.

### 1. Executive summary

Replace the 256-per-route expert computation (each route: input-row staging + scale loads +
GEMM dispatch inside the stock fused op) with an **expert-streamed program per MoE layer**:
the TPC fat node sorts the step's routes into per-expert batches once (a permutation over
≤ 256 ints — µs), stages each batch's rows contiguously, and the layer's expert compute
becomes one GEMM per ACTIVE expert (≤ 230 at @16, ~12 at bs1) reading that expert's weights
**once, contiguously, in FP8 in place** (the MME's native fp8 path — production already
reads E4M3 in place; no dequant), M = the batch's token count (1-16), K = 2048 (w13) /
4096 (w2) per expert-slice. The per-route fixed cost (300-350 µs/layer/row at the stock
op, DERIVED §0.4) becomes a per-expert setup (~1-2 µs MME descriptor + the weight stream).
The cookbook's v7/v8 kernels prove the per-route cost is reducible (47 → ~25 µs); the
streaming form removes the per-route axis entirely. **bs1 expert: 19.5 → 1.5-3 ms; @16
expert: ~130-150 → 20-30 ms (the HBM floor, §0.3). This single lever is ~+50-65% bs1 and
~+100-200% @16 in the step arithmetic (§0.7).**

### 2. Current limitation (MEASURED)

- Expert kernels 19.5 ms/step (slowest EP rank; mean 9.6) at bs1; 50.7 vs 30.7 at 4 streams
  (cookbook 8-rank trace, CORPUS-D traced).
- Production @16: 181.3 t/s = 166 ms step; the expert term is the bulk (~130-150 ms,
  DERIVED; D24 confirms with our own 8-rank @16 trace).
- Q17 REFUTED the layout lever on the stock op (bs1 −22%, @16 −21%): the stock
  `mixture_of_experts` per-route staging cost dominates regardless of bin width.
- Q16 REFUTED the gather-then-GEMM decomposition at 2-row shapes (the index_select copy +
  fp32 down GEMM dominated).
- Q21: the block-FP8 dequant TPC kernel core is BUILT and gate-passed (1.215e-4 / 3.62e-8
  parity; the E4M3 load-pipe expansion `v_f8_ld_tnsr_b` + `v_convert_f8_to_bf16_b`); the
  fused GEMV variant (dequant-in-registers + dot, reading E4M3 in place) is the staged
  next kernel — the TPC-side attack on the same per-route cost.

### 3. Proposed architecture

Per MoE layer, the node sequence (replacing the current fused-op subgraph of ~3.3k nodes,
census):

1. **TPC fat node `expert_sort_stage`** (one node, index space = the step's routes):
   - input: the router's top-8 ids + weights (already on-device — the router fast path is
     banked, Q18) + the activation rows.
   - work: (a) the noaux_tc top-k output is already a (route → expert) list; sort it into
     per-expert batches (counting sort over 288 bins — trivial); (b) stage each batch's
     activation rows into a contiguous `[max_batch, 4096]` buffer (≤ 16 rows × 8 KB =
     128 KB — fits SRAM); (c) write the per-expert batch descriptors (expert id, row
     count, row offsets) to an HBM descriptor table (288 × 16 B = 4.6 KB).
   - cost: ~10-50 µs (a 256-int sort + 128 KB copy on 24 TPCs).
2. **MME GEMM chain** — one `batch_gemm` node per active expert (the count is static at
   capture: 288 descriptors; inactive experts get M=0 and the compiler elides/zero-costs
   them — the stock fused op already emits per-expert GEMM structure, census: 1,496
   batch_gemm nodes in the decode recipe; the streaming form is the same node class with a
   different M distribution):
   - w13: `M=batch_rows, N=2×2048, K=4096` FP8 (the MME fp8 path, fp32 accumulate —
     production semantics); the block scales (128×128) are applied per the checkpoint
     contract — the Q21 kernel's scale handling is the reference (the scale rounding to
     BF16 before multiply is the documented contract; the MME-side variant applies the
     scale to the fp32 partial sums per k-block — the parity test, item 8, settles the
     exact semantics).
   - SiLU-and-mul (TPC micro-node or in the w2 staging — the cookbook's fused kernel does
     this in-register; as a node it is a 128 KB TPC op, ~5 µs).
   - w2: `M=batch_rows, N=4096, K=2048` FP8.
   - **back-to-back MME nodes with no sync between experts**: the Synapse compiler already
     pipelines the MME nodes inside the stock fused op; the streaming form relies on the
     same pipelining with a longer, sparser chain. The per-expert MME setup cost is the
     open measurement — **the first task (§8.1) is the micro-benchmark: N back-to-back
     batch_gemm nodes, M=1..16, K/N as above, FP8 — measure the per-GEMM overhead; the
     go/no-go threshold is ≤ 10 µs/expert at M ≤ 4** (below it, the streaming form wins
     decisively; above it, the TPC fused-GEMV variant (Q21) takes the lead and the MME
     chain is the fallback).
3. **TPC fat node `expert_combine`**: the per-token sum over its 8 expert outputs (the
   routed_scaling_factor 2.5 + shared-expert add) + the residual — the current combine is
   part of the 3.8 ms MoE-adjacent term; folded into one TPC node (~20-50 µs).

The **MTP draft** uses the same streamed structure (the draft's 8 routes sort into the
same table — the cookbook's "MTP pair overlaps: verify and draft share 2.4 of 8 experts"
is automatic in the streaming form: shared experts are read once for both rows).

**Why this respects the hardware model:** the MME stays a node (it cannot be called from
the TPC — §0.6); the TPC does only what TPCs are for (sorts, gathers, staging, SiLU,
combine — the arXiv 2501.00210 characterization: "vector gather-scatter operations or
activation functions"); the 48 MB SRAM holds the staged rows (128 KB) + the descriptor
table + the per-expert weight tile if the TPC variant streams (a 25.17 MB expert does NOT
fit SRAM — the TPC variant reads the weight stream directly from HBM in the load pipe,
the Q21 pattern: dequant-in-registers + dot, no materialization).

**The EP placement** (the Q17 part 2-3 machinery, recorded in TASKS.md): after the kernel
lands, the static placement recovers the residual imbalance (the cookbook simulation:
EP8 uniform-routing floor 17.37 → 15.26 EP4×TP2 with placement; the MTP-pair overlap ~1.0
ms). The layout A/B runs ONLY after the per-route cost is down (the Q17 refutation is the
cautionary ordering).

### 4. Technical feasibility

- **The MME fp8 GEMM with block scales**: production already runs the native fp8 fused
  weights path (the `torch.ops.hpu.mixture_of_experts.fp8_fused_weights[_dynamic]`
  overloads, `vllm_gaudi/extension/ops.py` — our port's `VllmMixtureOfExpertsOpFP8PerChannel`
  calls them; EXPERIMENTS.md: "reading E4M3 in place"). The per-expert GEMM nodes consume
  the same weight tensors + scales. The open question is the per-GEMM node overhead (§8.1
  micro-benchmark) — not the dtype.
- **The TPC fat node**: the in-tree Q21/AB-DEQ kernels (production-qualified, gate-passed)
  are the same mechanism at smaller scope; the g2-opt-manual ch.4/9 (CORPUS-D, pinned,
  hardware-verified on HL-225) is the programming-model reference (the index-space model,
  the SRAM discipline, the host glue). The `expert_sort_stage` is a 200-400 line TPC-C
  kernel; the `expert_combine` a 100-200 line kernel.
- **The static expert count at capture**: the graph is captured at a static shape — the
  288-descriptor table is the static part; the active count varies at runtime (the M=0
  elision). This is the same data-dependent-shape-with-static-shape pattern the current
  recipe already uses (the routing table is data, the graph is static).
- **Unsupported assumptions:** (i) the back-to-back MME node pipelining holds for a
  200+ GEMM chain (the stock fused op's internal structure is the closest precedent —
  the micro-benchmark measures it); (ii) the M=0 GEMM is free/cheap (if not, the active-
  expert count is fixed at capture per bucket: the decode bucket shapes already exist —
  the 16-row/32-row class — and the per-shape active-expert cap is a capture constant:
  bs1 → 16, @16 → 256); (iii) the compiler does not insert per-MME-node syncs in lazy
  mode (the lazy-mode hazard class — the contract test + the trace are the guard).

### 5. Expected performance impact (DERIVED, the step arithmetic §0.7)

- **bs1 expert: 19.5 ms → 1.5-3 ms** (weight stream 0.9-1.9 ms incl. draft + per-expert
  setup 12 active × 2 GEMMs × 1-10 µs × 42 layers = 0.1-1 ms + the sort/combine 0.5-2 ms).
  Best case (setup ≤ 2 µs): 1.5 ms. Worst (setup 10 µs, 70% HBM): 3.5 ms.
- **@16 expert: ~130-150 → 20-30 ms** (the HBM floor 19.6-28 ms at 70-100%; the per-expert
  setup: 230 × 2 × 1-10 µs × 42 = 1.9-19.3 ms — the setup IS the term at @16 if > 2 µs;
  the streaming form at 70% efficiency + 2 µs setup ≈ 22 ms).
- **Step totals (bs1):** 34.9 → with RA-053 alone: ~20-24 ms → **79-94 t/s** (AL 1.88).
  With RA-053 + RA-054 (attention 6.9→2.5) + RA-056 (AR 3.4→2) + RA-055 (boundary 2.5→0.7):
  **~14-17 ms → 110-134 t/s no-drafter.** The §0 headline range (188-288) applies at the
  EXPERT floor with ALL terms at floor (the 70% HBM + 2 µs setup + attention 1.5 + AR 1.5
  + boundary 0.7 → 9-12.5 ms → 150-210 t/s; the optimistic 188-288 assumes the TPC fused-
  GEMV variant beating the MME chain, §3).
- **@16 step:** 166 → ~35-45 ms (RA-053 + attention 3-4 + AR 2 + boundary 1) → **670-860
  t/s no-drafter at 8 HPUs**; the 800 target met at the optimistic end, met comfortably
  with a drafter (AL 4.5: 72 tok/step → 1600-2050 t/s).
- **4 HPUs:** bs1 ~198-288 (the attention weights 2× per HPU at TP4 — §0.7 table); @16
  560-770 (the expert floor 31.3 ms at 100% is the binding term — 800 needs the drafter
  or ≥ 85% HBM efficiency).

### 6. Implementation complexity

High. The `expert_sort_stage` + `expert_combine` TPC kernels (1-2 weeks including the
contract tests), the graph builder (the per-layer node sequence replacing the fused-op
subgraph — the FX-pass pattern from RA-052 is the instrument: match the fused-op subgraph,
replace with the streamed sequence, meta-preserving rewire, DCE, lint — the 1Cat
`fuse_projection_graph` template, corpus-verified), the micro-benchmark (1 day), the A/B
boots + gates (1-2 weeks). The MME-chain variant is lower-risk (stock node class); the TPC
fused-GEMV variant is the Q21 continuation (the core is built). Total: 4-6 weeks to the
first full A/B on one MoE layer class, 2-3 months to the full 42-layer bank + the layout
follow-up.

### 7. Risks

- **The per-expert MME setup exceeds the threshold** (the §8.1 micro-benchmark is the
  gate; if > 10 µs at M ≤ 4, the TPC fused-GEMV variant leads — the Q21 core is the
  fallback, and the cookbook's v7/v8 precedent says the TPC-side kernel reaches
  17.7-47 µs/route — still a 7-20× win over the stock ~300-350 µs).
- **Numerics:** the accumulation order changes (per-expert M=small GEMMs, fp32
  accumulate, different k-block scale application) → NOT bit-identical → the R003
  knife-edge class applies (the T1 invariance package + the T7 tie-aware gate policy —
  the owner decision item from rev 16 — is the acceptance instrument; the cookbook's
  precedent: their layout change was "not bit-identical; quality gates held: GSM8K
  49/50, needles, top-1 agreement 0.988" — the same gate stack we already run).
- **The lazy-mode hazard class** (four prior sampler bugs + the T3 refutation): the
  back-to-back MME chain + the staging buffers are a new lazy-mode surface — the
  contract test (bit-exact dequant/scale semantics on real weights) + the gate suite +
  the 30-min soak are the mitigation; the env gate (default OFF) is the revert.
- **Memory:** the staging buffer (128 KB) + the descriptor table (4.6 KB) per layer —
  negligible; the SRAM budget: the MME's operand staging is the compiler's (the 48 MB
  shared pool — the per-expert GEMM operand tiles are small at M ≤ 16: 16 × 4096 × 2B =
  128 KB + the weight tile streamed — the compiler's pipelining handles it as in the
  stock fused op).
- **EP imbalance residual** (the balls-in-bins floor): at bs1, 16 routes into 8 bins →
  the slowest bin varies 0-5 routes/step — the placement (post-kernel) + the MTP-pair
  sharing recover ~1 ms; the floor remains (~0.5-1 ms) — acceptable.

### 8. Implementation instructions

1. **The micro-benchmark FIRST (1 day, 1 card, no boot of the server):** N back-to-back
   `batch_gemm` (or `gemm`) nodes in one captured graph, FP8 weights (real checkpoint
   shard slices), M ∈ {1,2,4,8,16}, N/K as the expert shapes (4096→4096, 2048→4096),
   with and without the block-scale application; measure the per-GEMM wall time
   (the D22-style node timing or the profiler). Output: the per-expert setup cost —
   the go/no-go for the MME-chain vs TPC-GEMV lead.
2. **The 8-rank @16 trace (1 boot, the existing trace tooling):** the current
   production @16 step decomposition (the D24 gap — the cookbook's is 1-4 streams; our
   166 ms @16 expert term is DERIVED, not measured). Output: the confirmed @16 expert/
   attention/AR/boundary split — the target spec for the A/B.
3. **The TPC kernels (1-2 weeks):** `expert_sort_stage` (the counting sort + staging +
   descriptor table; the contract: the batch descriptors reproduce the router's routing
   exactly — CPU reference test on 10k random route sets) + `expert_combine` (the 8-way
   sum + 2.5 scale + shared add; contract: bit-exact vs the current combine on real
   activations). The Q21 fused-GEMV variant in parallel (the staged core; the per-route
   target: beat 25 µs at half width).
4. **The FX pass (the RA-052 instrument):** the match predicate over the fused-op
   subgraph (the census-confirmed topology: the 3.3k-node MoE lazy subgraph), the
   replacement with the streamed sequence, the env gate (`VLLM_HPU_EXPERT_STREAM=1`,
   default OFF), the fail-closed shape qualification (the decode bucket shapes only).
5. **The A/B boot (the gates):** one MoE layer class first (the 11 DSA layers — the
   smallest attention coupling), then all 42. Gates: the parity probe (1/8), R003 (5/5,
   the T7 instrument if the owner adopts it), the golden (its own reference — the
   numerics regime changes), the tools 8/8, the smoke 391, the 30-min soak. The fresh
   census (the node delta — the fused subgraph → ~3 nodes + the GEMM chain).
6. **The benchmark:** bs1 (expect 79-94 → 110-134 with the companion levers), @16
   (expect 670-860), the per-layer expert time from the 8-rank trace (the primary
   metric: 19.5 → 1.5-3 ms bs1), the @16 expert term (130-150 → 20-30 ms).
7. **Then (post-bank):** the EP layout A/B (the Q17 machinery, TASKS.md's recorded
   wiring: the group-slice hook at `RoutedExperts.load_weights` ~897, the config.py
   EP branch ~1231-1253, the router `map_ids` at `hpu_fused_moe.py` ~485/576) + the
   static placement (the cookbook's record-simulate-fit-validate method, their
   9.3K-step recorder).

### 9. Validation criteria

1. **The micro-benchmark gate:** the per-expert MME setup ≤ 10 µs at M ≤ 4 (else the TPC
   variant leads — recorded, not failed).
2. **The expert term (the primary metric):** the 8-rank trace shows the slowest-rank
   expert time bs1 ≤ 3 ms (the 19.5 baseline) and @16 ≤ 30 ms (the ~130-150 baseline);
   < 50% of the reduction at the first A/B → the match/staging is defective — re-derive
   from the trace, not the model.
3. **The step:** bs1 ≥ 90 t/s (RA-053 alone, the honest floor); @16 ≥ 600 t/s.
4. **The gates:** parity 1/8, R003 5/5 (or the T7 instrument with the measured top-2 gap
   reported), the golden (own reference), the tools 8/8, the smoke 391, the 30-min soak,
   the accuracy suite 40×5 ≤ 0.5 pp (the cookbook precedent: 0.988 top-1 agreement on
   a non-bit-identical change — our gate stack is stricter).
5. **The census:** the MoE subgraph node count 3.3k → ~3 + the GEMM chain count (the
   host-replay term drops with it — the rev-14 model, overlapped but real).

### 10. Novelty assessment

Not a retry of any closed experiment. Q16 (gather-then-GEMM) and Q17 (layout on the stock
op) are closed NEGATIVE — and their failures are the design constraints here (no per-route
copies; the layout comes after the kernel). Q21 (the block-FP8 GEMV core) is IN PROGRESS as
a per-route kernel — this proposal is its streaming generalization (the per-route axis
replaced by the per-expert batch axis), which is why the Q21 core is the substrate, not a
conflict. The stock fused op is itself a per-expert GEMM batch — the new element is the
**pre-sorted contiguous token batches + the per-layer TPC sort/stage/combine + the
back-to-back MME chain with minimal per-expert setup**, i.e. moving the routing work from
per-route (inside the op, 256×) to per-layer (outside, 1×). The cookbook's v7/v8 custom
kernels are the closest prior art (they cut the per-route cost 2×) — the streaming form is
the next step their data implies (the 17.7 µs quarter-width per-route floor is still 4-7×
the per-expert-stream floor at @16).

---

## RA-054 — The attention consolidation: the KDA and DSA regions as fat TPC kernels
### (6.9 ms → 1.5-3 ms; the 1M design)

**Status:** PROPOSED · **Complexity:** Medium-High (2-3 custom TPC kernels + the graph
rewiring) · **Effort:** 2-4 weeks.

### 1. Executive summary

The 6.9 ms attention term (MEASURED, same on every rank) is ~90% overhead, not bandwidth:
the attention weight GEMMs are 1.2 ms of it (2.4 GB/HPU); the KV/state reads are ~0.1 ms;
the remaining ~5.6 ms is the per-layer node/launch/sync overhead across 45 attention
regions (34 KDA: the conv-4 + gated-delta-net recurrence + output projection as a node
chain; 11 DSA: the indexer score + top-2048 select + the absorbed-latent gather + the W_UV
as a node chain). Each region collapses to ONE fat TPC kernel (the recurrence, the conv,
the indexer score, the top-k, and the absorbed attention are all TPC-natural work — the
arXiv characterization: "vector gather-scatter operations or activation functions") plus
the existing MME GEMM nodes for the projections (which stay — they are bandwidth-bound at
decode, not overhead-bound). **The DSA kernel doubles as the 1M-context design** (§0.5):
the indexer read at 1M (11 GB → 5.6 ms) is the only context-scaling term, and the
kernel's kpool-compressed read + the optional KV sequence-sharding (item 3c) bound it.

### 2. Current limitation (MEASURED)

- Attention 6.9 ms/step at 1 stream, 2.8K context (cookbook trace; identical on every
  rank — it is the per-rank local attention work, not the AR).
- The DSA absorbed decode is already the data-movement-friendly path (the stock
  `forward_mqa_sparse`: absorbed query, top-2048 latent slots, single latent cache
  `num_heads=1` — vllm-gaudi Tier B, `hpu_attn.py` `HPUMLAImpl`); the read is 2048 slots ×
  512 × 2B = 2 MB/layer — the kernel structure is right, the node decomposition is not.
- KDA: the per-token recurrence (64 heads × 128×128 state, head-sharded /8 = 8 heads/HPU)
  is ~33.5M MAC/token/layer → 6 µs at the TPC's 11 TFLOPS — 34 layers = 0.2 ms of actual
  compute; the state read+write is 8.5 MB/HPU (D18 MEASURED). The rest of the KDA region's
  cost is the node chain (the conv, the gate, the A/B projections, the o-projection as
  separate nodes + the lazy-mode boundaries).

### 3. Proposed architecture

1. **The KDA kernel** (one TPC fat node per KDA layer, index space = (head, token)):
   - reads: the input row (already staged by the previous region), the 8-head state
     (8 × 128 × 128 × 2B = 256 KB/HPU — SRAM-resident across the step? No — the state
     lives in HBM (the mamba state pool, per-block); the kernel reads the 256 KB,
     updates it (the gated delta rule: the per-head 128×128 rank-1 update + the decay),
     writes it back (256 KB × 2 = 0.5 MB/HPU/layer → 34 layers = 17 MB/step — the D18
     number), computes the conv-4 on the KV sequence (4 taps × 8 heads × 128 — trivial),
     and the attention output (q · state, 8 × 128 × 128 × 2 = 262K MAC — trivial).
   - the o-projection (16384→4096, BF16, 32 MB/HPU): stays an MME GEMM node (the
     bandwidth-bound GEMM) OR folds into the kernel's final stage as a TPC GEMV
     (32 MB read → 16 µs at 2 TB/s — the TPC does 11 TFLOPS: 134 MFLOP → 12 µs — the
     GEMV is cheaper as a TPC tail than as a separate MME node + sync; the A/B decides).
   - contract: the recurrence is SEQUENTIAL in the token (the state dependency) — one
     TPC point = one (head, token) with the state read-modify-write; the index space is
     parallel over heads × tokens (the tokens are independent at bs1; at @16 the 16
     sequences' tokens are independent — the state is per-sequence).
2. **The DSA kernel** (one TPC fat node per DSA layer):
   - the indexer: the per-position index-key read (kpool-4 compressed, ~1 KB/position) +
     the 128-dim score (the wq_b/wk projections are MME GEMM nodes BEFORE the kernel —
     they are weight-bound GEMMs, 16.8 MB each — or TPC GEMV tails; the A/B decides) →
     the kernel computes the 1M (or C) scores, the top-2048 select (the kpool tail-
     always-select flag honored — the config `index_kpool_always_select_tail: true`),
     and the absorbed attention: the 2048 × 512 latent gather (2 MB — the paged-cache
     gather: 2048 × 1 KB random reads — the TPC's gather strength) + the 64-head
     absorbed dots (64 × 2048 × (512+rope) × 2 = 134 MFLOP → 12 µs) + the W_UV (8.4
     MFLOP → 1 µs) + the output staging.
   - the kernel is the 1M design: the indexer read is the context-scaling term (11 GB at
     1M → 5.6 ms — §0.5); the latent read is context-independent (2048 slots).
3. **The graph rewiring**: the FX pass (the RA-052 instrument) replaces each attention
   region's node chain with [MME projection GEMMs] → [TPC fat kernel] → [MME o-projection
   (or the TPC tail)]. The region count: 45 kernels/step. The node count: the attention
   regions of the 20.5-26.4k-node recipe (the ~11% non-MoE share, ~2.5-3k nodes) → ~90
   nodes + the GEMMs.

**The KV sequence-sharding option (3c, for the 4-HPU 1M fit):** the DSA latent + index
keys are per-sequence and currently REPLICATED across TP ranks (2 GiB/HPU at 1M: the
latent 1 GiB + the index 1 GiB). Sharding them across the TP group along the sequence
(1/4 per HPU at TP4): the indexer scores its local 1/4 of positions (a 256K score read
per HPU at 1M — 25% of the term), the top-2048 is a global selection (a 2048 × 8B = 16 KB
all-gather of the (position, score) pairs — trivial), each HPU attends its local slice of
the 2048 (512 slots) and the outputs merge with the online-softmax 2-value reduction
(64 heads × 2 × 4B = 512 B — trivial) + the latent-output partial (64 × 512 × 2B = 64 KB
all-reduce — trivial). Net at 1M: the 5.6 ms indexer term → ~1.4 ms (TP4); the memory:
2 GiB → 0.5 GiB/HPU. This is the 4-HPU 1M-concurrency answer (the KV budget, RA-056).
It is a SECOND-phase item (the 32K production window does not need it; it changes the
AR count + the KV layout — the 1M boot (D17) is the test window).

### 4. Technical feasibility

- The TPC fat kernels: the Q21/AB-DEQ precedent (production-qualified TPC kernels in-tree);
  the g2-opt-manual ch.4 (the index-space model: the (head, token) index space is the
  standard work division; §4.2.9 "the index-space point is the cost unit").
- The KDA recurrence in TPC-C: the gated delta net is a rank-1 state update
  (S ← S·diag(decay) + (v − S·k)·k^T / (1 + k^T·S·k)-class arithmetic — the exact
  GLM-5.3-Flash KDA formulation is in the port's KDA layer code, `glm5_next_hpu` — the
  kernel transcribes the existing torch ops 1:1; the parity test is against the current
  recipe's KDA output on real sequences).
- The top-2048 select in TPC: a 1M-element top-k over 24 TPCs (42K elements/TPC — a
  2-pass threshold + collect, ~0.1-0.2 ms worst case at 1M; at 32K: ~5 µs). The kpool-4
  compression + the tail-always flag are the config's documented semantics (the stock
  indexer code is the reference — `nope_mla.py` + the indexer ops in the port).
- **Unsupported assumptions:** (i) the paged-cache gather (2048 random 1 KB reads)
  achieves ≥ 50% of HBM bandwidth on the TPC (the TPC's HBM access is the same shared
  subsystem — the Q21 kernel's measured per-route bandwidth is the calibration; the
  2048 × 1 KB = 2 MB read at 50% = 2 µs — even at 10% it is 10 µs — the risk is LOW);
  (ii) the MME projection GEMMs before the kernel stay pipelined with the fat kernel
  (the node dependency chain is the standard TPC-after-MME pattern — the current recipe
  already has it).

### 5. Expected performance impact (DERIVED)

- Attention: 6.9 → **2.5-3.5 ms** (the projection GEMMs 1.2 + the kernel work 0.3-0.5 +
  the state/KV 0.1 + the residual sync 0.5-1). Best case (o-projection as TPC tail,
  clean pipelining): 1.8-2.2 ms.
- bs1 step (with RA-053): −4 to −5.4 ms. @16: the attention scales ~2× with rows (the
  DSA gather × 16 seqs: 33 MB/layer × 11 = 367 MB → 0.2 ms; the indexer × 16: 0.5-8 ms
  by context) → −6 to −10 ms at 32K context.
- **1M decode bs1 (the new design):** the 32K step + 5.4 ms (indexer, §0.5) → with the
  full program: ~20-23 ms → **82-94 t/s at 1M context** (vs the 32K 110-134) — the 1M
  decode rate MEASURED-FOR-DESIGN for the first time (the needle at 1M already passes;
  the rate was the open number — RA-047's premise corrected).
- With 3c (4-HPU, 1M): the indexer term 5.4 → 1.4 ms → the 4-HPU 1M step ≈ the 4-HPU 32K
  step + 1.4.

### 6. Implementation complexity

Medium-High. Three TPC kernels (the KDA ~400-600 lines, the DSA ~500-800 lines with the
indexer, the projection tails ~200 each) + the FX pass (the RA-052 instrument, the second
application after the MoE) + the contract tests (the KDA recurrence parity on real
sequences — a 10k-token random walk vs the current recipe; the DSA top-k exactness vs
the stock indexer on real index keys) + the A/B boots. 2-4 weeks to the first KDA-layer
A/B; 4-6 weeks to the full 45-layer bank. The 3c sharding is a separate 2-3 week phase
(the KV layout + the merge ops + the 1M boot).

### 7. Risks

- **The KDA recurrence parity** (the primary risk): the rank-1 update in TPC-C must
  reproduce the torch ops' fp32/bf16 rounding boundaries EXACTLY (the T1 invariance
  package's discipline — the cookbook's "preserve FP32 accumulation order, BF16 rounding
  boundaries" is the 1Cat rule). The 10k-token parity test is the gate; any drift → the
  KDA kernel takes the torch op graph's exact op order (the transcription is 1:1 by
  construction — the risk is the TPC-C floating-point semantics vs torch on HPU — the
  AB-DEQ precedent resolved the same question for the dequant at parity).
- **The top-k tie-breaking** (the DSA select): ties in the index scores (BF16 quantized
  scores) — the stock indexer's tie-break must be replicated (the config's
  `index_kpool_always_select_tail` + the kpool ordering) — the exactness test on real
  keys catches it.
- **The lazy-mode hazard** (the fat kernels are new lazy-mode surfaces — the standard
  gate stack + the soak).
- **Memory:** none (the kernels stream; the staging is the existing activation buffers).
- **The 3c sharding:** changes the KV layout (the paged-cache geometry) + adds 2-3 small
  collectives per DSA layer — the 1M boot is the only test window; it is explicitly
  second-phase (the 32K window is unaffected).

### 8. Implementation instructions

1. **The KDA kernel first** (the 34 layers — the bigger count; the simpler math):
   transcribe the port's KDA layer ops 1:1 (the conv-4, the gate, the delta update, the
   output); the CPU/torch parity test on 10k random tokens (bit-exact or ≤ 1 ULP bf16 —
   the T3 bar); the TPC build (the Q21 build pipeline, the staged skeleton).
2. **The DSA kernel**: the indexer score + top-2048 + the absorbed attention (the stock
   `forward_mqa_sparse` is the reference for the absorbed math; the paged gather is the
   new element — the block-table walk in TPC-C); the exactness test: the selected 2048
   slots IDENTICAL to the stock indexer's on 100 real sequences (32K context) + the
   attention output ≤ 1 ULP.
3. **The FX pass** (the RA-052 instrument): the attention-region match (the census-
   confirmed topology), the replacement, the env gate (`VLLM_HPU_ATTN_FAT=1`, default
   OFF), the fail-closed shape qualification.
4. **The A/B boots:** one KDA layer → the 34 KDA layers → the 11 DSA layers. Gates: the
   parity probe, R003, the golden (own reference), the tools, the smoke, the soak.
   Benchmark: the attention term from the 8-rank trace (the primary metric: 6.9 →
   ≤ 3.5 ms), the step bs1/@16.
5. **Then (second phase):** the 3c KV sequence-sharding (the 4-HPU 1M fit) — the KV
   layout change + the merge ops + the 1M boot (D17 window) + the 1M needle re-run
   (ZEPHYR-7492 must still be retrieved) + the 1M decode rate bench (the RA-047
   measurement, now with a design).

### 9. Validation criteria

1. **The parity gates** (bit-exact or ≤ 1 ULP): the KDA 10k-token walk; the DSA top-2048
   slot identity on 100 real sequences + the output ≤ 1 ULP.
2. **The attention term:** the 8-rank trace shows ≤ 3.5 ms (bs1, the 6.9 baseline);
   < 50% of the reduction → re-derive from the trace.
3. **The step:** bs1 −4 ms (with RA-053 banked); the 1M decode rate MEASURED (the
   ra047_decode.py script exists on the server — the measurement task is now
   well-posed: expect 82-94 t/s at 1M with the full program).
4. **The gates:** the standard stack (parity 1/8, R003 5/5, the golden, the tools 8/8,
   the smoke 391, the 30-min soak, the accuracy suite ≤ 0.5 pp).

### 10. Novelty assessment

Not a retry: the attention has not been kernel-targeted in this project (the Q16/Q17/Q21
line was MoE; the cookbook's study measured the attention term (6.9 ms) but left it —
"measured but left alone" is their note on the top-p sorts, and the attention
consolidation is not in their lever list because their kernels were MoE-scoped). The
DSA absorbed path is already stock (the read structure is right) — the proposal is the
node-decomposition fix (the 45-region node chains → 45 fat kernels), the same class as
RA-053's MoE consolidation, and the 1M design falls out of the DSA kernel for free
(the indexer term, §0.5 — correcting round 7's RA-047 premise that the 1M decode was
"software-blocked/never measured": it is now bounded, designed, and measurable).

---

## RA-055 — The device-side step chain (the command ring): the last host sync, removed

**Status:** PROPOSED (the engine-level change; the owner's "the whole engine will be
rewritten" authorization applies) · **Complexity:** High (the vLLM HPU worker/engine-
core step loop + the graph plumbing) · **Effort:** 3-5 weeks to the first ring A/B.

### 1. Executive summary

The remaining host term after Q18 (the router fast path + early draft + device-side
rejection — all VERIFIED and in production) is the **step-boundary sync**: the vLLM
engine core submits the step graph, WAITS for the sampled token(s), runs the Python
scheduler (the next-step input build), and submits again — one host round trip per step,
~1-2.5 ms of it on the critical path (the boundary term, §0.2) + the Python scheduler
bookkeeping. The vendor gate (D14, MEASURED rev 16: "for HPU, speculative decoding is not
supported with async scheduling" — `vllm_gaudi/platform.py` l.276-278) blocks the
upstream async-scheduling fix. **The command ring bypasses the gate entirely without
touching the scheduler's semantics:** the host pre-submits a ring of K step-graphs at
boot (K = 4-8, the runtime's graph queue depth — verify, item 8.1); the token-id handoff
between steps is an **HBM mailbox** — the last node of step i writes the accepted token +
the draft rows into the input buffer that step i+1's first node reads (the same
fixed-address buffer mechanism vLLM's HPU graph capture ALREADY uses for the input
token-ids — the device writes it instead of the host). The device walks the ring at full
speed; the host's per-step work is a poll of the output mailbox (µs) + the Python
scheduler amortized over 2-4 steps (the scheduler processes the outputs of steps
i-3..i-1 in one burst while the ring runs steps i..i+3). **The boundary term 1-2.5 ms →
~0.3-0.7 ms, and the step cadence becomes device-bound at every batch size — including
the @16 shape where the Python per-request bookkeeping is 16× the bs1 cost.**

### 2. Current limitation (MEASURED)

- The boundary: ~5.5 ms pre-Q18 (cookbook trace) → ~1-2.5 ms post-Q18 (the three levers
  removed ~3.5 ms; our 34.9 ms step vs the trace's 39.0-39.4 with the same expert/
  attention/AR terms confirms the boundary is now the smaller term).
- D14 (rev 16, MEASURED): the vendor async-scheduling gate — the upstream fix (#1264, the
  engine-core fixes) is inert on our stack; the per-step submit-wait is the scheduler's
  only mode.
- The Python per-step cost: the vLLM engine core's step() (the schedule + the output
  build + the sampler input copy) — the apispy/engspy data on the server
  (`apispy_bench.json`, `engspy_raw.txt`) is the host-side profile (the Nov-14 trace is
  Python-only: 316,948 python_function events over 76 s — the per-step Python breakdown
  is extractable — item 8.2).
- The Strata precedent (CORPUS-D, the server's `strata-scout1/2` logs + the corpus): a
  serving engine where the device-side worker loop owns the step cadence (the
  "device-side execution" pattern) — the conceptual precedent that the host does not
  have to be in the per-step loop; the implementation here is the stock-runtime version
  (the graph ring + the mailbox), not Strata's fork.

### 3. Proposed architecture

**The ring (K = 4-8 step-graphs, all captured at the bucket shapes — the existing
capture machinery, no new shapes):**

1. **Capture:** the step graph G = [token-fetch → main forward (verify rows) → draft
   (the Q01v2 graphed draft) → rejection sampler (the Q18 device-side walk) →
   mailbox-write]. The token-fetch node reads the step's input token-ids + the block-
   tables from the fixed input buffer B (the existing capture input binding). The
   mailbox-write node writes: the accepted token-id(s) + the accepted-count + the
   draft-token + the finished-flag into the fixed output buffer O_i (a 64-byte struct).
2. **The handoff:** the host, at boot, submits G_0..G_{K-1} (the ring) — each G_i's
   input buffer B_i is the mailbox target of G_{i-1}: **the last node of G_{i-1} writes
   B_i** (the accepted token becomes step i's input; the draft rows are built
   device-side — the Q18 "last accepted row is 2i + accepted − 1" index, already
   deployed). The device executes G_0 → G_1 → ... at full speed with NO host
   involvement: each step's input is written by the previous step's last node.
3. **The host loop:** poll O_i (a 1-2 µs HBM read via the existing sync-free poll
   mechanism, or the runtime's event wait) → when O_i shows the step done: the Python
   scheduler processes O_{i-3}..O_{i-1} (the 3-step-oldest outputs — the finished
   requests, the token append, the next admission) and RE-SUBMITS the freed ring slot
   (G_{i-K} re-captured/re-bound for the newly admitted request or the next iteration of
   the continuing one — the re-bind is the host's only per-step write: the block-table
   update for a finished/new request, a small H2D). The steady state (all 16 requests
   running, no finish/admit): **the host does ZERO per-step work** — the ring walks
   itself; the host only polls for the finished-flag (which never fires mid-run).
4. **The finish/admit path:** a finished request frees its rows; the admission of a new
   one (or the continuation of a waiting one) requires a re-bind of one ring slot (the
   block tables + the request metadata — a ~50-200 µs host operation, amortized over
   the steps between finish/admit events — at 16 concurrent with 256-token generations,
   a finish/admit every ~8-16 steps).

**Why the mailbox is sound:** the token-id dependency (step i+1's input = step i's
accepted output) is a device→device HBM write→read with the graph's own node ordering
as the memory barrier (G_{i-1} completes before G_i starts — the ring's submission
order). No host sync, no new hardware, no driver change. The sampled token is produced
by the in-graph rejection sampler (the Q18 device-side walk — the float32-rounded
draws, 0 mismatches in 7,000 steps/rank, VERIFIED) — it never leaves the device except
through O_i for the host's bookkeeping.

**The MTP integration:** the draft is inside the graph (the Q01v2 pattern, in
production) — the ring's G already contains it. The acceptance walk + the draft-row
index are device-side (Q18). The ring changes ONLY the submission cadence (submit-once-
per-K-steps instead of submit-per-step) — the per-step device work is byte-identical to
production.

**The Python amortization:** the engine core's step() splits into (a) the scheduler
burst (process up to K pending outputs — the existing logic, batched) + (b) the ring
maintenance (the re-bind on finish/admit). (a) runs at the host's pace (0.5-1.5 ms
Python per 1-4 outputs — BELOW the device step cadence of 15-25 ms, so it never blocks
the ring); (b) is event-driven. The host is never on the critical path except at a
finish/admit (a few hundred µs, amortized).

### 4. Technical feasibility

- **The fixed-buffer graph input**: vLLM's HPU graph capture ALREADY binds the input
  token-ids/positions/block-tables to fixed HPU buffers (the host copies them before
  each replay — the standard `HPUModelRunner` capture path, in our tree). The mailbox
  is the same binding with a device-side writer — no new runtime feature.
- **The graph queue depth K**: the Synapse runtime executes submitted graphs in order
  from a submission queue (the depth is a runtime parameter — the vLLM HPU worker
  submits sequentially today; K = 4-8 pre-submitted graphs is within the runtime's
  normal queueing — **verify the depth at item 8.1**; if the stock depth is < K, the
  depth is a config knob or the ring degrades to K = the depth — the value is K-independent
  in the steady state, only the finish/admit re-bind latency scales with it).
- **The Q18 device-side rejection** is the prerequisite and is DEPLOYED (the sampled
  token is already a device value) — the ring consumes it as-is.
- **The Strata precedent** (the device-side loop ownership) + the 1Cat fork's
  single-card driver (the corpus Tier D — a minimal host↔HPU control path exists as a
  reference) bound the risk that the stock runtime cannot sustain a pre-submitted ring.
- **Unsupported assumptions:** (i) the runtime's graph queue executes K graphs
  back-to-back without a host re-arm between them (the normal queueing semantics —
  item 8.1); (ii) the mailbox write→read is ordered by the graph completion (the
  standard HBM coherence for graph-ordered access — the Q18 device-side walk already
  relies on device-ordered HBM access for the draft rows); (iii) the Python scheduler
  tolerates a K-step output latency (the outputs are buffered K deep — the user-visible
  token latency adds ≤ K × step-time ≈ 60-100 ms to the FIRST token of a burst... no:
  the outputs are polled per-step (O_i is readable as soon as G_i finishes — the
  streaming user sees tokens at the device cadence; the K-depth is only the
  SUBMISSION ring, not the output latency — the host polls O_i every step for the
  streaming append).

### 5. Expected performance impact (DERIVED)

- The boundary: 1-2.5 ms → **0.3-0.7 ms** (the poll + the amortized Python; the
  finish/admit re-bind ~0.1-0.3 ms per 8-16 steps).
- **bs1 step (with RA-053+054 banked):** −1 to −1.8 ms → the 14-17 ms → 12.2-15.2 ms
  → **123-154 t/s** (no drafter); with the drafter (AL 4.5): 296-369 t/s.
- **@16 step (with RA-053 banked):** the Python per-request bookkeeping at 16
  concurrent (the schedule + the 16 output appends per step — ~1-3 ms, the engspy data
  is the calibration, item 8.2) is amortized off the cadence → −1 to −3 ms → the
  35-45 ms → 32-42 ms → **720-940 t/s** (no drafter); the 800 target met in the
  optimistic half without a drafter, met everywhere with one.
- **The stability value:** the step cadence becomes device-locked (no host jitter in
  the cadence — the 16-stream P95 tightens; the soak's variance drops).

### 6. Implementation complexity

High. The graph plumbing (the mailbox nodes: the token-fetch rewrite + the mailbox-
write node — ~200-400 lines in the runner's capture path), the ring manager (the
pre-submission + the re-bind — ~300-500 lines in the HPU worker), the engine-core split
(the scheduler burst + the poll loop — ~300-500 lines in the engine core / the
multiproc executor — the server's `engine_core_loop.py` / `engine_core_stotal.py`
scratch files show the agent has already instrumented this seam), the streaming-output
path (the per-step poll → the token append → the user stream — the existing async-LLM
output path, adapted from per-submit to per-poll). 3-5 weeks to the first K=2 ring A/B
(the K=2 ring is the minimal form: it removes the submit-wait for the NEXT step even
before the full amortization).

### 7. Risks

- **The runtime queue semantics** (the primary risk): if the stock runtime re-arms per
  graph (a host doorbell per replay regardless of queueing), the ring degrades to the
  status quo — item 8.1 (the K-graph pre-submission micro-test, 1 day, no server boot:
  capture 8 graphs of the add1 kernel, pre-submit all 8, measure the per-graph gap vs
  the sequential submit) is the go/no-go.
- **The scheduler desync:** the K-step output buffering changes the engine core's
  step() contract (the outputs arrive in a buffer, not per-submit) — the vLLM engine
  core is the integration surface (the fork's `engine_core_loop.py` scratch shows the
  seam); the risk is a subtle desync (a finished request's last token lost/duplicated
  in the buffer) — the contract test: 10k steps with random finish/admit injection, the
  output sequence IDENTICAL to the sequential mode (the golden + the needle are the
  end-to-end guards).
- **The watchdog** (the long-running ring): the per-graph timeout applies per graph
  (each G_i completes in 15-25 ms — far below the vendor timeout) — the ring does not
  create a non-terminating node; the risk is LOW. (The TPC QM-kernel swap, RA-057, is
  the variant that WOULD touch the watchdog — it is deliberately not this proposal.)
- **The numerics:** byte-identical device work per step (the per-step graph is
  production's graph + the mailbox nodes, which carry values, not compute) — the
  golden/parity gates are expected UNCHANGED; any drift is a plumbing bug.
- **The debuggability:** a ringed step loop is harder to trace (the host timeline no
  longer brackets each step) — the polling loop stamps O_i reads with a host timestamp
  (the step-boundary markers for the existing trace tooling — the cookbook's overlay
  method keeps working).

### 8. Implementation instructions

1. **The ring micro-test FIRST (1 day, 1 card, no server boot):** capture 8 graphs of a
   trivial kernel with a mailbox handoff (graph i writes a counter to graph i+1's input
   buffer); pre-submit all 8; measure (a) the per-graph gap vs sequential submit (the
   queue depth in practice), (b) the mailbox correctness over 8 chained graphs. Go/no-
   go: the pre-submitted gap < 50% of the sequential gap AND the mailbox values exact.
2. **The host per-step profile (no HPU time, 0.5 day):** the Nov-14 Python trace
   (`profile/g2_462491...pt.trace.json.gz`, 316,948 python_function events) → the
   per-step Python breakdown (the scheduler, the sampler input build, the output
   append) at bs1 and @16 — the calibration for the amortization model. (The
   apispy/engspy data is the cross-check.)
3. **The mailbox nodes (1 week):** the token-fetch rewrite (the input buffer read
   stays; the source becomes the previous graph's write — the capture change) + the
   mailbox-write node (the 64-byte struct: the accepted token, the count, the draft
   row, the finished flag — a TPC micro-kernel, the Q21 pipeline).
4. **The ring manager (1-2 weeks):** the K-graph pre-submission at the bucket shapes;
   the re-bind path (the finish/admit: the block-table + metadata H2D + the slot
   re-capture — the existing capture machinery); the poll loop (the O_i read per step
   — the runtime's event wait or the DMA poll).
5. **The engine-core split (1 week):** the scheduler burst (the batched output
   processing) + the streaming path (the per-poll token append). The env gate
   (`VLLM_HPU_STEP_RING=1`, default OFF, K configurable).
6. **The A/B boots:** K=2 first (the minimal form), then K=4-8. Gates: the parity
   probe, R003, the golden (UNCHANGED — the device work is byte-identical), the tools,
   the smoke, the 10k-step finish/admit contract test, the 30-min soak (the 16-stream
   P95 is the stability metric), the needle (1M — the end-to-end guard).
7. **The benchmark:** the step cadence from the 8-rank trace (the boundary term: 1-2.5 →
   0.3-0.7 ms — the primary metric), bs1/@16 t/s, the 16-stream P95/P50 ratio (the
   jitter), the host CPU usage during steady state (expect ~0 at 16 concurrent).

### 9. Validation criteria

1. **The micro-test:** the pre-submitted per-graph gap < 50% of sequential (else the
   proposal degrades — recorded with the measured gap).
2. **The boundary term:** the 8-rank trace shows ≤ 0.7 ms of host-with-device-idle at
   the step boundary (the 1-2.5 ms baseline).
3. **The cadence:** the device step period is host-invariant (the host CPU pinned/slowed
   in a control run — the step period unchanged within 2%).
4. **The contract:** the 10k-step finish/admit test — the output sequence IDENTICAL to
   sequential mode; the needle at 32K and 1M pass; the golden unchanged.
5. **The performance:** bs1 ≥ 123 t/s (with RA-053+054 banked); @16 ≥ 720 t/s; the
   16-stream P95/P50 ≤ 1.15 (the jitter guard).

### 10. Novelty assessment

Not a retry: D14 (the vendor async gate) closed the UPSTREAM path (the engine-core
async scheduling) — the ring is a different mechanism (the submission-cadence change
with the device-side handoff, not the scheduler's async mode) that the gate does not
touch (it needs no async scheduling: the scheduler stays synchronous, it just stops
being per-step). The Q18 levers (the device-side rejection + the early draft) are the
prerequisites and are banked — they removed the per-step DATA dependencies on the host
(sampled token, draft indices); the ring removes the per-step SUBMISSION dependency.
The Strata/1Cat device-side-execution pattern is the conceptual precedent (the host
out of the per-step loop); the stock-runtime ring + mailbox is the implementation that
keeps the production tree (no engine fork, no driver change).

---

## RA-056 — The collective reduction: the AR count and the 4-HPU endgame topology

**Status:** PROPOSED (the layout/design program; the layout A/B is gated on RA-053) ·
**Complexity:** Medium (the analysis + the 4-HPU design) / the layout A/B is Low
(the Q17 machinery exists) · **Effort:** 1-2 weeks (the design) + the A/B boots.

### 1. Executive summary

Two related items. **(a) The AR count:** 162 all-reduces/step × ~20 µs = 3.4 ms
(MEASURED) — "a lazy-mode collective ends a recipe, so each layer replays as two
recipes and two all-reduces" (cookbook). The RA-053 streaming op removes the MoE-
combine AR (the expert outputs combine in the local `expert_combine` TPC node after the
intra-group exchange — at EP4×TP2 the exchange is a 2-rank group AR, ~10 µs, not the
8-rank 20 µs; at EP8 it is the all-to-all over the 3-ports-per-peer mesh — the
all_to_all at 1-2 MB messages: ~50-100 µs per the small-message data, BUT it replaces
the per-layer MoE AR + it pipelines over the 3 ports per peer). The net: 162 → ~90-100
collectives, and the per-collective cost drops where the group shrinks. **(b) The 4-
HPU endgame:** the final deployment is 4 HPUs (384 GiB total; the model is 328 GiB +
KV + workspace — the fit is tight, ~4-10 GiB of KV budget per HPU at 32K, less at 1M
without the RA-054 3c sharding). The topology choice (EP4×TP1 vs EP2×TP2 vs EP1×TP4)
is settled by the per-route cost × the bin count at the RA-053 kernel (the cookbook's
method: "measure the per-route fixed cost at each slice width; it decides whether EP ×
TP pays" — their 47/25/17.7 µs at full/half/quarter width is the calibration; our
FP8 streaming kernel's numbers replace it at the micro-benchmark).

### 2. Current limitation (MEASURED)

- AR: 162/step × ~20 µs (24 KB messages) = 3.4 ms (cookbook trace); the lazy-collective
  recipe-split doubles the per-layer recipe count (322 launches/step at bs1).
- The 4-HPU decode profile (Q14v2, MEASURED): 41.3-45.1 t/s cache-ON (vs 53.9 at 8) —
  the 2× weight read per HPU (the attention/dense are 2× at TP4 vs TP8) + the 2× expert
  read (72 local experts at EP4 vs 36 at EP8).
- The 4-HPU capacity: 328 GiB model / 4 = 82 GiB/HPU + KV; the 96 GiB HBM leaves ~4-10
  GiB/HPU for KV + workspace — the 32K window at 16 concurrent: ~4.4 GiB/HPU (the D18
  KV census: the KDA state 8.5 MiB/card/request + the DSA latent + the indexer at 32K
  — the agent's D18/D19 measurements); the 1M window: ~2 GiB/HPU/sequence REPLICATED
  (the RA-054 3c cuts it to ~0.5) — the 1M 4-HPU concurrency is 2-4 sequences without
  3c, 8-16 with.

### 3. Proposed architecture

**(a) The collective reduction (post-RA-053):**
- The MoE combine: the local TPC node (the RA-053 `expert_combine`) replaces the
  8-rank AR of the expert outputs; the inter-rank exchange is the EP all-to-all (the
  token activations only: bs1 16 rows × 8 KB = 128 KB/layer; @16 256 rows × 8 KB = 2 MB
  /layer — the small-message all_to_all, ~50-100 µs — BUT it is ONE per layer replacing
  the MoE AR, and it pipelines over the 3 ports per peer (the 262.5 GB/s unidirectional
  per-card, VENDOR-derived) — the net per-layer collective cost at EP8: ~50-100 µs
  (a2a) vs ~20-40 µs (the 2 ARs) — a LOSS at EP8; at EP4×TP2: the 2-rank group exchange
  (~10 µs) + no 8-rank AR — a WIN. **The collective reduction is EP-layout-
  dependent: it pays at EP4×TP2, not at EP8** — the ordering constraint (RA-053
  kernel → EP4×TP2 A/B → the collective win) is the Q17 lesson applied to the
  collectives.
- The attention ARs (the o-projection output, the pre-MoE hidden): unchanged (they are
  the TP ARs — the 24 KB class, ~20 µs each — the floor for TP; only a sequence-
  parallel attention (the RA-054 3c merge, 2-value + 64 KB) would restructure them —
  second-phase).

**(b) The 4-HPU endgame (the design, the numbers §0.7):**
- **Topology: EP4×TP1 for the MoE (72 full experts/HPU) + TP4 for the attention/dense**
  (the hybrid the Q17 machinery supports: the group layout with t=1... the group is
  4 ranks of 1 — the `GLM_MOE_TP=1` limit of the Q17 patch, or the plain EP4). The
  per-route cost at full width (47 µs v7-class; our streaming kernel's number TBD) ×
  the bins (16 routes → 4 bins, busiest 6.3/mean 3.95 — the cookbook) vs EP2×TP2
  (half width 25 µs, 2 bins, busiest ~9-10) vs EP1×TP4 (quarter 17.7 µs, 1 bin, all 16
  local): at the streaming kernel (the per-route → per-expert, the width cost drops):
  **EP4×TP1 wins at bs1** (fewer bins, and the streaming makes the width penalty
  vanish); **at @16 the expert term is bandwidth-identical across layouts** (288
  distinct / 4 = the same 1.81 GB/layer/HPU regardless of the EP/TP split — §0.3) →
  the @16 choice is the collective cost → EP4×TP1 (the 2-rank... no: EP4 = 4 groups of
  1 rank — the all-to-all is the full 4-way, 3 ports per peer... at 4 HPUs the mesh is
  4-way (3 peers × 3 ports = 9 ports scale-up — the same per-peer wiring).
- **The fit (the hard constraint):** 82 GiB model/HPU + the KV. At 32K, 16 concurrent:
  ~4.4 GiB → 86.4 GiB → fits with ~9 GiB workspace headroom (the production 8-HPU
  workspace is ~10 GiB/HPU — the 4-HPU boot's workspace check is the first 4-HPU task).
  At 1M: the RA-054 3c sharding (2 → 0.5 GiB/HPU/sequence) is REQUIRED for > 2
  concurrent 1M sequences — the 4-HPU 1M design = the 3c + the indexer-sharded DSA
  kernel.
- **The numbers (DERIVED, the §0.7 table):** bs1 198-288 t/s no-drafter (the attention
  2× at TP4 is the binding extra term: 2.4 → 4.8 GB/HPU → +1.2 ms); @16 560-770 (the
  expert floor 31.3 ms at 100% HBM is binding — the 800 target needs the drafter
  (AL 4.5 → 1120-1540) or ≥ 85% HBM efficiency).

### 4. Technical feasibility

- (a): the EP all-to-all is a stock HCCL op (the small-message weakness is MEASURED —
  the 9.1 GB/s @ 16 MB is the bandwidth; the 1-2 MB messages are latency-bound ~50-100
  µs — the micro-benchmark item 8.1 measures the per-layer a2a at the decode sizes);
  the Q17 group-layout machinery is in the repo (the wiring sites located, TASKS.md).
- (b): the 4-HPU boot exists (`serve_mtp_tp4.sh`, the Q14v2 profile) — the fit check +
  the workspace check are boot tasks; the EP4×TP1 layout is the Q17 part-1 patch (the
  model-generic group config) + the plain EP4 (no group TP) — the loader wiring is the
  Q17 recorded site.
- **Unsupported assumptions:** (i) the 4-HPU workspace headroom (~9 GiB) is sufficient
  (the 8-HPU production workspace is ~10 GiB/HPU — the 4-HPU graphs are the same count
  but the per-graph workspace is shape-dependent — the boot check); (ii) the EP4×TP1
  per-route cost at the streaming kernel is ≤ the EP8 (the micro-benchmark at the 4-
  HPU shape — the 4-HPU A/B is the measurement).

### 5. Expected performance impact (DERIVED)

- (a) at 8 HPUs, post-RA-053, with EP4×TP2: the AR term 3.4 → ~1.5-2 ms (the 2-rank
  group exchanges + the 2 remaining TP ARs per layer, ~90-100 collectives × 10-20 µs);
  the recipe count 322 → ~180-200 (the lazy-collective split halved) — the host-
  prefetch term drops with it (overlapped, but the boundary tightens).
- (b) 4-HPU: the table in §0.7 (bs1 198-288; @16 560-770 no-drafter; the drafter
  answers the 800 @16).

### 6. Implementation complexity

Medium (the design + the 4-HPU fit check are boot/analysis tasks; the EP4×TP2 A/B is
the Q17 machinery — Low once RA-053 is banked). The 4-HPU 1M (the 3c) is the RA-054
second phase.

### 7. Risks

- (a) the all-to-all small-message latency exceeds the AR savings at EP8 (the micro-
  benchmark is the gate — the design is EP4×TP2-scoped by construction, so the risk is
  bounded: if EP4×TP2's collective cost is also worse, the AR stays at 3.4 ms — the
  status quo, not a regression).
- (b) the 4-HPU workspace overflow (the boot check is the gate; the fallback: the
  per-shape graph reduction — the bucket set shrinks at 4 HPUs, the capture count
  drops — the T1b pinned-bucket machinery, in the tree).
- The layout's numerics (the Q17 note: "partial sums are added in a different order" —
  the R003/T7 discipline, as in RA-053).

### 8. Implementation instructions

1. **The micro-benchmark (1 day, with the RA-053 item 8.1):** the per-layer collective
   cost at the decode message sizes: the 8-rank AR (24 KB), the 4-way/2-rank a2a
   (1-2 MB), the 2-rank group AR — the per-collective latency table (the small-
   message regime the gaudi-benchmark data brackets: 91 GB/s @ 16 MB AR, 9.1 @ 16 MB
   a2a — the 1-2 MB point is the gap to fill).
2. **The 4-HPU fit check (1 boot):** the EP4×TP1 boot at 32K, 16 concurrent — the HBM
   census (the model + KV + workspace per HPU), the decode bench (the Q14v2 re-run with
   the current lever stack — the 41.3-45.1 baseline is pre-Q18), the needle.
3. **The EP4×TP2 A/B (post-RA-053, 8 HPUs):** the Q17 machinery (the TASKS.md wiring
   sites) + the static placement (the cookbook's record-simulate-fit-validate) — the
   A/B vs the EP8 streaming baseline (the expert term + the AR term + the step).
4. **The 4-HPU 1M (second phase, with the RA-054 3c):** the KV layout + the sharded
   indexer + the 1M boot + the needle (ZEPHYR-7492) + the 1M decode rate.

### 9. Validation criteria

1. **The collective table:** the per-collective latency at the decode sizes MEASURED
   (the design input; the a2a ≤ 100 µs at 1-2 MB is the EP4×TP2 go condition).
2. **The 4-HPU fit:** the HBM census ≤ 92 GiB/HPU at 32K/16-concurrent (the 4 GiB
   headroom) — else the bucket-set reduction (the T1b machinery).
3. **The EP4×TP2 A/B (post-RA-053):** the expert term ≤ the EP8 streaming baseline AND
   the AR term −1.5 ms AND the step −2 ms (the combined (a) win).
4. **The 4-HPU numbers:** the bs1 ≥ 150 (the 198-288 DERIVED floor, the honest bar is
   the measured floor), the @16 the measured expert-floor confirmation (31.3 ms at
   100% — the efficiency measurement).

### 10. Novelty assessment

The AR count has not been targeted (the 3.4 ms is measured in the cookbook trace and
carried in our rev-14 model as "the 322 replays + ≥ 162 AR boundaries" without a
reduction design). The 4-HPU topology is the Q14v2 profile's open question (the 41.3-
45.1 vs 53.9 gap was attributed to the 2× weight read — the EP/TP split choice was
never A/B'd; the Q17 refutation on the stock op closed the layout question at the OLD
kernel — it reopens at the RA-053 kernel by the Q17's own recorded condition:
"relevant again under the Q21 custom kernels").

---

## RA-057 — The driver-level option (the last resort, scoped): the raw submission path

**Status:** RESEARCH (the spike design; NOT the first move — the ring (RA-055) achieves
~95% of the host-term value in userspace) · **Complexity:** Architectural rewrite (the
driver/firmware boundary) · **Effort:** 1-2 weeks (the spike) + months (the full raw
stack, only if the spike says the ring is insufficient).

### 1. Executive summary

The owner authorized the driver level. The honest assessment: **the residual host term
after the ring (RA-055) is ~0.3-0.7 ms of polling + the amortized Python — the driver
rewrite buys ~0.3-0.7 ms, at the cost of owning the firmware/driver boundary.** The
driver-level work is therefore scoped as a RESEARCH SPIKE (the raw submission
feasibility, 1-2 weeks) that answers: can we submit a graph (or a raw DSA command
sequence) without the Synapse runtime's per-replay host path — and if so, what is the
per-step floor? The corpus holds the building blocks (the finding
`gaudi2-kernel-dev-map.md`: "source building blocks for raw Gaudi2 command submission"
— with the open questions explicitly noted: "its minimum firmware and host dependencies
remain open") and the 1Cat single-card driver (the corpus Tier D — a minimal
Gaudi-2 host↔device control path, the precedent that someone has already built it).
The TPC QM-kernel swap (replacing the TPC's work-queue consumer with a step-loop
program — the fully device-resident variant) is the deeper level: the driver's
`gaudi_run_tpc_kernel` (VENDOR Tier B, `gaudi.c`: the per-TPC kernel base-address
registers + the execute command — "should be used only during initialization and/or
after reset, when there are no active users") shows the TPC main program IS swappable
at init — the QM loop is the device-resident step walker (no graph replay at all).
That is the "rewrite the drivers" end-state — and it is the highest-risk item in the
project (the firmware ABI, the watchdog, the vendor support surface). **Recommendation:
run the spike; adopt only if the ring leaves > 2 ms of host on the critical path at @16
(the engspy calibration, RA-055 item 8.2) — otherwise the ring is the end-state and the
driver work is parked as the documented fallback.**

### 2. Current limitation (MEASURED/CORPUS)

- The ring (RA-055) removes the per-step submission; the residual is the poll + the
  Python (0.3-0.7 ms) — the driver rewrite's target is that residual + the runtime's
  per-graph overhead (the 74 µs/launch class, which the ring already amortizes).
- The corpus building blocks: the `gaudi2-kernel-dev-map.md` finding (the raw command
  submission sources; the open firmware/host dependencies); the driver's per-TPC kernel
  launch path (the CFG registers: the kernel base, the icache invalidate/prefetch, the
  QM sync-object, the execute bit — the Tier B `gaudi.c` + `tpc0_cfg_regs.h`/`masks.h`);
  the 1Cat single-card driver (Tier D, the minimal-stack precedent); the 1Cat
  REFERENCE_NOTES (the MME-is-node constraint, the ~2 µs/node heuristic, the "count
  physical nodes and measure native replay" discipline).

### 3. Proposed architecture (the spike scope)

1. **The raw graph submission test (week 1):** bypass the Synapse runtime's replay API:
   take a captured graph's command buffer (the runtime's compiled artifact — the
   `ht`/synapse graph object) and submit it via the driver's submission path directly
   (the user-space driver's queue doorbell — the `habanalabs` user-space driver's
   submission entry, the corpus's driver source) — measure: (a) does it execute
   (the add1 kernel, the K-chain from the RA-055 micro-test), (b) the per-submission
   host cost vs the runtime's 74 µs class, (c) the queue depth (the raw queue vs the
   runtime's). Output: the per-step floor of the raw path — the number that decides
   adoption.
2. **The QM-loop feasibility note (week 2, paper + 1 day of driver source reading):**
   the TPC QM kernel (the work-queue consumer) as the step walker: the design (the
   control block in HBM: the step descriptor; the QM loop: read the descriptor →
   dispatch the step's DSA work (the MME/DMA command sequences — the raw command
   format from the runtime's compiled graph) → write the mailbox → loop); the
   blockers (the firmware's QM expectations, the watchdog's per-TPC timeout — the
   driver's `HL_DEVICE_TIMEOUT_USEC` class, the sync-object protocol the QM kernel
   manages); the verdict: feasible-with-vendor-cooperation / feasible-standalone /
   blocked — the decision input for the owner.

### 4. Technical feasibility

- The raw submission: the user-space driver is open source (the corpus's
  `drivers.accel.habanalabs.kernel` + the user-space driver in the same tree) — the
  submission path is readable; the 1Cat single-card driver proves the minimal path
  exists (Tier D). The risk: the runtime's compiled graph format is a vendor ABI
  (the firmware consumes it — the format is stable across the 1.24.x line but
  undocumented — the corpus's decompiled libSynapse (Tier C) is the reference for
  the command structure).
- The QM swap: the driver's init-time kernel path (Tier B) shows the mechanism; the
  QM kernel's internal protocol (the work-queue descriptor format, the sync objects,
  the error paths) is the decompiled-firmware territory (the corpus's firmware
  strings + the decompiled libSynapse bracket it; the full protocol is the open
  question the spike's note must answer).

### 5. Expected performance impact (DERIVED)

- If the raw path reaches the floor: the per-step host → the poll only (~0.1-0.2 ms)
  — vs the ring's 0.3-0.7 ms: **the gain is 0.1-0.5 ms/step** (bs1: +0.5-1.5% on a
  12-15 ms step; @16: +0.2-0.8% on a 32-42 ms step). The QM-loop variant (no graph
  replay at all): the same floor + the elimination of the per-graph compile at boot
  (the 5.4-8.1 s/graph — the 277-graph boot's ~25 min → the QM boot is the raw
  program load — the boot time, not the step, is the gain).
- **The honest verdict: the driver rewrite is a 0.1-0.5 ms/step lever — the smallest
  in the round. It is authorized, scoped, and parked behind the ring by design.**

### 6. Implementation complexity

Architectural rewrite (the driver/firmware boundary) — the spike is the only
justified commitment now (1-2 weeks, no production risk: it is a standalone test
binary + a source-reading note).

### 7. Risks

- The firmware ABI (the raw command format) is undocumented — the corpus's decompiled
  libSynapse (Tier C) + the 1Cat driver (Tier D) bracket it; a format mismatch = the
  spike's answer is "needs the vendor" — acceptable (the spike's purpose is the
  verdict).
- The QM swap: the watchdog (a non-standard QM loop trips the device timeout — the
  driver's `hl_poll_timeout` class), the error paths (a step fault outside the
  runtime's error handling = the device hang — the recovery is the HPU reset), the
  vendor support (a custom QM kernel is outside the supported surface — the
  production-qualification risk for a SHIPPING system).
- The time: months-class if adopted — against a 0.1-0.5 ms payoff. The recommendation
  stands: park unless the ring underperforms.

### 8. Implementation instructions (the spike)

1. **Week 1:** the raw submission test (the add1 K-chain, the per-submission cost,
   the queue depth) — the test binary against the add1 kernel (the g2-opt-manual
   ch.9 worked example, in the corpus) + the runtime's graph object. The output: the
   per-step floor number + the queue depth.
2. **Week 2:** the QM-loop note (the driver source reading: the QM kernel's launch
   path, the work-queue descriptor format from the decompiled libSynapse, the
   timeout/watchdog surface) — the verdict: feasible-with-vendor / standalone /
   blocked.
3. **The decision rule:** adopt the raw path only if (a) the ring's measured residual
   (RA-055 item 8.2, the @16 host term) > 2 ms AND (b) the raw path's floor < 0.5 ms
   of the residual. Else: park, document, done.

### 9. Validation criteria

1. **The spike output:** the per-submission raw cost MEASURED (vs the runtime's 74 µs
   class) + the queue depth + the QM verdict — the three numbers the decision rule
   consumes.
2. **No production change** (the spike is standalone; the production tree is untouched
   unless the decision rule fires).

### 10. Novelty assessment

New territory for this project (no prior experiment touched the submission path). The
corpus work (the kernel-dev-map finding, the 1Cat driver, the driver's Tier B launch
path) makes it a scoped research item rather than a guess — and the scoped payoff
(0.1-0.5 ms) makes the "park behind the ring" recommendation the intellectually honest
one: the owner authorized the driver level, the investigation is done, the arithmetic
says the ring is 95% of the value, and the spike is the 5% check with a 1-2 week
price.

---

## §0.7 — The target table (DERIVED, the full step arithmetic)

Step = E (expert, slowest rank) + A (attention) + M (MoE-adjacent/dense) + R (collectives)
+ H (host boundary), per HPU; t/s = AL × 1000 / step_ms (bs1: AL 1.88; @16: AL 1.88 × 16
seqs = 30.08 tok/step; the drafter class: AL 4.5 → 72 tok/step @16).

| Config | E | A | M | R | H | step ms | t/s no-drafter | t/s drafter (AL 4.5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **8 HPU bs1 — production (MEASURED structure)** | 19 | 6.9 | 3.8 | 3.4 | 2.5 | 34.9-36 | 53.9 (MEASURED ✓) | ~135 |
| **8 HPU bs1 — P1: +RA-053** | 1.5-3 | 6.9 | 2.5 | 3.4 | 2.5 | 16.8-18.3 | 103-112 | 246-268 |
| **8 HPU bs1 — P2: +RA-054+056** | 1.5-3 | 2.5 | 2.5 | 2 | 2.5 | 11-12.5 | 150-171 | 360-409 |
| **8 HPU bs1 — P3: +RA-055 (the program)** | 1.5-3 | 2.5 | 2.5 | 2 | 0.7 | 9.2-10.7 | **176-204** | 410-489 |
| **8 HPU bs1 — P4: + the expert floor (70% HBM, 2 µs setup)** | 0.9-1.5 | 1.8 | 2 | 1.5 | 0.7 | 6.9-8.2 | **229-272** | 549-652 |
| **8 HPU @16 — production** | ~130-150 | ~14 | 3.8 | 3.4 | 3 | 154-174 (166 ✓) | 181 (MEASURED ✓) | ~460 |
| **8 HPU @16 — P1: +RA-053** | 20-30 | ~14 | 2.5 | 3.4 | 3 | 42.9-52.9 | 570-700 | 1360-1680 |
| **8 HPU @16 — P3: +RA-054+055+056** | 20-30 | 3-4 | 2.5 | 2 | 1 | 28.5-39.5 | **760-1055** | 1820-2530 |
| **4 HPU bs1 — P3** | 1.5-3 | 4.5 (2× TP4) | 2.5 | 2 | 0.7 | 11.2-12.7 | **148-168** | 354-405 |
| **4 HPU bs1 — P4** | 0.9-1.5 | 3.5 | 2 | 1.5 | 0.7 | 8.6-9.7 | **194-219** | 464-523 |
| **4 HPU @16 — P3** | 31.3-44.7 (the floor) | 3-4 | 2.5 | 2.5 | 1 | 40.3-54.7 | **550-746** | 1315-1790 |
| **8 HPU 1M bs1 — P3** | +0.4 (the indexer) | +5.4 (RA-054 §0.5) | 2.5 | 2 | 0.7 | 15-18.6 | **101-125** | 242-300 |

Reading the table: **the 8-HPU targets (200 bs1 / 800 @16) are met without a drafter at
P3 (the program) in the bs1 optimistic half and the @16 range, and met everywhere with
a drafter of AL ≥ 4.5** (the DFlash2-class number, or the DSpline/3090 class — the RA-049
options re-opened with the relaxed bar: AL ≥ 4.0 suffices for both 8-HPU targets at P3).
**The 4-HPU targets: 200 bs1 is met at P4 (the expert floor) — 194-219, borderline at P3
(148-168); 800 @16 needs the drafter (1315-1790 at AL 4.5) or ≥ 85% HBM efficiency on
the expert stream.** The drafter's role shifts from "the critical path to both targets"
(rev 16) to "the 4-HPU @16 answer + the 200-beyond margin" — the RA-049 decision item
stands, with the arithmetic updated.

**The no-drafter floor (what the program buys alone):** 176-272 t/s bs1 (8 HPU, 3.3-5.0×
the current 53.9), 760-1055 t/s @16 (8 HPU, 4.2-5.8× the current 181.3), 148-219 bs1 /
550-746 @16 (4 HPU). Even at the conservative ends: 3-4× the current single-stream and
aggregate.

---

## §10 — Execution order (the do-first sequence for the implementation agent)

1. **The RA-053 micro-benchmark (1 day, 1 card, no server boot):** the per-expert MME
   GEMM setup cost (the M=1..16, K/N expert shapes, FP8, back-to-back nodes) — the
   go/no-go for the MME-chain vs TPC-GEMV lead. **This is the first HPU task of the
   round; it costs one afternoon and decides the RA-053 design split.**
2. **The 8-rank @16 trace (1 boot, the existing tooling):** the D24 gap — the
   production @16 step decomposition (the expert/attention/AR/boundary split at 16
   concurrent — the cookbook's is 1-4 streams; our @16 expert term is DERIVED, not
   measured). The target spec for the @16 A/Bs.
3. **The RA-055 ring micro-test (1 day, 1 card, no server boot):** the K-graph
   pre-submission gap + the mailbox correctness — the go/no-go for the ring (the
   second biggest lever; it is independent of #1/#2 — run it the same week).
4. **The TPC kernels (the RA-053 `expert_sort_stage`/`expert_combine` + the RA-054
   KDA kernel, in parallel — 2-4 weeks):** the contract tests first (the CPU
   references: the 10k-route sort, the 10k-token KDA walk, the 100-sequence DSA top-k).
5. **The FX pass + the first A/B boots (the RA-053 one MoE layer class, then 42; the
   RA-054 one KDA layer, then 45):** the gate stack per proposal §9.
6. **The ring integration (the RA-055 K=2 → K=4-8, 3-5 weeks):** after #5's first
   A/Bs (the ring on the stock recipe is valid too — it is independent; the order is
   by integration risk, not by dependency).
7. **The EP4×TP2 A/B (the RA-056, post-#5):** the Q17 machinery + the placement.
8. **The 4-HPU fit check (the RA-056, 1 boot):** the EP4×TP1 boot at 32K/16 — the HBM
   census + the bench + the needle.
9. **The RA-049 drafter decision (owner):** now with the §0.7 arithmetic — the bar is
   AL ≥ 4.0-4.5 (relaxed from 4.9-5.2), the role is the 4-HPU @16 answer + the margin.
10. **The RA-057 spike (only if #6's measured @16 host residual > 2 ms):** the raw
    submission test + the QM note.

## Appendix — the config byte model (the derivation inputs)

`/data/models/GLM-5.3-Flash/config.json` (server, 2026-11-23): hidden 4096; 45 layers
(34 KDA at the layer list, 11 DSA at 3,7,...,43; 3 dense-MLP first, 42 MoE); 288
routed experts (top-8, moe_intermediate 2048, routed_scaling 2.5, sigmoid scoring,
noaux_tc top-k, n_group=1/topk_group=1 — the Q18 router fast path qualifies); 1 shared
expert; KDA 64 heads × 128 head_dim, conv-4; DSA kv_lora_rank 512, 64 heads × 256 NoPE
(qk_rope_head_dim 0, mla_use_nope), index 32×128, top-2048, kpool-4 compressed, tail-
always-selected; block-FP8 [128,128] e4m3 (the experts + the dense-4..44 FFN... the
modules_to_not_convert list: all attention/KDA/hc/indexer/router/dense-0-2/embedding/
lm_head are BF16 — the FP8 set = the MoE experts); vocab 154,880; max_position
1,048,576. Per-expert: 3 × 4096 × 2048 = 25.17 MB FP8. Expert total: 288 × 25.17 MB ×
42 = 302 GB (of the 328 GiB checkpoint). Attention BF16 per HPU (head-sharded /8):
~2.4 GB/token. lm_head 1.27 GB (/8 = 159 MB). KDA state 8.5 MiB/HPU/request (D18).
DSA latent per token: 512 × 2B = 1 KB; the top-2048 read: 2 MB/layer. Indexer per
position: ~1 KB (kpool-4 compressed). HBM: 1.945 GB/s (the compiler cost model,
CORPUS-C); the measured envelope 1.696-2.147. MME: 432 bf16 / 865 fp8 TFLOPS (the
peaks; 99.3% at 8192³ MEASURED by the ISCA'25 study); thin GEMM M=16, K=N=4096: 37.3
bf16 / 67.8 fp8 (CORPUS-D table). TPC: 24/chip, 11 TFLOPS bf16, 2048-bit SIMD. SRAM:
48 MB/chip shared. Mesh: 21 × 100 Gbps RoCEv2, 3 ports/peer (VENDOR); AR busbw 250.5
@ 2 GB / 91 @ 16 MB, a2a 218 @ 2 GB / 9.1 @ 16 MB (MEASURED, gaudi-benchmark).
