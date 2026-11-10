# RA-012 — Fused dequant-GEMM for non-MoE block-FP8 linears + mHC Sinkhorn TPC kernel

**Status:** PROPOSED (research architect, 2026-11-09) · **Priority:** 3 (BG-1/BG-4 decide the device term first) · **Complexity:** Medium-High (Part A) / Low (Part B)
**Parent:** `BOTTLENECK_RESEARCH.md` (Section 3, RA-012) · **Targets:** B2 (recipe count), the non-MoE device weight traffic
**Prerequisite checks:** BG-1 (per-layer device trace — is the dequant→GEMM fused today?). **BG-4 RESOLVED 2026-11-09 (live process environ on g2):** `GLM5_NEXT_MHC_SINKHORN=kernel` is set in the running server → the Sinkhorn TPC kernel is ON; Part B below is a no-op to close (node-count baseline via BG-1 only). Also verified live: `GLM_FP8_MXK=0` — the running dequant path is the TORCH dequant (`dequant_block_fp8_weight_naive` → BF16 → `F.linear`), not the TPC dequant op (AB-DEQ tested the op in a separate gated experiment; the adopted config is the torch path).

---

## 1. Executive summary

Two device-side recipe/traffic eliminators for the non-MoE path. **Part A:** the deployed tree dequantizes
every non-MoE block-FP8 linear to BF16 and then runs a BF16 GEMM (`ops.py:868-956`, verified on g2) —
~135 linears/step (42 attention projections × 45 layers… precisely: QKV+O per DSA/KDA layer + 3 dense
layers + 42 shared experts), each = 2 recipes (dequant + GEMM). If the recipe compiler does NOT fuse the
pair (UNVERIFIED — BG-1), each step moves ~9.8 GB/rank of weight traffic (1.96 GB FP8 read + 3.93 GB
BF16 write + 3.93 GB BF16 read) instead of ~1.96 GB → ~3.2 ms device at bs=1. A fused
`block_fp8_dequant_gemm` TPC op (dequant-in-register, FP32-accumulate GEMV over K) replaces each pair
with one op: −135 recipes, −7.9 GB/rank traffic. It is the dense-linear sibling of the in-progress Q21
MoE kernel and reuses the same production-qualified op infrastructure (B7 load recipe, mxk build,
variant library). **Part B:** the mHC mixing (4 streams, 20 Sinkhorn iterations, all 45 layers) — the
DSV4.1 port (identical mHC design) measured its 4×4 Sinkhorn loop at ~3,400 device nodes per step and
removed them bit-exactly with a tiny TPC kernel. **RESOLVED 2026-11-09 (live process environ on g2):**
`GLM5_NEXT_MHC_SINKHORN=kernel` is set in the running server — the kernel is ON, the node storm does
not exist on this tree; Part B is a no-op to close (record the mHC node baseline from BG-1's trace).
**Part A is the live value of this proposal.**

## 2. Current limitation (evidence)

| Fact | Value | Source |
|---|---|---|
| The non-MoE FP8 path | `apply_block_fp8_linear_hpu` → `apply_block_fp8_linear_hpu_dequant` (ops.py l.927-960): dequant via `dequant_block_fp8_weight_naive` (l.868-915: `weight.to(bf16) * scale.to(bf16)` chain, or the TPC `block_fp8_dequant_glm` op behind `GLM_FP8_MXK` — same BF16 material) → **BF16 `torch.nn.functional.linear` (l.954-956)** | deployed tree `vllm_gaudi/extension/ops.py` (read on g2) |
| Linear count | ~135 dequant+GEMM pairs/step (attention QKV/O projections across 45 layers + 3 dense layers + 42 shared experts) | DERIVED from the model geometry (45 layers, the architecture in the diagnosis) |
| Weight traffic per step (if unfused) | FP8 read 1.96 GB + BF16 write 3.93 GB + BF16 read 3.93 GB ≈ 9.8 GB/rank vs 1.96 GB fused → Δ7.9 GB → Δ~3.2 ms at 2.45 TB/s | DERIVED (the per-rank non-MoE FP8 weight mass from the checkpoint geometry; the 2.45 TB/s is the MEASURED effective HBM figure) |
| Recipe cost | 135 pairs = 270 single-purpose recipes vs 135 fused → −135 recipes × ~43 µs ≈ −5.8 ms host replay (of which RA-011 removes most from the critical path — the two compose) | D1 (43 µs/region); DERIVED |
| AB-DEQ | the TPC dequant op ALONE is parity (42.7 vs 41.4 t/s) — the dequant op is not the cost; the extra traffic (or its absence) is the open question | project data (EXPERIMENTS.md) |
| The fused pattern exists | 1Cat `block_fp8_dequant_gaudi2.c`: E4M3 load-pipe expansion (`v_f8_ld_tnsr_b` SW_UNPACK) + `v_convert_f8_to_bf16_b`, in-register dequant, FP32 accumulation — the Q21 kernel's w13/w2 phases use exactly this pattern | GaudiGrimoire tier D; the Q21 staging in this repo |
| The mHC node storm precedent | DSV4.1: "a 4x4 Sinkhorn loop cost about 3,400 nodes per step, and a tiny TPC kernel removed them bit-exactly" | GaudiGrimoire tier D (gaudi-cookbook DSV4.1 port) |
| The GLM mHC design | 4 streams, 20 Sinkhorn iterations, all 45 layers (same mixing family as DSV4.1) | the model architecture (diagnosis + the port source) |
| The mHC Sinkhorn mode in the LIVE tree | **`GLM5_NEXT_MHC_SINKHORN=kernel` is set in the running server process** (verified 2026-11-09 via `/proc/<pid>/environ` on g2) — the TPC kernel is ON; the port registers the kernel library "inert unless `GLM5_NEXT_MHC_SINKHORN=kernel`" (`vllm_gaudi/__init__.py:7`, verified) | live process + deployed tree (read on g2, 2026-11-09) |
| The live dequant gate | **`GLM_FP8_MXK=0` in the running process** — the adopted non-MoE path is the torch dequant (`dequant_block_fp8_weight_naive` → BF16 → `F.linear`); the TPC dequant op (AB-DEQ) is a separate gated experiment, not the live path | live process environ (verified 2026-11-09) |

**The open question that sizes Part A:** does the lazy recipe compiler already fuse
`dequant → F.linear` into one device operation (one HBM pass) at compile time? If yes, Part A's device
term is a wash (the value is the −135 recipes only); if no, Part A is worth ~3.2 ms device + the recipes
at bs=1. **BG-1's HBM-bandwidth-during-the-linear-phase measurement decides this** — the dequant GEMM
kernel must NOT be started before BG-1 (or until the recipe-count value alone justifies it, which the
RA-011 landing makes marginal).

## 3. Proposed architecture

### Part A — the `block_fp8_dequant_gemm` TPC op

```
input:  weight  E4M3 [N,K] contiguous (N%128==0, K%256==0 — the checkpoint layout)
        scales  F32  [N/128, K/128]  (the block scales, same as the dequant op)
        x       BF16 [M,K]  (M ≤ 64 — decode row counts ≤32 verify + draft)
        bias    BF16 [N] optional
output: y       BF16 [M,N]
kernel: for each output tile: load x tile (BF16) + E4M3 weight tile via the load pipe
        (v_f8_ld_tnsr_b SW_UNPACK → F8 in-register)
        multiply by the 128×128 block scale (F32, one per block)
        accumulate over K in F32
        store BF16
        (the 1Cat pattern: dequant-in-register, no BF16 weight material, no intermediate global write)
```

- One op = one recipe per linear (vs two today): the dequant and the GEMM never materialize the BF16
  weight in HBM.
- The padded-weight path (`do_unpad`) stays in torch (the op cannot unpad — same constraint as the
  existing dequant op; the unpad is a one-time load-time op, not per-step).
- Prefill (large M) is out of scope for iteration 1: the MME GEMM path handles prefill today and is
  not the bottleneck; the op is decode-M (≤64) only. (If a later MME-fused variant is wanted, the op
  contract is unchanged.)

### Part B — the mHC Sinkhorn TPC kernel — **CLOSED (verified ON, 2026-11-09)**

- RESOLVED: the live server runs `GLM5_NEXT_MHC_SINKHORN=kernel` (process environ, g2, 2026-11-09).
  The 20-iteration PyTorch loop is NOT the live path; the DSV4.1-class node storm is already eliminated
  on this tree.
- Remaining action: none, except recording the mHC node count from BG-1's trace as the baseline
  (expected: ~1 node/layer for the kernel, not ~3,400/step). If the trace ever shows the loop's node
  signature (a config regression), the cookbook kernel enablement is the known fix (the DSV4.1
  bit-exact precedent).

## 4. Technical feasibility

- **The dequant-in-register GEMV is a demonstrated pattern on this exact hardware:** the Q21 kernel's
  w13/w2 phases do it for the expert weights (staged in this repo); the 1Cat `block_fp8_dequant_gaudi2.c`
  proves the E4M3 load-pipe expansion (`v_f8_ld_tnsr_b`) on Gaudi2 TPC (tier D). The dense version is
  SIMPLER than Q21 (one [N,K]×[K,M] GEMV, no routing gather, no expert index table).
- **The op infrastructure is production-qualified:** the B7 load recipe (the op lib loads in production,
  8/8 workers, `block_fp8_dequant_glm` visible in `torch.ops.custom_op`), the `mxk_op_lazy.cpp`
  registration pattern, the variant library, the `GLM_FP8_MXK` fail-closed guard — all exist and are
  exercised by AB-DEQ (the dequant op serves healthy captured graphs; the GEMM variant inherits the
  same qualification path).
- **Precision contract:** FP32 accumulation over K with 128×128 block scales — the same contract as the
  Q21 kernel (dequant-qualified via AB-DEQ). The accumulation ORDER differs from the stock
  BF16 `F.linear` (which accumulates the dequantized BF16 weight) → the parity gate is the accuracy
  suite + the arithmetic-sanity protocol, NOT bit-identity vs the stock path (see Section 9).
- **Part B:** CLOSED — verified ON in the live tree (2026-11-09); the DSV4.1 bit-exact precedent
  (identical mHC design) is the evidence for why the node storm is not a live cost here.

**Unsupported assumptions:** (A1) the lazy compiler does NOT already fuse the pair today (BG-1 decides —
if it does fuse, Part A's device term is zero and the proposal narrows to the recipe count); (A2) the
TPC GEMV at M≤32 is not compute-bound at the E4M3 load rate (the load pipe is the bottleneck — the 1Cat
kernel's design; at M=1 the FLOPs are trivial; at M=32 the F8 load-pipe throughput must cover 32
accumulations per weight element — verify in the on-device micro-benchmark, the AB-DEQ protocol step).

## 5. Expected performance impact (all ESTIMATED until measured)

| Scenario (BG-1 outcome; Part B closed — kernel ON) | Part A device | Recipes | bs=1 step effect |
|---|---|---|---|
| Unfused today (full Part A value) | −3.2 ms | −135 | **−3 to −4 ms** (45.1 → ~41-42 pre-RA-011; composes with RA-011: the device term shrinks under the host ceiling) |
| Fused today (Part A narrows to recipes) | 0 | −135 | −1 to −2 ms (the host replay −5.8 ms, of which RA-011 removes most from the critical path) |

- **Conservative:** −1 to −2 ms/step bs=1 (the traffic is already fused; the recipe-count value only).
- **Optimistic:** −4 ms/step bs=1 (the traffic is unfused — the full Part A device term).
- @16: the weight read is M-independent (the same ~9.8 vs ~1.96 GB/rank); ESTIMATED −3 ms of the 53 ms
  device if unfused (0 if fused — the recipe-count value only).
- The recipe-count reduction COMPOSES with RA-011 (which hides the loop) — RA-012 shrinks the loop RA-011
  is offloading, and (with RA-012C/Q21's shared-expert absorption) it lowers the ~163-recipe structural
  floor toward ~115-138.

## 6. Risks

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | Accumulation-order divergence vs the stock BF16 path (FP32 over K with block scales ≠ the BF16-dequantized accumulation) | numerical | The gate is the accuracy suite + the arithmetic-sanity protocol (the AB-DEQ pattern), NOT bit-identity vs stock; the 0.5 pp budget applies; the dequant-qualified AB-DEQ result bounds the dequant half of the risk |
| R2 | TPC GEMV compute-bound at M=32 (A2) | performance | The on-device micro-benchmark FIRST (the AB-DEQ protocol: a standalone op bench at M=1/8/32 before any wiring); if compute-bound, the MME variant is the fallback (same contract) |
| R3 | Padded-weight path (`do_unpad`) | correctness | Stays in torch (unchanged); the op contract rejects non-contiguous/padded weights (fail-closed, the existing guard pattern) |
| R4 | Graph capture interaction | stability | The dequant op already captures healthy graphs (AB-DEQ); the GEMM variant follows the same path; the capture test is in the qualification protocol |
| R5 | ~~Part B bit-exactness~~ — CLOSED: the kernel is verified ON in the live tree (2026-11-09); no Part B work remains | — | If a future config regression revives the loop (the ~3,400-node signature in a trace), re-enable the kernel gate — the DSV4.1 bit-exact precedent applies |
| R6 | BG-1 shows the traffic is already fused → Part A's device value evaporates | scope | The proposal is written to survive: the recipe-count value (−135) remains; the decision gate is explicit (Section 8 step 1) |

## 7. Dependencies

- B7 load recipe — done (production-qualified).
- The mxk build system + variant library + `GLM_FP8_MXK` guard — done (Q21/AB-DEQ infrastructure).
- **BG-1** (the per-layer device trace + the HBM bandwidth during the linear phase) — the Part A
  decision measurement. **BG-4 RESOLVED (2026-11-09):** the Sinkhorn kernel is ON in the live tree
  (`GLM5_NEXT_MHC_SINKHORN=kernel`, process environ) — Part B closed.
- Composes with RA-011 (loop shrinking) and Q21 (shared infrastructure; the shared-expert absorption
  RA-012C is a Q21 design note, not this op).

## 8. Implementation instructions

**Decision measurements FIRST (do not start the GEMM kernel before these):**
1. BG-1: the 8-rank torch profiler (CPU+HPU) with per-layer ranges; record (a) the KDA/DSA/MoE/mHC/
   non-MoE-linear device split, (b) the HBM bandwidth during the non-MoE-linear phase, (c) the mHC node
   count per step, (d) the per-rank MoE kernel times. (This is the diagnosis's own Section-10 item —
   share it with the RA-011 boot family.)
   **Decision rule:** linear-phase traffic ≈ 9.8 GB/step (unfused) → Part A full value; ≈ 1.96 GB
   (fused) → Part A narrows to recipes. (Part B is already closed — the kernel is ON, verified
   2026-11-09; the trace's mHC node count is recorded as the baseline only.)
2. (BG-4 — DONE 2026-11-09: live process environ shows `GLM5_NEXT_MHC_SINKHORN=kernel`.) No further
   action; if BG-1's trace ever shows the ~3,400-node loop signature, that is a config regression —
   re-set the env var and re-verify.

**Part A (only if BG-1 confirms the traffic, or the recipe value alone justifies it):**
3. The standalone op bench FIRST (AB-DEQ protocol): a micro-benchmark of the `block_fp8_dequant_gemm`
   op at M=1/8/32 against the current (dequant + F.linear) pair — the performance gate before any
   wiring (R2).
4. Implement `custom_op::block_fp8_dequant_gemm` in the mxk op family (contract in Section 3; the
   registration follows `mxk_op_lazy.cpp`; the load recipe is the B7 one — no new load-order work).
5. Wire into `apply_block_fp8_linear_hpu_dequant` (ops.py l.927-960) behind the existing `GLM_FP8_MXK`
   fail-closed guard (same pattern as the dequant op: on failure, fall back to the torch path, log
   once). NOTE: the LIVE config is `GLM_FP8_MXK=0` — the A/B arm sets the gate ON; the torch path is
   the arm-0 baseline.
6. Qualification: (a) output vs a CPU reference within the FP8 block-scale tolerance (the arithmetic-
   sanity protocol), (b) capture test (a graph captures + replays with the new op — the AB-DEQ
   precedent), (c) stable serving (the soak).
7. A/B at bs=1 + @16 (same-boot protocol; parity = the suite, NOT bit-identity vs stock — R1). Record
   EXP-RA012a. Update the recipe count (the wrapper's verbose logging: the cached-graph count before/
   after — expect −135).

**Diagnostics to capture:** the linear-phase HBM traffic in the trace (the mechanism proof: 9.8 → 1.96
GB), the cached-graph count, the per-layer device split after the change, the op micro-benchmark
numbers.

## 9. Validation criteria

- **Part B (closed):** no validation required — the kernel is verified ON (2026-11-09). The BG-1 trace
  records the mHC node count as the baseline (expect the kernel's ~1 node/layer signature, not the
  loop's ~3,400/step).
- **Part A:**
  1. Output vs a CPU reference within the block-scale tolerance (the arithmetic-sanity protocol; the
     AB-DEQ precedent).
  2. The linear-phase HBM traffic in the trace drops ~4× (9.8 → ~1.96 GB) — the direct mechanism proof.
  3. bs=1 + @16 A/B: no regression vs the current stack; target ≥ −3 ms/step at bs=1 (the conservative
     bound); the cached-graph count drops by ~135.
  4. Graphs capture + replay with the op; stable serving through the soak.
  5. The accuracy suite within the 0.5 pp budget (R1 — the suite gate, not bit-identity).

## 10. Novelty assessment

- **Part A:** no prior experiment fused the dequant INTO the GEMM for the non-MoE linears. AB-DEQ
  explicitly stopped at the dequant op (and proved the dequant-alone value is zero — 42.7 vs 41.4 t/s
  parity). Q16 (the gather+GEMM decomposition) was for the MoE path (expert weight gathering) —
  different problem. Q21 is the MoE sibling (in progress) — this is its dense-linear version. The
  unfused-traffic hypothesis has been flagged in the diagnosis (the "verify the unfused dequant
  hypothesis" item) but never measured — BG-1 is the first measurement.
- **Part B (closed 2026-11-09):** the question (kernel vs loop in the deployed tree) is resolved —
  the live server runs the kernel (`GLM5_NEXT_MHC_SINKHORN=kernel`, process environ). The DSV4.1
  precedent (the identical mHC design, ~3,400 nodes removed bit-exactly) explains why this was never a
  live cost on this tree. No further work.
- **Why Part A deserves investigation now:** the B7 infrastructure (which made fused MoE work possible)
  is production-qualified; BG-1 is a cheap decision measurement (shared with the RA-011 boot family);
  the upside is the second-largest device term after MoE (~3.2 ms if unfused) + the −135-recipe host
  loop shrinkage; and it narrows the recipe floor that caps B2.
