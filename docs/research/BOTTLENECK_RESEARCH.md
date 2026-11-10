# BOTTLENECK_RESEARCH — GLM-5.3-Flash on 8× Gaudi 2 (independent research architect)

**Date:** 2026-11-10 (revision 1.5 — see Section 9 for the revision log; RA-018 added 2026-11-09;
RA-019/020/021 + round-2 audit added 2026-11-10)
**Primary input:** `kb_staging/current_bottleneck.md` (local v1, 2026-11-08 13:01) and its superseding internal
revision `kb_staging/current_bottleneck_final.md` (rev 3, 2026-11-08 19:08 — includes the D1/D1-16 py-spy
profiles and the B7 resolution). The sanitized server copy at `publish_work/fork/current_bottleneck.md`
(2026-11-08 21:03) reflects the pre-D1 state; the rev-3 file is the most information-rich diagnosis and is
treated as authoritative here.
**Method:** independent re-derivation of the byte/step models; source inspection of the deployed tree
(`hpu_model_runner.py`, `ops.py`, the port `model.py`, the vendor `habana_frameworks` venv on g2) and of the
vendor C++ bridge (tier B via GaudiGrimoire); GaudiGrimoire corpus research; web research on the
AMD/NVIDIA GLM-5.3-Flash implementations (upstream vLLM PR #53906, project-maya, gaudi-llm-notes ports).
**Numbering:** RA-001…RA-010 / RJ-01…RJ-12 / D1…D13 belong to
`docs/research/EXTERNAL_RESEARCH_PROPOSALS.md` (rev 5). This document continues the global namespace:
proposals RA-011…, rejections RJ-13…, gaps BG-1…
**Labeling:** MEASURED (project data) / DERIVED (arithmetic from verified data) / ESTIMATED (engineering
model) / SPECULATIVE (insufficient support).

---

## SECTION 1: Executive Summary

### 1.1 Where the step goes (independent re-derivation, cross-checked to the rev-3 diagnosis)

bs=1, adopted stack, 8 HPU: **45.1 ms/step = 24 ms host + ~21 ms device, with ~zero overlap** (DERIVED:
the sum of the D1 host split (24 ms MEASURED) and the device estimate (21 ms DERIVED) reproduces the
measured step to 0.1 ms — the host and device pipelines are serialized). At 16 streams: 61 ms host +
~53 ms device ≈ 114 ms MEASURED step — again serialized.

This serialization is the single most load-bearing fact in the diagnosis, and the rev-3 document does not
state it explicitly. It has a concrete cause verified in the vendor source: the model wrapper calls
`wrap_in_hpu_graph(..., disable_tensor_cache=False)` **without the `asynchronous` argument, so every
decode-step replay is SYNCHRONOUS** — the host thread blocks inside the vendor C++ replay until the device
finishes, then does the sampler/draft/metadata host work before the next step can start. The vendor
documents the opposite mode for exactly this purpose ("This flag can be used along with stream and async
D2H to copy outputs for reducing host overheads" — docs.habana.ai, tier B), and the C++ bridge contains an
entire thread-pool branch (`PT_HPU_ENABLE_HPUGRAPH_THREAD` + `PT_HPU_QUEUE_SYNLAUNCHES`) that moves the
~322-recipe launch loop off the calling thread (tier B, `HPUGraph.cpp::replayGraph`). **None of this is
used by the current stack. That is RA-011, and it is the highest-value new item in this document.**

### 1.2 The most important bottlenecks (independent ranking)

| # | Bottleneck | bs=1 share | @16 share | Verdict |
|---|---|---|---|---|
| 1 | **Host/device serialization (synchronous replay)** | hides ~17-21 ms of device under ~0 ms of host | hides ~53 ms of device under ~0 ms | NEW — not in the diagnosis; the diagnosis's B2 is the *symptom* (14 ms replay loop) but the *cause* is the sync boundary. RA-011 |
| 2 | B1: MoE all-experts weight streaming (38.1 GB/card/step = 15.5 ms) | 34% | 14% (but see correction C1: only ~59% of experts are touched at bs=16 → Q21 still saves ~6 ms there) | CONFIRMED as stated for bs=1; CORRECTED at bs=16 |
| 3 | B2: the 322-recipe replay loop (14 ms bs=1 / 42 ms @16, vendor C++ per-recipe cost ~43 µs) | 31% | 37% | CONFIRMED; the per-recipe cost is in the vendor C++ (`ExecuteCachedGraph`), not Python — the diagnosis's "C++ execution loop" user-space direction is misaimed (see C5). RA-011 + RA-012 + Q21 attack it |
| 4 | B5: the eager MTP draft dispatch (7.5 ms bs=1 / 10.6 ms @16) | 17% | 9% | CONFIRMED (D1 leaf-level). Note: it is nearly FIXED in row count (7.5→10.6 from 1→16 rows), not 1 ms/stream as the rev-1 doc implied — the per-stream marginal is mostly the replay loop's row-scaling. RA-004 (graphed draft) remains the fix |
| 5 | B3: per-stream worker host (3.8 ms/stream linear, R²>0.99) | 8% | 54% | CONFIRMED as measured scaling; the COMPONENT ATTRIBUTION in rev-1 is wrong (sampler/scheduler were the suspects; D1-16 relocated it to worker replay staging + draft + sampler). The dominant per-stream host term is the replay loop's row growth (~2.6 ms/stream) — RA-011 removes it from the critical path, RA-012/Q21 shrink the loop |
| 6 | B4: 162 ARs (3.4 ms device) + the recipe-boundary floor they impose | 7% | ~5% | CONFIRMED; RE-RANKED UP: the AR `mark_step()` boundaries are a HARD FLOOR on the recipe count (~163 cached graphs), which caps how far B2 can shrink (see C4). RA-012 Part C + the Q21 shared-expert fusion attack the AR count |
| 7 | B7: op-loading registrar collision | — | — | RESOLVED (load-order recipe; the legacy-API op lib runs in production, 8/8 workers) |

### 1.3 The three strongest new proposals

1. **RA-011 — Asynchronous HPU-graph replay with the vendor launch thread pool** (one-line change at
   `_maybe_wrap_in_hpu_graph` + two env vars; optional third env var). Hides the entire host post-work
   (sampler + draft dispatch + metadata, ~12 ms bs=1 / ~20 ms @16) AND the 322-recipe C++ launch loop
   under the device execution. ESTIMATED: bs=1 45.1 → ~26-30 ms (+45-70%); @16 114 → ~63-68 ms
   (+70-80%). Complexity LOW (change) / MEDIUM (verification). No prior attempt; the mechanism is
   tier-B-verified in the vendor bridge. Detail: `RA-011-async-replay.md`.
2. **RA-012 — Fused dequant-GEMM for the non-MoE block-FP8 linears (+ the mHC Sinkhorn check —
   RESOLVED: the kernel is ON in the live tree, verified 2026-11-09)**
   (extends the production-qualified mxk op infrastructure to a dense `dequant_gemm` op). Eliminates the double weight traffic of
   dequant→BF16→GEMM on ~1.96 GB/rank of FP8 linears (~3.2 ms/step if unfused — to be confirmed by the
   per-layer trace, BG-1) and cuts ~135 recipes. ESTIMATED: −1 to −4 ms/step bs=1 (Part A only; Part B
   closed — `GLM5_NEXT_MHC_SINKHORN=kernel` verified in the live process). Complexity
   MEDIUM-HIGH (kernel work). Detail: `RA-012-fused-dequant-gemm-sinkhorn.md`.
3. **RA-013 — Expert placement by routing histogram** (compute the per-expert routing profile; re-shard
   the expert→rank mapping to equalize per-rank load; pure permutation, zero accuracy impact). Attacks B5
   (EP imbalance, slowest rank 2× mean) and sharpens the Q21 byte model under real (non-uniform) routing.
   Precedent: project-maya's `expert_counts.txt` per-layer slot allocation (35.5→32.9 ms/token there).
   ESTIMATED: −2 to −5 ms at bs=16-32; enables a tighter Q21 design. Complexity LOW-MEDIUM. Detail:
   `RA-013-expert-placement.md`.

(Also: RA-014 — the 2-4K prefill bucket-transition fix, config-only, closes the one unmet prefill band
sub-case. Detail: `RA-014-prefill-buckets.md`. And RA-015 — the DSA indexer port, priority 5, for 1M-context
fidelity.)

### 1.4 The most significant architectural limitations

1. **The lazy-mode execution model fragments the decode step into ~322 cached graph regions** (≥~163 by the
   AR boundaries alone — "lazy collectives fragment decode graphs… each all-reduce cuts the layer into
   separate cached graphs", gaudi-cookbook Beam port, tier D; confirmed by the D1 count). Every region pays
   a ~43 µs vendor C++ replay cost (hashing + staging + launch). This is a structural property of the
   vendor lazy compiler + HPUGraph wrapper, and it sets the floor of the host term: **even with perfect op
   fusion the host replay floor is ~163 × 43 µs ≈ 7 ms** (DERIVED). The floor is only removable by (a) the
   async thread pool (RA-011 — moves the loop off the critical path), (b) cutting the AR count (RA-012C /
   Q21 shared-expert fusion), or (c) a vendor change.
2. **The stock `mixture_of_experts` op reads all 36 local experts every step** (B1) — unchanged, the
   bs=1 physical blocker (119 t/s ceiling). Q21 remains the fix; this document's correction C1 shows its
   @16/4-HPU value is larger than the diagnosis credits.
3. **The non-MoE path dequantizes every block-FP8 linear to BF16 before a BF16 GEMM** (ops.py:868-956,
   verified in the deployed tree) — unless the recipe compiler fuses it (UNVERIFIED, BG-1), this doubles
   ~1.96 GB/rank of weight traffic per step. AB-DEQ's parity result proves the TPC dequant op alone is
   cost-neutral; only a fused dequant-GEMM (RA-012A) or verified compiler fusion removes the extra traffic.
4. **The 800 t/s @16 and 200 t/s bs=1 targets are reachable but ONLY as combinations** — no single lever
   gets there. The combined end-state model (Section 7) shows 800@16 inside the envelope after
   RA-011 + Q21 + RA-012 + ~50% of the attention/compute term, and 200 bs=1 after RA-011 + Q21 + RA-012 +
   RA-004 + attention/sampler/AR trims — the diagnosis's conclusion that "all four bottlenecks must be
   addressed simultaneously" is confirmed, with the host-side half of the combination identified more
   precisely here.

### 1.5 Round-2 state (2026-11-10 — the post-ABCONFIG-WIN bottleneck model)

The implementation agent's round-2 arc (D1-3 → PFG+MHC OOM → ABOTSIS-REF → AB3-HYPA → MEMLOG →
**ABCONFIG-WIN** → the gather_sum port) changed the state of the project:

1. **The prefill objective is MET** (5,546-7,723 t/s across 0.5-8K, +209-277% vs eager) via the
   prefill-optimized profile: tensor cache OFF (`GLM_TENSOR_CACHE_OFF=1`) + prefill graph threshold
   8,192. The CAP-8192/PFG+MHC/AB3 OOMs are worked around, not fixed — the root cause is now
   identified (MEMLOG): the mHC-prep FP32 chain materializes ~20 × [rows, 4, 4096] FP32 per layer
   (the 184-MiB class, 165.7 GB of the 667.7 GB capture volume; the 101 GB watermark that OOMed).
2. **The cost is visible in decode:** bs=1 37.3 (−12% vs the round-1 adopted 42.4 — the RA-001
   tensor-cache win forfeited for the prefill fix), @16 259.5 (parity), suite 97.5% (R003 — the
   **4th flip source: the tensor-cache state itself**; the taxonomy: Q01v2 fusion drift, RA-011
   in-flight consumption, MHC-ROWS kernel reduction order, ABCONFIG-WIN cache state).
3. **The gather_sum port (RA-018/T1.1) is applied to the venv** (env-gated off) but deviates from the
   corpus patch in three ways (the missing 16 MiB size cap, the bf16 sum, the DSV4.1-documented
   all_gather stale-value hazard at batch >1) — RA-021 fixes these BEFORE the pending A/B. The
   gsum probe (0.0%, wall_s 0.0) is an instrumentation artifact (probed a non-serving server;
   re-probe after the A/B). [REV 1.5.1 UPDATE: the A/B RAN overnight (GSUM-T1) — the row-order
   mechanism is REFUTED on our tree (no R003 flip on the hccl baseline), gather_sum is NEUTRAL
   (R003 still flips, decode parity 43.1/43.7) — the 5th flip source: the AR order itself. The
   deployed per-phase patch (RA-020's mechanism, attempted 23:41) has three defects (D1-D3) —
   do not re-run as-is; the corrected implementation is the runner-side env route with the
   >2,048 row-count signal. Detail: `RA-019-round2-theories.md` §7.]
4. **The ops layer failed once** (CRASH-INCIDENT): the OOM-crash spam bloated the journal daemon
   (tens of GB RSS on the 125 GB host) → the card-reset path (host-resident MMU page pool) failed →
   hard power recoveries. Lessons recorded: the journal rate-limit, the host-RAM headroom as a
   first-class constraint, the reset-failure signature.

**The new top-3:** RA-021 (the gather_sum hardening — prerequisite for the pending A/B; hours),
RA-020 (the per-phase tensor cache — the vendor reads `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` per call
(tier B); the 3-line runner change recovers the −12% decode now; falsifiable prediction: the suite
returns to 100% — the R003 unflip), RA-019 (the full mHC-prep TPC kernel — the structural fix that
removes the OOM class and the profile split; the DSV4.1 `control_mme` family is the pattern, the
source is in the corpus and on this host). Full detail: `RA-019-round2-theories.md`; ranking:
Section 7.

---

## SECTION 2: Independent Bottleneck Assessment

For each bottleneck in the rev-3 diagnosis: confirmation, challenges, alternative explanations, missing
information, fundamental-vs-removable verdict.

### B1 — MoE expert-weight streaming (38.1 GB/card/step)

**Confirmed (bs=1):** the stock op reads all 36 local experts unconditionally (op source tier B + the
model reproduces the measured 42.4 t/s under the all-read assumption); the floor arithmetic
(38.1 GB ÷ 2.45 TB/s = 15.5 ms → 119 t/s ceiling) is correct; the AIP 0-24% signature is consistent.

**Challenged (bs=16/32):** the diagnosis claims "at bs=16: no improvement (all 36 local experts are
already touched at 128 routes)" (B1) and "all 36 local experts touched per layer per rank" (Section 8,
using 32 verify rows × 8 = 256 routes). Both are wrong (correction C1, below). The balls-in-bins
expectation at bs=16/k=1 is ~59% of local experts touched, not 100%. The Q21 routed-only kernel therefore
still saves ~4-6.4 ms/step at bs=16 on 8 HPUs and ~8-12.7 ms at bs=16 on 4 HPUs — not zero.

**Challenged (internal inconsistency):** B1's "Expected behavior" paragraph gives the bs=1 routing-aware
floor as "16 of 36 local experts → 44% → 6.8 ms" — that is the bs=16 figure mislabeled as bs=1. Section 8
of the same document gives the correct bs=1 figure (16-24 distinct experts GLOBALLY → 2-3/rank → 2.1-3.1
GB → 0.86-1.26 ms). Adopt Section 8 (correction C2).

**Alternative explanation for the AIP signature:** the AIP 0-24% is read as "device memory-bound during the
MoE phase". It is equally consistent with "device idle waiting on the host" (the serialization fact of
Section 1.1). The missing HBM-bandwidth-during-MoE-read measurement (their Section 10) distinguishes the
two; it is a prerequisite for trusting the 15.5 ms figure as DEVICE time rather than step-attributed time.
(BG-1.)

**Missing information:** per-layer device time split; HBM bandwidth during the MoE phase; the actual
routing skew (uniform balls-in-bins is an idealization — BG-2).

**Verdict:** FUNDAMENTAL for bs=1 under the stock op (physically blocks 200 t/s); REMOVABLE via Q21
(routing-aware fused kernel — in progress, B7 unblocked). At bs≥16 the same kernel still saves 40%+ of the
weight time (C1), making Q21 more load-bearing for 800@16 than the diagnosis credits.

### B2 — Host-launch pipeline (322 recipe replays, 14 ms bs=1 / 42 ms @16)

**Confirmed (D1):** 93.3% of `execute_model` host time in the `HPUGraph.replay` frame; staging ~1.6% of
the bucket; the replay LOOP dominates. The tensor cache removed per-replay allocation (RA-001, +10.7%).

**Challenged (attribution):** py-spy attributes the time to `torch/hpu/graphs.py:75` — but that line is a
single native call, `_hpu_C.replay(self.hpu_graph, [], asynchronous)` (verified in the deployed venv).
The 14 ms is spent inside the vendor C++ `SingleHPUGraph::replay` → `replayGraph` →
`HbLazyTensor::ExecuteCachedGraph` per-recipe loop (tier B source). Consequence: the diagnosis's B2
direction "C++ execution loop: replacing the Python model runner with a C++ executor" is misaimed — the
per-recipe cost is already C++, and a user-space executor cannot remove it (rejection RJ-14). The correct
levers are (a) offloading that loop to the vendor thread pool (RA-011, tier-B-verified mechanism),
(b) cutting the recipe count (RA-012, Q21, RA-008 from the companion document), (c) a vendor
optimization request.

**Challenged (the recipe floor):** the 162 AR `mark_step()` boundaries impose a structural floor of
~163 cached graph regions (Beam port: "each all-reduce cuts the layer into separate cached graphs";
Qwen notes: "~100-190 small recipes (two all-reduce graph splits per layer × 48 layers)"). The
diagnosis's "realistic target: ~50-100 launches" is only reachable if the AR count drops from 162 to
~50-100 (or the async offload makes the floor irrelevant — which is what RA-011 does). The floor is the
reason B4 (AR count) is co-primary with B2, not rank 5 (correction C4).

**Alternative explanation for the @16 growth (42 ms):** the replay loop's row scaling (~2.6 ms/stream)
comes from per-region input staging that grows with rows (each cached graph's input set includes
row-scaled tensors) — D1 shows staging is only 1.6% of the bs=1 bucket, so at bs=16 the growth is more
likely in the per-region hash/spec work and the H2D scale conversions (`replayV3`'s
`maybe_convert_tensor_to_h2d` path, tier B). BG-1's trace should separate these.

**Missing information:** the per-recipe cost breakdown (hash vs staging vs launch) at bs=16; the actual
cached-graph count on the FP8 tree (the 322 figure is from the mxfp4 study — D1 on our tree confirmed
the bucket share but the region count on our exact tree is still the mxfp4 carry-over, flagged in the
diagnosis itself).

**Verdict:** REMOVABLE — the serialization wrapper (RA-011) removes most of its step-level impact without
touching the loop; the loop itself shrinks via op fusion (RA-012, Q21) and the AR count (RA-012C).

### B3 — Per-stream worker host cost (3.8 ms/stream)

**Confirmed (D1-16):** the EngineCore scheduler is 99.4% message-queue wait — correctly DISPROVEN as the
per-stream bottleneck; the cost is worker-local. The linear fit (40.7 + 3.8N, R²>0.99) is a solid
MEASURED scaling fact.

**Challenged (component attribution):** the rev-1 document attributed the marginal to "sampler +
scheduler + logprob processing ~1.5-2.0 ms/stream". D1-16 shows the worker host @16 = replay 68.6%
(~42 ms) / draft 17.4% (~10.6 ms) / sampler 9.9% (~6 ms) / meta ~3.4% (~2 ms). The dominant per-stream
host term is the **replay loop's row growth** (~2.6 ms/stream), not the sampler (~0.4 ms/stream). The
bench scripts request no logprobs (verified: no `logprobs` in `benchmarks/`), so the "logprob processing"
attribution in rev-1 was wrong on two counts. The draft term is nearly fixed (7.5→10.6 ms from 1→16
rows, +3 ms for 15 extra rows ≈ 0.2 ms/stream marginal), not ~1 ms/stream.

**Consequence:** RA-011 (async) removes the replay-loop term from the critical path entirely (the loop
runs in the vendor thread pool while the device executes) — this attacks the LARGEST part of B3, which
the diagnosis's B3 lever list (graph the draft + batch the sampler) only attacks in part. Graphing the
draft (RA-004) and batching the sampler remain useful but are secondary to the async fix for the
per-stream term.

**Missing information:** the replay-loop row-scaling mechanism (BG-1); the sampler's internal split
(9.9% of worker host — device-side sampling is adopted, so this is host output processing; the
per-stream floor there is small).

**Verdict:** REMOVABLE — re-targeted from "sampler/scheduler optimization" to "async host (RA-011) +
recipe reduction (RA-012/Q21) + draft graphing (RA-004)".

### B4 — All-reduce latency (162 × ~20 µs = 3.4 ms) + EP imbalance

**Confirmed (count + latency):** 162 ARs/step (46 layer boundaries × ~3.5); 20-30 µs at 24 KB (independent
8-card Gaudi2 measurement, comm-gaudi2-8card.json). The EP imbalance (slowest rank 19.5 vs mean 9.6 ms)
is from the mxfp4 study — a different precision/model; re-measure on the FP8 tree (BG-1) before sizing
B5's high-batch penalty.

**Challenged (severity ranking):** the diagnosis ranks B4 at 5 because it sees only the 3.4 ms device
cost. The ARs' STRUCTURAL cost — the ~163-recipe floor on the replay loop (Section 2/B2) — makes the AR
count a first-class host bottleneck: each removable AR saves ~43 µs of host replay per step at every
batch size, plus its 20 µs device latency. The per-layer AR budget (~3.5) is attackable: the shared-expert
AR can be absorbed into the Q21 MoE kernel (the shared expert is just an extra always-routed expert;
fuse its contribution before the single FFN AR) → ~2.5-3 ARs/layer → ~115-138 ARs. (RA-012 Part C; Q21
design note.)

**Alternative:** sequence-parallelism (RS+AG instead of 2 ARs) is unverified on this stack and changes
activation memory significantly — not worth it while the AR device cost is 3.4 ms of a 114 ms step and the
host floor is handled by RA-011. Rejected for now (RJ-19).

**Verdict:** the 3.4 ms device cost is MEDIUM and largely hides under RA-011; the AR COUNT is HIGH
(structural recipe floor + B5 load balancing). Re-ranked up.

### B5 — Eager MTP draft dispatch (7.5 ms bs=1 / 10.6 ms @16)

**Confirmed (D1 leaf level):** op dispatch ~21% of the bucket, `mark_step` closures 10.5%, `dynamic_quant`
7.5%, native `SiluAndMul.forward_native` fallback, metadata 1.6%. ~15-20 eager ops.

**Challenged (the capture blockers):** the diagnosis says the draft "cannot be wrapped in a separate HPU
graph without the storage conflict (the 'Neither storage attached' error)" and that the private-storage
cores are "implemented in our port but blocked until B7 resolved (now unblocked)". Two refinements:
(i) the storage-conflict error is the VENDOR's input-storage assertion (tier B, `replayV3`: "Neither
storage attached to input tensor, not its view") — it bites when the same storage is claimed by two
graphs; the abotsis reference's solution (draft modules held BY REFERENCE, one graph, `dry_run=False`) is
the proven pattern (EXTERNAL_RESEARCH_PROPOSALS RA-004 T1, A014/A018 READY). (ii) D1's leaf list shows
the draft's OWN ops include non-lazy-friendly pieces (`dynamic_quant` extension op, the native
SiluAndMul fallback) — even before the storage question, those ops are per-recipe boundary generators.
The draft-graph work (RA-004/Q01) must shim them (static-scale quant op; the fused clamped-SwiGLU that
abotsis ships for the HPU MoE path, `VLLM_GLM_FUSED_CLAMP_MOE` — tier D, same model).

**Missing information:** whether the deployed tree's mHC Sinkhorn is the TPC kernel or the PyTorch loop
(BG-4) — if the loop, the draft's 15-20 ops is an undercount (the mHC mixing adds many nodes per layer;
DSV4.1 measured ~3,400 nodes/step for its 4×4 Sinkhorn loop alone, removed bit-exactly by a tiny TPC
kernel — tier D, gaudi-cookbook DSV4.1 port).

**Verdict:** REMOVABLE via RA-004 (graphed draft chain — the untried cell of the project's own Q-matrix,
reference-measured at MTP-4 on the same model class). RA-011 provides an interim win: the draft's 7.5-10.6
ms of HOST dispatch overlaps the verify's device execution even while the draft stays eager.

### B7 — Op-loading registrar collision

**Confirmed RESOLVED:** the load-order recipe (load the legacy-API op lib after full HPU init; the
serving path does this naturally) is verified in production (q21h boot, 8/8 workers, `block_fp8_dequant_glm
VISIBLE in torch.ops.custom_op`); AB-DEQ shows the wired dequant path is at parity (production-qualified).
One hygiene note (N2, below): the EXPERIMENTS.md final section ("collision FINAL — the registrar is
unavoidable in userspace .so") predates the q21h result and states the superseded conclusion as the
last word in the file.

### Corrections register (to current_bottleneck.md)

| ID | Location | Correction |
|---|---|---|
| C1 | B1 "Performance impact" + Section 8 "@16" | At bs=16/k=1 the verify processes 32 rows × 8 routes = 256 global routes → per rank ≈32 routes into 36 local experts → expected distinct ≈ 36(1−e^(−32/36)) ≈ 21/36 = **59%**, not 100%. (Even the B1 text's own "128 routes" figure — which forgets the ×2 MTP verify rows — gives 36(1−e^(−16/36)) ≈ 36%, still not 100%.) At bs=32: ≈88%; all-touched only at bs≥~55 (k=1). Consequence: Q21 saves **~4-6.4 ms/step at @16 (8 HPU)** and **~8-12.7 ms at @16 (4 HPU)**, not 0; the 800@16 roofline and the 4-HPU candidate numbers both rise. (DERIVED, uniform-routing idealization; real skew measured per BG-2.) |
| C2 | B1 "Expected behavior" | The bs=1 routing-aware floor is 16-24 distinct experts GLOBALLY (2 verify rows × 8 + 1 draft row × 8, partial overlap) → 2-3/rank → 2.1-3.1 GB/card → **0.86-1.26 ms**, not "16 of 36 local → 44% → 6.8 ms" (the latter is the bs=16 figure). Section 8 of the same document has it right; B1's paragraph is the error. |
| C3 | Section 3 "Model loading" | "the lm_head is NOT sharded (replicated)" is wrong for the deployed port: `model.py:411` uses `ParallelLMHead` (vocab-parallel, TP-sharded); only `PPMissingLayer` is the PP fallback (l.419). Minor (lm_head bytes are small at TP8), but the weight-mass table's per-rank breakdown should use the sharded figure. |
| C4 | Section 9 ranking | B4 is rank 5 on device-cost grounds, but the AR `mark_step` boundaries impose the ~163-recipe structural floor on B2 (Section 2/B2) — the AR count is a first-class HOST bottleneck and a prerequisite for B2's "50-100 launches" target. Re-rank B4 up (co-primary with B2). |
| C5 | B2 "Potential optimization directions" | "C++ execution loop: replacing the Python model runner with a C++ executor" — the per-recipe cost is already in vendor C++ (`ExecuteCachedGraph`), not Python (the py-spy frame at graphs.py:75 is the native-call boundary). A user-space C++ executor cannot remove it. The working levers are the vendor thread pool (RA-011), the recipe count (RA-012/Q21/RA-012C), and a vendor request. |
| C6 | Section 1.2-3 + B5 | The eager draft's host cost is ~7.5 ms FIXED + ~0.2 ms/stream (7.5→10.6 from 1→16 rows), not ~1 ms/stream as the rev-1 "per-stream" attribution implied. The per-stream marginal is dominated by the replay loop's row growth (~2.6 ms/stream). |
| C7 | Section 5 (bs=1 model) | The bs=1 step (45.1 ms) ≈ host (24) + device (21) with ~zero overlap — the serialization fact (Section 1.1) should be stated in the document; it is the enabling condition for RA-011 and explains why the AIP is 0-24%. |

### Notes (non-corrections)

- **N1 — the mxfp4 carry-over:** the 322-region count, the 74 µs/launch figure, the 162-AR count, and the
  EP-imbalance numbers are all from the GLM-5.3 (non-Flash) MXFP4 step-anatomy study. D1 re-measured the
  HOST split on the FP8 tree (done, good); the DEVICE-side per-layer split is still pending (their
  Section 10, = BG-1 here). The weight-read floor (15.5 ms) is FP8-correct from the checkpoint geometry.
- **N2 — EXPERIMENTS.md hygiene:** the file's final section states "the registrar is unavoidable in
  userspace .so … ARCHITECTURAL property … not fixable in userspace" — superseded the same day by the q21h
  boot (B7 RESOLVED; the op lib runs in production). A future reader taking the last word gets the wrong
  conclusion. (The implementation agent's record; flagged, not edited.)
- **N3 — the Q01 queue item** (TASKS.md) still carries the stale "accuracy gate FAILED (97.5%)" framing;
  per the companion document (rev 5), Q01b/Q15 established the 97.5% as a cross-boot artifact and the
  draft graphs as accuracy-neutral. The split-mode +9-11% result and the Q19 non-stacking with the Q18
  levers remain the operative facts.

---

## SECTION 3: New Research Hypotheses (RA-011 … RA-021)

Full detail files: `RA-011-async-replay.md`, `RA-012-fused-dequant-gemm-sinkhorn.md`,
`RA-013-expert-placement.md`, `RA-014-prefill-buckets.md`, `RA-016-dspine-drafter.md`,
`RA-017-ecf8-lossless-weights.md`, `RA-018-mtp4-theories.md`, `RA-019-round2-theories.md`
(RA-019/020/021 + the round-2 audit, 2026-11-10).

---

### RA-011 — Asynchronous HPU-graph replay with the vendor launch thread pool

**ID:** RA-011 (new)
**Title:** Move the decode-step replay off the host critical path: `asynchronous=True` +
`PT_HPU_ENABLE_HPUGRAPH_THREAD=1` + `PT_HPU_QUEUE_SYNLAUNCHES=1` (+ optional
`PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1`)

**Target bottleneck:** B2 (the 322-recipe replay loop, 14 ms bs=1 / 42 ms @16) + B3 (per-stream host) +
the serialization fact (C7) — i.e., the HOST side of the step at every batch size.

**Current limitation:** `_maybe_wrap_in_hpu_graph` (hpu_model_runner.py:1118-1120) wraps the main model
with `htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(...), disable_tensor_cache=False)` — **no
`asynchronous` argument, so the vendor default `False` applies**. Every decode step: the host thread
runs the entire 322-recipe C++ launch loop (14 ms bs=1 / 42 ms @16) AND blocks until the device finishes
the step's work, then performs the sampler (~2.5-8 ms), the eager draft dispatch (~7.5-10.6 ms), and the
metadata work (~2 ms) before the next step can begin. The measured step (45.1 ms bs=1; 114 ms @16) equals
host + device with ~zero overlap (C7). The runner's own design comment (hpu_model_runner.py:4381) says
"Everything is done asynchronously - the only sync point is the place where we copy the generated tokens
back to the host" — the replay is the one piece that does not honor that design.

**Technical hypothesis:** the vendor bridge already implements the needed offload:
1. `HPUGraph.replay(..., asynchronous=True)` → `_hpu_C.replay(graph, [], True)` →
   `SingleHPUGraph::replayGraph(..., async=true)` (tier B, `backend/habana_device/HPUGraph.cpp`).
2. Inside `replayGraph`: `bool queue_in_thread_pool = async &&
   GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD) && GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES);` — with
   both env vars set, the ENTIRE `HbLazyTensor::ExecuteCachedGraph` (the 322-recipe loop) is enqueued on
   the vendor's `SingleTonExecThreadPool` and the calling thread returns after staging. (tier B source,
   read directly.)
3. `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1` skips the per-replay `RecipeArgumentSpec` recomputation
   (reuses `cached_rarg_psh_`) — the input-hash work D1 identified inside the loop. (tier B source.)
4. Vendor documentation (tier B, docs.habana.ai HPU Graph APIs): `asynchronous=True` — "This flag can be
   used along with stream and async D2H to copy outputs for reducing host overheads."

With the replay off the critical path, the per-step host critical path becomes: input staging
(`copy_to(cached.graph_inputs, inputs)` — ~1.6% of the replay bucket ≈ 0.2-1 ms) + the sampler/draft/
metadata host work (~12 ms bs=1 / ~20 ms @16, now running WHILE the device executes step N) + the token
D2H sync (the runner's designed sync point). The step time approaches max(host, device) + sync ε.

**Proposed architectural change:** (1) one-line change: `asynchronous=True` in `_maybe_wrap_in_hpu_graph`
(l.1120) for the main model wrap (l.5021); leave the drafter wrap (l.5060) synchronous for the first
iteration (it only matters once the draft is graphed — RA-004). (2) env: `PT_HPU_ENABLE_HPUGRAPH_THREAD=1`,
`PT_HPU_QUEUE_SYNLAUNCHES=1` on the serving boot. (3) optional second iteration: `PT_HPU_DISABLE_HPUGRAPH_
REPLAY_HASHCHECK=1` (the graph cache is keyed by `input_hash(inputs)` — shapes/layouts are stable per
bucket, so skipping the inner hash check is the intended use; gate it behind the same A/B). (4) No runner
logic changes: the stream-ordered staging (the H2D `copy_to` is enqueued before the replay on the same
stream) preserves device-side ordering; the token D2H already provides the step boundary sync; the eager
draft ops are enqueued on the stream after the verify graph work (ordered by the stream).

**Why it should work:** the mechanism is not inferred — it is the vendor's own implementation path, read
in the tier-B bridge source; the vendor documents the flag for host-overhead reduction; the D1 data shows
exactly the work that gets overlapped (24 ms host / 21 ms device, serialized). The only unverified
element is the interaction with `disable_tensor_cache=False` (retained in/out tensors) — analyzed in
Risks below; the Qwen-notes failure modes (hangs, poisoned server) are the failure signatures to watch,
not the expected behavior (they involved different mechanisms: engine-level `--async-scheduling` and
compile-mode AR-split, both rejected here as separate items, RJ-13/RJ-18).

**Gaudi 2 feasibility:** verified in source (tier B) AND in the DEPLOYED build (g2, 2026-11-09): the
`wrap_in_hpu_graph(module, asynchronous=False, disable_tensor_cache=False, ...)` signature confirmed in
`/opt/glm53/venv/.../habana_frameworks/torch/hpu/graphs.py` (l.840; `replay(self, asynchronous=False)`
l.67), and the env var strings `PT_HPU_ENABLE_HPUGRAPH_THREAD`, `PT_HPU_QUEUE_SYNLAUNCHES`,
`PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK` all present in the deployed 1.24.1 bridge .so (strings
scan); the live process environ does NOT set them (verified) — the premise holds.
`PT_HPU_ENABLE_LAZY_COLLECTIVES` (auto-enabled, platform.py:432 — companion doc rev 4) means the
in-graph collectives are the supported TP path, so nothing about the async offload changes the
collective semantics (the HCCL work still executes on the device in stream order).

**Supporting research:**
- Tier B: `habana_frameworks/torch/hpu/graphs.py` (wrapper: `copy_to` + `replay(asynchronous)`; the
  cache is keyed by `input_hash`; `CachedParams` stores the `asynchronous` flag per entry);
  `backend/habana_device/HPUGraph.cpp` (`replayGraph` thread-pool branch, the `SetExecutionInProgress` /
  `SetTensorDataNullOpt` bookkeeping — note: in the thread-pool branch the output tensors are NOT
  nulled, which is the tensor-cache-friendly path); docs.habana.ai (flag semantics).
- Tier D (cautionary): gaudi-llm-notes Qwen3.8-Flash-Next — `--async-scheduling` (vLLM engine flag, a
  DIFFERENT mechanism) hangs; "fused 6-layer regions … each partition carries far more tensors" (larger
  regions are not automatically faster — does not apply here: we are not changing region sizes);
  gaudi-cookbook DSV4.1 — "HPU graph caches key on the input's view state … re-captured every warmed
  graph" (watch the graph count during soak — a re-capture storm would show as HBM growth + latency
  steps).
- Project: D1 (host split), C7 (serialization), RA-001 (tensor cache adopted — the precondition),
  Q18 levers (early-draft enqueue already assumes the device keeps working after host dispatch).

**Expected performance impact (ESTIMATED until measured):**
- bs=1: host critical path 24 → ~3-5 ms (staging + D2H + the small residual); step ≈ max(host ~4, device
  ~21-28) + ε ≈ 22-29 ms → 42.4 → **~65-85 t/s (+50-100%)**. Conservative (the thread-pool enqueue adds
  host work back, partial overlap): +30-50%. Optimistic (device tail also shrinks via the draft overlap):
  +80-100%.
- @16: host critical path 61 → ~5-10 ms; step ≈ max(~10, device ~53) + ε ≈ 55-60 ms → 257 → **~430-470
  t/s (+70-85%)**. Conservative: +40-60%.
- 4-HPU bs=1 (33.3): similar relative effect (the host term is rank-local and unchanged in size): est.
  50-65 t/s.
- The numbers above assume the device-time estimates hold (BG-1 will tighten them). All ESTIMATED.

**Conservative estimate:** +30% bs=1 / +40% @16 (thread-pool overhead + imperfect overlap).
**Optimistic estimate:** +100% bs=1 / +85% @16 (full overlap; the device tail also benefits from the
eager draft running under the verify).

**Implementation complexity:** LOW for the change (1 line + 3 env vars, 1 boot). MEDIUM for the
verification protocol (parity + soak + the graph-count watch). No kernel work.

**Risks:**
- **Correctness (highest):** the tensor-cache interaction — with retained out tensors, step N+1's
  `copy_to` staging writes the cached INPUT tensors while the device may still be reading them from step
  N. Mitigated by stream ordering (the H2D is enqueued after N's work on the same stream) — but this
  holds only if the staging is a stream-ordered H2D (it is: `copy_to` into the cached device tensors) and
  not a direct host write into a mapped buffer (it is not). Verification: greedy bit-identity on the
  accuracy suite (same-boot A/B protocol) is the gate; any divergence = stop, bisect (main-model wrap
  only vs + env vars).
- **The thread-pool branch's host bookkeeping** (`SetExecutionInProgress` on in/out tensors from the pool
  thread while the host thread prepares the next step) — a vendor-internal race possibility; the vendor
  ships this as a supported path, but our exact configuration (tensor cache on + in-graph collectives)
  is untested in the corpus. Mitigation: the 30-60 min soak at 16 streams + the graph-count/HBM watch
  (N3 re-capture signature) before adopting; a vendor question (BG-5) in parallel.
- **Failure signatures to watch** (from the corpus): hangs at the first request (the async-scheduling
  mode's signature — different mechanism, but the symptom class), garbage/`!!!!` output (the
  AR-split=0 signature — watch for stale-value patterns at batch>1, the DSV4.1 all_gather-stale lesson),
  HBM growth (re-capture storm).
- **Numerical:** none expected (no math change; pure scheduling). The dry_run caveat does not apply
  (we stay in the tensor-cache-on, non-dry-run path).

**Dependencies:** none (independent of Q21/RA-012/RA-004; composes with all of them). Precondition: the
adopted tensor-cache config (RA-001) — already in place.

**Previous experiments checked:** RA-001 (tensor cache — the enabling config, adopted); Q18 levers
(early-draft enqueue — compatible, enqueues more device work after the host returns); EXP-Q19 (draft
graphs + levers non-stacking — unrelated; RA-011 does not graph the draft); the Qwen-notes
`--async-scheduling` failure (vLLM engine flag, different mechanism — RJ-18 distinguishes them).
**Why different from previous attempts:** no prior experiment touched the wrapper's `asynchronous`
argument or the thread-pool env vars; the B2 work so far (RA-001) attacked per-replay ALLOCATION, not the
serialization. This is the first item that attacks the host/device BOUNDARY itself.

**Implementation starting point:**
1. `ports/glm-5-3-flash/glm5_next_hpu/...` is not involved — the change is in the DEPLOYED tree's
   `vllm_gaudi/v1/worker/hpu_model_runner.py:1118-1120` (`_maybe_wrap_in_hpu_graph`): add
   `asynchronous=True` to the `wrap_in_hpu_graph` call. Keep it env-gated (e.g. `GLM_HPU_GRAPH_ASYNC=1`)
   for the A/B.
2. Boot the A/B arms (fresh boot each, same-boot-protocol): arm 0 = current; arm 1 = +`asynchronous=True`;
   arm 2 = arm 1 + the two thread-pool env vars; arm 3 = arm 2 + the hashcheck env var. (If the
   implementation prefers fewer boots: arm 0 vs arm 2 first — the thread pool is where the loop cost goes;
   async-without-pool still blocks less but the host loop stays on the critical path.)
3. Per arm: (a) greedy bit-identity vs arm 0 on the accuracy suite subset (the same-boot protocol),
   (b) bs=1 + 16-stream bench (the standard reps), (c) 30-60 min soak at 16 streams, (d) HPUGraph cache
   count + HBM watch during the soak (verbose wrapper logging: `verbose=True, log_frequency=100` is
   available in the wrapper — the re-capture signature), (e) a timeline trace (torch profiler CPU+HPU)
   to measure the ACTUAL overlap ratio (BG-3).
4. Record EXP-RA011; update the bottleneck doc's Section 5 with the measured overlap.

**Validation criteria:** (a) greedy parity vs arm 0 (bit-identical on the suite subset; any flip = fail);
(b) bs=1 ≥ +30% and @16 ≥ +40% over arm 0 (below that = the overlap is not materializing; inspect the
trace before iterating); (c) soak: no hang, no garbage, no HBM growth >1 GB, graph count stable;
(d) the trace shows host post-work (sampler/draft frames) overlapping device execution of step N
(the qualitative confirmation of the mechanism).

**Confidence level:** HIGH on the mechanism (tier-B source + vendor docs + the D1/serialization evidence);
MEDIUM on the magnitude (the overlap efficiency is unmeasured — ESTIMATED range above).

---

### RA-012 — Fused dequant-GEMM for the non-MoE block-FP8 linears + mHC Sinkhorn TPC kernel

**ID:** RA-012 (new)
**Title:** Kill the double weight traffic of the non-MoE FP8 path and the mHC node storm

**Target bottleneck:** B2 (recipe count — the ~135 dequant+GEMM pairs + the mHC nodes) + the device
weight traffic of the non-MoE linears (~1.96 GB/rank FP8 read + 3.93 GB BF16 write + 3.93 GB BF16 read
per step IF unfused — ~7.1 ms device vs ~0.8 ms fused; UNVERIFIED whether the recipe compiler already
fuses — BG-1 is the deciding measurement).

**Current limitation (verified in the deployed tree):** `vllm_gaudi/extension/ops.py`
`apply_block_fp8_linear_hpu` → `apply_block_fp8_linear_hpu_dequant` (l.927-960): dequantizes the block-
FP8 weight (`dequant_block_fp8_weight_naive`, l.868-915: a `weight.to(bf16) * scale.to(bf16)` chain — or
the TPC `block_fp8_dequant_glm` op behind `GLM_FP8_MXK`, which produces the SAME BF16 material) and then
runs a BF16 `torch.nn.functional.linear` (l.954-956). This applies per step to every non-MoE FP8 linear:
the attention QKV/O projections, the 3 dense layers, the 42 shared experts (~135 linears total, each = 2
recipes: dequant + GEMM). AB-DEQ (parity: 42.7 vs 41.4 t/s) proves the TPC dequant op ALONE is
cost-neutral — the extra BF16 traffic (or its compiler fusion) is the open question, and only a
dequant+GEMM FUSED op removes the traffic either way.
Second part: the mHC mixing (4 streams, 20 Sinkhorn iterations, ALL 45 layers) — the DSV4.1 port (same
mHC design) measured the 4×4 Sinkhorn loop at ~3,400 device nodes per step and removed them bit-exactly
with a tiny TPC kernel (tier D, gaudi-cookbook DSV4.1 "What it taught about Gaudi"). The cookbook GLM
port ships the "optional Sinkhorn TPC kernel" — **RESOLVED 2026-11-09 (live process environ on g2):
`GLM5_NEXT_MHC_SINKHORN=kernel` is set in the running server — the kernel IS ON**; the 20-iteration
PyTorch loop is not the live path (BG-4 closed; the mHC node count is recorded as a baseline in
BG-1's trace only).

**Technical hypothesis:** (A) a `block_fp8_dequant_gemm` TPC op — read E4M3 + FP32 block scales, dequant
in-register (the 1Cat `v_f8_ld_tnsr_b` SW_UNPACK + `v_convert_f8_to_bf16_b` pattern, tier D), accumulate
the GEMM in FP32 over K — replaces each (dequant, F.linear) pair with ONE op: −135 recipes and −3.9 GB/
card of intermediate BF16 traffic per step (if unfused today). The Q21 MoE kernel is the batched-per-
expert version of exactly this op; the dense linear version is simpler (one [N,K]×[K,M] GEMV at decode
row counts ≤32). (B) **CLOSED (2026-11-09):** the Sinkhorn TPC kernel is verified ON in the live tree
(`GLM5_NEXT_MHC_SINKHORN=kernel`, process environ) — the DSV4.1-class node storm (~3,400 nodes/step)
does not exist on this tree; no Part B work remains (the BG-1 trace records the mHC node baseline).
Also verified live: `GLM_FP8_MXK=0` — the running dequant path is the torch dequant (`dequant_block_
fp8_weight_naive` → BF16 → `F.linear`); the TPC dequant op (AB-DEQ) is a separate gated experiment.

**Proposed architectural change:** (A) extend the mxk op family (the production-qualified infrastructure:
B7 load recipe, `mxk_op_lazy.cpp` pattern, the variant library) with `custom_op::block_fp8_dequant_gemm`
(contract: contiguous 2-D E4M3 [N,K], N%128==0, K%256==0, F32 scales [N/128,K/128], BF16 input [M,K],
M≤64; output BF16 [M,N] (+ bias)); wire it into `apply_block_fp8_linear_hpu_dequant` behind the existing
`GLM_FP8_MXK` guard (same fail-closed pattern as the dequant op). (B) source-check the deployed tree's
mHC path (`mhc.py` in the port) for the TPC Sinkhorn kernel gate; if absent, port the cookbook kernel
(bit-exactness gate vs the loop).

**Why it should work:** the dequant-in-register GEMV is the same demonstrated pattern as the Q21 kernel's
w13/w2 phases (in-register E4M3 dequant, FP32 accumulation, the block-scale multiply — all staged in the
repo and production-qualified at the dequant level, AB-DEQ); the DSV4.1 MoE variant family
(`moe_fp8_prequant`, `direct_finalize`, `prefetch_w2`, `horizontal` — tier D) is the tuning playbook.
The Sinkhorn part is a proven bit-exact node elimination on the identical mHC design (DSV4.1).

**Gaudi 2 feasibility:** TPC GEMV at decode M (≤32 rows) is the regime the mxk core already runs; the
1Cat block-FP8 dequant kernel (tier D) proves the E4M3 load-pipe expansion on Gaudi2; MME is the
alternative for the GEMM at larger M (prefill — out of scope for this proposal, decode only). The op
infrastructure (B7) is production-qualified.

**Supporting research:** ops.py l.868-956 (deployed tree, read directly); AB-DEQ (parity — the dequant
op alone is neutral, so the fused GEMM is the only path to the traffic saving); 1Cat
`block_fp8_dequant_gaudi2.c` (tier D, the load-pipe pattern); DSV4.1 variant family (tier D); DSV4.1
Sinkhorn lesson (tier D: ~3,400 nodes/step removed bit-exactly); gaudi-cookbook GLM port README (the
optional TPC Sinkhorn kernel exists for this model class).

**Expected performance impact (ESTIMATED):**
- Part A: if unfused today (BG-1 to confirm): −3.2 ms device bs=1 (7.1→0.8+3.9… precisely: 1.96 GB FP8
  read instead of 9.8 GB total traffic → −7.9 GB → −3.2 ms at 2.45 TB/s) + −135 recipes (−~5.8 ms host
  replay at 43 µs/recipe, of which RA-011 removes most from the critical path anyway — the two compose:
  RA-011 hides the loop; RA-012A shrinks it and the device term). If already compiler-fused (BG-1
  negative for the traffic): the device term is a wash and the value is the −135 recipes only.
- Part B: **CLOSED (2026-11-09)** — the Sinkhorn TPC kernel is verified ON in the live tree
  (`GLM5_NEXT_MHC_SINKHORN=kernel`, process environ); the loop is not the live path. The BG-1 trace
  records the mHC node count as the baseline only.
- Combined ESTIMATED: −3 to −4 ms/step bs=1 (traffic unfused — the full Part A device term); −1 to −2 ms
  (traffic already compiler-fused — the −135 recipes only).

**Conservative estimate:** −1 to −2 ms/step bs=1, −135 recipes.
**Optimistic estimate:** −4 ms/step bs=1, −135 recipes.

**Implementation complexity:** MEDIUM-HIGH (a new TPC kernel — but the dequant pipeline, the op
registration, the load recipe, and the build system are all production-qualified from Q21/AB-DEQ; Part B
is CLOSED — the kernel is verified ON in the live tree, 2026-11-09).

**Risks:** numerical (FP32 accumulation across K with 128×128 block scales — the same precision contract
as the Q21 kernel, already qualified at the dequant level; the GEMM accumulation order changes vs the BF16
F.linear → the parity gate must be the suite, not bit-identity against the stock path — use the
arithmetic-sanity + suite-accuracy protocol); the padded-weight `do_unpad` path must stay in torch (the
op cannot unpad — same constraint as the dequant op); capture interaction (the op is already captured in
graphs under AB-DEQ — the dequant op serves healthy graphs; the GEMM variant inherits the same
qualification path).

**Dependencies:** B7 load recipe (done); the mxk build system (done); BG-1 (the per-layer trace decides
Part A's device term BEFORE kernel work — do not start the GEMM kernel until BG-1 confirms the
unfused-traffic hypothesis, or until the recipe-count value alone justifies it).

**Previous experiments checked:** AB-DEQ (dequant op parity — the direct predecessor; shows the op
infrastructure works and the dequant-alone value is zero); Q16 (the gather+GEMM decomposition —
different: Q16 gathered expert weights for the MoE op; this is a dense fused dequant-GEMM, no gather,
no intermediate); Q21 (the MoE kernel — this proposal is its dense-linear sibling and shares its
infrastructure); C015 (the KDA decay-dot kernel — no overlap).
**Why different:** no prior experiment fused the dequant INTO the GEMM for the non-MoE linears (AB-DEQ
explicitly stops at the dequant). The mHC-node-count question is resolved (kernel ON, verified 2026-11-09
in the live process) — no Part B experiment needed.

**Implementation starting point:**
1. BG-1 first (the per-layer device trace — already on the diagnosis's missing-measurements list): the
   KDA/DSA/MoE/mHC/linear split + the HBM bandwidth during the non-MoE linear phase. Decision: if the
   non-MoE linear phase shows ~9.8 GB of traffic (unfused), Part A's device term is confirmed.
2. (BG-4 — DONE 2026-11-09: live process environ shows `GLM5_NEXT_MHC_SINKHORN=kernel`; Part B closed.
   The BG-1 trace records the mHC node baseline.)
3. Part A: the `block_fp8_dequant_gemm` op per the contract above; on-device qualification (the AB-DEQ
   protocol: parity perf, arithmetic sanity, correct output vs a CPU reference, stable serving); then the
   A/B at bs=1 + @16. NOTE: the live config is `GLM_FP8_MXK=0` (torch dequant path) — the A/B arm sets
   the gate ON.
4. Record EXP-RA012a; update the recipe count (the wrapper's verbose logging gives the cached-graph
   count).

**Validation criteria:** Part B: CLOSED — no validation required (kernel verified ON, 2026-11-09); the
BG-1 trace records the mHC node baseline (expect the kernel's ~1 node/layer signature, not the loop's
~3,400/step). Part A: (a) the output vs a CPU reference within the FP8 block-scale
tolerance (the AB-DEQ arithmetic-sanity protocol), (b) the HBM traffic of the linear phase in the trace
drops ~4× (the direct mechanism proof), (c) bs=1 and @16 bench: no regression vs the current stack,
target ≥ −1 to −2 ms/step at bs=1 (the conservative bound), (d) graphs capture with the new op (the
AB-DEQ precedent).

**Confidence level:** HIGH that the op is buildable/qualifiable (the infrastructure is proven); MEDIUM on
the device term (BG-1 decides it); Part B is CLOSED — the Sinkhorn kernel is verified ON in the live
tree (2026-11-09), so the DSV4.1 precedent no longer carries any open work.

---

### RA-013 — Expert placement by routing histogram (EP load balancing + Q21 byte-model input)

**ID:** RA-013 (new)
**Title:** Re-shard the expert→rank assignment by measured routing popularity

**Target bottleneck:** B5 (EP imbalance — the slowest-rank penalty, ~10 ms at high batch in the mxfp4
study; re-measure per BG-1) + the Q21 byte model (the distinct-expert count under real routing skew).

**Current limitation:** the expert→rank mapping is the checkpoint's shard order (rank r holds experts
[36r, 36r+36) — the architecture diagram in the diagnosis). The router's per-expert popularity is
non-uniform (the load-balancing loss makes it CLOSE to uniform, not uniform). Two consequences:
(i) per-rank per-step token load is skewed → the slowest-rank GEMV tail (B5); (ii) the Q21 routed-only
byte model (balls-in-bins under uniformity, C1) is an idealization — a skewed distribution hits FEWER
distinct experts than uniform (Zipf-like concentration), so the Q21 @16 saving is larger than the 40-50%
estimated, and the bs=1 hot set is much narrower than 8 uniform draws.
Precedent (tier D, project-maya "strata" GLM port): "VRAM slots per layer by routing share
(`expert_counts.txt`) | 35.5 -> 32.9 ms/token" — a per-layer popularity-proportional allocation measurably
helped on their 2×V100 machine; they also maintain `data/expert-profile-glm.bin` (a precomputed expert
access profile) — the routing-profile artifact is standard practice for this model class.

**Technical hypothesis:** compute the per-layer, per-expert route histogram from a representative traffic
sample (100-1,000 decode steps of serving, or a CPU-side router pass over a prompt set — the router is a
small GEMM [4096→288] + top-8, cheap off-device); sort experts per layer by popularity; reassign rank r
the experts at stride positions in the sorted list (rank 0 gets the hottest, then the next-hottest to
rank 1, … — interleaving equalizes the expected per-rank load). This is a PURE PERMUTATION of identical
expert weights: zero accuracy impact, zero byte-count change, a load-time change to the EP shard map (the
safetensors→rank assignment at load).

**Proposed architectural change:** (1) the histogram tool (a logging hook in the router fast path —
`GLM_ROUTER_FAST=1` already computes the top-8 on-device; a debug mode that accumulates per-layer
histograms into a CPU tensor over N steps, dumped to a file — or the CPU-router alternative); (2) the
re-sharded load map (an env-gated permutation applied at expert-shard load); (3) the A/B: current map vs
re-sharded map, same-boot protocol, bs=1 + @16 + @32 bench + the per-rank MoE kernel times from the
trace (the slowest/mean ratio is the direct metric).

**Why it should work:** the EP imbalance is a balls-in-bins statistic over the per-rank route counts;
interleaving by popularity minimizes the variance of the per-rank route sum (the standard load-balancing
argument; the skew is mild — the load-balancing loss keeps it near-uniform — so the expected gain is
modest but certain in sign, and it also TIGHTENS the Q21 byte model: the measured distinct-expert
distribution replaces the uniform idealization).

**Gaudi 2 feasibility:** pure data-placement change at load time; no kernel, no collective, no graph
change (the expert weights land at the same per-rank HBM addresses, permuted content). The only
interaction: the Q21 kernel's per-rank expert index table must use the new map (it will, once built —
the kernel reads the routing indices, not the shard order).

**Supporting research:** project-maya GLM port (the routing-share allocation + the profile artifact + the
measured 35.5→32.9 effect, tier D); the B5 measurement (mxfp4 study: slowest rank 19.5 vs mean 9.6 ms —
re-measure on our tree per BG-1); the Q17 EP4×TP2 refutation (different lever — layout, not placement;
placement helps at ANY layout).

**Expected performance impact (ESTIMATED):** the B5 tail: from 2× mean toward ~1.2-1.5× mean (the
interleaving residual) → −2 to −5 ms/step at bs=16-32 (the high-batch regime where the compute tail
matters; at bs=1 the effect is ≤1 ms). The Q21 byte model: the skewed distinct-expert count is
EXPECTED to be 5-20% below the uniform estimate at bs=16 (DERIVED from typical router skew; the
histogram measures it exactly) → the Q21 @16 saving rises from ~4-6.4 ms toward ~5-8 ms.

**Conservative estimate:** −1 to −2 ms at @16; the byte-model input alone is valuable for Q21 design.
**Optimistic estimate:** −4 to −5 ms at @16-32 + a tighter Q21 design.

**Implementation complexity:** LOW-MEDIUM (the histogram hook is a small debug addition; the re-sharded
load map is a permutation of the existing shard-load code; no kernel work).

**Risks:** negligible correctness risk (pure permutation — the parity gate is trivially expected to
pass; if it fails, the load map is buggy, not the math). The only operational risk: the permutation must
be pinned (a fixed seed from the histogram) so that A/B arms and the deployed config use the same map;
record the map in the experiment record. The histogram sample must match the serving traffic
distribution (code-heavy vs chat — use the bench prompt set + a serving sample).

**Dependencies:** BG-1 (the per-rank kernel times to size the B5 effect on our tree) or at minimum the
histogram (independent); composes with Q21 (the byte model) and is layout-independent (works at TP8+EP8
and TP4+EP4).

**Previous experiments checked:** Q17 (EP4×TP2 — a LAYOUT change, refuted on the stock op; placement is
orthogonal and helps under any layout); B5's mxfp4 measurement (the effect exists, magnitude to
re-measure); no prior placement experiment in the project (checked EXPERIMENTS.md — no expert-placement
entry).
**Why different:** no prior experiment re-sharded the expert→rank assignment; Q17 changed the
parallelism topology, not the assignment.

**Implementation starting point:**
1. The histogram: extend the router fast path (`GLM_ROUTER_FAST`) with a debug accumulator (per-layer
   per-expert route counts over N=1,000 steps, dumped as a .npz) — or the CPU-router pass over the bench
   prompt set (simpler, no server change; the router GEMM + top-8 in torch on CPU over a few hundred
   prompts × 45 layers is minutes of CPU time).
2. The re-sharded map: sort per layer, interleave to ranks, env-gate the load-time permutation.
3. A/B (same-boot protocol): current vs re-sharded at bs=1 / @16 / @32; the per-rank MoE kernel times
   from the trace (BG-1's data doubles as the before/after).
4. Record the histogram + the map in EXP-RA013; feed the distinct-expert distribution into the Q21
   byte model.

**Validation criteria:** (a) parity (greedy identical — the permutation is exact); (b) the per-rank
   kernel-time slowest/mean ratio drops (target: from ~2× toward ≤1.5× at @16); (c) @16 bench ≥ current
   (no regression) with a target of +1 to +3%; (d) the histogram is recorded (the Q21 input) regardless
   of the bench outcome.

**Confidence level:** HIGH that the histogram is informative (the profile-artifact precedent); MEDIUM on
the step-time gain (the B5 magnitude on our FP8 tree is unmeasured — BG-1).

---

### RA-014 — Prefill 2-4K bucket-transition fix (config-only)

**ID:** RA-014 (new)
**Title:** Close the 2,538-2,787 t/s dip at 3.5-4K prompts (the 2048→4096 lin-bucket transition)

**Target bottleneck:** the one unmet sub-case of the prefill objective (3-6K t/s): "Prefill (warm,
3.5-4K) 2,538-2,787 t/s — slightly below — the 2048→4096 lin-bucket transition" (scorecard, MEASURED).

**Current limitation:** the lin bucketing steps the query dimension by 128 up to 2048, then by 2048
(2048, 4096, 6144…). A 3.5K-token prompt lands in the 4096 bucket → ~15% padding waste at the bucket
edge, and worse, the 2048→4096 TRANSITION is where the measured 30% dip sits (2,538-2,787 vs
3,338-5,697 at 0.5-2K and 6-8K). The 6-8K band being MET again (3,782-5,663) confirms the dip is
transition-specific, not a long-context cliff.

**Technical hypothesis:** the transition dip is the bucket-edge padding waste + the first-use compile of
the 4096 bucket mid-bench (the 13-15 s lazy first-use compilation, if the 4096 bucket was not pre-warmed
in the bench's warm-up set). Both are config-addressable: (a) finer query buckets in the 2-4K range
(512-step: 2048, 2560, 3072, 3584, 4096 — or 256-step), which cuts the edge waste from 15-50% to ≤6%;
(b) pre-warming the intermediate buckets in the boot's warm client (the pre-warm protocol already exists
for the decode ladder — extend it to the 2-4K prefill shapes).

**Proposed architectural change:** config only: the lin-bucket query step in the 2048-4096 range (the
`VLLM_PROMPT_*` bucketing knobs the tree has — the Q04 bucket-pinning work already identified them) +
the warm-client extension. No code, no kernel.

**Why it should work:** the bucket mechanism is the project's own (E015-A: +1.4% decode, warmup 3×
faster — the bucketing is proven); the dip's location (exactly the 2048→4096 edge) matches the padding-
waste + cold-compile signature; the 6-8K recovery confirms no other mechanism is at work in that band.

**Gaudi 2 feasibility:** pure serving config; the only cost is HBM for the extra graph shapes (each
decode/prompt shape's graph workspace — the 25.5 GiB first-reclaimable-block figure from the companion
document applies to the 8K-row PREFILL graphs; the 2.5-4K query shapes are far smaller — and the pool has
~28 GB free at decode (96 − 68 GB used)).

**Supporting research:** the scorecard dip (MEASURED); E015-A (the bucketing mechanism, project data);
the warm-protocol (RESEARCH-001 addendum-9, companion document); the gpt-oss/GLM prefill bucket
behavior (no contrary evidence in the corpus).

**Expected performance impact (ESTIMATED):** the 3.5-4K band from 2,538-2,787 toward 3,300-4,500 t/s
(the padding-waste removal alone recovers ~15%: 2,787 × 1.15 ≈ 3,200; the cold-compile removal depends
on the warm protocol). This brings the prefill objective to MET across the whole 0.5-8K range.

**Conservative estimate:** +15% in the dip band (padding only).
**Optimistic estimate:** +50-60% (padding + cold-compile both removed).

**Implementation complexity:** LOW (config + warm-client list; 1 boot; ~10 min of benching).

**Risks:** negligible; the extra bucket shapes add boot warm time (minutes) and graph HBM (small). The
only failure mode: if the dip is NOT padding/cold-compile (e.g., a KDA state-chunk alignment effect at
the 2048 boundary — the KDA state is per-128-token chunks, a divisor of all the proposed bucket sizes,
so no), the A/B will show no movement and the dip's cause goes to BG-6.

**Dependencies:** none (independent of everything; can share any boot family).

**Previous experiments checked:** E015-A (lin bucketing adopted — this is a refinement of the same
mechanism at the 2048-4096 edge, not a re-test); C015 (the KDA decay-dot kernel — prefill TTFT, no
overlap); no prior experiment targeted the 3.5-4K dip specifically (checked EXPERIMENTS.md — the dip is
in the scorecard, not in any experiment).
**Why different:** no prior attempt adjusted the bucket step in the 2-4K range or pre-warmed those
shapes; the dip has been recorded but never attacked.

**Implementation starting point:**
1. Read the current lin-bucket config (the boot scripts in `kb_staging/` — `boot_*.sh` — carry the
   `VLLM_PROMPT_*`/capture args; the adopted values are in the diagnosis header).
2. Arm A = current; Arm B = 512-step query buckets 2048-4096 + the 2560/3072/3584 shapes added to the
   warm client. 1 boot, A/B, the prefill ladder (0.5K/2K/3.5K/4K/6K/8K, warm, gen=1, 3 reps/depth —
   the scorecard protocol).
3. Record EXP-RA014; update the scorecard row.

**Validation criteria:** the 3.5-4K band ≥ 3,000 t/s (the objective band's floor) with no regression at
the other depths (within run-to-run noise); the warm-up log shows the intermediate shapes captured
pre-bench (the cold-compile signature removed).

**Confidence level:** MEDIUM-HIGH (the mechanism is inferred from the dip's location — ESTIMATED gain;
the A/B is cheap enough that the evidence quality is not a concern).

---

### RA-015 — DSA indexer port (long-context fidelity + KV-read cap) [priority 5]

**ID:** RA-015 (new)
**Title:** Port the sparse-DSA indexer (index_topk 2048) from the GPU implementation

**Target bottleneck:** none of B1-B5 directly — the 1M-context FIDELITY gap and the long-context KV-read
growth. The diagnosis's scorecard shows 1M context MET (1,062,400 tokens, needle at 860,739), so this is
a robustness/fidelity item, not a throughput item.

**Current limitation (verified in the corpus):** the GPU implementation of GLM-5.3-Flash (upstream vLLM
PR #53906, tier D) runs DSA as SPARSE MLA — "the indexer selecting key pools (FlashInfer, aiter)",
`index_topk 2048`. The Gaudi port (gaudi-cookbook, our fork's origin, tier D) runs DENSE MLA — "dense
MLA, indexer skipped: the same function up to 2,048 prompt tokens, an approximation beyond (needle gates
to the served window)". Inside the 2,048-token window the two are identical (top-2048 of ≤2,048 keys =
all keys); beyond it, our port reads ALL context keys while the model was TRAINED with the 2048-key
indexer selection. The needle test (100% at 860,739) proves recall survives the approximation; it does
NOT bound the quality delta on generation-heavy long-context workloads (the 0.5 pp gate is measured at
short context).

**Technical hypothesis:** implement the indexer on Gaudi: a small attention-like op per DSA layer (11
layers): the query against the index keys (the index projection is low-dim — the 64-128 dim index
vectors, a fraction of the 512-dim latent), a top-2048 selection (a top-k on 1M scores — an MME/TPC
reduction), and a gather of the selected latent vectors into the existing MLA path. Effects: (a) the
long-context behavior converges to the model's trained design (fidelity at 500K-1M); (b) the per-step
KV read at long context caps at ~2048 keys/layer instead of growing with context (at 1M: 1M×576×2B/8
ranks ≈ 1.7 GB/rank today → 2048×576×2B×(1+index) ≈ a few MB — the 0.7 ms/step at 1M today becomes
~0.05 ms, and the 1M-context decode margin widens for the 4-HPU config where the KV headroom is tight,
BG-7).

**Proposed architectural change:** the indexer op (new, MME or TPC: a [1, d_idx] × [d_idx, ctx] score
GEMV + top-2048 + gather — the top-k on 1M is the novel piece; a two-stage top-k (block argmax → merge)
is the standard pattern) + the DSA attention metadata change (the block tables for the 2048 selected
keys) + the KV layout (the index keys must be kept in the paged cache — they already are, as part of the
MLA latent page; only the selection logic is new).

**Why it should work:** the GPU stack runs exactly this (FlashInfer sparse-MLA with the indexer — a
mature kernel family); the index-key selection is a standard top-k problem with no Gaudi-specific
obstruction (MME reductions handle 1M-element top-k trivially); the MLA path downstream is unchanged.

**Gaudi 2 feasibility:** the MME top-k/reduction ops are standard (tier B ops); the gather into the paged
latent is the same pattern the KV machinery already does; the novel work is the two-stage top-k kernel +
the metadata plumbing. MEDIUM-HIGH complexity; the 1M-context accuracy slice (BG-6) is the gate for
whether the fidelity gain is real.

**Supporting research:** upstream vLLM #53906 (the GPU implementation — tier D, the algorithm reference);
gaudi-cookbook GLM port README (the dense-approximation disclosure — tier D); the 1M needle result
(project data — recall is intact under the approximation).

**Expected performance impact:** NOT a decode-throughput item at the 32K window (the two paths are
identical there — zero step-time change in the operating window). At 500K-1M: the KV read caps (the
0.7 ms/step at 1M on 8 HPU becomes negligible; on 4 HPU it halves the long-context decode cost); the
fidelity delta is the real product (unquantified until BG-6 measures the dense-vs-sparse quality gap).

**Conservative estimate:** the 4-HPU 1M-context decode cost halves; the fidelity gap (if any) closes.
**Optimistic estimate:** as above + the 4-HPU 1M capacity margin improves (the index keys are the same
latent pages — no extra KV, only the selection logic).

**Implementation complexity:** MEDIUM-HIGH (a new kernel family + metadata plumbing + the 1M-context
accuracy slice).

**Risks:** the top-k determinism (tie-breaking at the 2048 boundary — pin a deterministic order for
parity gates); the gather's page-crossing pattern (the 1Cat DFlash2 notes describe the same class of
per-token page-extent handling — a known-hard-but-solved pattern on this stack); the fidelity question is
OPEN (BG-6 — if the dense approximation is already within the 0.5 pp budget at 1M, this proposal's value
is only the 4-HPU margin and drops to priority 6).

**Dependencies:** BG-6 (the 1M-context quality slice) to justify the work; the KV metadata plumbing is
shared with nothing else (independent).

**Previous experiments checked:** none in the project (the indexer has never been attempted on Gaudi;
the port deliberately skipped it at port time).
**Why different:** not a re-test — the feature does not exist in the Gaudi port; it is a port of the
GPU implementation's attention algorithm.

**Implementation starting point:**
1. BG-6 first (the 1M-context accuracy slice: the suite's long-context items at 500K/1M under the current
   dense path — if within the 0.5 pp budget, re-scope RA-015 to the 4-HPU margin only).
2. Read the upstream indexer implementation (vllm#53906 `vllm/models/glm5next` — the index projection
   shapes, the top-k, the gather) — the algorithm reference.
3. Prototype the two-stage top-2048 on MME with a 1M-element input (a standalone micro-benchmark first:
   the per-layer budget is ~20-50 µs at 1M context).
4. Wire into the DSA layer behind an env gate; the 1M-context A/B (needle + the quality slice) before
   adoption.

**Validation criteria:** (a) bit-identity at ≤2048 context (the two paths are defined-identical there —
any divergence = a bug); (b) at 1M: the needle 10/10 + the quality slice ≥ the dense path (the fidelity
direction) or the measured gap quantified; (c) the per-step KV read at 1M in the trace caps at the
2048-key figure; (d) the 32K-window decode is unchanged (the operating window is unaffected).

**Confidence level:** MEDIUM on the mechanism (a standard algorithm, but the 1M top-k + page gather on
this exact stack is unproven); LOW-MEDIUM on the value (the fidelity gap is unmeasured — BG-6 decides it).

---

### RA-016 — DSpine drafter: the strongest published block-drafter, conditional port plan (added 2026-11-09)

**ID:** RA-016 (new, user-requested investigation) · **Status:** PROPOSED — CONDITIONAL (no
GLM-5.3-Flash DSpine draft checkpoint exists yet)
**Detail file:** `RA-016-dspine-drafter.md` (full spec, sources, port plan)

**What it is:** DSpine ("Draft in Parallel, Condition Through Depth: Adjacent Causal Injection",
Tencent, arXiv 2609.36173, Apache 2.0) — a parallel block drafter (b16) where gated adjacent
injection at every draft layer carries predecessor predicted-features into successors, so causal
conditioning unfolds over depth while all positions draft in parallel. Measured: +3.7-6.9% acceptance
over DFlash2 at equal training budget (Qwen3-8B), +8-27% acceptance and +15-20% end-to-end throughput
vs the official DFlash2-b8 checkpoint (Qwen3.8-27B); draft checkpoints are 1-2B, 5-layer, Apache-2.0
licensed (`tencent/dspine` HF collection, actively maintained — 27B draft released 2026-11-08).
**The user's question answered: yes, DSpine beats DFlash2 on acceptance** — by a few percent
(same-budget) to ~20% (vs the b8 checkpoint), not by a multiplier.

**Why it is conditional:** the drafter is a TRAINED model; no GLM-5.3-Flash DSpine checkpoint exists
(checked the collection 2026-11-09). Routes: (a) track the collection (near-zero cost; plausible next
target given the model's popularity), (b) train via SpecForge (PR #1038, public training code — a
project decision, weeks + GPUs), (c) proceed with RA-009 (DFlash2-GLM checkpoint exists today, but
CC BY-NC-ND 4.0 — a production-license caveat).

**The Gaudi 2 path (corpus-verified substrate):** the HPU integration is the same class as the 1Cat
DFlash2 HPU path (`hpu_dflash2.py`'s `propose_query_block` — all drafts in one pass over a fixed query
block, device-side selection; tier D) with `num_speculative_tokens=15`, and DSpine's draft is
architecturally SIMPLER to port (plain dense 5-layer transformer; the injection is inside its forward,
no dynamic conv / non-causal quirks). The KDA candidate-slot machinery for k=15 is already generic in
the tree (the abotsis `_kda_spec_decode`: candidate slot j = state after token j, resume from
num_accepted — the origin of our MTP port; the runner layout `[groups, bs, 1+num_spec+1]` is
generic). The port work is SHARED with RA-009: a GLM block-drafter integration serves both
checkpoints; DSpine then becomes a checkpoint swap + the k=7→k=15 slot extension.

**Expected impact (ESTIMATED):** ~3.0-3.6 tokens/step bs=1 (vs ~2.7-3.3 for the DFlash2 port) →
~10-25% over RA-009 once both are ported; ~75-95 t/s bs=1 on the current stack (vs 42.4 MTP k=1);
~480-600 t/s @16 (the +4-7% acceptance is nearly pure at high batch). Cost: 16 candidate KDA states
per request per layer (~2× the DFlash2 8-slot reserve — the 1Cat max_num_seqs≤16 bound must be
re-derived; possible tightening). License advantage: Apache 2.0 drafts vs the NC-ND DFlash2 GLM
checkpoint — potentially the right drafter for the PRODUCTION deployment.

**Priority:** 7 (drafter-class slot; after RA-011/Q21/RA-012/RA-004). **Action now:** track the
`tencent/dspine` collection; do RA-009 first (shared machinery); activate the RA-016 port plan
(checkpoint audit → drafter class → proposer → k=15 slots → validation → A/B) on checkpoint release
or a training decision.

---

### RA-017 — ECFP8 (ECF8): lossless exponent-concentrated FP8 weights — closes BG-7 (the 4-HPU 1M fit) + cuts the weight floor (added 2026-11-09, user-requested)

**ID:** RA-017 (new, user-requested: "a long dive into getting a version of GLM 5.3 flash that is
quantized using the new and novel ECFP8. Estimate the impact of performance and how it could be
implemented.") · **Status:** PROPOSED — Phase 0 (the 1-day CPU-only measurement) can start immediately;
the kernel work stacks on the Q21/RA-012A fused-dequant-GEMM skeleton
**Detail file:** `RA-017-ecf8-lossless-weights.md` (full spec, format decode, risks, validation)

**What it is:** ECF8 (arXiv 2510.02676, ICLR 2026, Rice/Lambda; code github.com/zeyuyang8/ecf8) —
**lossless** compression of native-FP8 (E4M3) weights via Huffman coding of the 4-bit exponent fields
(α-stable SGD distributions → ~2-3 bits/exponent; theoretical limit ≈ FP4.67) + a packed 4-bit
sign/mantissa stream. Bit-exact by construction (the bytes into the GEMM are identical to today's).
Measured on released FP8 LLMs: 9.8-14.8% weight reduction (DeepSeek-R1-0528 671B MoE 14.8%,
Qwen3-235B MoE 14.4%, Llama-70B 13.4%); planning number for our checkpoint: **~13-15% on the FP8
mass (≈12% of the full 328.3 GB), pending Phase 0**.

**The decisive correction:** the paper's throughput gains (11.3-177.1%) are MEMORY-CAPACITY effects
(smaller weights → larger batches under a fixed memory budget), NOT per-token decode gains. The paper's
own inference mode (JIT layer-wise decompression into a scratch buffer) would multiply per-step HBM
weight traffic ~2.86× on our HBM-resident system (0.855× read + 1.0× write + 1.0× read vs 1.0×) —
**rejected here (RJ-20)**: bs=1 floor 15.5 ms → ~44 ms.

**The value for this project:**
1. **4-HPU final deployment (the binding value — closes BG-7):** weights stored COMPRESSED in HBM
   (328.3 → ~282 GB) + a **fused decompress-in-register GEMM** (the RA-012A/Q21 TPC skeleton + the
   Huffman decode in the load path) → the 1M-context KV (≈13 GB) + graphs (≈25 GB) fit with ~64 GB
   margin instead of the current ~8-18 GB. The cleanest 4-HPU fit solution that ALSO cuts per-card
   weight bandwidth 14.5% — superior to FP8-KV (the RJ-10 needle risk, no bandwidth help) and to a
   window reduction (product trade-off).
2. **8-HPU (secondary):** the fused kernel cuts the weight floor 14.5% (bs=1 all-read 15.5 → 13.3 ms;
   Q21 routed-only floors 0.86-1.26 → 0.74-1.08 ms; @16 routed 9.1 → 7.8 ms). Modest alone — the
   8-HPU system is not memory-constrained, so there is no capacity effect.
3. **Zero accuracy risk by construction** (lossless → the 0.5 pp gate is a non-event; Phase 1 verifies
   byte-identity on our checkpoint).

**The make-or-break question (a microbenchmark, not a paper argument):** the Huffman decode is a
sequential per-element ALU chain (~8-15 ops/element: 1-2 LUT lookups + 64-bit window shifts + nibble
assembly); Gaudi2 TPC GEMVs are issue-bound at small M (Strata: 237 GB/s = 10% of HBM peak). At bs=1
(M=2) the decode ALU rivals the FMA work → the bs=1 gain may be a fraction of the byte saving; at @16
(M=32-48) the FMA dominates → the full 14.5% materializes. **Go/no-go gate (Phase 2a):** standalone TPC
decode GB/s + the fused op vs the stock skeleton at M=1/2/8/16/32/48 (TPC vs MME — the MME variant has
more ALU headroom, the Q21 D4 tile-sweep discipline). Even a decode-limited kernel retains the 4-HPU
FIT value (the weights sit compressed in HBM either way).

**Gaudi 2 specifics:** (a) Gaudi2's native E4M3 read caps at 240 (exponent fields 0-7, corpus
minimax-m3 codec) — our checkpoint's exponent histogram likely lives in {0..7} → a smaller Huffman
space → our compression may be slightly BETTER than the paper's 16-value numbers (Phase 0 measures);
(b) the corpus lossless-re-encoding lesson (minimax-m3: "a lossless format change can be the fastest
kernel change") validates the class, but the "no new kernel" shortcut does NOT apply — the compressed
form is a bitstream, so the decode stage is mandatory; (c) the fused-dequant-GEMM skeleton (RA-012A/
Q21) is the substrate — ECF8 adds the decode stage + a ≤4 KB LUT in TPC local memory + the per-span
byte-offset table (the span-flush extension: byte-align the bitstream every 256-element TPC lane span
at encode time → no per-thread `gaps` metadata, ~0.35% overhead).

**Expected impact (ESTIMATED):** see the detail file's table. Headlines: 4-HPU 1M fit margin ~8-18 GB →
~64 GB (BG-7 closed); 8-HPU @16 routed floor 9.1 → 7.8 ms; conservative case (decode issue-bound at
small M) = the 4-HPU fit value only; worst case (Phase 0 compression <8% on our block-scaled
checkpoint) = close after Phase 0 (1 day, no HPU time, no build cost).

**Implementation complexity:** Phase 0-1 LOW (offline encoder + bit-exact verification, CPU-only);
Phase 2 MEDIUM-HIGH (the kernel — but ~70% of the skeleton work is the Q21/RA-012A op); Phase 3 MEDIUM
(the 4-HPU loader reads the ECFP8 archive — the stock checkpoint stays as-is, rollback is trivial).

**Dependencies:** the Q21/RA-012A fused-dequant-GEMM skeleton (Phase 2 only — Phases 0-1 have none,
start NOW); updates BG-7's headroom arithmetic with the Phase 0 number. Composes orthogonally with
Q21 (factors multiply: routed-only × 0.855), RA-011, RA-013, RA-004/009/016 (a future FP8 drafter
could take the same treatment).

**Validation criteria:** (1) Phase 1 byte-identity of every tensor (hard gate — any mismatch aborts);
(2) Phase 2a fused op beats the stock op's effective GB/s at the target M (go/no-go per M);
(3) 4-HPU: 1M context serves at TP4 with pool headroom ≥ prediction −20%, needle 10/10, quality slice
within budget (expected trivially), decode t/s within 10% of the stock-loader arm at M=1; (4) 8-HPU
(secondary): bs=1 weight term −5%+ in the per-layer trace with greedy bit-identity (the strong
lossless A/B gate).

**Novelty:** not attempted in the project (no ECF8/DFloat entry in TASKS/EXPERIMENTS/DECISIONS;
GaudiGrimoire search 2026-11-09: no ECF8 content — nearest corpus neighbors: the minimax-m3/kimi-k2-7
lossless MXFP4→FP8 re-encodings (format re-encoding, no entropy coding) and the DFloat11 lineage in the
paper's related work). Different from RJ-16 (MXFP4 LOSSY — ECF8 is lossless, no accuracy study),
Q21 (orthogonal, multiplies), RA-012A (the skeleton ECF8 extends), RJ-15 (still rejected at 8 HPU —
ECF8-compressed HBM-resident at 4 HPU is the superior tier anyway). The rare proposal in this program
that cannot regress the 0.5 pp gate.

**Why it deserves investment:** (a) the user asked; (b) it is the cleanest answer to BG-7 — the FINAL
deployment's binding constraint — with a composing side benefit (−14.5% weight bandwidth); (c) the
risk is front-loaded and cheap: Phase 0 = 1 day of CPU with a hard gate, Phase 2a = a microbenchmark
with a go/no-go per M — expensive work starts only if the measurement says the physics work; (d) zero
accuracy risk by construction.

---

### RA-018 — Making MTP=4 work on Gaudi 2: the numerics-invariance package, the chain graph, and the acceptance ceiling (added 2026-11-09, user-requested)

**ID:** RA-018 (new) · **Status:** PROPOSED — the T1 sub-package (numerics invariance) is the
highest-value TG fix after B1; it unlocks the banked +20% draft graph, the +29% async replay follow-up,
the +19% MHC prefill, and MTP=4 itself · **Complexity:** T1 Low-Medium (1-2 boots) / T2 Medium /
T3 Medium (kernel) / T4 High (drafter class) / T5 calc+1 boot / T6 Medium (runner) / T7 policy
**Detail:** `RA-018-mtp4-theories.md` (the full audit + wall analysis + theories + evidence index).

**Executive summary:** the implementation audit (RA-018 §0, 2026-11-09) found that three structurally
unrelated optimizations — graph-captured draft (+20% banked), async replay (+29% banked), mHC row cap
(+19% at the 2944 bucket) — were each rejected by the SAME single accuracy-suite item, R003
(bat-and-ball, a trap question answering "10" instead of "5"), deterministically. The GaudiGrimoire
corpus (abotsis `LAYER0_DELTA_FINDINGS` §10-12, the third GLM port's patch 51) proves the mechanism at
boundary level: HCCL all-reduce is not row-invariant (chunk-order rank summation → per-row ULP at
decode), the KDA state carries it through all 45 layers, and MoE routing flips at near-ties. The
corpus fix — `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (70-line portable patch, compile-safe, default
inert) + pinned decode buckets — achieves 12/12 serial-vs-concurrent parity with no measurable decode
cost (14.7 vs 13.6 tok/s) in the reference fork. **Our tree lacks both.** RA-018's theories: T1 port
the invariance package (the unlock); T2 capture the whole 4-step MTP draft chain as ONE graph per
decode bucket (kills the 4× full-device-sync seam that broke the Q19 early-draft pipelining — the
W1/W3 walls); T3 fuse the KDA verify conv-chain (a per-token PYTHON loop × 34 KDA layers ≈ 1,190 small
captured ops/step at k=4 — the port's `kda_layer.py` spec branch, verified in source); T4 treat the
acceptance ceiling as a separate problem — the MTP head's 3.33/5 measured acceptance is BELOW the
4.2 the DECISIONS roofline requires for 200 t/s bs=1, so "MTP=4" must be redefined as "≥4 tokens/step"
and the block-drafter route (1Cat DFlash2-on-Gaudi: one-pass 7-draft + 8-token block verify, HPU
top-16 selector, proven on Gaudi; DFlash2-GLM/DSpine checkpoints) is the 200-bs=1 path; T5 shows k=4
is HBM-feasible at ns16-32 (candidate-slot arithmetic: ~1.1 GB/rank ns16, ~2.2 GB ns32 — DERIVED, the
wall is long-context block-table carry, W8, not decode HBM); T6 completes the host side (the RA-011
follow-up: async-D2H output staging — the rejection cause was the host consuming in-flight graph
outputs, exactly the vendor-documented "along with async D2H" scope); T7 is the policy theory: the
0.5 pp gate (2 items of 40) auto-rejects on ONE knife-edge item and cannot distinguish a 1-ULP mode
flip from a regression — an MTP-aware, tie-aware gate (per-flip top-2-gap attribution, measured, not
asserted) is required to convert the documented residual into adoption.

**The ranked plan (RA-018 §3):** (1) T1.1 gather_sum port + T1.2 pinned decode buckets + the rowdep
probe (8 identical prompts → distinct-text count; expect ~7 distinct pre-fix) — one boot family, days;
(2) T1.3 captured-core numerics alignment (recipe diff of a captured draft core vs eager; fp32
accumulators in the draft cores as the cheap first attempt) + re-run the Q01v2 protocol — the +20%
draft graph is adopted or the residual is characterized; (3) T5 the k=4 HBM instrumented boot (ns16,
pool headroom + the 80-row capture log) — shared boot family; (4) T2 the chain graph (CPU contract
test first, the `check_draft_graph.py` pattern; then the Q11 protocol k=2 → k=4 + the early-draft A/B);
(5) T3 the conv-chain closed-form fusion (CPU-pinned by `check_kda_spec.py`); (6) T7 the gate policy
(owner decision, with the per-flip gap data from steps 2/4); (7) T6 the async-D2H output-path
redesign (after Q21 lands so the device term is small enough that overlap is the binding question at
bs=1); (8) T4 the block-drafter scoping in parallel (RA-009/016 line). **Dependency spine:** Q21 + T1
→ T2/T5 → T3 → T6 → the 200 bs=1 envelope; T4 in parallel for the acceptance ceiling; T7 gates every
adoption from step 2 onward.

**End-state physics (all ESTIMATED, RA-018 §2.6):** bs=1 k=4 (MTP head): Q21 + chain graph + async →
step ≈ max(host ~6-8, device ~7-12) ≈ 8-12 ms → 3.33/0.010 ≈ **165-220 t/s** — the 200 target is
inside the MTP-4 envelope at its top; with the T4 block drafter (acceptance ~4-5) → 200-280 t/s.
@16 k=4: the full 38.1 GB weight read is unchanged (all 36 local experts touched at 80 verify rows —
the C1 routing saving does not apply at high batch), the tokens/step rise 1.85 → 3.33 → net ≈
**300-450 t/s** (the abotsis reference measured 185.9 @16 on a base ~2.5-3× slower — consistent).
800@16 at k=4 still requires the Q21 launch/AR reduction + the ECF8 weight-floor cut (RA-017) on top
of the acceptance — it is NOT inside the stock-weight-read envelope.

**Validation criteria (RA-018 §4):** functional — acceptance ~3.3/5 ±10% vs the reference parity,
zero degeneration at 2K+, the block-boundary × 5-row tests, the S=5 churn/chaining parity (A022
extended), the MTP-001 gate list under k=4; performance — bs=1 ≥ 60 t/s (1.5× the adopted 42.4),
@16 ≥ 300 t/s, the per-layer trace shows the draft marginal at ~device-only (T2) and the conv-chain
small-op count at ~1/layer (T3); accuracy — the T7 gate (suite ≥ 99.5% or all flips attributed
near-tie; golden ≥ 10/12 with the residual attributed; MTP-4-vs-MTP-1 parity: no new flips beyond
the documented class); stability — the production-qualification soak, the stale-worker hygiene
(`pkill -9 -f VLLM::Worker` before every boot), the first-batch 80-row shape-capture cost logged.

**Novelty:** MTP=4 (A012) is NOT INVESTIGATED in the project — "blocked on the draft-execution lever
(A014) because eager per-step cost grows with depth"; Q11 (k=2/k=4 revival) was REFUTED under the
EAGER draft and is explicitly "revisit only under the in-graph draft" — T2 IS that in-graph draft,
extended from one step (the Q01 private-storage cores) to the whole 4-step chain. The T1 package has
never been attempted in the project (B022 ported the abotsis determinism doc's analysis but not the
patch; RESEARCH-001 addendum 2 flagged both knobs as "free wins" — untested). T3's conv-chain fusion
is new (the recurrence is already single-call; the conv chain is the per-token Python loop, verified
in `kda_layer.py`). T4 is the reframing (RA-009/016 gain a concrete Gaudi2 mechanism: the 1Cat
`propose_query_block` one-pass draft). T7 is new as policy (the implementation agent recorded the
open question after the third R003 flip; RA-018 makes it enforceable with the measurement protocol).

### Round 2 (2026-11-10): the post-ABCONFIG-WIN proposals — full detail in `RA-019-round2-theories.md`

Context for all three: the implementation agent's round-2 work (D1-3, AIP-util, PFG+MHC OOM,
ABOTSIS-REF, AB3-HYPA, MEMLOG, **ABCONFIG-WIN adopted**, CRASH-INCIDENT) moved the state to:
**the prefill objective is MET (5,546-7,723 t/s across 0.5-8K)** at the cost of **decode bs=1 37.3
(−12% — the RA-001 tensor-cache win forfeited)** and **the suite 97.5% (R003 — the 4TH flip source:
the tensor-cache state itself)**. The gather_sum port (RA-018/T1.1) is applied to the venv, env-gated
off; the gsum probe returned an instrumentation artifact (0.0%, wall_s 0.0 — probed a non-serving
server); the real A/B is pending.

### RA-019 — The full mHC-prep TPC kernel (the structural fix for the workspace OOM class)

**ID:** RA-019 (new, 2026-11-10) · **Complexity:** MEDIUM (weeks — the kernel work; the pattern, build
system, op registration, and load-order recipe all exist)
**Target:** the 184-MiB mHC-prep FP32 class (165.7 GB of the 667.7 GB capture volume — the confirmed
workspace hog behind the CAP-8192/PFG+MHC/AB3 OOMs) + the adopted profile's decode regression.
**Hypothesis:** fuse the ENTIRE `mhc_pre` chain (FP32 prep + 20-iteration Sinkhorn + epilogue — the
existing TPC kernel covers only the 8-MiB loop class; `mhc.py`'s docstring: the wrapper "repeats the
stock ops in the stock order and swaps only the loop") into one TPC/MME program, the DSV4.1
`control_mme` family pattern (G19). **Expected (ESTIMATED):** capture workspace 101 GB peak → ~20-30 GB
(the agent's figure; my refinement: ~15-25 GB from the 184-MiB class alone, pending the memlog
attribution of the 128-MiB padded class); the >2,048-row prefill graphs capturable with the tensor
cache ON + full KV → **decode 42.4 AND the prefill fix in one profile** (the profile split disappears);
the 2,944/4,096-bucket prefill extends beyond MHC-ROWS +19% (the prep ops leave the graph — D1-3's
~1,800-op re-tracing is a fraction of the nodes); the 4-HPU HBM headroom; the OOM-crash class (the
CRASH-INCIDENT trigger) shrinks. **Risks:** the prep's FP32 arithmetic must be preserved exactly
(bit-identity or the ULP class expands — the DSV4.1 control ops carry explicit f32 variants for this);
the >1024-row path re-validation (MHC-ROWS proved the loop kernel at 8192 rows). **Validation:** the
memlog re-capture at 2,944 rows (the 184-MiB class gone, peak < 40 GB); the capture with cache ON +
full KV (OOM gone); the prefill ladder (≥ ABCONFIG-WIN); the decode bench (42.4 recovered); the suite
A/B; the 4-HPU boot.

### RA-020 — The per-phase tensor cache (the quick decode recovery; days)

**ID:** RA-020 (new, 2026-11-10) · **Complexity:** LOW (the 3-line change to the staged `GLM_TENSOR_CACHE_OFF`
patch)
**Target:** the −12% decode regression (37.3 → ~42.4) without losing the ABCONFIG-WIN prefill fix.
**Mechanism (G20, tier B):** the vendor `wrapped_hpugraph_forward` reads `PT_HPUGRAPH_DISABLE_TENSOR_CACHE`
**on every call** (not at wrap time); the runner knows the phase at every call (the l.3737
`use_graphs`/`phase` site). Per-phase: decode calls env "0" (cache ON — the decode shapes retain their
tensors, the RA-001 win); prefill calls env "1" (cache OFF — the prefill tensors free after replay, the
ABCONFIG-WIN fix; the prefill graphs are still captured/replayed — the cache-OFF dry-run mode
ABCONFIG-WIN already proved at +209-277%). **Expected (ESTIMATED):** bs=1 ~42.4 (+12%), prefill
unchanged, @16 parity-to-better. **Falsifiable accuracy prediction:** the decode retention state is
bit-identical to the pre-ABCONFIG-WIN adopted profile (100% suite) → **the suite should return to 100%
(R003 unflips)**; if not, the 4th flip source is deeper than the retention state (a T7 data point
either way). **Risk:** the chunked-prefill mixed step replays a decode entry with env "1" for that one
call (transient re-cache the next step — measure in the A/B). **Sequencing:** BEFORE RA-019 (recovers
the win now; RA-019 subsumes it structurally — the toggle stays as the 4-HPU HBM-budget knob).

### RA-021 — The gather_sum port hardening + A/B protocol (fix BEFORE the pending A/B)

**ID:** RA-021 (new, 2026-11-10) · **Complexity:** LOW (hours — a venv patch)
**Target:** the three deviations of the applied gather_sum port from the corpus patch (G14):
(1) **no size cap** — the corpus falls back to hccl for buffers >16 MiB because captured-graph sites
retain working buffers; as written, the port would all_gather the prefill ARs (512 MB × 162 at 8,192
rows, retained) → **guaranteed OOM at the >2,048-row captures** (the exact failure class just
recorded); (2) **the sum is bf16** — the corpus sums in fp32 (free at decode size); (3) **the
all_gather stale-value hazard** (G5/G6, DSV4.1 docs, tier D): "vllm-gaudi's `all_gather` calls
`mark_step`; inside a captured HPU graph that gave stale values for batches above 1. An all-reduce
replaced it" — our v0.26.0 is the DSV4.1-era all_gather, and the port makes every AR an all_gather
inside captured decode graphs at every batch. **Protocol:** the cap + fp32 sum first (the venv patch);
then, before enabling `GLM_ALLREDUCE_GATHER_SUM=1`: the rowdep probe (8 identical prompts — the
stale-value detector) + a bs=8 parity check; then the A/B on the CURRENT adopted profile: the suite
(prediction: on the cache-off profile gather_sum may NOT move R003 — the 4th flip source is the
cache-state class, orthogonal to AR row order; the full 100% may need RA-020 + RA-021 together — the
result either way is the empirical test of T1 on our tree), decode parity bs=1/@16 (corpus: 14.7 vs
13.6 = noise), the prefill ladder (the >16 MiB ARs stay hccl — the prefill parity degrades to today's
behavior, a wash).

---

## SECTION 4: Transformative Architectural Proposals

**The from-scratch question (asked per the directive):** if I designed this inference engine from scratch
for Gaudi 2, what would I change?

A from-scratch engine for GLM-5.3-Flash on Gaudi 2 would have: (1) ONE launch per step per rank — the 45
layers as a small number of fused TPC/MME programs with the HCCL collectives inside them, the MoE as a
routing-aware fused dequant-GEMV (routed experts only, shared expert absorbed), the draft as a second
program (a block-diffusion drafter or the graphed MTP chain), the sampler on-device; (2) an async host
that enqueues and moves on, syncing only for the token IDs; (3) expert placement by routing profile;
(4) no intermediate BF16 materialization anywhere (in-register dequant in every GEMM).

**Every one of those four is achievable on the current vllm-gaudi stack without replacing it:**
(1) = Q21 (the fused MoE) + RA-012 (the fused linears + the Sinkhorn) + RA-004/RA-009 (the draft
programs) — the "one launch" ideal degrades gracefully to "~100-170 regions" and, with RA-011, the
region count stops being the binding cost; (2) = RA-011 exactly; (3) = RA-013; (4) = RA-012A + Q21.
The vLLM v1 substrate (the 1M-context KV machinery, the hybrid KDA/MLA cache, the MTP candidate-slot
state, the scheduler, the accuracy gates, the vendor's lazy-compile + HPUGraph investment) is too valuable
to rebuild — a strata/Maya-style custom C++ engine (RJ-17) would re-implement the KV/state machinery
against the vendor bridge from scratch for a benefit (fewer host launches) that RA-011 delivers by
configuration. **The transformative change is therefore at the execution layer UNDER the engine, not a
replacement of the engine** — and the specific execution-layer changes are RA-011/012/013 + the in-
progress Q21 + the queued RA-004/009. There is no identified opportunity that requires leaving
vllm-gaudi.

**The two genuinely open architectural questions** (not answerable from current evidence — they are the
vendor-side items):
1. **Can the AR `mark_step` boundaries be relaxed in lazy mode?** (The 163-region floor.) The compile-
   mode env var (`PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT`) does not apply to lazy mode (verified: the
   consumer is `dynamo/compile_backend/passes.py` only). A vendor question with the D1 numbers (BG-5) —
   if the lazy compiler can keep an AR inside the surrounding cached graph (the HCCL work is
   stream-ordered; the `mark_step` may be a compile-unit-size heuristic, not a semantic requirement),
   the floor drops from ~163 to ~50 and B2's host term halves again.
2. **Can the per-recipe C++ cost (~43 µs) be reduced by the vendor?** (The hashing + spec work per
   region.) `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK` removes part of it (user-controllable, RA-011);
   the rest is vendor-internal. A quantified request (322 regions × 43 µs × every step × 8 ranks) is
   worth filing with the D1 evidence.

**Execution-engine alternatives considered and rejected:** see Section 5 (RJ-14, RJ-17, RJ-18, RJ-19).

---

## SECTION 5: Rejected Hypotheses

| ID | Idea | Why rejected | What would reopen it |
|---|---|---|---|
| RJ-13 | `PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0` as a lazy-mode recipe-count lever (the Qwen-notes 2× result) | The env var is consumed ONLY in `habana_frameworks/torch/dynamo/compile_backend/passes.py` (verified in the deployed venv: `config.py:66` + the two pass consumers) — the torch.compile path, not lazy mode. Our stack is lazy mode; the flag is a no-op here. The Qwen result (2× on an 8-layer model, poisoned batched decode on the real model) came from a compile-mode stack. The transferable lessons are kept (large regions are not automatically faster — "38.8→30"; split=0 poisons batched decode; in-graph `index_put`/`index_copy_` state writes are mis-ordered — the last is a live warning for RA-004's KDA-candidate-slot graphing). | A lazy-mode equivalent of the flag appears in a vendor release; or the vendor confirms (BG-5) that the AR `mark_step` in lazy mode is a heuristic, not semantic |
| RJ-14 | A user-space C++ executor replacing the Python model runner (the diagnosis's B2 direction) | The per-recipe cost (14 ms bs=1 / 42 ms @16) is inside the VENDOR C++ (`ExecuteCachedGraph` per-region loop, tier B) — the py-spy frame at `graphs.py:75` is the native-call boundary, not Python loop cost. Rewriting the runner in C++ removes the ~ms of Python the runner does, not the 43 µs × 322 vendor loop. RA-011 (the vendor thread pool) offloads that loop with zero rewrite. | The vendor loop is proven NOT to offload (RA-011 fails on the thread-pool branch) AND a trace shows the user-space Python is still a significant residual |
| RJ-15 | Expert tiering to RAM/SSD (the project-maya model: hot experts in VRAM, warm in RAM, cold on NVMe) | Not applicable: all 328 GB of block-FP8 weights fit in 8×96 GB HBM (68 GB used at decode) AND in 4×96 GB (384 GB total vs ~360-370 GB with KV+graphs at 4 HPU — tight but fits, BG-7). HBM bandwidth (2.45 TB/s/card) is ~2-4× the consumer-PCIe-VRAM aggregate that makes tiering necessary on Maya's machines; the RAM/SSD tiers would be SLOWER than HBM at every access. What IS transferable from Maya is the routing-profile idea (RA-013) — kept, as placement not tiering. | A future config where weights + KV exceed HBM (e.g., a 2M-context 4-HPU deployment) |
| RJ-16 | MXFP4 (4-bit) expert weights as the base case for the Q21 kernel | The 0.5 pp accuracy gate is the binding constraint and the published checkpoint is block-FP8; the consumer 4-bit quants of this exact model (Maya's, EXL3-4.05bpw) keep 97.7-99.2% zero-shot match and 83-90% next-token agreement — usable for a consumer app, not demonstrably inside a 0.5 pp gate vs the FP8 reference. The MXFP4 variant HALVES the weight bytes (19.0 GB/card → 7.75 ms floor at full read; ~0.43 ms routed at bs=1) and is therefore a strong STRETCH variant AFTER Q21 lands in FP8 — but it must not gate or delay the FP8 kernel, and it needs its own accuracy study. | The FP8 Q21 kernel lands AND the 200 t/s bs=1 target is still short after RA-011+RA-012+RA-004 (in which case the 2× weight-byte headroom becomes worth the accuracy study) |
| RJ-17 | A from-scratch custom engine (strata/Maya-style C++ engine on the Gaudi bridge) | The vLLM v1 substrate carries the 1M-context hybrid KV machinery, the MTP candidate-slot KDA state, the scheduler, the accuracy gates, and the vendor's lazy-compile/HPUGraph investment. Rebuilding it against the bridge from scratch (as Maya did for CUDA/ROCm) costs quarters and re-opens every correctness gate, for a benefit — fewer host launches — that RA-011 delivers by configuration on the current stack. Section 4 shows the from-scratch design degrades to RA-011/012/013 + Q21 + RA-004/009 on the existing engine. | A vendor-stack ceiling is proven (the lazy-compile + HPUGraph path cannot reach the target step anatomy even with all levers) — at that point the question is worth reopening |
| RJ-18 | vLLM engine-level `--async-scheduling` | Known broken on this stack class: the Qwen3.8-Flash-Next notes record "first request hangs (`RPC call to sample_tokens timed out`)"; the Qwen3.8-27B recipe lists "Async scheduling + spec decode | Disabled in vllm-gaudi". It is a DIFFERENT mechanism from RA-011 (engine-loop scheduling vs graph-replay offload) — RA-011 does not use the engine flag and keeps the engine's step loop synchronous (only the replay is async). The hang/garbage failure signatures from RJ-18's failures are included in RA-011's verification protocol as watch items. | vllm-gaudi ships a fixed engine async-scheduling for v1 (check the release notes at any stack upgrade) |
| RJ-20 | ECF8 in its paper's JIT layer-wise decompression mode (compressed weights + per-step scratch-buffer decompression via forward hooks) | The JIT mode is an OFFLOAD pattern (weights compressed on host/low-memory device, decompressed layer-by-layer into a reused scratch buffer). On our HBM-resident system it costs 0.855× read (compressed) + 1.0× write (scratch) + 1.0× read (scratch) = **2.855× the baseline weight traffic** (bs=1 floor 15.5 ms → ~44 ms; @16 routed 9.1 → ~26 ms). The paper's throughput wins from this mode are memory-CAPACITY effects (larger batches under a fixed memory budget) — we are not memory-constrained at 8 HPU (416 GB free) and at 4 HPU the FIT is solved by the fused-decode variant (RA-017) without the traffic penalty. | A deployment where the weights do NOT fit in HBM even compressed (e.g., a 2M-context 4-HPU config, or a sub-HBM expert tier) — in which case the JIT mode's scratch buffer becomes the streaming mechanism of record |
| RJ-19 | Reducing the AR count by batching/fusing all-reduces across layers, or sequence parallelism (RS+AG) | The per-layer AR semantics (the residual stream must be reduced before the next layer reads it) cannot be merged across layers without a layer restructuring; SP on Gaudi is unverified in the corpus and changes activation memory significantly. The AR device cost is 3.4 ms of a 114 ms step and largely hides under RA-011's overlap; the AR COUNT's structural cost (the recipe floor) is attacked by the cheaper RA-012C (shared-expert absorption into the Q21 kernel: 3.5 → ~2.5-3 ARs/layer) and the vendor question (BG-5). Full AR elimination is not worth its risk before those. | The device-side trace (BG-1) shows the 3.4 ms AR time NOT hiding under the overlap (i.e., the HCCL work extends the device tail beyond the host), or the vendor answers BG-5 negatively and the recipe floor is confirmed binding |

---

## SECTION 6: Research Gaps

| ID | Question | Why it matters | How to resolve | Effort |
|---|---|---|---|---|
| BG-1 | The per-LAYER device-time split on the FP8 tree (KDA vs DSA vs MoE vs mHC vs the non-MoE linears) + the HBM bandwidth during the MoE weight read + the per-rank MoE kernel times | Decides RA-012A's device term (is the dequant→GEMM unfused today? the ~7.1 GB vs ~1.96 GB traffic question), sizes B5 on our tree (the slowest/mean rank ratio), confirms the 15.5 ms MoE floor as DEVICE time, and gives the before/after for RA-011's overlap measurement | The 8-rank torch profiler (CPU+HPU) with per-layer ranges (the diagnosis's own Section-10 item — this document adopts it as the first measurement of the next boot family) | 1 boot |
| BG-2 | The routing histogram: per-layer, per-expert route counts under serving traffic (the skew, the hot set, the distinct-expert distribution at bs=1/16/32) | RA-013's input (the re-sharding map); the Q21 byte model under real skew (C1's uniform idealization → the measured distribution); the bs=1 hot-set size (the expert-caching question in the diagnosis's B1) | The router histogram hook (RA-013 step 1) or the CPU-router pass over the bench prompt set; no HPU time for the CPU route | ~1 day |
| BG-3 | The ACTUAL host/device overlap ratio after RA-011 (the timeline: host post-work frames vs device execution of step N) | Converts RA-011's ESTIMATED range into MEASURED; identifies any residual serialization (e.g., the thread-pool enqueue cost, a hidden sync) | The torch profiler timeline from RA-011's validation arm (already in the protocol) | with RA-011 |
| BG-4 | **RESOLVED (2026-11-09, live process environ on g2):** `GLM5_NEXT_MHC_SINKHORN=kernel` is set — the Sinkhorn TPC kernel is ON in the deployed tree; the 20-iteration PyTorch loop is not the live path | RA-012B closed (no Part B work); the mHC node count from BG-1's trace is recorded as the baseline only (expect the kernel's ~1 node/layer, not the loop's ~3,400/step) | Done — process environ + `vllm_gaudi/__init__.py:7` (the kernel registration, "inert unless `GLM5_NEXT_MHC_SINKHORN=kernel`") read on g2 | closed |
| BG-5 | Vendor question: (a) is the AR `mark_step` in LAZY mode a semantic requirement or a compile-unit heuristic (can an AR live inside the surrounding cached graph)? (b) is the async replay (thread pool + tensor cache + in-graph collectives) a supported combination? (c) can the per-region replay cost (~43 µs) be reduced? | (a) sets the recipe floor (163 today) — the only way below it besides op fusion; (b) de-risks RA-011's adoption; (c) the B2 long-term lever | A vendor support request with the D1 numbers (14 ms/42 ms, 322 regions, 43 µs/region) + the RA-011 A/B results as they land | async |
| BG-6 | The 1M-context quality delta: dense-MLA (current port) vs sparse-DSA (the model's trained design) — the 0.5 pp gate at 500K/1M | RA-015's go/no-go: if the dense approximation is already within budget at 1M, RA-015 drops to the 4-HPU-margin value only | The suite's long-context slice at 500K/1M under the current path (the needle is already 100% at 860,739 — the generation-quality items are the missing piece) | 1 boot (shared with BG-7's config if 8-HPU) |
| BG-7 | The 4-HPU 1M-context KV fit: the KV pool headroom at TP4 (96 − 76.3 GB weights − graphs/workspace ≈ 8-11.7 GB) vs the 1M-token KV requirement (~13 GB at the measured 1.06M-token pool geometry on 8 HPU, scaled to 4 ranks) | The FINAL deployment is 4 HPU and the context objective is 1M — if the KV does not fit at 4 HPU, the options must be decided BEFORE the 4-HPU freeze. **Options ranked (2026-11-09, RA-017 added):** (1) **ECF8-compressed weights + fused decode (RA-017)** — weights 328.3 → ~282 GB in HBM, 1M KV fits with ~64 GB margin, AND the per-card weight bandwidth drops 14.5%; the only option that also improves bandwidth; zero accuracy risk (lossless). (2) FP8 KV (the RJ-10 needle risk) — KV 13 → 6.5 GB, no weight-side change, carries a quality risk. (3) The RA-015 indexer KV cap — the long-context KV READ drops, but the 1M-context KV POOL still needs to fit (it only helps the decode cost, not the fit). (4) A reduced window — a product-level trade-off. | RA-017 Phase 0 (the 1-day CPU compression measurement on our checkpoint — the number that sets option (1)'s margin) + the KV-pool math from the Q14 4-HPU config + the per-token KV size from the checkpoint geometry (no HPU time); confirm with a 1M-context boot at TP4 if the math is within ±20% | calc + 1 day CPU (RA-017 Phase 0) + optional 1 boot |

---

## SECTION 7: Ranked Recommendations

Ranked by (expected impact × feasibility × evidence quality), with the sequencing constraint that each
item is independently bootable:

**Status delta (2026-11-10 round-2 audit — `RA-019-round2-theories.md` §1; supersedes the 2026-11-09
delta below for the items it covers):**
- **ABCONFIG-WIN — ADOPTED as the prefill-optimized profile:** tensor cache OFF + prefill threshold
  8,192 → **the prefill objective MET (5,546-7,723 t/s across 0.5-8K, +209-277% vs eager)**; cost:
  decode bs=1 37.3 (−12% — the RA-001 win forfeited), @16 259.5 (parity), suite 97.5% (R003 — the
  4TH flip source: the tensor-cache state itself). The OOM root cause is identified (MEMLOG: the
  184-MiB mHC-prep FP32 class, 165.7 GB of the 667.7 GB capture volume; 101 GB peak watermark) —
  the fix paths are now RA-019 (structural) and RA-020 (per-phase retention), not further config.
- **RA-018/T1.1 — the gather_sum port is APPLIED to the venv** (env-gated off), with THREE deviations
  from the corpus patch found in audit (the missing 16 MiB size cap → prefill-OOM risk; the bf16 sum
  vs the corpus fp32; the DSV4.1-documented all_gather stale-value hazard at batch >1) — **RA-021
  (the hardening) must land before the A/B enables it.** The gsum probe (0.0%, wall_s 0.0) is an
  instrumentation artifact (probed a non-serving server); the real A/B is pending.
- **New proposals:** RA-019 (the full mHC-prep TPC kernel — the structural fix; the DSV4.1
  `control_mme` pattern, G19), RA-020 (the per-phase tensor cache — the vendor reads the env per
  call, G20 tier B; the 3-line change; the falsifiable 100%-suite prediction), RA-021 (the
  gather_sum hardening + A/B protocol).
- **New findings:** N4 (the eager prefill dip is partly RANK DESYNC, not just host dispatch — the
  AIP-util comparison: the eager 2,944-class runs 4-19% AIP wildly uneven across ranks, the graphed
  2,048-class 25-33% uniform; capture re-locks the ranks — a secondary rationale for T2's chain
  graph and a caution for every future eager hot path, including the 4-HPU TP4 config); N5 (the
  k=4 bs=1 expert-union weight floor is 3-5× the k=1 floor: 40-60 distinct global experts per layer
  for 9 correlated rows → 5-8/rank → 5.3-8.5 GB → 2.2-3.5 ms — the Q21 k=4 design target; the exact
  number is a no-HPU per-row route hook); G21 (the stock `VLLM_GRAPH_RESERVED_MEM`/`VLLM_GRAPH_PROMPT_
  RATIO` size the KV pool only — they do not cap graph capture; do not look for a config fix there).
- **Ops (CRASH-INCIDENT):** the OOM-crash spam → journal daemon bloat (tens of GB RSS on the 125 GB
  host) → the card-reset path failed (host-resident MMU page pool) → hard power recoveries. The
  journal rate-limit + the host-RAM headroom pre-flight are now standing items (item 7, do-first).

**Status delta (2026-11-09 audit — RA-018 §0.4; this re-orders the table below):**
- **RA-011 — EXECUTED + REJECTED (EXP-RA011):** +29% bs=1 (54.2 median — the mechanism and my bs=1
  estimate CONFIRMED), @16 PARITY (my +70-85% @16 was wrong — the @16 step is device+per-row bound),
  rejected on R003 deterministic flips — exactly my R3 risk (in-flight consumption of the graph's
  cached outputs). The follow-up is the async-D2H output-path redesign (RA-018/T6), not a retry of the
  flag.
- **RA-012 — SKELETON LANDED (AB-DEQ):** the TPC dequant op is production-qualified at parity (the
  dequant swap alone does not move decode, as the doc predicted); the fused MoE GEMV (Q21) is the
  remaining work; Part B (Sinkhorn) closed.
- **RA-013 — DISPROVEN as proposed (BG-2):** routing is near-uniform (top-10 6.5%, 80%-coverage
  168-182/288) — nothing to re-shard. The histogram's real value (validating the Q21 uniform byte
  model) materialized instead.
- **RA-014 — MECHANISM REFUTED, FIX PATH RE-SCOPED:** the dip is the graph-threshold cliff (>2048
  prefill buckets run EAGER by `_use_graphs` config), not padding/cold-compile; the fix needs an HBM
  budget (CAP-8192 recorded the OOM at 8192; the viable paths: capture 4096 + KV ~640, or the
  prefill-workspace optimization).
- **Q01v2 draft graphs — +20% BANKED, REJECTED** on the captured-region fusion drift (R003 5/5,
  mode-independent whole=split=1/12). **MHC-ROWS — +19% at the 2944 bucket, REJECTED** on R003 3/3.
  **Three structurally different optimizations now sit behind one knife-edge item** — RA-018/T1 (the
  invariance package) + T7 (the gate policy) are the unlock.
- **B7 — RESOLVED** (load-context artifact; the Q21 kernel track is unblocked and the AB-DEQ skeleton
  is deployed). **Q14v2 — reproduced** (TP4+EP4: bs=1 41.7 +25%, @16 180.8 +59%, 100% suite,
  2026-11-09 independent boot).

| Rank | Item | Type | Expected effect (ESTIMATED) | Effort | Sequencing |
|---|---|---|---|---|---|
| 1 | **RA-018/T1** (the numerics-invariance package: port `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (the 70-line corpus patch, env-gated default-off) + pin the decode buckets (`VLLM_DECODE_BS_BUCKET_*`, in-tree, untested) + captured-core numerics alignment (recipe diff / fp32 accumulators in the draft cores)) + the rowdep probe | NEW (2026-11-09 audit) | Kills the two fixable ULP sources that trip the 0.5 pp gate (HCCL AR row order + bucket-ladder shape numerics) — the row-position ULP that carries through the KDA state and flips near-ties (R003). Unlocks in one boot family: the **+20% draft graph (Q01v2, banked)**, the **+29% RA-011/T6 async follow-up (banked)**, the **+19% MHC prefill (banked)**, and the MTP=4 accuracy path (T2/T5). Corpus reference: 7 distinct texts → 1, 7/12 → 12/12 parity, no decode cost (14.7 vs 13.6) | 1-2 boots (days) — the patch is corpus-sourced and compile-safe; the alignment is a measurement + a small code change | **FIRST (post-audit)** — it is the cheapest item and the precondition for adopting three banked optimizations and for MTP=4; T7 (the gate policy) is decided in parallel with its results |
| 2 | **Q21/RA-002** (the routing-aware fused MoE GEMV — in progress, B7 unblocked; the AB-DEQ skeleton is production-qualified) | existing | bs=1: 15.5→0.86-1.26 ms weight floor (the 200 t/s physical prerequisite); @16: −4 to −6.4 ms (8 HPU) / −8 to −12.7 ms (4 HPU) — LARGER than the diagnosis credits (C1); BG-2 VALIDATED the uniform byte model (near-uniform routing) | weeks (the kernel) | Continue; add the shared-expert absorption to the design (RA-012C — the 3.5→2.5-3 ARs/layer); the k=4 expert-union measurement (RA-018 §2.1: count the 9-row route union from the 22M-route archive — no HPU time) feeds its k=4 sizing |
| 3 | **RA-012** (fused dequant-GEMM linears; Sinkhorn check RESOLVED — kernel ON in live tree) | NEW | −1 to −4 ms/step bs=1 (Part A only, BG-1 decides); −135 recipes | kernel work (weeks) | Start with BG-1 (the decision measurement); Part B is closed (verified 2026-11-09) |
| 3b | **RA-018/T2+T5** (the chain graph: capture the whole 4-step MTP draft chain as ONE graph per pinned decode bucket — 1 replay + 1 seam instead of 4×(replay + FULL sync); + the k=4 HBM instrumented boot) | NEW (2026-11-09 audit) | Kills the W1+W3 walls (eager draft cost scaling with k; the seam sync that broke the Q19 early-draft pipelining). The Q11 k=2 refutation was written "revisit only under the in-graph draft" — this IS that change. k=4 goes live: bs=1 ≥ 60 t/s (1.5× the adopted 42.4), @16 ≥ 300 t/s (ESTIMATED 300-450; the abotsis reference 185.9 @16 on a ~2.5-3× slower base); T5 arithmetic: k=4 fits at ns16-32 (~1.1-2.2 GB/rank candidate slots — DERIVED, one instrumented boot confirms) | CPU contract test (days) + 1 shared boot + the k=2→k=4 sweep (weeks) | AFTER T1 (the accuracy regime) and alongside the Q21 k=4 sizing; the pinned-bucket recipe from T1.2 bounds the graph count |
| 3c | **RA-011 → RA-018/T6** (async replay follow-up: the async-D2H output staging that fixes the RA-011 rejection — the host reads a pinned copy of the consumed outputs, never the in-flight tensor-cache slots) | NEW (2026-11-09 audit; RA-011 executed + rejected on exactly this hazard) | +29% bs=1 is BANKED (EXP-RA011 measured); the D2H redesign makes it adoptable; the candidate-slot state semantics make the resume safe. End-state with Q21+T2: step ≈ max(host ~6-8, device ~7-12) → the 200 bs=1 envelope (165-220 t/s at MTP-4 acceptance; 200-280 with the T4 drafter) | runner change (weeks; the output-read path in `hpu_rejection_sampler.py` + sampler + logprobs) | AFTER T1 (the accuracy regime) + Q21 (the device term must be small enough that the overlap is the binding question at bs=1); the vendor thread-pool branch is tier-B verified and +29% measured |
| 4 | **RA-004** (the graphed draft chain — T0 probe → T1/T2, companion doc) | existing | −7.5 to −10.6 ms HOST bs=1/@16 (the draft dispatch → one replay); k=2-4 tokens/step upside | days (T0) + weeks (T1/T2) | After RA-011 (RA-011 already hides most of the eager-draft host cost — T0's measurement is then the GRAPHING decision, not an emergency); the KDA-candidate-slot graphing must respect the in-graph state-write caution (RJ-13 lesson) |
| 4b | **RA-017** (ECFP8/ECF8 lossless weight compression — the 4-HPU fit + the weight floor) | NEW (2026-11-09, user-requested) | 4-HPU: 1M-context KV fit margin ~8-18 GB → ~64 GB (**closes BG-7**); 8-HPU: weight floor −14.5% (@16 routed 9.1 → 7.8 ms; bs=1 all-read 15.5 → 13.3 ms) — the small-M decode ALU cost may eat part of it (Phase 2a gate); zero accuracy risk (lossless) | Phase 0: 1 day CPU (start NOW — the measurement decides BG-7's strategy); Phase 1: days (bit-exact verification); Phase 2: weeks (kernel, after the Q21/RA-012A skeleton); Phase 3: the 4-HPU loader before the 4-HPU freeze | Phase 0-1 independent (no HPU); Phase 2 after the Q21/RA-012A fused-dequant-GEMM skeleton exists (it extends that op); Phase 3 before the 4-HPU freeze |
| 5 | **RA-013** (expert placement by routing histogram) | NEW | −1 to −5 ms at @16-32; the Q21 byte-model input (the skew) | ~1 day (histogram + re-shard A/B) | Independent; do the histogram early (BG-2 feeds the Q21 design regardless of the A/B outcome) |
| 6 | **RA-014** (2-4K prefill bucket fix) | NEW | the prefill objective MET across 0.5-8K (2,538-2,787 → ~3,200-4,500 in the dip band) | 1 boot, config-only | Any boot family; independent |
| 7 | **RA-009/RA-010** (DFlash2 drafter / ngram probe — companion doc) | existing | the drafter-class upgrade (the 1-pass block draft; community-standard for this model per project-maya's production use) | weeks (RA-009, High) / 1 boot (RA-010) | RA-010 (ngram, config-only) can slot into any boot; RA-009 after RA-004 lands (shares the draft-graph machinery) |
| 7b | **RA-016** (DSpine drafter — CONDITIONAL: no GLM checkpoint yet; Apache-2.0 drafts vs NC-ND DFlash2) | NEW (2026-11-09) | +10-25% over the DFlash2 port once ported (~3.0-3.6 tokens/step bs=1; ~480-600 t/s @16); the production-license advantage | tracking (near-zero) + weeks (the port, shared with RA-009) | Track `tencent/dspine`; do RA-009 first (shared machinery); activate on checkpoint release or a SpecForge training decision |
| 8 | **BG-6/BG-7 + RA-015** (1M-context fidelity + 4-HPU KV fit + the indexer) | NEW | the 1M objective's robustness at the 4-HPU final deployment | calc + 1-2 boots | Before the 4-HPU freeze; RA-015 only if BG-6 shows a fidelity gap |

**The combined end-state model (all ESTIMATED, for the 800@16 and 200 bs=1 targets):**

@16 (8 HPU), current 114 ms → with RA-011: ≈ max(host ~10, device ~53) ≈ 55-60 ms (~430-470 t/s) → +
Q21 (device −6): ≈ 50-55 ms (~470-510) → + RA-012 (device −3 to −4 if unfused, host loop −30-50%): ≈ max(~7, ~43)
≈ 44-48 ms (~520-560) → + the attention/compute −50% (the diagnosis's own target row; kernel work on the
DSA/KDA decode paths — not a new proposal here, it is the remaining device term): ≈ max(~7, ~31) ≈ 33-37
ms → **780-860 t/s: the 800@16 target is INSIDE the combined envelope**, reached by four credible items
(RA-011, Q21, RA-012, attention kernels), each independently validated. With RA-004's k=2-3 on top
(+15-40% tokens/step at +2-4 ms/step), the margin widens to ~900-1,100 t/s.

bs=1 (8 HPU), current 45.1 ms → RA-011: ≈ max(~4, ~24) ≈ 25-28 ms (~65-75 t/s) → + Q21 (device 24→~10):
≈ max(~4, ~10) ≈ 11-12 ms (~150-170) → + RA-012 (device −3 to −4 if unfused, host −1 to −3): ≈ 7-9 ms (~200-260)
→ + RA-004 graphed draft (host −2 to −4): ≈ 5-7 ms → **the 200 t/s bs=1 target is INSIDE the envelope
after RA-011 + Q21 + RA-012 + RA-004**, with the attention/sampler/AR trims (the diagnosis's remaining
rows) providing the margin. The diagnosis's conclusion (all four bottlenecks simultaneously) is confirmed;
this document's addition is that the host-side half is RA-011 + RA-012, and RA-011 alone is the largest
single step in the whole program.

4-HPU (the final deployment): the same levers apply with 2× the per-card weight bytes; the Q21 routed-
only saving is proportionally larger there (C1: −8 to −12.7 ms at @16), RA-011 is rank-local (unchanged
effect), and the 1M-context KV fit is BG-7's question. The companion document's RA-005 physics (the
4-HPU 800 target requires the deep-spec + full host stack) stands; this document does not change that
conclusion.

**What the implementation agent should do first (concrete order — 2026-11-10 round-2 audit,
supersedes the 2026-11-09 list below):**
1. **RA-021 (the gather_sum hardening):** add the 16 MiB size cap (env-tunable; the >cap ARs fall
   back to hccl — the prefill parity stays at today's behavior, a wash) + the fp32 sum to the
   applied venv port. Hours. Then the pre-enabling gates: the rowdep probe (8 identical prompts)
   + a bs=8 parity check (the DSV4.1 stale-value detector) — THEN run the pending A/B on the
   current adopted profile (suite + decode parity bs=1/@16 + the prefill ladder). The suite
   prediction is documented in RA-021 (on the cache-off profile gather_sum may not move R003 —
   the 4th flip source is orthogonal; the result is the empirical test of T1 either way).
2. **RA-020 (the per-phase tensor cache):** the 3-line change to the staged `GLM_TENSOR_CACHE_OFF`
   patch (decode calls env "0" / prefill calls env "1", set at the l.3737 phase site before the
   wrapped forward). Recovers the −12% decode (37.3 → ~42.4) with the prefill fix kept; the
   falsifiable prediction: the suite returns to 100% (the R003 unflip — the decode retention state
   is bit-identical to the pre-ABCONFIG-WIN profile). Days.
3. **Q21 continues** (the fused MoE GEMV; add the shared-expert absorption — RA-012C; the k=4
   expert-union sizing now has the N5 target — 2.2-3.5 ms weight floor — and the no-HPU
   measurement: a per-row top-8 route hook in the CPU router, 100-request decode run, 9-row union
   per layer).
4. **RA-019 (the full mHC-prep TPC kernel):** the structural fix (the OOM class, the profile split,
   the 4-HPU headroom). Pattern: the DSV4.1 `control_mme` family (G19 — the source is in the corpus
   and on this host at /opt/dsv41/1Cat-vLLM-Gaudi); infrastructure: the existing GLM kernel
   registration (`GC_KERNEL_PATH`, the op lib, the B7 load-order recipe). Weeks; start after step 2
   (step 2 banks the decode win; step 4 subsumes it structurally).
5. **RA-018/T1.2 + T2** (pin the decode buckets; then the chain graph — CPU contract test first,
   the Q11 protocol k=2 → k=4) — after the step-1 A/B result (the accuracy regime first); the N4
   desync re-lock is the secondary win to measure (AIP uniformity under the chain graph).
6. **RA-018/T7 (the gate policy):** now with the 4th flip source — add the BASELINE definition:
   the current adopted profile's suite profile = the reference ("regression" = new flips vs it),
   since the adopted profile itself is at 97.5%.
7. **Ops (CRASH-INCIDENT lessons, cheap, now):** the systemd journal rate-limit
   (RateLimitIntervalSec/RateLimitBurst or the vacuum policy) + the host-RAM headroom check in the
   pre-flight (the driver reset path needs the host-resident MMU page pool).
8. The remaining items continue: **BG-1** (the per-layer device trace), **RA-012 Part A** (after
   BG-1), **RA-011/T6** (the async-D2H redesign), **RA-009/RA-016/T4** (the drafter scoping),
   **RA-017 Phase 0** (1 day CPU), **RA-014** (now subsumed by RA-019/020 — the prefill is MET;
   keep only as the >8,192-token band question).

**What the implementation agent should do first (concrete order — 2026-11-09 audit, superseded by
the 2026-11-10 list above; retained for history):**
1. **RA-018/T1.1 + T1.2** (port the corpus `gather_sum` patch, env-gated default-off; pin the decode
   buckets; run the rowdep probe — 8 identical prompts → distinct-text count, expect ~7 distinct
   pre-fix, 1 post-fix). [T1.1 the port is DONE (applied 2026-11-10, env-gated off) — see RA-021 for
   the three deviations to fix first.] One boot family; days.
2. **RA-018/T1.3** (captured-core numerics alignment — recipe diff of a captured draft core vs eager;
   fp32 accumulators in the draft cores as the cheap first attempt — then re-run the Q01v2 protocol:
   golden + suite + the +20% re-confirm). The draft graph is adopted or the residual is characterized.
3. **RA-018/T5** (the k=4 HBM instrumented boot at ns16: pool headroom + the 80-row capture log +
   the candidate-pool allocation) — share step 2's boot family.
4. **RA-018/T7** (the gate policy — submit the owner decision WITH the per-flip top-2-gap data from
   steps 2-3; the I015 boundary-diagnostics port is READY and strengthens every parity claim).
5. **Q21** continues (the AB-DEQ skeleton is deployed and production-qualified; the fused GEMV is
   next; add the shared-expert absorption — RA-012C — to the design; the k=4 expert-union sizing is
   a no-HPU measurement over the 22M-route archive).
6. **BG-1** (the per-layer DEVICE-side trace — still pending; the decision measurement for RA-012A
   and the B5 sizing).
7. **RA-018/T2** (the chain graph — CPU contract test first, then the Q11 protocol k=2 → k=4 + the
   early-draft A/B) → **T3** (the KDA conv-chain closed-form fusion, CPU-pinned) → **T6** (the
   async-D2H output-path redesign).
8. **RA-009/RA-016/RA-018-T4** (the block-drafter scoping — the DFlash2-GLM/DSpine checkpoint
   decision + the 1Cat HPU machinery port assessment) — in parallel, not blocking 1-7.
9. **RA-014** — re-scoped (no longer config-only): the >2048 prefill buckets run eager by the graph
   threshold; the fix needs the capture-size/KV HBM budget (CAP-8192's viable paths: capture 4096 +
   KV ~640, or the prefill-workspace optimization).
10. **RA-017 Phase 0** (1 day CPU, no HPU — independent; the measurement decides BG-7's strategy).

**Pre-audit list (superseded; retained for history):**
1. **RA-011** (next boot family: arm 0/2 A/B + parity + soak + trace = BG-3). It is days, not weeks, and
   it re-baselines every number in the program.
2. **BG-1** (the per-layer trace — share RA-011's boot family; it is the decision measurement for RA-012
   and sizes B5 for RA-013).
3. **BG-2** (the routing histogram — no HPU time; start the CPU-router pass now; it feeds the Q21 kernel
   design while the kernel is being written). [BG-2 now DONE — the histogram ran, near-uniform, RA-013
   disproven, the Q21 byte model validated.]
4. **Q21** continues (add the shared-expert absorption — RA-012C — to the design doc; it is the AR-count
   lever).
5. **RA-014** slots into any boot (config-only).
6. (RA-012 Part B — CLOSED 2026-11-09: the live process runs `GLM5_NEXT_MHC_SINKHORN=kernel`; no
   action. RA-012 Part A starts only after BG-1 confirms the unfused-traffic hypothesis.)

---

## SECTION 8: GaudiGrimoire Findings

(Compact citations; the companion document's rev-5 log covers the earlier rounds. All corpus text is
quoted data; tiers as labeled.)

| # | Finding | Tier / source | Influence on this document |
|---|---|---|---|
| G1 | `SingleHPUGraph::replayGraph` (C++): `queue_in_thread_pool = async && PT_HPU_ENABLE_HPUGRAPH_THREAD && PT_HPU_QUEUE_SYNLAUNCHES` → the entire `ExecuteCachedGraph` (the per-region launch loop) is enqueued on the vendor `SingleTonExecThreadPool`; `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK` reuses the cached `RecipeArgumentSpec` (skips the per-replay hash work); in the thread-pool branch the output tensors are NOT nulled (the tensor-cache-friendly path) | B / `backend/habana_device/HPUGraph.cpp` (gaudi-pytorch-bridge) | RA-011's mechanism — the whole proposal is built on this source; the env var names are exact |
| G2 | `wrapped_hpugraph_forward` (wrapper): the tensor-cache-on replay path is `copy_to(cached.graph_inputs, inputs)` (stream-ordered staging) → `cached.graph.replay(asynchronous)` → returns the CACHED `graph_outputs`; the cache is keyed by `input_hash(inputs)`; `CachedParams` stores the `asynchronous` flag per entry | B / `habana_frameworks/torch/hpu/graphs.py` | RA-011's race analysis (stream ordering makes the staging safe) and the hashcheck-env justification (stable key per bucket) |
| G3 | Vendor docs: `wrap_in_hpu_graph(..., asynchronous=False, ...)` — "If True, replays for the HPU Graphs will be launched asynchronously… This flag can be used along with stream and async D2H to copy outputs for reducing host overheads" | B / docs.habana.ai (HPU Graph APIs for Inference) | RA-011 — the vendor documents the flag for exactly the host-overhead purpose claimed here |
| G4 | Qwen3.8-Flash-Next (compile-mode stack): "the decode step is launch-bound… ~100-190 small recipes (two all-reduce graph splits per layer × 48 layers)"; `PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0` + fused 6-layer regions = 2× on an 8-layer model but "breaks batched decode on the real model (poisoned server, `!!!!`)"; fused regions WITH default splits = SLOWER (38.8→30: "each partition carries far more tensors"); in-graph `index_put`/`index_copy_` state writes "silently mis-ordered/dropped"; per-request metadata must be passed as graph inputs; `--async-scheduling` hangs | D / gaudi-llm-notes `models/qwen3.8-flash-next/README.md` | The AR-split env var is compile-mode (RJ-13); the large-region caution (do not expect bigger regions to help — RA-011 keeps region sizes); the state-write caution is a live warning for RA-004's KDA-candidate-slot graphing; the hang signature is in RA-011's watch list |
| G5 | Beam-1.0-501B port: "Lazy collectives fragment decode graphs. Each all-reduce cuts the layer into separate cached graphs, so a model with cheap layers is bound by host launch time; fewer tensor inputs per MoE launch (EP) helped more than any device-side change"; "vllm-gaudi's `all_gather` calls `mark_step`; inside a captured HPU graph that gave stale values for batches above 1. An all-reduce replaced it" | D / gaudi-cookbook `docs/ports/beam-1-0.md` | The recipe FLOOR is structural (C4, the 163-region bound); the "fewer tensor inputs per MoE launch" precedent supports the Q21 routed-only + shared-expert-absorbed design (RA-012C); the all_gather-stale caution applies to any new in-graph collective pattern |
| G6 | DeepSeek-V4.1-Flash port: "Many-small-op decode is launch-bound. Count device nodes before optimising compute: a 4x4 Sinkhorn loop cost about 3,400 nodes per step, and a tiny TPC kernel removed them bit-exactly"; "HPU graph caches key on the input's view state… re-captured every warmed graph in serving, and memory grew" | D / gaudi-cookbook `docs/ports/deepseek-v4-1-flash.md` | RA-012B context: the mHC Sinkhorn is the same 4×4/20-iteration design in GLM-5.3-Flash — and the live tree already runs the TPC kernel (`GLM5_NEXT_MHC_SINKHORN=kernel`, verified 2026-11-09 in the process environ), so the node storm is NOT a live cost here (BG-4 closed); the view-state re-capture signature is in RA-011's soak watch list |
| G7 | `PT_HPU_ENABLE_LAZY_COLLECTIVES` (kzawora-intel, vendor): "enables usage of collective operations (mostly allgather and allreduce in vLLM's case) in HPUGraphs… required for using tensor parallelism with HPUGraphs" | D / vllm-fork PR #200 comment (corpus web) | Confirms the in-graph ARs in our stack are the supported path — RA-011 changes no collective semantics; also (companion doc rev 4) auto-enabled at `platform.py:432` |
| G8 | Upstream vLLM GLM-5.3-Flash (PR #53906): GPU-only subpackage (`vllm/models/glm5next/nvidia/`) — Triton chunked+recurrent KDA kernels (NVIDIA/AMD), DSA as SPARSE MLA "with the indexer selecting key pools (FlashInfer, aiter)" (`index_topk 2048`), CUDA/TileLang/Triton mHC kernels, `Glm5NextMTP` in the speculative-decoding registry; the Gaudi port (our fork's origin) runs dense MLA "indexer skipped: the same function up to 2,048 prompt tokens, an approximation beyond" | D / gaudi-cookbook port README + gaudi-llm-notes patch series | RA-015 (the indexer port — the algorithm reference is the GPU implementation); the 1M-context fidelity gap (BG-6); the MTP-on-GPU existence (context for RA-004/009 — the drafter head is a first-class vLLM citizen) |
| G9 | project-maya (mw00, 2.7K stars, MIT): a C++ GGUF engine running GLM-5.3-Flash (321B) on consumer GPUs with (a) three-tier expert placement (VRAM/RAM/NVMe) driven by a per-expert routing profile (`data/expert-profile-glm.bin`, `expert_counts.txt` — "VRAM slots per layer by routing share | 35.5 -> 32.9 ms/token"), (b) DFlash2 as the GLM drafter in production configs ("Maya-S / Maya-M-Derisked (DFlash2)… 148.6 tokens/s" on 2×RTX 5090 + RTX 4090, 32K context), (c) the MTP head measured at "draft == next greedy token 64-68% on teacher-forced doc/code text, 83-99% in free generation", (d) pipelined draft-verify (the head half drafts token N+2 while the tail verifies N+1; recurrent states saved/restored on a miss), (e) 4-bit quants keeping 97.7-99.2% zero-shot match | D / github.com/mw00/project-maya (README + `bench/results/2026-10-06-glm-spec/README.md`) | RA-013 (the routing-profile placement precedent + the measured effect); RA-009 corroboration (DFlash2 is the community's chosen drafter for THIS model, in a production engine); the MTP acceptance range cross-checks our 73-98% (companion doc); RJ-15 (tiering not applicable — all weights fit in HBM); RJ-16 (4-bit as stretch, not base case); the pipelined-draft concept converges on RA-004/RA-009 (no separate proposal needed) |
| G10 | gaudi-llm-notes GLM-5.3-Flash-Uncensored-FP8 on Gaudi2 (a SECOND independent Gaudi2 port, vllm-gaudi v0.29.0, 8×, TP8/EP8, no MTP): single-stream 17.7 t/s (57 ms/token, flat 4K→255K); **742 output t/s at 128 concurrent (4K context)**; context verified to 262,144 | D / gaudi-llm-notes `models/glm-5.3-flash/README.md` | The hardware envelope: 742 t/s aggregate on the same 8-card class WITHOUT MTP, WITHOUT the project's levers — at 128 rows. The per-row marginal at high row counts is sub-linear (their 128-row step ≈ 172 ms vs the project's 3.8 ms/stream fit at ≤32 rows) — the row-scaling curve bends, which matters for the ns32/64 saturation readings; the 800@16 target (with MTP + levers) is well inside the demonstrated envelope |
| G11 | Qwen3.8-27B recipe "Do not": "Concurrent batch 2 in one compiled decode graph | Dynamo crash"; "Async scheduling + spec decode | Disabled in vllm-gaudi"; MTP k=2/k=3 on that hybrid: "Collapse (~3 tok/s) or garbage" | D / gaudi-llm-notes `models/qwen3.8-27b-fp8/RECIPE.md` | RJ-18 (the engine-async distinction); context for the Qwen-hybrid MTP-depth failures (a DIFFERENT model — the GLM MTP depth question remains the project's own Q11/RA-004 measurement, not this evidence) |
| G12 | vllm-gaudi issue (corpus web): prefix-caching chunk boundaries vs GDN state-block boundaries (128-token scheduler chunks vs 896-token state blocks — "Ending chunks on state-block boundaries… restores the hits with outputs identical to a miss") | D / vllm-gaudi issues (corpus web) | Context for RA-014: the KDA state chunk (128 tokens) is a divisor of all proposed bucket sizes (no interaction); the general lesson — state-block-aligned boundaries matter on this stack — is recorded in case the RA-014 A/B shows no movement (BG-6 fallback) |
| G13 | Lazy mode (vendor ToO): "the Intel Gaudi PyTorch bridge internally accumulates ops into a graph, executing them lazily - only when a tensor value is needed… This mode will be deprecated in subsequent releases" (v1.21.0+ not default; enabled via `PT_HPU_LAZY_MODE=1`) | B / docs.habana.ai (Theory of Operations) | Context: the lazy-mode execution model is the substrate of the 322-region structure (C4) AND is on a deprecation track — a future stack upgrade (the companion doc's Q on v0.30.0) may change the region economics; the RA-011/012 levers are lazy-mode-specific and should be re-validated at any upgrade |
| G14 | **HCCL all-reduce is not row-invariant** (the AR row-order ULP, proven at boundary level): "HCCL all-reduces each chunk of the `[bs, hidden]` buffer in a chunk-dependent rank order; at decode time a chunk is one decode row, so a row's residual after o_proj/MoE down-proj depends on its position in the batch by about 1 bf16 ULP… GLM-5.3 carries the ULP in the KDA recurrent state of every layer after the first and MoE routing flips at near-ties." Fix: `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (all_gather the partials, sum in fixed rank order in fp32; 70-line portable patch, compile-safe, default `hccl` = inert; buffers >16 MiB fall back to hccl because captured-graph sites retain working buffers) → 8 identical prompts: 7 distinct texts → 1; serial-vs-concurrent parity 7/12 → 12/12 **with pinned decode buckets** (`VLLM_DECODE_BS_BUCKET_MIN/STEP/MAX`, already in our tree, untested); no measurable decode cost (14.7 vs 13.6 tok/s). The "everything else exonerated" list: tensor cache, mark_steps, lazy-acc-par, KDA state handling, MoE routing, fp8 activation scaling | D / abotsis-vllm-gaudi fork `LAYER0_DELTA_FINDINGS` §10-12 + gaudi-llm-notes `patches/optional/51-hpu-allreduce-order-invariant.patch` (the portable version) | **RA-018/T1 — the core evidence.** The root cause of the near-tie inversion class that trips our 0.5 pp gate on R003; the patch is env-gated and default-inert; our tree lacks both the fix and the pinned buckets (RESEARCH-001 addendum 2 flagged the buckets as an untested "free win"); the 16 MiB cap means co-batched prefill (>16 MiB) stays hccl — prefill first-token parity degrades to today's behavior (a wash) |
| G15 | **The one-call MTP spec recurrence + closed-form candidate states** (the reference for the KDA verify path): `_kda_spec_decode` / `chunk_kda_spec_states` — the per-position candidate states are a CLOSED-FORM expression (one causal-masked einsum over the stacked window instead of L sequential steps), the state at every candidate position is preserved via `einsum((S_j × S_0^T)(g_j·g_t) + K·K^T(exp(g_j−g_t))·v_new_t)`; the test target is explicit: "L=2 is MTP-1, L=5 is the MTP-4 target" | D / abotsis fork `glm5_next.py` + `hpu_kda_eager.py` | **RA-018/T3 — the reference.** Our port's recurrence already matches (single call, `hpu_recurrent_kda(..., return_all_states=True)` — verified in `kda_layer.py`); our CONV chain does not (a per-token Python loop of elementwise ops inside the captured graph — the dispatch-bound part). The closed-form trick is the fusion target (one bmm over the padded window instead of ~35 small ops/token/layer); the vendor's `hpu_fused_recurrent_gated_delta_rule` raises `NotImplementedError` for the spec path in v0.26.0/0.30.0 (G18) — the custom path is the only one |
| G16 | **The DFlash2-on-Gaudi block-drafter machinery** (the one-pass block draft on HPU): "Qwen DFlash2 proposes 7 draft tokens in ONE pass, HPU top-16 selector, target verifies an 8-token block" — a graphable `propose_query_block` over a fixed query block; rollback-safe conv+GDN state checkpoints ("each active request reserves eight rollback checkpoints per GDN layer", max_num_seqs ≤ 16, greedy only, Qwen3.8 target only today). Strata's `dflash.hpp` (Qwen3.8-Flash-Next, single Gaudi2 card): the draft mechanism reuses the TARGET's hyper-connection stream residuals (layers 5/15/25/35/45, 4 streams, averaged, concatenated → fc → draft K/V); block=4 acceptance 2.83-3.26, block 8 = 2.9 tokens/round | D / 1Cat-vLLM-Gaudi (DFlash2-on-Gaudi) + Strata `dflash.hpp` (gaudi2-single-card-strata) | **RA-018/T4 — the Gaudi2 proof of the block-drafter class.** The acceptance-ceiling answer (the MTP head's 3.33/5 < the 4.2 the DECISIONS roofline requires for 200 bs=1): one draft pass instead of four, competitive acceptance, and the mHC-residual mechanism is architecturally aligned with GLM-5.3-Flash's 4-stream hyper-connections. Open: the DFlash2-GLM acceptance on Gaudi is UNMEASURED in the corpus; the license split (DFlash2-GLM CC BY-NC-ND vs DSpine Apache 2.0, RA-016) |
| G17 | **The MTP-head acceptance ceiling, measured:** the abotsis depth sweep — MTP-4 is the OPTIMAL depth on the same model class (acceptance 3.33 of 5; depth 5: +0.1 accepted at higher cost; depth 3: nothing); the MTP-4 production recipe (NSPEC=4, MAXSEQ=8/16, GMU 0.35, `VLLM_COMPACT_GDN=0`, pinned decode buckets, draft-attn-in-graph, warmup 4-5 min ~17 GiB graphs/card; concurrency 15.5/56.0/106.7/185.9 @1/4/8/16); the long-context wall: 262K trial → ~6 t/s (each speculative lane carries the whole block table, 10,240 blocks/step × 5 lanes) | D / abotsis fork `docs/features/glm53_flash.md` (§26 depth sweep, the recipe, the 262K trial) | **RA-018/W7+W8+T5 — the reference numbers.** The 3.33/5 is the acceptance our k=4 must reach (expect 3.1-3.4 on our stack); the 185.9 @16 is the @16 consistency check (our base is ~2.5-3× faster → 300-450 ESTIMATED); the ~6 t/s at 219K is the long-context k=4 wall (the block-table-sharing lever, W8); the class-level warning (G11: the Qwen hybrid's MTP k=2/k=3 "Collapse or garbage") is why the per-verification (T7) and the state-safety (the candidate slots, A022) matter |
| G18 | **The vendor GDN spec path is unimplemented:** the stock `hpu_fused_recurrent_gated_delta_rule` raises `NotImplementedError` for the speculative-decoding path (v0.26.0; the issue is open on v0.30.0); the community request is the multi-token variant (`hpu_fused_recurrent_gated_delta_rule` spec decode / a "multi-token GDN decode"); the vendor's fused single-token GDN decode op does exist (v0.24.1+) and our port's custom KDA path is the only route for the spec verify | D / vllm-gaudi issue (corpus web) + vllm-gaudi v0.26.0 tree | **RA-018/T3 context.** No vendor fix to wait for on our stack version; the TPC custom kernel (the Q21/B7 infrastructure — the op-lib load order is now proven in production) is the deployment path; track the multi-token GDN variant for the v0.30.0 upgrade decision (it would supersede both the conv-chain and the recurrence custom work) |

| G19 | **The DSV4.1 mHC control-op family** (the full-mHC-branch-as-custom-ops pattern): `deepseek_v41_overlap.py` (`independent_mhc_nodes`/`split_mhc_consumers` — extracts the closed, pure mHC branches so they can run in an overlapping partition) + `control_mme.cpp` (includes `hpu_dsv41_control_mme_pt2.cpp` — the PT2 registration the AB-DEQ dequant op already runs in production); the op list: `control_mme_f32`, `control_rrms_unpack/parallel/swizzled/bf16_weight` (residual RMS norm), `control_gemv`, `control_batch4_f32`, `control_prefetch_f32`, `mhc_gates_f32/positive`, `deepseek_v4_sinkhorn4` | D / 1Cat-vLLM-Gaudi `vllm_gaudi/compilation/deepseek_v41_overlap.py` + `csrc/deepseek_v41_unique/pytorch/control_mme.cpp` | **RA-019's pattern.** The mHC-prep chain (the 184-MiB MEMLOG class) is exactly the branch family this infrastructure fuses into a few custom ops; the f32-suffixed variants exist to preserve the FP32 arithmetic exactly (the bit-identity risk management); the PT2 path is production-proven in our tree (AB-DEQ) |
| G20 | **The tensor-cache policy is read per call** (not at wrap time): `wrapped_hpugraph_forward(cache, ..., disable_tensor_cache, ...)` does `env_tensor_cache = os.environ.get("PT_HPUGRAPH_DISABLE_TENSOR_CACHE")` inside the per-forward path, `disable_tensor_cache = disable_tensor_cache if env_tensor_cache is None else env_tensor_cache == "1"`; the cache-OFF capture is the `dry_run=True` mode (no allocation at capture); with cache OFF "tensors … are replaced with empty tensors after replaying" | B / gaudi-pytorch-bridge `python_packages/habana_frameworks/torch/hpu/graphs.py` `wrapped_hpugraph_forward` | **RA-020's mechanism.** A runner-side env toggle per call (the phase is known at the l.3737 site) gives per-phase retention: decode ON (the RA-001 win) / prefill OFF (the ABCONFIG-WIN fix) with the prefill graphs still captured and replayed. The staged `GLM_TENSOR_CACHE_OFF` patch (global OFF) becomes a 3-line conditional |
| G21 | **The stock graph-memory envs size the KV pool; they do NOT cap graph capture:** `VLLM_GRAPH_RESERVED_MEM` (default 0.1 — 10% of usable memory reserved for graphs, the rest for KV) + `VLLM_GRAPH_PROMPT_RATIO` (default 0.5 — the prefill/decode graph-memory split); verified PRESENT in our v0.26.0 (`hpu_worker.py:410`) — the reservation is a KV-sizing budget computed after the profile run; graph captures allocate past it unbounded (the 101 GB watermark OOMs prove the captures ran past any reservation) | B / vllm-fork `README_GAUDI.md` + habana-docs (tier B docs) + verified in `/opt/glm53/vllm-gaudi/vllm_gaudi/v1/worker/hpu_worker.py` | Do NOT look for a config-level fix for the prefill-workspace OOM in these envs — they bound the KV, not the captures. The levers are structural: RA-019 (remove the hog) / RA-020 (bound the retention). Also: with `--num-gpu-blocks-override 1280` the KV pool is fixed regardless — the reservation interacts with the auto-sizing path only |

**Cross-checks performed outside the corpus (this round):** deployed venv source reads on g2 —
`habana_frameworks/torch/hpu/graphs.py` (the wrapper: l.41-160 the `HPUGraph` API; l.635-770
`wrapped_hpugraph_forward` — the `copy_to` + `replay(asynchronous)` path, the `input_hash` cache key,
the `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` env override, the `bypass_hpu_graphs`/`warmup_mode` kwargs;
l.773-900 `wrap_in_hpu_graph`/`wrap_in_hpu_graph_func` — the `asynchronous` parameter, default False),
`habana_frameworks/torch/dynamo/compile_backend/config.py:66` (`PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT`,
dynamo-only consumer) + `passes.py:242/292` (the two consumers), `vllm_gaudi/v1/worker/hpu_model_
runner.py` (l.1118-1120 `_maybe_wrap_in_hpu_graph` — no `asynchronous` argument; l.5021 main-model
wrap; l.5060 drafter wrap; l.4370-4400 the async-until-token-D2H design comment), `vllm_gaudi/
extension/ops.py:860-960` (the dequant path: `dequant_block_fp8_weight_naive` + the TPC-dequant gate +
the BF16 `F.linear`), the port `model.py` (l.312 `VocabParallelEmbedding`, l.411 `ParallelLMHead` —
the C3 correction), `benchmarks/` (no logprobs). Web: mw00/project-maya (README + the GLM spec bench
README), the upstream vLLM #53906 evidence via the corpus port docs.

---

## SECTION 9: Research History

| Date | Rev | Author | Findings / changes |
|---|---|---|---|
| 2026-11-09 | 1 | independent research architect (bottleneck-elimination directive) | First edition of `BOTTLENECK_RESEARCH.md`. Input: `current_bottleneck.md` (v1) + its superseding `current_bottleneck_final.md` rev 3 (D1/D1-16 profiles, B7 resolution) + the EXPERIMENTS.md/TASKS.md audit (q21h boot, AB-DEQ, the NS16/32/64 curve) + deployed-tree source inspection + GaudiGrimoire (13 findings, G1-G13) + web (project-maya, the GPU GLM-5.3-Flash implementation). New proposals: RA-011 (async replay + vendor thread pool — the single highest-value new item; tier-B mechanism), RA-012 (fused dequant-GEMM linears + mHC Sinkhorn TPC kernel), RA-013 (expert placement by routing histogram — the project-maya precedent), RA-014 (2-4K prefill bucket fix), RA-015 (DSA indexer port, priority 5). New rejections: RJ-13 (the AR-split env var is compile-mode-only), RJ-14 (user-space C++ executor misaimed — the loop is vendor C++), RJ-15 (RAM/SSD expert tiering — all weights fit in HBM), RJ-16 (MXFP4 base case — stretch only, 0.5 pp gate), RJ-17 (from-scratch engine — the execution-layer changes deliver the benefit on the existing stack), RJ-18 (engine `--async-scheduling` — broken on this stack class; distinct from RA-011), RJ-19 (AR batching/SP — not worth the risk before RA-011/RA-012C/BG-5). Corrections to the diagnosis: C1 (the @16 weight read is ~59% touched, not 100% — Q21's @16/4-HPU value is larger than credited), C2 (the B1 bs=1 figure is mislabeled — Section 8's is correct), C3 (the lm_head is sharded — `ParallelLMHead`), C4 (the AR count is a structural recipe floor — B4 re-ranked), C5 (the per-recipe cost is vendor C++, not Python — the C++-executor direction misaimed), C6 (the draft host term is ~fixed +0.2/stream, not ~1/stream), C7 (the host/device serialization fact — 45.1 ≈ 24+21). New gaps: BG-1..BG-7. The combined end-state model (Section 7) places 800@16 inside the envelope after RA-011+Q21+RA-012+attention trims (~780-860 t/s) and 200 bs=1 after RA-011+Q21+RA-012+RA-004 (~200-260 t/s pre-margins). |
| 2026-11-09 | 1.1 | independent research architect | Live-tree verification pass on g2 (deployed venv + running process): (1) RA-011 premises CONFIRMED in the deployed build — `wrap_in_hpu_graph(module, asynchronous=False, ...)` signature in `/opt/glm53/venv/.../habana_frameworks/torch/hpu/graphs.py` (l.840), and the env var strings `PT_HPU_ENABLE_HPUGRAPH_THREAD` / `PT_HPU_QUEUE_SYNLAUNCHES` / `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK` present in the deployed 1.24.1 bridge .so; the live process does NOT set them. (2) **BG-4 RESOLVED:** live process environ shows `GLM5_NEXT_MHC_SINKHORN=kernel` — the Sinkhorn TPC kernel is ON; RA-012 Part B closed (no work), RA-012's live value is Part A only (−1 to −4 ms/step bs=1, BG-1 decides); ranking table, end-state model, and detail file updated accordingly. (3) Live dequant gate: `GLM_FP8_MXK=0` — the running non-MoE path is the torch dequant + BF16 `F.linear` (the TPC dequant op, AB-DEQ, is a separate gated experiment); RA-012 Part A's A/B arm sets the gate ON. (4) The wrap-site check in the companion doc's cross-check line was imprecise — the LAZY-mode wraps (l.1118-1120, main l.5021 + drafter l.5060) use `disable_tensor_cache=False` (tensor cache ON, as RA-001/RA-011 state); `disable_tensor_cache=True` at l.5030 is the non-lazy branch (inactive under `PT_HPU_LAZY_MODE=1`). |
| 2026-11-09 | 1.2 | independent research architect | **RA-016 added (user-requested investigation of github.com/Tencent/DSpine):** DSpine (arXiv 2609.36173, Apache 2.0) — the strongest published block drafter: gated adjacent causal injection through draft depth, b16 blocks, +3.7-6.9% acceptance over DFlash2 at equal budget, +8-27% vs the official DFlash2-b8 checkpoint (Qwen3.8-27B), 1-2B 5-layer Apache-2.0 draft checkpoints on the `tencent/dspine` HF collection (active; 27B draft released 2026-11-08; **no GLM-5.3-Flash draft yet**). Confirms the user's hypothesis (better than DFlash2 — by a few percent, not a multiplier). The Gaudi 2 port path is corpus-grounded: the 1Cat DFlash2 HPU `propose_query_block` pattern (tier D) with k=15 + the abotsis KDA candidate-slot machinery (generic in `spec_len`; origin of our MTP port); DSpine's draft is a plain dense transformer (simpler to port than DFlash2's conv/non-causal draft). Status: CONDITIONAL — track the collection, do RA-009 first (shared port machinery), activate on checkpoint release or a SpecForge training decision. Full spec: `RA-016-dspine-drafter.md`; ranking table row 7b added. |
| 2026-11-09 | 1.3 | independent research architect | **RA-017 added (user-requested deep dive into "ECFP8" = ECF8, arXiv 2510.02676, ICLR 2026, github.com/zeyuyang8/ecf8):** lossless exponent-concentrated FP8 weight compression (Huffman-coded E4M3 exponent fields, ~2-3 bits/exponent, + packed 4-bit sign/mantissa stream; bit-exact by construction). Deep dive read the full paper (Theorem 2.1 entropy bounds, Section 3 format + Algorithm 1 decode pseudocode, Tables 1-3, appendices) and the repo. Key findings: (1) the paper's throughput gains (11.3-177.1%) are MEMORY-CAPACITY effects (larger batches under a fixed memory budget), NOT per-token decode gains — its JIT layer-wise decompression mode would multiply our HBM-resident weight traffic 2.855× (bs=1 floor 15.5 → ~44 ms) → **RJ-20 (new rejection)**; (2) the value for us: weights stored COMPRESSED in HBM (328.3 → ~282 GB at the ~13-15% FP8-mass estimate, pending Phase 0) + a FUSED decompress-in-register GEMM (the RA-012A/Q21 TPC skeleton + the Huffman decode in the load path, ≤4 KB LUT in TPC local memory, span-flush byte alignment) → **closes BG-7** (the 4-HPU 1M-context KV fit: margin ~8-18 GB → ~64 GB) AND cuts the per-card weight floor 14.5% (8-HPU @16 routed 9.1 → 7.8 ms) — the only 4-HPU option (vs FP8-KV/RJ-10, the RA-015 cap, a window cut) that also improves bandwidth, with zero accuracy risk (lossless); (3) the make-or-break is a microbenchmark: the decode is a sequential per-element ALU chain (~8-15 ops/element) competing with the issue-bound TPC GEMV (Strata: 237 GB/s at M=1) → Phase 2a go/no-go per M (TPC vs MME); (4) Gaudi2's native E4M3 read caps at 240 (exponent fields 0-7, corpus minimax-m3 codec) → our checkpoint's exponent space may be smaller than the paper's 16-value assumption → compression may be slightly better (Phase 0 measures); (5) the repo has no visible LICENSE (R3) — re-implement the encoder from the paper's spec, reference the repo for validation only. Phases: 0 (1 day CPU — compression rate + exponent histogram + losslessness on our checkpoint; the 8% gate) → 1 (bit-exact verification) → 2 (the kernel, after the Q21/RA-012A skeleton; 2a microbenchmark first) → 3 (the 4-HPU loader before the 4-HPU freeze). Full spec: `RA-017-ecf8-lossless-weights.md`; ranking table row 4b added; BG-7 options ranked; RJ-20 added. |
| 2026-11-09 | 1.4 | independent research architect | **RA-018 added (user-requested: audit the implementation agent's completed work, analyze results + reasons for failure, and build multiple theories for making MTP=4 work, with GaudiGrimoire support).** Full audit of the post-rev-1.3 work (TASKS/EXPERIMENTS/ports source + the G2 server fork state): ADOPTED — A001 MTP k=1 (+15.1% bit-identical), RA-001 (+10.7% bs=1), NS16/NS32 (257/400-452 aggregate), the Q18 levers (+42-46% aggregate, the 16-stream MTP crossover GONE), AB-DEQ (the TPC dequant op production-qualified at parity — the Q21 skeleton deployed), **B7 RESOLVED** (the registrar collision was a load-context artifact; the Q21 kernel track unblocked), Q14v2 (TP4+EP4 reproduced: +25% bs=1, +59% @16, 100% suite), BG-2 (routing near-uniform — **RA-013 DISPROVEN as proposed**, the Q21 uniform byte model VALIDATED). REJECTED — Q01v2 draft graphs (+20% banked; captured-region fusion drift flips R003 5/5, mode-independent whole=split=1/12), **RA-011 async replay (+29% banked; R003 flips 5/5 — the in-flight consumption of the graph's cached outputs, exactly my R3 risk; the @16 +70-85% estimate was WRONG — parity; the follow-up is the async-D2H redesign)**, MHC-ROWS (+19% at the 2944 bucket; R003 3/3 — the THIRD structurally different optimization to flip exactly R003), Q11 k=2 under levers (−11%/−29% — refuted, "revisit only under the in-graph draft"), Q16 (−45% bs=1), Q17 (−22%), B023 (+0% — stack-specific), C015 (no improvement — stack-specific), CAP-8192 (OOM — the prefill-workspace HBM budget; RA-014 re-scoped). **The failure pattern: three unrelated optimizations flip one knife-edge suite item (R003, bat-and-ball, a trap question) deterministically — a 1-ULP distribution-mode flip, not a regression.** GaudiGrimoire (G14-G18, new findings): the AR row-invariance proof at boundary level (abotsis `LAYER0_DELTA_FINDINGS` §10-12: HCCL chunk-order rank summation → per-row ULP → KDA state carries it → MoE routing flips at near-ties; `VLLM_HPU_ALLREDUCE_MODE=gather_sum` + pinned decode buckets → 12/12 parity, no decode cost; the 70-line portable patch is in the corpus, our tree lacks it), the one-call spec recurrence + closed-form candidate states (the T3 fusion reference; our conv chain is the per-token Python loop — verified in `kda_layer.py`), the DFlash2-on-Gaudi one-pass block-drafter machinery (the T4 Gaudi2 proof; the mHC-residual mechanism is architecturally aligned with GLM), the MTP-head acceptance ceiling (3.33/5 measured, the depth sweep, the 262K ~6 t/s block-table wall), the vendor GDN spec-path gap (the custom path is the only route; track the multi-token variant). **Theories T1-T7 (detail: `RA-018-mtp4-theories.md`):** T1 the numerics-invariance package (gather_sum port + pinned buckets + captured-core alignment — the unlock for the three banked optimizations AND MTP=4; rank 1 post-audit), T2 the chain graph (one graph per decode bucket for the whole 4-step draft — kills the seam-sync wall that broke Q19; the Q11 "in-graph draft" revisit condition), T3 the KDA conv-chain fusion (the spec_len=5 small-op storm), T4 the acceptance ceiling (3.33 < the 4.2 the DECISIONS roofline requires for 200 bs=1 — "MTP=4" redefined as ≥4 tokens/step; the block-drafter route), T5 the k=4 HBM budget (fits at ns16-32: ~1.1-2.2 GB/rank candidate slots — DERIVED; the wall is long-context block-table carry), T6 the async-D2H output staging (the RA-011 rejection fix; the end-state physics: 165-220 t/s bs=1 at MTP-4 acceptance, 200-280 with the T4 drafter), T7 the tie-aware MTP-aware gate policy (the 0.5 pp bar auto-rejects on one knife-edge item — the gate, not the numerics, is the binding wall; per-flip top-2-gap attribution makes it enforceable). Ranking table re-ordered (rows 1, 3b, 3c added; the status delta block added); the do-first list re-ordered (post-audit); the proposal scorecard (RA-011 directionally confirmed / RA-013 disproven / RA-014 re-scoped) recorded in RA-018 §0.4. |
| 2026-11-10 | 1.5 | independent research architect | **Round-2 audit + RA-019/020/021 (detail: `RA-019-round2-theories.md`).** Audited the post-rev-1.4 arc on g2 (commits through b101646 + the venv + the live server state): **ABCONFIG-WIN ADOPTED** — the tensor-cache-OFF + prefill-8192 profile METS THE PREFILL OBJECTIVE (5,546-7,723 t/s across 0.5-8K, +209-277% vs eager) at the cost of decode bs=1 37.3 (−12% — the RA-001 win forfeited), @16 parity, suite 97.5% (R003 — the 4TH flip source: the tensor-cache STATE itself; the taxonomy is now Q01v2 fusion drift / RA-011 in-flight consumption / MHC-ROWS kernel reduction order / ABCONFIG-WIN cache state — every structural change to the decode residual stream flips it). Root cause of the CAP-8192/PFG+MHC/AB3 OOMs identified by MEMLOG: the mHC-prep FP32 chain (~20 × [rows,4,4096] FP32/layer; the 184-MiB class = 165.7 GB of the 667.7 GB capture volume; 101 GB peak watermark). **New proposals:** RA-019 (the full mHC-prep TPC kernel — fuse the ENTIRE mhc_pre chain, the DSV4.1 `control_mme` family pattern, G19; removes the OOM class + the profile split + gives the 4-HPU headroom; MEDIUM, weeks), RA-020 (the per-phase tensor cache — the vendor reads `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` PER CALL (G20, tier B), so the 3-line runner toggle (decode "0"/prefill "1" at the l.3737 phase site) recovers the −12% decode now with the prefill fix kept; LOW, days; FALSIFIABLE PREDICTION: the suite returns to 100% — the R003 unflip, since the decode retention state becomes bit-identical to the pre-ABCONFIG-WIN 100% profile), RA-021 (the gather_sum port hardening — the applied venv port deviates from the corpus patch in THREE ways: the missing 16 MiB size cap (prefill-OOM risk: 512 MB × 162 ARs at 8,192 rows, retained), the bf16 sum vs the corpus fp32, and the DSV4.1-documented all_gather stale-value hazard in captured graphs at batch >1 (G5/G6) — fix BEFORE the pending A/B enables `GLM_ALLREDUCE_GATHER_SUM=1`; the A/B protocol adds the rowdep probe + a bs=8 parity check as pre-enabling gates; the suite prediction is documented: on the cache-off profile gather_sum may NOT move R003 — the flip sources are now orthogonal). **New findings:** N4 (the eager prefill dip is partly RANK DESYNC — the AIP-util comparison: the eager 2,944-class runs 4-19% AIP wildly uneven across ranks, the graphed 2,048-class 25-33% uniform; graph capture re-locks the ranks; implications: capture-first for every hot path, the MTP eager draft has the same signature at the AR boundaries (a secondary win to measure for T2's chain graph), the 4-HPU TP4 has the smaller desync surface), N5 (the k=4 bs=1 expert-union weight floor is 3-5× the k=1 floor — 9 correlated rows → 40-60 distinct global experts/layer → 5-8/rank → 5.3-8.5 GB → 2.2-3.5 ms at 2.45 TB/s; the Q21 k=4 design target; the exact number is a no-HPU per-row top-8 route hook, 100-request decode run), G21 (the stock `VLLM_GRAPH_RESERVED_MEM`/`VLLM_GRAPH_PROMPT_RATIO` (verified present in our v0.26.0, `hpu_worker.py:410`) size the KV pool only — they do NOT cap graph capture; do not look for a config-level OOM fix there). **Ops (CRASH-INCIDENT lessons):** the journal rate-limit + the host-RAM headroom pre-flight are standing items (the driver reset path allocates a host-resident MMU page pool; the 125 GB host with the crash spam has no margin). The gsum probe (0.0%, wall_s 0.0) is an instrumentation artifact (probed a non-serving server — the boot was mid-load) — re-probe after the A/B. Do-first list re-ordered (round-2 list supersedes the 2026-11-09 list): 1. RA-021 → 2. RA-020 → 3. Q21 (+ the N5 hook) → 4. RA-019 → 5. T1.2/T2 → 6. T7 (+ the baseline definition: the current adopted profile's suite profile = the reference) → 7. ops → 8. the rest. RA-014 is subsumed (the prefill is MET) — kept only as the >8,192-token band question. |
| 2026-11-10 | 1.5.1 | independent research architect | **Post-audit update (the overnight results landed while rev 1.5 was being written; detail: `RA-019-round2-theories.md` §7).** (1) **GSUM-T1 (T1.1 A/B, ran 03:13): REFUTED as the unlock, NEUTRAL otherwise** — the rowdep probe shows NO R003 flip on the HCCL baseline (the row-order ULP mechanism, the corpus G14 theory, is REFUTED as the flip cause on our tree); the gather_sum arm: R003 still flips (suite 97.5%) with decode parity (43.1/43.7 — the corpus no-cost claim confirmed); the flip re-characterized as config-numerics sensitivity — the 5th flip source: the AR order itself (the gather_sum change flips R003 even with the fixed-rank-order sum). The DSV4.1 all_gather stale-value hazard was tested at batch >1 and did not manifest on our v0.26.0. T1.2 DEFERRED as the predictable null. T1.3 downgraded (whack-a-mole); **T7 (the gate policy) is now the primary unlock** for the three banked optimizations (Q01v2 +20%, RA-011 +29%, MHC-ROWS +19%), with the baseline definition (the current adopted profile's suite profile = the reference — the adopted profile itself is at 97.5%). (2) **The per-phase tensor cache (RA-020) was ATTEMPTED (boot_perphase.sh, 23:41) and the boot failed on infra** (`synStatus=26 Device acquire failed` — wedged cards, the CRASH-INCIDENT class) before the deployed patches were ever executed — and audit of the deployed patches found three defects: D1 (the phase signal `total_num_scheduled_tokens > 0` is True on every serving step — decode requests contribute ≥1, `scheduler.py:1065` verified — so no per-phase distinction), D2 (the flag is set on the HpuModelRunner but read from `orig_fwd.__self__` = the HpuModelAdapter, which has no `__getattr__` proxy → the flag is always False → the patch is inert → the boot would OOM at the 2,944-row capture, the PFG+MHC failure mode), D3 (the patch only toggles `mark_user_inputs`, not the dry_run/disable_tensor_cache capture mode that controls the intermediate-tensor workspace — the MEMLOG OOM class; the vendor's cache-OFF path does dry_run=True + mark_user_inputs). Predicted outcome of the boot as deployed: OOM or decode 37.3 — NOT the both-wins its comment claims; do not re-run as-is. **RA-020 corrected:** the vendor reads `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` per call (graphs.py:697, tier B — G20) and the per-call value drives the full cache-OFF path incl. the derived dry_run (l.699-700) → the fix is a RUNNER one-liner (the env set at the l.4266 site, signal = `total_num_scheduled_tokens > 2048` — the demonstrated OOM boundary; the decode class is ≤ ~64 rows at ns32) + the graphs.py:706 patch REVERT; the 100%-suite (R003 unflip) prediction stands as the gate. (3) k4b (the MTP k=4 experiment) in progress at update time. Do-first order: see the rev-8 handoff note. |
