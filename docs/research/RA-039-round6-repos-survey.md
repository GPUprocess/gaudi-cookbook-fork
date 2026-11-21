# RA-039..RA-048 — Round 6 (2026-11-21, revision 15) — the repos-survey round

Independent performance research for GLM-5.3-Flash on 8× / 4× Intel Gaudi 2.
Author: independent research architect (research-only; no HPU runs, no implementation).
Companion to `EXTERNAL_RESEARCH_PROPOSALS.md` (master handoff) and `BOTTLENECK_RESEARCH.md`.

## 0. What changed this round (the evidence base)

Four external repositories were surveyed in depth this round (full read-only surveys;
none of this code is ours, none is proposed for verbatim import except where stated):

1. **abotsis/wright** (Tier D; AGPL-3.0-only or commercial license) — the "Wright" native
   engine and its GLM-5.3-Flash 4-card reference deployment (v0.3.0, 2026-11-16):
   **94 tok/s bs=1 (no MTP), 404 @16, 514 @64** on 4× Gaudi 2; device-side sampling,
   16-step engine overlap, context-agnostic compiled-at-startup graphs, host-LRU prefix
   cache (285 ms 90K agent turn), paged-arena KV.
2. **rimb05/glm-on-gaudi2** (Tier D) — the most complete production GLM-5 deployment on
   Gaudi 2 that exists: GLM-5.3 "744B" (the 78-layer, 753B-param, 256-expert sibling of
   our model) on 2×8 cards (PP2×TP8×EP) with **vllm-gaudi v0.30.0 @ 058f107 +
   gaudi-cookbook @ 1540263** (14-patch GLM-5 port). Techniques: INT8 per-group latent KV
   (`LatentInt8Cache`, 0.6% error, 2× capacity, reads faster, writes +3-4 ms/step),
   **MTP k=2 draft chain in a single HPU graph (pad-block writes + selection reuse +
   device rejection; +13-23% vs k=1, acceptance 0.73)**, DSA v2 sparse decode (4096-key
   threshold), fastmeta numpy block list, disk recipe cache in production, curated
   1M-window bucket files, twin-model A/B methodology, and the documented
   "agreement statistics, not token identity" near-tie validation protocol.
3. **JChen-318/GaudiInfer** (Tier D) — native C++20 single-card engine (llama.cpp-derived,
   direct Synapse C++ graph API, no Python on the serving path), 205 tok/s bs=1 on a
   27B FP8 dense. Transferable mechanisms (not code): **WY-form gated-delta-net verify
   kernel** (2 MME batch GEMMs + 1 TPC recurrence kernel, full algorithm in
   `gdn_mtp_wy_f32_gaudi2.c`), **deferred state commit** (verify reads state read-only +
   stashes the block; a tiny accept recipe advances state over accepted rows only,
   double-buffered SSM/conv), **pre-enqueue step-loop overlap** (next round's draft
   enqueued after commit, before host work; single stream), batched verify/commit across
   slots, fused norm+quant/act+quant TPC kernels with the exact fp8_143/240/bf16
   rounding chain, two-stage vocab top-k + FP8-candidate verify head, and two
   measurement-discipline negative results (fused small kernels can *lose* to
   compiler-overlapped graph ops; merged cross-slot attention cores were *slower*).
4. **JChen-318/gaudi2-single-card-driver** (Tier D) — bring-up record documenting a real
   **1.24.1 driver bug** (host-resident PMMU treats a DMA/IOVA address as physical →
   host-memory corruption → kernel panic) active only under an IOMMU *translated* domain;
   `iommu=pt` (identity) is the verified workaround. Mechanism cross-verified against the
   canonical driver source in GaudiGrimoire (Tier B: `gaudi2.c`
   `prop->pmmu.host_resident = 1`).

Direct verification on our own stack this round (read-only; server + local):

- **V-1 (async scheduling is in our allow-list):** `/opt/glm53/vllm/vllm/config/vllm.py`
  l.1045-1085 (server-verified 2026-11-21): async scheduling is enabled by default
  (`async_scheduling: bool | None = None` → enabled unless incompatible) and the
  incompatibility check explicitly **allows MTP**: "async scheduling is only supported
  with EAGLE/MTP/Draft Model/NGram GPU/DSpark kind of speculative decoding". Our v0.26.0
  tree ships `HPUAsyncScheduler`
  (`vllm_gaudi/v1/core/sched/hpu_async_scheduler.py`), and our fork already carries two
  fixes for exactly this path: `fixes/async-inputs` (upstream vllm-gaudi#1919) and
  `fixes/async-discard-counter` (vllm-gaudi#1794/#1803). **Whether the scheduler is
  actually ACTIVE in our current MTP boot, and what the device-reject `.cpu()` sync at
  `rejection_sampler.py:273` is waiting on, is unverified** → D14, and the core of RA-039.
- **V-2 (KV dtype):** `ports/glm-5-3-flash/serving/serve_mtp.sh` has **no
  `--kv-cache-dtype`** → KV cache is bf16 (the model dtype). FP8/INT8 KV is therefore a
  pure capacity gain available to us (RA-042). Habana documents `--kv-cache-dtype
  fp8_inc` as a first-class serving option (Tier B, vllm-hpu-extension calibration
  README).
- **V-3 (ops checks, no action needed):** our node runs `iommu=pt intel_iommu=on` with
  all Gaudi devices in the **identity** IOMMU domain (cmdline + `/sys/bus/pci/devices/
  */iommu_group/type`, server-verified) → the 1.24.1 PMMU panic is NOT active on us
  (RJ-24). Host CPU governor is already **`performance`** at 3.4 GHz (RJ-25).
  `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` is already set in our production environment
  (rev-11 RA-032 NO-OP finding, re-confirmed by the vllm-fork plugin
  `load_general_plugins` auto-set, Tier D) (RJ-26).
- **V-4 (model geometry, server-verified 2026-11-21 from
  `/data/models/GLM-5.3-Flash/config.json`):** 45 layers + 1 MTP layer (layer 45,
  `index_share_for_mtp_iteration: true`); 34 KDA layers (`num_heads 64`, `head_dim 128`,
  `short_conv_kernel_size 4`) at indices {0,1,2,4,5,6,8,…,44}; 11 DSA layers at {3,7,11,
  15,19,23,27,31,35,39,43} (MLA: `kv_lora_rank 512`, `qk_nope_head_dim 256`,
  `qk_rope_head_dim 0`, `v_head_dim 256`, `q_lora_rank 1536`; indexer: 32 heads × 128
  dim, top-2048, `index_kpool 4`); MoE: 42 sparse layers, 288 routed top-8,
  `n_shared_experts 1`, `moe_intermediate_size 2048`; 3 dense layers (intermediate
  12288); `vocab_size 154880`, `hidden_size 4096`, `max_position_embeddings 1048576`,
  mHC `hc_mult 4`.
- **V-5 (upstream sampler fixes):** the gaudi-cookbook (our upstream; pinned 1540263 in
  the corpus = the same commit rimb05 runs) ships `fixes/topk-subset-sampler` (v0.26.0:
  top-k/top-p over `torch.topk(K)`, K ∈ {64,128,256,512,1024} covering the batch max
  top_k, then per-row threshold + exponential-race sampling — no full-vocab sort on the
  decode path) and `fixes/seeded-sampling`. Our fork's `fixes/` has **seeded-sampling
  but not topk-subset-sampler** (local listing, 2026-11-21) → cherry-pick candidate in
  RA-039.
- **V-6 (shared-expert overlap is an explicit upstream no-op on HPU):** vllm-gaudi
  `hpu_fused_moe.py::_launch_shared_experts_overlap` (Tier D, 2f1aaea): "On Gaudi every
  shape is a no-op — upstream gates the multi-stream path on
  `current_platform.is_cuda_alike()`". Our model has `n_shared_experts 1` in all 42
  sparse layers → RA-045 (gated probe).

Settled project physics carried into this round (rev 14, unchanged): P-HOST at the
hardware level; replay floor 25-45 ms at bs=1-class; recipe anatomy 20.5-26.4k nodes,
MoE 87-89% of nodes; adopted stack bs=1 42.4 / @16 258-260 (MTP k=1); the 15.6 ms
post-term (95/95 py-spy samples on the `.cpu()` D2H sync at `rejection_sampler.py:273`);
k=4 acceptance measured 2.68 tok/step (67.1%); the graphed-chain draft cost ~0.8
ms/pass device vs 6.4 ms/pass eager (MTP4-T0/rev-14); Q14v2 TP4 = 90% of TP8 bs=1
(41.7), @16 180.8; 1M needle 100% at 860,739 tokens; R003 knife-edge suite item
(0.75 logit gap flips); 97.5% suite gate.

### Ranks (rev 15)

| Rank | ID | One-line | Class | Status |
|---|---|---|---|---|
| 1 | **RA-048** | Owner decision: the 4-HPU production engine question (Wright gap table: their 4 cards = 94 tok/s bs=1 vs our 41-45; AGPL-3.0 or commercial license) | decision | PROPOSED (owner) |
| 2 | **RA-040** | Graphed chained MTP draft (k=2) with pad-block-write + selection-reuse invariants (rimb05 production design; materially different from the rejected Q01/Q01v2) | DIRECT/ARCH | PROPOSED |
| 3 | **RA-039** | Step-loop overlap program: async-scheduler audit (D14) + pre-enqueued draft (GaudiInfer pattern) + topk-subset-sampler port — takes the 15.6 ms post-term off the critical path | DIRECT | PROPOSED |
| 4 | **RA-043** | WY-form KDA verify/decode kernel + deferred-commit state design (the 34-layer node-count program; template + algorithm specified in GaudiInfer source) | ARCH | PROPOSED |
| 5 | **RA-041** | KDA prefix cache via block-page state checkpoints — cookbook `gdn-prefix-cache` fix port + GLM-sized checkpoint lattice (state is 68-90 MB/block, so not every block) | DIRECT | PROPOSED |
| 6 | **RA-042** | INT8 per-group latent KV for the 11 DSA layers (+ fp8_inc fallback, hybrid-precision option) — the BG-7 4-HPU fit + 1M headroom answer | DIRECT | PROPOSED |
| 7 | **RA-047** | DSA v2 sparse decode + the unmeasured 1M decode rate (D17 gate) — the 1M workstream | ARCH | PROPOSED |
| 8 | **RA-044** | Q21 kernel build + design addendum (g2-opt-manual compiler flags, the poison-byte contract, the GaudiInfer negative-result discipline) | OPS addendum | PROPOSED |
| 9 | **RA-046** | Production-config env A/B sweep (fastmeta, graph-reserved-mem, QKV-slice threshold, 1M bucket-file merge) | OPS | PROPOSED |
| 10 | **RA-045** | Shared-expert multi-stream overlap probe (upstream HPU no-op; D19 microbench gate first) | ARCH (gated) | PROPOSED (gated) |

Execution order (dependencies, not ranks): **(1) RA-048 is an owner decision — route it
now, it does not consume HPU time.** (2) D14 + D15 + D17 + D18 in one measurement
window (no HPU-time items: D14 is a config/log read, D18 is a shape read + arithmetic;
D15 is the graph_editor characterization; D17 is one 1M boot window). (3) RA-044 is a
2-day addendum to the already-running Q21 — fold in immediately. (4) RA-039 after D14
(its audit is the first step). (5) RA-040 after the RA-018/T1 drift fix lands (the
prerequisite for ANY draft-graph adoption) — the pad-block invariants are additive.
(6) RA-041/RA-042/RA-047 are independent tracks; RA-042 is the 4-HPU decision input.
(7) RA-046 items ride any boot family (config-only). (8) RA-045 only if D19 ≥ 3%.

---

## RA-048 — Owner decision: the 4-HPU production engine question (the Wright gap)

**1. Executive summary.** The Wright reference deployment serves GLM-5.3-Flash on
4× Gaudi 2 at **94 tok/s bs=1 (no MTP), 404 @16, 514 @64** (Tier D, v0.3.0,
2026-11-16) — ~2× our measured 4-HPU decode (35.6-45.1) on the single-stream metric, and
~1.6× at @16 (404 vs our ~246-258 @16 on 8 HPUs; our 4-HPU @16 is ~180 per Q14v2).
Wright is a native C++ engine (no PyTorch, direct Synapse graph API, ~50-60 launches per
forward vs our 20k-node graph) with device-side sampling and 16-step engine overlap;
its structural advantage is the host term — their step host cost is O(1 ms) vs our
settled 25-45 ms replay floor. The decision: **(a)** evaluate Wright (AGPL-3.0-only,
or a commercial license from the author — `COMMERCIAL.md`: "The author also grants
commercial licenses for Wright") as the 4-HPU production engine, with our 8-HPU vLLM
stack remaining the development vehicle; **(b)** continue the in-tree vLLM lever stack
(RA-038/039/040/043/044) and accept that 4-HPU production stays on the vLLM port;
**(c)** dual-track (our stack for 8-HPU development and the MTP=4 owner directive,
Wright evaluation as a bounded spike). This proposal is the decision document, not an
implementation.

**2. Current limitation.** Our 4-HPU decode profile (Q14v2, MEASURED): bs=1 41.7
(90% of the 8-HPU 42.4), @16 ~180.8. The 4-HPU multi-row host was measured ~90 ms
(rev-5 consequence 4). The settled physics (rev 14) says the 4-HPU bs=1 base case after
the full lever stack is 130-250 tok/s **if** the host falls to 7-16 ms and the routing
kernel hits 89-100% of spec — i.e. the 4-HPU target depends on the two hardest
remaining levers (RA-002 kernel + RA-038-A node collapse) landing near their best cases.
Wright's 94 tok/s bs=1 at 4 cards (no MTP, i.e. with k=1-equivalent acceptance of 1.0
and no draft cost) already exceeds our *measured* 4-HPU number by ~2×, with MTP-2-class
speculation (their DSpine b16 on the 27B; our MTP layer at k=2 would add ~1.2-1.3×)
pointing at ~110-120+ tok/s bs=1 on 4 cards in that engine [DERIVED].

**3. Proposed "architecture".** No architecture change — a decision + a bounded
evaluation: (i) the owner picks (a)/(b)/(c); (ii) if (a) or (c): a 1-2 week Wright
spike on our 4 spare HPUs (8-HPU node, 4-card isolation via HSA_VISIBLE_DEVICES or the
driver's device selection): boot GLM-5.3-Flash (they already ship it — v0.3.0), run the
project's parity suite (greedy + sampled, the R003 item, the 1M needle, degeneration),
and record the 94/404/514 claims as MEASURED-or-refuted on our hardware/driver (1.24.1);
(iii) the license posture is recorded: AGPL-3.0 obligations apply to *any* use of the
engine including network service (section 13); a commercial license is available from
the author (contact in `COMMERCIAL.md`). The in-tree vLLM work continues regardless —
the 8-HPU development target (200 bs=1 / 800 @16) is not claimed by Wright (their 8-lane
numbers are on a 27B dense, not GLM-5.3-Flash at 8 cards).

**4. Technical feasibility.** Wright is a working product on this exact model family and
ASIC (Tier D, self-reported; the 4-card GLM-5.3-Flash table is from v0.3.0 docs dated
2026-11-16 — two weeks before this round). Feasibility of (ii) is high: the spike uses
their release, not our code. The risk is that their numbers are single-vendor
self-reports on their node (driver/SW build may differ from our 1.24.1); the spike
converts Tier D → MEASURED.

**5. Expected performance impact.** If (a): 4-HPU production decode 2-2.4× (94 →
measured) with no dependence on RA-002/RA-038-A landing; the 200-bs=1/800-@16 4-HPU
targets become plausible-with-speculation on their engine [DERIVED, wide band]. If (b):
no change; the 4-HPU targets remain gated on the in-tree stack per rev-14 physics.
Worst case of (a): the spike refutes the numbers on our node (driver delta, the EP
layout on 4 cards, their KV pool math at our block sizes) → the decision reverts to (b)
at the cost of 1-2 weeks.

**6. Implementation complexity.** None (decision) → Low (the spike: their boot script +
our suite, 4 cards, 1-2 weeks).

**7. Risks.** License (AGPL network clause; the commercial license is the clean path for
production). Two-engine maintenance burden if (a) is adopted for production while
development stays on vLLM — mitigated by (c) keeping Wright production-only and the
vLLM port as the source of truth for correctness work (the parity suite runs on both).
Wright is a single-maintainer project (bus factor). Their engine is single-node
4-card-native; it does not help the 8-HPU development targets at all.

**8. Dependencies.** Owner decision. For (ii): 4 HPUs for 1-2 weeks, the checkpoint
(already on the node), the parity suite (already in the fork), driver 1.24.1 parity
with their build (record the delta).

**9. Implementation instructions.** (1) Route this section to the owner with the gap
table (below) and the license note; (2) if (a)/(c): isolate 4 HPUs, obtain the Wright
v0.3.0 release, boot with their GLM-5.3-Flash preset, run: greedy 32K single-stream
(×3 runs), the project parity suite (greedy + sampled, ≥97.5% gate), the R003 item, the
1M needle, @16 aggregate (16 concurrent, 2K context), 1H soak; (3) record everything as
EXP-W* entries per `docs/BENCHMARK_POLICY.md`; (4) do not modify the implementation
agent's branches — the spike is a separate worktree/container.

**10. Validation criteria.** The decision is "validated" when the owner picks an option
and (for (a)/(c)) the spike's parity-suite result is recorded: Wright is viable for
production if suite ≥ 97.5% with all flips attributed AND bs=1 ≥ 80 tok/s on 4 cards on
our node (their 94 within node-to-node tolerance); otherwise (b).

**11. Novelty assessment.** Never raised before this round (the Wright repo appeared in
the research scope 2026-11-20). It is not a repeat of any failed experiment: it is an
external reference implementation that changes the drafter/host-overlap picture
entirely (no-draft 94 > our MTP-on 45 at 4 cards). It deserves the owner's attention
because the 4-HPU production target is a hard project requirement ("Final HPU
allocation: 4 Gaudi 2 HPUs") and the in-tree path to it is gated on the two hardest
levers.

**Gap table (all Tier D unless marked; our side MEASURED):**

| Metric | Wright 4-card (self-reported, v0.3.0 2026-11-16) | Ours 4-HPU (MEASURED, Q14v2/rev-5) | Ours 8-HPU (MEASURED, adopted) |
|---|---:|---:|---:|
| bs=1 decode | 94 tok/s (no MTP) | 41.7 (MTP k=1) | 42.4 (MTP k=1) |
| @16 decode | 404 tok/s | ~180.8 | 258-260 |
| @64 decode | 514 tok/s | — (ns32 cap; owner scope 16) | — |
| 90K agent turn TTFT (prefix cache) | ~285 ms | prefix caching OFF (engine constraint) | same |
| Engine | C++ native, ~50-60 launches/step, device sampling, 16-step overlap | vLLM v1 + vllm-gaudi v0.26.0, 20.5-26.4k nodes/recipe, host floor 25-45 ms | same |

---

## RA-040 — Graphed chained MTP draft (k=2) with the pad-block-write + selection-reuse invariants

**1. Executive summary.** Re-attempt the graph-captured MTP draft at k=2 under the
invariant design that rimb05 runs in production on the GLM-5.3 family (v0.30.0,
16 cards): one HPU graph containing **both** draft passes; the draft's KV writes target
**PAD blocks** (the draft never writes real KV state — zero side effects on committed
state); expert/indexer **selection is reused** from pass 1; rejection stays on device.
Their measured effect: 62.7 → 70-77 tok/s (+13-23%) at k=2 vs k=1, acceptance 0.73
(Tier D). Our own k=2 numbers: acceptance 2.25/3.0 (RJ-11, MEASURED), eager k=2 probe
+10-17% bs=1 (7aaeaa5, MEASURED), graphed-chain device cost ~0.8 ms/pass (rev 14,
MEASURED). The k=2 latency mode is already staged (`serve_mtp_k2ns32.sh`) but eager.
Graphing it under the new invariants is the single highest-value decode-latency item
remaining, and it is the explicit owner directive ("get MTP=4 working better"; k=2 is
the stepping stone RA-038-B D1/D5 build on).

**2. Current limitation.** Our rejected draft graphs (Q01, Q01v2) flipped the R003
suite item; the 11-13 note records the working hypothesis: "speculative decoding
contract break (draft must never change content) — probably a verify/rollback
interaction with the graph-captured draft". The rev-14 corpus work pinned R003's
mechanism to the HCCL all-reduce row-order ULP + the bucket-ladder shape numerics
(RA-018/T1, the drift fix is the prerequisite for ANY draft-graph adoption). What is
NOT yet established is whether our captured draft also *wrote real KV/state during the
draft passes* (the eager path writes draft-token KV that must be rolled back on
rejection). If the graph-captured draft retained that write-then-rollback structure, the
draft's state writes perturb the committed state's numerics during capture and replay —
a content-changing violation of the spec-decode contract, exactly the class rimb05's
pad-block invariant eliminates.

**3. Proposed architecture.**
- **Capture:** one graph per bucket containing [verify forward (k+1 rows/req) → draft
  pass 1 (MTP layer) → draft pass 2 (MTP layer, consumes pass 1's hidden state)] with
  the pass-2 input wired from pass 1's output inside the graph (the rimb05 chain; the
  A014/RA-004-T1 chain structure at k=2 — the same design, shallower).
- **Invariant I-1 (pad-block writes):** the draft passes' KV slot mapping targets a
  dedicated PAD block region (the bucket's reserved rows), never the request's live KV
  blocks. The draft's attention reads the *committed* KV (read-only) + its own in-graph
  draft tokens; its writes are discarded. Rejected rows therefore leave all real state
  bit-identical (the contract holds by construction, not by rollback).
- **Invariant I-2 (selection reuse):** pass 2 reuses pass 1's router/indexer selection
  (rimb05: "selection reuse"; our `index_share_for_mtp_iteration: true` config field
  indicates the checkpoint was trained with MTP iterations sharing the indexer — the
  model-side precedent). This halves the draft's indexer/router work and removes one
  dataflow dependency between the passes.
- **Rejection:** stays on the existing device-reject path (GLM_SPEC_DEVICE_REJECT,
  adopted). The accept count is computed on device over the k=2 verify rows; the
  async-token-discard machinery (our fork's `async-discard-counter` fix) handles the
  accounting if async scheduling is active (RA-039).
- **Execution flow (per step, bs=1):** host builds the step (input ids from the
  persistent device buffer if async, else the normal path) → one graph replay [verify +
  2 draft passes] → device reject → (RA-039: pre-enqueue the next step's graph before
  the host consumes the accept count) → host bookkeeping.

**4. Technical feasibility.** (a) The chain-graph structure is proven: the A014
reference (abotsis, Tier D) runs exactly this at k=4-5 on the same model family with
measured acceptance 3.33/5.0 and 12/12 parity — RA-004-T1's port target; rimb05 runs it
at k=2 in production on v0.30.0 with the pad-block invariant. (b) The MTP module in our
port already runs two passes eagerly in the k=2 config (7aaeaa5) — the capture is a
re-sequencing of existing nodes, not new numerics. (c) The pad-block write is a slot-
mapping change (the slot-mapping tensor the attention backends consume is already
graph-input in the bucketing regime) — a capture-time constant per bucket. (d)
`index_share_for_mtp_iteration: true` in the checkpoint config (V-4) shows the
reference model shares the indexer across MTP iterations — selection reuse is the
intended mode, not an approximation. **Unsupported assumptions:** that our captured
draft's R003 flip was state-write-induced rather than (only) the AR-row-order ULP — the
drift fix (RA-018/T1) is required regardless; this proposal's I-1/I-2 are additive
invariants, and if the drift fix alone clears R003, I-1/I-2 still hold as the
contract-safe design (the validation below tests them explicitly).

**5. Expected performance impact.**
- bs=1 (primary): k=1 adopted = 42.4 tok/s at 1.88 tok/step. k=2 graphed = 2.25
  tok/step (MEASURED acceptance) at a marginal draft cost of ~0.8 ms/pass device
  (rev-14 MEASURED) vs the eager 6.4 ms/pass. Step time k=1 ≈ 72.5 ms (MEASURED, rev
  14 decomposition); k=2 graphed ≈ 72.5 + 2×0.8 + verify-Δ ≈ 75-78 ms → **2.25/76 ≈
  29.6... correction: 2.25 tok/0.076 s ≈ 29.6?** No — k=1 is 1.88 tok/0.0725 s = 25.9
  effective... The adopted 42.4 tok/s is the *served* rate including the acceptance
  already. Recompute consistently: k=1: 42.4 tok/s = 1.88 tok/step ÷ 44.3 ms/step
  effective (42.4 × 0.0443). Hmm — the 72.5 ms is the *iteration* including host; the
  served 42.4 implies 44.3 ms/token... the clean DERIVED statement: k=2 graphed
  improves the accepted-tokens-per-iteration by 2.25/1.88 = 1.197× at a
  +1.6-3.2 ms/iteration cost (2 × 0.8-1.6 ms, the rev-14 graphed-chain figure band).
  Net: **42.4 × 1.197 × (72.5/75.7) ≈ 45.5-46.5 tok/s bs=1 (+7-9%)** [DERIVED; the
  eager-probe measured +10-17% with the 6.4 ms/pass cost, so the graphed expectation is
  at or below the eager-probe win — conservative band +7-12%]. Best case (draft fully
  pipelined per RA-039, marginal cost → 0): ~50.7 (+19%). Worst case (replay sync
  re-appears, RJ-07 class): neutral or −5% → reject and keep eager k=2.
- @16: the rev-14 T3 decomposition (36 ms/pass per-row split) suggests the @16 k=2 is
  negative (the draft does not amortize at 16 rows) — this proposal is a **bs=1/low-
  batch latency mode**; the RA-038-B D5 batch-conditional k* (k=2 for bs ≤ 4, k=1 for
  the large buckets) is the integration path. Expected @16: unchanged (k=1 buckets).

**6. Implementation complexity.** Medium-High. The chain capture is days (the A014 port
  is RA-004-T1's staged work — this is k=2 of the same structure); I-1 (pad-block slot
  mapping per bucket) is a capture-time constant + the reserved PAD block allocation in
  the KV pool (a few hundred rows — negligible); I-2 (selection reuse) is a wiring
  change in the MTP module (the field exists in config; the eager k=2 path may already
  do it — verify in source, do not assume). The validation protocol is the expensive
  part (below).

**7. Risks.** Correctness: the R003 item (mitigations: the RA-018/T1 drift fix first;
  I-1 removes the state-write channel; the KV-pool bit-compare test below catches any
  residual). The PAD block must never alias live request blocks (allocator: allocate
  once at init, outside the free pool — the capture binds its address). Replay sync
  (RJ-07): the draft-in-graph removes the separate draft replay; the residual sync is
  the single graph's — expected to behave like the adopted single-graph regime.
  HBM: +2 draft passes of activations in one graph (the tensor-cache retention, D3
  class — the rev-8 OOM lesson: check the capture's HBM delta in the first boot; the
  PAD blocks + the pass-2 intermediate are the new terms). Accuracy: I-2 selection
  reuse changes pass-2's expert set vs a fresh route — the checkpoint's
  `index_share_for_mtp_iteration: true` says this is the trained mode; the acceptance
  number (2.25/3.0) was measured with the current eager path — re-measure under I-2 (if
  acceptance drops >5%, I-2 is the cause and is dropped, keeping I-1).

**8. Dependencies.** The RA-018/T1 drift fix (prerequisite — do not boot the draft graph
  before it). The RA-039 async audit (D14) — the accept-count path must be known
  (device vs host sync) before the pre-enqueue ordering is added. The bucketing
  infrastructure (the k=2 buckets are already staged: `serve_mtp_k2ns32.sh`).

**9. Implementation instructions.** Relevant sources: `ports/glm-5-3-flash/glm5_next_hpu/
  glm5_next_hpu/mtp.py` (the MTP module — the draft passes, the selection; 551 lines in
  the current tree), the attention backends' slot-mapping consumption
  (`vllm_gaudi/attention/oot_mla.py` for the DSA layers; the KDA layers' state write
  path in `kda_layer.py` — **verify which of the two the draft passes touch: the draft
  must be read-only on both**), the bucketing code (`ports/glm-5-3-flash/...` per the
  Q01v2 work — the agent's graph-capture patch series). Sequence: (1) source-read
  verification of the draft's current write set (D14-adjacent, no HPU time); (2) the
  PAD block allocation + the slot-mapping override for the draft rows (env-gated,
  default OFF); (3) the chain capture (k=2, the A014 structure) with I-1/I-2; (4) the
  CPU contract test first (the `check_draft_graph.py` pattern, RA-018/T2 step: chain
  graph output == 2-step eager chain, bit-exact on the verify logits and the accept
  count); (5) one HPU boot: greedy + sampled parity, the KV-pool bit-compare (below),
  the R003 item ×3, the degeneration check at 2K+, the bs=1 A/B (k=1 adopted vs k=2
  graphed, ≥5 runs each); (6) if green, the @16 check (expected neutral — the k* split).
  Diagnostics: the per-layer trace of the k=2 graph (the marginal pass cost vs 0.8
  ms), the accept-count histogram vs the eager 2.25/3.0, the HBM delta at capture.
  Benchmark: bs=1 served tok/s (the 42.4 reference), @16 aggregate (the 258-260
  reference), the suite.

**10. Validation criteria.** ADOPTED if: (i) CPU contract test bit-exact; (ii) suite
  ≥ 97.5% with R003 3/3 (or all flips attributed per RA-018/T7); (iii) KV-pool
  bit-compare: after each draft-containing step, the live request KV blocks + KDA state
  are bit-identical to the no-draft run's (the I-1 proof — the strongest single test);
  (iv) bs=1 ≥ 45.5 tok/s (the +7% floor) over 5 runs; (v) @16 within noise of 258-260;
  (vi) 1H soak clean. REJECTED if (iii) fails (the invariant is broken — do not ship),
  or (iv) < 44 (neutral — keep eager k=2 as the latency mode), or the R003 flips are
  unattributable.

**11. Novelty assessment.** The draft graph was attempted twice (Q01, Q01v2) and
  rejected on R003 + the contract-break note. This is a materially different hypothesis
  for the same experiment: (1) the drift fix is now a separate prerequisite (the rev-14
  corpus work pinned R003 to the AR row-order + bucket numerics — the Q01/Q01v2 attempts
  predate it); (2) the pad-block-write invariant removes the draft's state side effects
  by construction — the "draft must never change content" contract the 11-13 note
  identified as the suspected break — and is proven in production on the same model
  family (rimb05, Tier D); (3) the selection-reuse invariant has the checkpoint config
  field as precedent. The rejection of *eager* k=2 (RJ-11) is orthogonal and stands —
  this is the graphed mode RJ-11's reopen condition named ("a draft-execution mode
  where the marginal draft pass costs ≤ 1 graph replay").

---

## RA-039 — The step-loop overlap program: async-scheduler audit + pre-enqueued draft + sampler-chain reduction

**1. Executive summary.** Our measured 15.6 ms post-term (95/95 py-spy samples on the
`.cpu()` D2H sync at `rejection_sampler.py:273`, MEASURED) is a **device-idle window**
in every iteration: the device has finished the step, and the host is draining the
sampler output before the next step launches. Three independent mechanisms can remove
it from the critical path, and this round established that the first of them may
already be half-built in our stack: **(a) the async-scheduler audit** — vLLM's
async-scheduling (which overlaps step N+1's launch with step N's output handling)
explicitly **allows MTP** (V-1), our tree ships `HPUAsyncScheduler`, and our fork
already carries the two fixes that path needs (`async-inputs`, `async-discard-counter`)
— yet whether it is ACTIVE in our MTP boot, and whether our device-reject path's `.cpu()`
sync defeats it, is unverified (D14). **(b) the pre-enqueued draft** (GaudiInfer
pattern #5/#15, Tier D): enqueue the next round's work *before* the host consumes the
current round's output — single stream, overlap by ordering, no multi-stream.
**(c) the sampler-chain reduction**: port the cookbook's `topk-subset-sampler` fix
(V-5) — the device-side sampler currently sorts the full 154,880-vocab per row; the fix
reduces it to a single `torch.topk(K)` with K ≤ 1024, cutting the device time the `.cpu()`
waits for. RA-038-B D3 (the sampler D2H tail → 2-4 ms via the F2 piggyback) attacks the
*payload*; RA-039 attacks the *serialization* — both are worth doing, and (a) is the
structural fix that makes the post-term irrelevant even if the payload stays 15.6 ms.

**2. Current limitation.** Step decomposition (rev 14, MEASURED): 72.5 ms iteration =
28.3 ms launch/replay + 27.3 ms sync (device work) + 15.6 ms post-term (sampler D2H +
parse), serialized. The 15.6 ms is 21.5% of the iteration at bs=1, and it is per-
*iteration* — at @16 the same 15.6 ms sits in a ~115 ms step (13.5%) but it is still
dead device time per step. The device-reject lever (GLM_SPEC_DEVICE_REJECT, adopted)
already moved the *accept walk* to device; what remains is the final `.cpu()` of the
sampler's output tensor + the host parse.

**3. Proposed architecture.**
- **(a) Audit (D14, step 1, no HPU time):** from one MTP boot: the engine config dump
  (is `async_scheduling` true?), the engine-core log line for the scheduler class
  (`HPUAsyncScheduler` vs `Scheduler`), and the worker's effective env. Then, with
  async confirmed on/off, one py-spy + torch.profiler window at bs=1: what does the
  `.cpu()` at `rejection_sampler.py:273` wait for (the sampler op chain's device
  completion? a host-side parse loop?), and does step N+1's launch begin before or
  after it returns? This converts "15.6 ms post-term" from a label into a causal
  chain.
- **(b) Pre-enqueued next-step launch (the GaudiInfer pattern):** after the verify
  graph (+ draft graph, once RA-040 lands) is enqueued, enqueue the *next* step's
  graph replay **before** the host blocks on the accept count / sampled tokens. The
  device then runs [step N verify+draft+sample → step N+1 verify+draft+sample] with no
  host gap; the host does its bookkeeping (scheduler, token accounting) while the
  device is already on N+1. Correctness constraints: (i) single stream (no multi-
  stream — the GaudiInfer discipline; our workspace/collective assumptions are
  single-stream); (ii) step N+1's input ids must be device-resident and correct —
  this is exactly what our fork's `async-inputs` fix provides (the persistent
  `input_ids_hpu` buffer seeded from host, overwritten by device-scattered sampled
  tokens); (iii) the accept count that determines N+1's row count is computed on
  device and consumed by the *scheduler* via the async-token-discard mechanism (our
  `async-discard-counter` fix) — the host never needs the count before N+1 launches;
  (iv) the per-rank draft is independent (no collectives inside the draft) so the
  pre-enqueue does not reorder cross-rank communication — **verify this in the trace**
  (the MTP module's op list; the `index_share` path must not insert an all-reduce
  between ranks in the draft). If any collectives exist in the draft, the pre-enqueue
  scope shrinks to "pre-enqueue the verify part only" or the item is rejected.
- **(c) Sampler-chain reduction (the port):** cherry-pick `fixes/topk-subset-sampler`
  from the cookbook (pinned 1540263, corpus-verified; targets v0.26.0) into our fork's
  `fixes/` + the `apply_all.py` profile. Mechanism (Tier D, the fix's README): one
  `torch.topk(K)` per step with K the smallest of {64,128,256,512,1024} covering the
  batch's max top_k; per-row top_k/top_p applied within those K values; the token
  sampled by the standard exponential race. At our decode (top_k=1 greedy, top_p=1.0
  for the suite; the production sampled mode uses top_p 0.95 class) K=128-256 covers —
  the full-vocab sort (154,880 × rows) is replaced by a 256-wide topk + a masked
  threshold. The device time the `.cpu()` waits for drops by the sort's share —
  expected to cut the 15.6 ms by 40-70% [DERIVED: the topk over 154,880 is the
  dominant sampler op at 32 rows; the exponential race over K is O(K log K)].
  `seeded-sampling` is already in our fork (V-5) — confirm it is in the apply profile.

**4. Technical feasibility.** (a) V-1/V-2/F-3 above: the allow-list includes MTP, the
  scheduler class exists, the two supporting fixes exist in our fork — the async path
  is the vendor-intended mode for MTP-on-HPU, not a hack. The open question is purely
  "is it on, and does our device-reject patch respect it" — D14 answers both. (b)
  GaudiInfer runs the identical ordering pattern in production (single stream, event-
  gated staging, commit-before-draft, Tier D); their measured 29.5 vs 37.5 ms verify
  round vs their own vLLM-gaudi baseline includes this overlap [Tier D, their node].
  The pattern is graph-enqueue ordering only — no new ops, no new streams. (c) The
  fix is ready-made for our exact vLLM version (v0.26.0 profile), Apache-2.0, from our
  own upstream.

**5. Expected performance impact.**
- (a)+(b) full (async active + pre-enqueue): the 15.6 ms post-term leaves the
  critical path → iteration 72.5 → ~57-60 ms at bs=1 → **42.4 → 51-56 tok/s (+20-32%)**
  [DERIVED; the rev-14 physics: the post-term is currently 21.5% of the iteration, so
  removing it is at most +27% before interaction effects; the band accounts for the
  residual host bookkeeping that cannot overlap (the scheduler decision) — expect the
  overlap to hide 70-100% of the 15.6 ms]. At @16: 115 ms → ~100-105 ms → 258-260 →
  **~295-310 tok/s (+15-20%)** [DERIVED].
- (c) alone (sync scheduler remains): 15.6 → ~5-9 ms → bs=1 42.4 → 46-48 (+8-13%),
  @16 258-260 → 270-280 (+5-8%) [DERIVED].
- Combined worst case: the async path is active but our device-reject's `.cpu()`
  *forces* the sync (the fix: move the accept-count D2H to the small count tensor, or
  restructure the reject to stay on the async contract) → still +10-20% from (c).
  Worst-of-worst: the async scheduler is OFF and cannot be turned on without a vendor
  change (MTP + our patch stack interaction) → (c) alone, +8-13% bs=1.

**6. Implementation complexity.** (a) Low (audit, no code). (b) Medium (the enqueue-
  ordering change in the runner's step loop, env-gated; the draft-independence check;
  the interaction with the device-reject patch — a new correctness surface, the same
  class as the "async replay" rejection but this is the vendor scheduler path, not the
  replay thread). (c) Low (a cherry-picked fix + profile line; the fix ships its own
  verify).

**7. Risks.** (b) is the risk-bearing item: the async contract requires that step N+1's
  launched graph is correct *without* knowing step N's accept count — the padded-row
  design (the `async_tokens_to_discard` machinery) handles this in vLLM's model, but
  our device-reject patch inserted a `.cpu()` that may exist precisely because the
  MTP accept count was not on the async contract — the audit must establish whether the
  `.cpu()` is (i) the vendor's MTP-async requirement (then it stays, and (b) is
  limited), (ii) our patch's choice (then (b) is open), or (iii) a leftover (then (b)
  is open + the sync is free to remove). Correctness: the async-inputs fix's e2e test
  (the repeated-token symptom) must be run — our MTP + device-reject + async is a
  combination none of the three fixes was individually tested against. Accuracy: no
  arithmetic changes in (a)/(b); (c) changes the sampling op — greedy (top_k=1) is
  bit-exact (topk(1) over 1024 vs 154,880 is the same argmax); **sampled** mode must
  pass the distribution-preservation check (the exponential race over the top-K subset
  is exact only if K covers the support the thresholds can reach — the fix's K-choice
  logic handles this; verify with the I004/I005 study on the suite's sampled prompts).
  Stability: the pre-enqueue adds one step of in-flight state (the D3-class HBM term —
  check the capture's HBM delta, the rev-8 OOM lesson).

**8. Dependencies.** D14 (the audit) gates everything. (b) depends on the draft's
  collectivity check (trace, no HPU time) and is sequenced after RA-040's chain graph
  exists (pre-enqueueing a 2-pass chain is the design point; pre-enqueueing the k=1
  verify-only graph is a smaller first step — do that first). (c) is independent and
  can land immediately (it is a fix port).

**9. Implementation instructions.** (1) D14: one MTP boot, config dump + log grep +
  env dump (no HPU time beyond a normal boot); then one py-spy window (the existing
  RESEARCH-001 addendum tooling) at bs=1 to time the `.cpu()` wait composition. (2) (c):
  `fixes/topk-subset-sampler` from the cookbook pinned 1540263 (the corpus has the
  README + the patch) → `apply_all.py --profile v0.26.0` line → the fix's own verify +
  greedy bit-exactness + the I004/I005 sampled check → the bs=1/@16 A/B (the 15.6 ms
  re-measured via the py-spy method). (3) (b): the env-gated `GLM_PRE_ENQUEUE_NEXT=1`
  in the runner step loop — after the verify (+draft) graph enqueue, before the
  accept-count `.cpu()`: enqueue the next step's graph (the k=1 form first); the
  in-flight bookkeeping via the existing async-token-discard path; the trace check
  that no collective lands between the two enqueues on any rank. (4) Diagnostics: the
  per-iteration timeline (the existing launch-timestamp patch,
  `kb_staging/hpu_model_runner_launch_ts.py`) before/after — the overlap is visible as
  the next launch timestamp preceding the current step's `.cpu()` return.

**10. Validation criteria.** (a) D14 recorded (async on/off + the `.cpu()` wait
  composition, MEASURED). (c) ADOPTED if greedy bit-exact, the sampled suite passes
  I004/I005, and the post-term drops ≥ 5 ms (bs=1 py-spy re-measure). (b) ADOPTED if:
  suite ≥ 97.5% (R003 3/3), the repeated-token e2e (the async-inputs fix's test)
  clean, bs=1 ≥ 51 tok/s over 5 runs, @16 ≥ 290, the HBM delta at capture ≤ 2 GiB, 1H
  soak clean, and the trace shows the next launch preceding the `.cpu()` return in
  ≥ 90% of steps.

**11. Novelty assessment.** The async-scheduler question has been *touched* (the rev-14
  note "the async-scheduling parity" — the parity finding was about the *device*
  behavior of async vs sync, and RA-032's lazy-collectives NO-OP is related but
  different) but never *audited on our MTP boot* — D14 is new. The pre-enqueued-draft
  pattern is new to the project (GaudiInfer Tier D; the "async HPU-graph replay"
  rejection RJ-01 is a different mechanism — the vendor replay thread, measured
  regression; this is in-order enqueue on the existing stream). The topk-subset-sampler
  port is new (the fix exists upstream; it is not in our fork's fixes/ — V-5). No
  overlap with RA-038-B D3 (the payload piggyback) — complementary.

---

## RA-043 — The WY-form KDA kernel + deferred-commit state design (the 34-layer node program)

**1. Executive summary.** 34 of our 45 layers are KDA (gated delta net). In the current
decode graph each KDA layer is a *chain* of graph ops (width-4 conv, the gating
norms, several small GEMMs/elementwise ops, the state update) — dozens of nodes per
layer per step, at 1.4 µs/node replay cost, in a 20.5-26.4k-node recipe (rev 14).
GaudiInfer (Tier D) demonstrates the alternative for the same architecture class
(gated delta net, verify/decode): **WY form** — two MME batch GEMMs
(`rmat = X·S0ᵀ` the initial-state precompute, `gram = X·Xᵀ` the key-gram) plus **one
TPC recurrence kernel** (`gdn_mtp_wy_f32`: the WY QR decomposition of the delta-rule
recurrence, the state checkpoint in a second kernel phase) replacing the whole chain —
for a 16-token verify block. Combined with their **deferred-commit** design (verify
reads the state read-only + stashes the block's inputs into a double-buffered stash; a
tiny accept recipe advances the state over *accepted* rows only, flipping the state
side), this is (i) the second node-collapse class after MoE (RA-038-A) — the KDA chain
is the largest non-MoE node consumer, and (ii) the *correctness substrate* for
RA-040's partial-accept draft (the commit semantics that make "rejected rows leave
state bit-identical" a small kernel, not a rollback) and for RA-041's checkpoint
lattice (the state is explicit and stash-addressable).

**2. Current limitation.** The KDA chain's node count is unmeasured (D15) but bounded:
the recipe is 20.5-26.4k nodes with MoE at 87-89% (rev 14, MEASURED) → KDA + attention
+ small ops share the remaining 11-13% ≈ 2.3-3.4k nodes ≈ 65-95 nodes/layer across
45 layers → the 34 KDA layers plausibly hold 1.2-2.0k nodes (30-60% of the non-MoE
share) [DERIVED — D15 converts this to MEASURED]. At 1.4 µs/node that is 1.7-2.8 ms of
replay per step — and the node collapse changes *device* behavior too (fewer small
ops = fewer MME/TPC dispatches; the rev-14 settled physics says the node count IS
device time, P-HOST). The KDA state update is also the reason our port cannot do
prefix caching (the in-place state write during the forward — the engine constraint)
and complicates spec-decode rollback (the 11-13 contract-break note).

**3. Proposed architecture.**
- **The WY kernel (new TPC kernel, our geometry):** GaudiInfer's algorithm (Tier D,
  `gdn_mtp_wy_f32_gaudi2.c` comments + `src/gdn.cpp`): for a T-token block over a gated
  delta-net layer: (1) `gdn_spec_prep` (TPC): the width-4 causal conv + SiLU in FP32,
  laying out X = [k; q] per key head and V per value head; (2) two MME batch GEMMs per
  key head: `rmat = X·S0ᵀ` (the initial state against every k_t/q_t) and
  `gram = X·Xᵀ` (the key-key gram matrix); (3) the TPC kernel runs the WY recurrence in
  lanes (the delta-rule solved in WY form: the Householder/QR product form of
  `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, which the WY decomposition turns
  into stable sequential updates over the T×T gram) and, in a second phase, writes the
  checkpointed state `S_T`. **Our geometry (V-4):** 64 heads (GaudiInfer: 16 key /
  48 value), head_dim 128 for both K and V (theirs 128/128), conv 4, plus the GLM
  specifics the kernel must absorb: the per-head gate (the KDA gating — the
  `f_a_proj`/`g_a_proj`/`fused_qkvbfg_a_proj` weight families in the config), the 4-
  stream mHC interaction (the hc mixing is *outside* the KDA state — verify in
  `kda_layer.py` that the state itself is the plain [64, 128, 128] recurrence; the
  config's `A_log`, `dt_bias`, `b_proj` are the delta-rule parameters). The kernel is
  new work (their geometry constants 16/48/128/128 do not transfer), but the algorithm
  and the VLIW structure (lanes, the lane-XOR token loop, LUT-free exp) are the
  template. The Q21 pipeline (the same AOT tpc-clang build + GC_KERNEL_PATH
  registration + the PT2 custom-op adapter) is the vehicle.
- **The deferred-commit design (the state management):** the KDA state per layer per
  request becomes a **double-buffered persistent tensor** (`state_side_a/b`, 64×128×128
  bf16 = 2 MB/layer, 34 layers = 68 MB/request × 2 sides = 136 MB/request — HBM-
  trivial at our pool sizes). The decode/verify forward reads `state_side_current`
  read-only (a graph *input*, like the KDA state is today) and stashes the block's
  (q, k, v, g, β, conv-window) into a per-slot stash region. The **accept recipe** (a
  tiny graph, ~10-30 nodes: the WY state update over the accepted rows + the conv-
  window write + the side flip, batched across all requests in one launch — GaudiInfer
  runs every slot's recurrence "as extra heads of one chunked delta rule" in
  `spec_commit_batch`) commits the accepted tokens. Rejected rows: state untouched by
  construction; the DSA KV rows of rejected tokens stay masked (the current behavior).
  At bs=1 the accept recipe is one slot; at @16 it is 16 slots × 34 layers in one
  batched launch — the commit is O(1) launches regardless of batch (the GaudiInfer
  batched-commit property, Tier D).
- **Integration:** the WY path replaces the KDA chain in (i) the decode graph (T=1:
  their `gdn_packed_decode_f32` single-token variant is the T=1 template — one kernel
  per layer), (ii) the MTP verify graph (T=2 at k=1, T=3 at k=2 — the WY block form),
  (iii) the prefill chunk path (their chunked prefill WY for 2048-token chunks —
  optional phase 2). The existing chunked-delta-rule path stays env-gated as the
  fallback (the A/B discipline).

**4. Technical feasibility.** The architecture is proven on the same op class (gated
  delta net) by GaudiInfer in production (Tier D, their 27B dense runs it as the
  default verify path). The Synapse mechanics (persistent tensors, the double-buffer
  flip as a graph-input rebind, the batched commit as one recipe) are the standard
  graph model our port already uses (the KDA state is a persistent per-request tensor
  today). The TPC kernel is new but bounded (the template is fully specified; the
  geometry is a parameter change + the GLM gate absorption); the Q21 team has the
  exact build/registration pipeline. **Unsupported assumptions:** (i) that the WY
  precompute GEMMs are cheaper at our geometry than at theirs — their H=16/HV=48 means
  the rmat GEMM is [T, 2·128]×[2·128, 128·48] per key head; ours at 64 heads is larger
  (the rmat GEMM is [T, 256]×[256, 128·64] per head — **measure the MME time before
  committing**, D15-adjacent microbench); if the precompute dominates, the WY form
  degenerates to "1 kernel + 2 big GEMMs" vs the chain's "many small ops" and the win
  is the node count only (still worth ~1.7-2.8 ms replay at D15's mid estimate, but
  the device-time claim must be re-derived). (ii) That the mHC 4-stream mixing does
  not touch the KDA state internally (source verification in `kda_layer.py` — no HPU
  time). (iii) Bit-compatibility of the WY recurrence with the current chunked path
  within the accuracy gate — the WY form is a different summation order; expect the
  R003-class near-tie sensitivity (V-21: rimb05's "agreement statistics, not token
  identity" — the paged/recurrence summation order legitimately flips near-ties), so
  the validation uses the suite with the flip-attribution protocol (RA-018/T7), not
  token identity.

**5. Expected performance impact.** Node count: if the KDA chain is 1.2-2.0k nodes
  (D15) and the WY+commit form is 3-6 nodes/layer/step (2 GEMMs + 1-3 kernels + the
  amortized commit share), the cut is ~80-90% of the KDA nodes ≈ 1.0-1.8k nodes ≈
  **1.4-2.5 ms/step replay at bs=1** (1.4 µs/node) → on the 25-45 ms floor that is
  3-10% of the step → **bs=1 42.4 → 44-46 tok/s (+5-8%)** [DERIVED, mid estimate;
  best case if D15 shows the KDA share at the top of the band + the device-time
  overlap of the 2 GEMMs with the TPC kernel per the overlap law: +8-12%]. At @16 the
  node-count cut is the same in absolute ms (the floor is per-recipe, not per-row) but
  the step is 115 ms → +1.5-2.5% on 258-260 (small) — **the @16 value is indirect**:
  the deferred commit is what makes RA-040's k=2 graph safe at @16, and the batched
  commit removes the per-slot state-update launches the current per-slot path would
  add at k=2. Strategic value: this is the node-class RA-038-A's ablation (T2) will
  price — the KDA class is the second ablation arm after the MoE dequant class.

**6. Implementation complexity.** High (weeks): the TPC kernel (the Q21-class effort,
  new geometry, VLIW tuning), the double-buffer state migration (the runner + the
  capture + the allocator), the accept-recipe graph (new), the prefill-chunk phase 2
  (later). The design risk is front-loaded by the D15 measurement + the rmat-GEMM
  microbench (no HPU time for either beyond a microbench window).

**7. Risks.** Correctness: the WY recurrence vs the current chunked path is a
  numerics change (the gate: suite + flip attribution + the bit-compare of the state
  tensors in a controlled single-step test: WY state after T tokens == chunked state
  after T tokens within fp32/bf16 tolerance — the tolerance band is the design
  parameter; if it exceeds the 0.5 pp gate's sensitivity, the WY kernel's math moves
  to FP32 throughout at the cost of the state dtype). HBM: +136 MB/request (the
  double buffer) + the stash regions (T × (q+k+v+g+β+conv) per slot ≈ 2 × ~2 KB × 34
  layers — trivial). The commit recipe is a new graph in the capture set (one more
  recipe per bucket — the warmup cost, the RA-006 class). The TPC kernel development
  risk (the VLIW tuning; the Q21 experience is the prior). If the rmat microbench
  shows the precompute is ≥ the chain's device time, the proposal degrades to
  node-count-only (re-rank to RA-038-A's ablation arms).

**8. Dependencies.** D15 (the per-layer node census — the graph_editor tooling staged at
  380303f), the rmat-GEMM microbench (a TPC bench, no full model), the Q21 TPC pipeline
  (shared), RA-038-A T2 (the ablation prices this class — sequence: T2 arms 0-2 first,
  the KDA arm after), RA-040 (the deferred commit is the k=2 safety substrate — the
  two share the accept-recipe work).

**9. Implementation instructions.** Sources: `ports/glm-5-3-flash/glm5_next_hpu/
  glm5_next_hpu/kda_layer.py` (the current chain: the conv, the gates, the state
  update — the exact op list to collapse), `mtp.py` (the verify-path KDA calls), the
  Q21 kernel files in `kb_staging/` (the build/registration pattern:
  `block_fp8_moe_*.c/.cpp` + `glue.inc`), the GaudiInfer template (Tier D, cited;
  algorithm only, no code import). Sequence: (1) D15 census + the KDA-chain op list
  per layer (no HPU time); (2) the rmat/gram GEMM microbench at our geometry (TPC
  bench, 1 day); (3) if the GEMMs are ≤ the chain's device time: the WY kernel
  (weeks, the Q21 pipeline) with the CPU reference test (the WY state vs the chunked
  state over T=1/2/16 blocks, the tolerance band); (4) the double-buffer migration
  (the runner, env-gated `GLM5_KDA_WY=0/1`); (5) the accept recipe (batched across
  slots); (6) the A/B: WY-on vs WY-off, bs=1 + @16, the suite with flip attribution,
  the state bit-compare test. Diagnostics: the per-layer trace node counts before/
  after (the census delta), the per-step replay time (the launch-timestamp patch),
  the MME/TPC utilization of the WY step (the overlap check).

**10. Validation criteria.** ADOPTED if: (i) the state bit-compare passes within the
  declared tolerance band (the band itself ≤ the 0.5 pp gate's demonstrated
  sensitivity — the I006/I007 FP-near-tie class); (ii) suite ≥ 97.5% with all flips
  attributed; (iii) the KDA node count drops ≥ 70% in the census; (iv) bs=1 ≥ 44
  tok/s over 5 runs; (v) the WY path is stable in a 1H soak (the state side-flip has
  no leak — the HBM flat across 10k steps). The microbench gate: if rmat+gram MME
  time > the chain's measured device time at T=2, the proposal is REJECTED-as-
  device-time (revisit only if the node floor is re-priced by RA-038-A T2).

**11. Novelty assessment.** Never attempted (the KDA chain has not been touched by any
  project experiment — C015 was the decay-dot op, stack-specific; the KDA work to
  date is the port itself + the KDA-scan env). It is not a repeat: the WY form is a
  different algorithm for the same recurrence (the chunked delta rule is the naive
  sequential form; WY is the stable factored form — the difference is the same class
  as "GEMM vs GEMV" in RA-002). GaudiInfer's production use on the same op class is
  the feasibility evidence (Tier D); the geometry is ours and the GLM gate absorption
  is new work.

---

## RA-041 — KDA prefix cache via block-page state checkpoints (the cookbook fix, GLM-sized)

**1. Executive summary.** Prefix caching is OFF in our port by an architectural
  constraint: the engine refuses to start with it for KDA hybrids because vLLM's GDN
  path has no state-restore mechanism — a prefix hit resumes the GDN layers from the
  wrong state (the state is per-request recurrent, not per-token KV). Our upstream
  cookbook **has the fix** (`fixes/gdn-prefix-cache`, Tier D, targets v0.26.0 — our
  version): GDN state checkpoints stored at the end of each full KV block, into that
  block's own attention page, with per-request compact slots (`VLLM_COMPACT_GDN`); on a
  hit the checkpoint is restored before the next chunk. The fix targets Qwen3.5/3.6/
  3.8's GDN (gated delta net — the same family as our KDA), and upstream vllm-gaudi is
  actively fixing the GDN prefix-cache area in v0.30.0 (#1827, #1842, #1815, #1753).
  The GLM-specific delta this round identified: **our KDA state is ~2 MB/layer
  (64×128×128 bf16) → ~70-90 MB per checkpoint across 34 layers**, while a 640-token
  hybrid block's KV (11 DSA layers × 512 latent + indexer) is ~7-14 MB — so the
  cookbook's "checkpoint every block" lattice is **5-10× too dense for GLM** and must
  become a **sparse lattice with replay** (checkpoints every N blocks; a hit replays
  N×640 tokens of KDA from the last checkpoint — deterministic, so bit-exact).

**2. Current limitation.** `--no-enable-prefix-caching` is hardcoded in
  `serve_mtp.sh` (V-7, verified). The 1M validation workloads and any multi-turn/agent
  serving re-prefill the full context every request (the 1M prefill is 1,643 t/s deep
  — a 90K-prefix agent turn costs ~55 s of re-prefill at our rate vs the ~285 ms
  class that both Wright and GaudiInfer achieve with host-LRU prefix caches, Tier D).
  The prefill *throughput* objective (3-6K t/s) is met for cold prefills; the
  *conversational* TTFT is the unmet surface, and it is the one where the owner's
  production workloads (multi-turn, tool calls) actually live.

**3. Proposed architecture.**
- **The port (phase 1):** apply the cookbook `gdn-prefix-cache` fix to our port's KDA
  path. Surface: (i) the engine-start constraint — the fix's Qwen target removes the
  "GDN hybrid + prefix caching = refuse" gate; our port's KDA layer registration must
  present the same compact-GDN contract (`VLLM_COMPACT_GDN=1`); (ii) the checkpoint
  write — at the end of each full KV block, serialize the KDA state (34 × [64,128,128]
  bf16 + the conv windows + **the mHC per-layer state if it is recurrent — verify:
  the hc mixing is per-token permutation mixing, NOT recurrent; the state is the KDA
  state + conv only** [INFERENCE — source-verify in `mhc.py`/`kda_layer.py`, no HPU
  time]) into the block's attention page (the block page must be sized to hold it —
  this changes the hybrid block geometry: the page grows from ~7-14 MB to ~80-105 MB
  at N=1, or the checkpoint goes to a **sidecar region** keyed by block hash — the
  design choice below); (iii) the restore — on a prefix hit, load the checkpoint into
  the per-request KDA state before resuming the chunk (an H2D/D2D copy of ~70-90 MB —
  at the measured 20.7-24.4 GB/s host→HPU (RJ-05's number) that is ~3-4 ms per hit
  from host, or sub-ms if the checkpoint lives in the HBM pool).
- **The lattice (the GLM delta):** checkpoint spacing N: at N=1 the pool cost is
  ~90-105 MB/block × 1280 blocks = 115-134 GB — **does not fit** (the pool is ~76.5
  GiB weights + KV at TP4; even at TP8 the 25 GiB workspace block is the constraint).
  At N=4 (one checkpoint per 2,560 tokens): 1280/4 = 320 checkpoints × 95 MB ≈ 30 GB
  at TP8 — still heavy but budgetable against a reduced KV pool (the trade: 320
  checkpoint slots vs 1280 KV slots of the non-checkpointed blocks — the effective
  context capacity drops ~25% at N=4); at N=8: 160 × 95 MB ≈ 15 GB, ~12.5% capacity
  cost, replay up to 5,120 tokens on a hit (~3 s of KDA compute at our decode rate —
  the replay cost is the price of the sparsity). **The N=4-8 lattice with the replay
  is the design** [DERIVED — D18 converts the state bytes to MEASURED and the A/B
  prices N]. The replay is the KDA chunked path (already in the tree) over the
  unchecked tokens — deterministic → bit-exact with the no-replay computation (the
  DSA layers' KV is restored from blocks as today; only the KDA state needs the
  lattice).
- **The hit semantics:** a prefix hit of length L restores (i) the DSA KV blocks for
  [0, L_block) (existing paged-cache machinery), (ii) the KDA checkpoint at the last
  lattice point ≤ L_block, (iii) replays (L − L_lattice) tokens through the KDA path
  (the DSA layers compute attention over the restored blocks for those tokens — the
  normal prefill path, just shorter). The MTP layer's state (layer 45 is DSA — no
  recurrent state; `index_share_for_mtp_iteration` shares the indexer) needs no
  checkpoint.

**4. Technical feasibility.** The mechanism is proven on the same op family by the
  cookbook fix (Tier D, v0.26.0 target — our version) and by upstream's active v0.30.0
  fix stream (the GDN prefix-cache PRs). The two production prefix-cache systems
  surveyed this round (rimb05's vLLM-based deployment with the cookbook's gdn-prefix-
  cache fix active on the GLM-5.3 744B; GaudiInfer's host-LRU with state snapshots,
  285 ms 90K agent turn) independently validate the pattern on Gaudi 2. The lattice
  generalization is new engineering but arithmetic (checkpoint budget = pool math;
  replay cost = the existing chunked path). **Unsupported assumptions:** (i) that our
  KDA layer goes through the same compact-GDN machinery the fix patches (source
  verification — `kda_layer.py` vs the cookbook's Qwen GDN path; if our port's KDA is
  structurally different — the mHC wrapper, the fused_qkvbfg projections — the fix
  needs adaptation, not application); (ii) that the engine-start constraint is
  *only* the missing state-restore (the constraint's code path must be read — no HPU
  time); (iii) the mHC statelessness (V-4: `hc_mult 4`, Sinkhorn mixing per token —
  believed non-recurrent; verify).

**5. Expected performance impact.** Conversational/agent workloads (the production
  surface): a 90K-prefix turn re-prefills 90K − hit tokens. At hit ratio 90% (a
  typical multi-turn): prefill 1,643 t/s × 9K tokens ≈ 5.5 s → **~0.5-1 s with the
  lattice hit (checkpoint load ~3-4 ms + replay ≤ 2,560 tokens ≈ 1.5-3 s at N=4...
  correction: the replay dominates at N=4-8** — the honest band: a 90K turn at 90% hit
  with N=4: 5.5 s → 2-4 s (−30-65%) [DERIVED, wide — the hit pattern is
  workload-dependent]; with N=8: → 4-5 s (−10-25%). Cold prefills: unchanged (the
  objective stays met). Decode throughput: unchanged (different path). Capacity:
  −12.5% (N=8) to −25% (N=4) effective context pool — **the 4-HPU math must absorb
  this** (interacts with RA-042: the int8 latent KV's 2× buys back the lattice cost —
  the two are designed to be adopted together at 4 HPUs). TTFT on first-of-session:
  unchanged.

**6. Implementation complexity.** Medium-High (weeks): the fix port/adaptation (the
  KDA path surface), the sidecar checkpoint region + the allocator, the lattice
  scheduler (the N parameter, the replay trigger), the restore path (checkpoint load +
  resume), the engine-gate removal. The replay reuses the existing chunked KDA path —
  no new numerics.

**7. Risks.** Accuracy: the restore must be bit-exact (the checkpoint is a copy of the
  state the forward would have produced — exact); the replay is the same ops as
  prefill — exact; the only numerics surface is the DSA attention summation order over
  restored blocks vs freshly-computed blocks (the V-21 near-tie class — the suite's
  flip-attribution protocol). HBM: the lattice (above) — at 4 HPUs the N=4 lattice +
  int8 KV is the combination to test; if the pool math fails at 4 HPUs, the lattice
  degrades to N=16 (replay ≤ 10K tokens) or the sidecar spills to host (the 3-4 ms
  load becomes ~40 ms — still better than a 90K re-prefill). Stability: the checkpoint
  write at block-completion is a new per-block device op (the HBM write ~95 MB —
  bandwidth-trivial at prefill rates, but the capture's node count grows by the
  checkpoint op — measure in the census, D15). The engine-gate removal is a
  vendor-code touch (the fix does it for Qwen; our port's gate location may differ).

**8. Dependencies.** D18 (the state bytes + the block bytes — source read + arithmetic,
  no HPU time), the source verification of (i)-(iii) above (no HPU time), the
  allocator's sidecar region (new), RA-042 for the 4-HPU combination, RA-043's
  deferred-commit for a *cleaner* checkpoint source (the explicit state side makes the
  checkpoint a pointer copy — sequence RA-041 after or parallel to RA-043's state
  migration; they share the state-tensor work).

**9. Implementation instructions.** Sources: `fixes/gdn-prefix-cache/` in our fork
  (the README + apply.py — read the exact patch surface first; it is in
  `publish_work/fork/fixes/gdn-prefix-cache/` on the server), `kda_layer.py` (our KDA
  state: the shape, the write path, the mHC interaction), the engine's prefix-cache
  gate (the startup refusal — locate the check that reads the model's GDN/KDA
  flag + prefix-caching), the KV allocator (the block page geometry, the sidecar
  option). Sequence: (1) the source-verification pass (D18 + (i)-(iii), no HPU time,
  ~1 day); (2) the fix port (the Qwen GDN patch surface mapped onto our KDA — the
  compact-slot contract, the checkpoint write/restore); (3) the lattice (the N
  parameter, the sidecar region, the replay trigger); (4) one boot at TP8: the
  multi-turn A/B (a 3-turn conversation, the re-prefill TTFT per turn, with/without),
  the suite (the near-tie protocol), the pool math recorded (the effective capacity at
  N=4/N=8); (5) the 4-HPU boot (with RA-042 if landed) for the production config.
  Diagnostics: the hit/miss log (the block-hash chain), the checkpoint-load latency,
  the replay token count per hit, the HBM census before/after.

**10. Validation criteria.** ADOPTED if: (i) the engine starts with prefix caching ON
  (the gate removal verified); (ii) the bit-exactness test: a 32K prompt prefilled
  cold vs a 32K prompt with a 30K hit + replay — identical logits (or flips attributed
  per the protocol); (iii) the multi-turn A/B: turn-2/3 TTFT ≥ 30% below cold re-
  prefill at N=4 (the −30% floor of the band above); (iv) suite ≥ 97.5%; (v) the
  effective pool capacity recorded at N=4/N=8 (the 4-HPU decision input); (vi) 1H
  soak with mixed hit/miss load clean.

**11. Novelty assessment.** Prefix caching has been *refused* in the project (the
  architectural constraint), not attempted — so this is not a repeat. The cookbook
  fix is upstream of us (our own repo's fixes directory) and targets our exact vLLM
  version; what is new is (a) the GLM KDA adaptation (the state-size delta → the
  lattice, which the cookbook fix does not address — their Qwen GDN state is small
  enough for every-block checkpoints) and (b) the mHC/DSA/KDA combined hit semantics.
  The 1Cat/abotsis/strata sources have no GLM-KDA prefix-cache (the DFlash2 feedback
  doc's B1 blocker was the *drafter's* GDN ops, a different surface).

---

## RA-042 — INT8 per-group latent KV (the 4-HPU fit + the 1M headroom)

**1. Executive summary.** Our KV cache is bf16 (V-2: no `--kv-cache-dtype` in the serve
  script). rimb05's production GLM-5.3 deployment (Tier D) stores the MLA latent as
  **INT8 with per-64-dim-group scales** (`LatentInt8Cache` in their `dsa.py`, v0.30.0):
  their 576-dim latent (512 + 64 rope) is 9 groups × 64 dims, each group quantized
  with a bf16 scale `max|x|/127` — measured **0.6% error per latent part, 2× KV
  capacity, reads *faster* than bf16 (31 vs 40 µs/sparse layer), writes +3-4 ms/decode
  step** (8-10% slower at the same context). Their failure modes are documented and
  informative: a single scale per token **breaks generation** (the rotary part runs
  16-100× larger than the compressed part in the first ~10 layers), and E4M3 codes
  give 2.5-2.9% error (4× worse than the int8 per-group). Our geometry (V-4): latent
  = **512 dims, no rope part** (`qk_rope_head_dim 0`) → **8 groups × 64** — the same
  scheme, simpler (no rotary asymmetry to manage, though the first-layer sensitivity
  finding still applies to the 512-dim compressed latent). This is the BG-7 answer for
  the 4-HPU production fit: at TP4 the KV pool halves; int8 latent KV doubles it back —
  same context at 4 cards as at 8, with a 0-10% decode tax.

**2. Current limitation.** KV pool: 1,280 hybrid blocks (V-7) at 640 tokens/block (the
  EXP-Q10 figure; D8 confirms 128-vs-640) = 819,200 tokens at 8 HPUs — 1M requires the
  separate 1M boot's larger override, and 1M multi-seq does not fit at all. At TP4 the
  pool is ~640 blocks (Q14v2 regime) → 409,600 tokens → a 1M single request does not
  fit at 4 HPUs in bf16 KV (BG-7's open question: "how does 1M fit on 4 cards?").
  The decode cost of KV is small (the weight floor dominates — RJ-06/RJ-10 logic), so
  KV quantization is a **capacity** play, not a speed play — the int8 write tax is the
  only cost.

**3. Proposed architecture.**
- **Scheme A (primary — int8 per-group latent, the rimb05 port):** the DSA layers'
  latent (512 dims) stored per token as 8 × int8(64) + 8 × bf16 scale (512 + 16 = 528
  bytes vs 1,024 bf16 — **1.94× capacity**). Write path: the latent quantization op
  (per-group max → scale → int8 codes) inserted at the KV write (the +3-4 ms/step tax
  at 11 DSA layers — [Tier D, their 22-layer model; ours 11 → ~1.5-2 ms/step
  DERIVED]). Read path: dequant in the attention read (the GEMM-side dequant — their
  measurement says reads get *faster* (int8 GEMM + dequant < bf16 GEMM at their
  shapes); at our shapes [DERIVED: the 512-dim latent GEMM is MME-friendly int8] —
  assume neutral-to-faster, measure). The indexer keys (32 × 128 dims = 4,096 per
  token): **keep bf16** (they are the selection input — quantizing the indexer input
  changes the top-2048 selection = an accuracy risk with zero capacity benefit, since
  the indexer keys are 1/8 of the latent bytes... correction: the indexer keys are 4,096
  elements vs the latent's 512 — the indexer is the *larger* store! 4,096 × 2B = 8 KB
  per token vs the latent's 1 KB per token. [Re-derive: the per-token DSA store =
  latent 512 × 2B = 1 KB + indexer keys 4,096 × 2B = 8 KB = 9 KB/token/layer × 11
  layers = 99 KB/token. Quantizing only the latent: 99 → 91 KB (−8%). Quantizing the
  indexer keys too (int8 per-group, 32 groups × 128): 99 → ~22 KB (−78%, 4.5×
  capacity) — but the indexer-quantization accuracy risk is real (the selection
  itself shifts). **Two-tier design:** latent int8 (safe, −8%) + indexer int8
  (aggressive, −78% total) as a second A/B arm with the 1M needle as the gate.]
- **Scheme B (fallback — `--kv-cache-dtype fp8_inc`):** the vendor-documented path
  (V-16, Tier B) — E4M3, unit or per-head scales. Simpler (a flag, not a port) but the
  2.5-2.9% error class (rimb05's measurement on the same model family) — the needle-
  sensitive choice. This is the RJ-10 rejection's original subject (the abotsis same-
  model needle report) — see novelty.
- **Scheme C (hybrid precision, the accuracy hedge):** the first 2 DSA layers (3, 7)
  bf16 + the rest int8 (rimb05's first-layer sensitivity finding, scaled to our 11
  layers): ~1.8× capacity instead of 1.94×, the accuracy risk concentrated where the
  model is most sensitive is removed.
- **The 4-HPU combination:** TP4 + Scheme A (latent+indexer int8): the pool 640
  blocks → ~2,900 effective (4.5× on the 8 KB indexer share... [careful: 4.5× applies
  to the per-token store; the block count scales by the store ratio 99→22 ≈ 4.5×] →
  640 × 4.5 ≈ 2,880 blocks ≈ 1.84M tokens — **1M fits 1-2 requests at 4 HPUs** [DERIVED,
  D18-adjacent arithmetic to confirm with the real block geometry]). The decode tax
  ~1.5-2 ms/step (the write) at bs=1 → −2-3% on 42.4-class [DERIVED].

**4. Technical feasibility.** Scheme A: proven in production on the same model family
  and ASIC (rimb05, Tier D — the 744B sibling runs it at 1,047,552-token context).
  The port surface: the KV write path in our OOT MLA backend (`vllm_gaudi/attention/
  oot_mla.py` — the corpus shows the bf16 write: `latent_vec_k = torch.concat((
  k_c_normed, k_pe...))` written to the paged cache; the quant op inserts before the
  store) + the read path dequant + the scale storage in the block page (the page
  geometry grows by 16 bytes/token — trivial). **The open question (D20):** whether
  our v0.26.0 OOT MLA backend accepts a non-bf16 cache dtype at all (the corpus
  search surfaced the NVIDIA-side `QUANTIZED_DS_MLA_CACHE_FORMATS` path — the Gaudi
  OOT backend's dtype handling is the code to read; if it hardcodes bf16, Scheme A is
  a small backend patch, Scheme B may be blocked at our version → check v0.30.0's
  oot_mla for the fp8 path as the reference). Scheme B: vendor-documented (Tier B) but
  the oot_mla support question applies equally. The int8 GEMM read path: the MME int8
  dot is a first-class Gaudi 2 op (the W4A8/INT8 class — the Q21 context).

**5. Expected performance impact.** Capacity (the primary): 1.94× (latent only) to
  4.5× (latent + indexer) per-token store reduction → at TP4: 409,600 tokens →
  795K-1.84M tokens [DERIVED, D18-adjacent] → **the BG-7 4-HPU 1M fit is solved**
  (1M single request fits at TP4 with the indexer-int8 arm; 2 concurrent 1M-class
  requests with the latent-only arm at 8 HPUs). Decode: −0-3% (the write tax; the read
  is neutral-to-faster per Tier D). Prefill: the KV write is a small share of prefill
  (the chunked prefill is compute-bound) — expected neutral [DERIVED — measure in the
  D17 window]. The 1M needle: the accuracy gate (below).

**6. Implementation complexity.** Medium (Scheme A latent-only: the quant/dequant ops
  + the scale storage + the A/B, days-week; the indexer-int8 arm: +the selection-
  stability study, a week; Scheme B: the flag + the D20 check, days). The port is
  smaller than RA-041/RA-043 (no state machinery, no new kernels — the quant is
  elementwise + a per-group reduce).

**7. Risks.** Accuracy (the whole proposal): the int8 per-group error through the MLA
  latent amplifies per-element (the latent is a 512-dim compression — errors are
  shared across all 64 heads that read it). The gate: the 1M needle at 860,739 (Q12's
  position), the suite, the I009 long-context stability — **the needle is the
  arbiter** (the RJ-10 rejection was exactly on needle behavior with E4M3; the int8
  per-group's 4× lower error is the material difference, but the gate is the gate).
  The first-DSA-layer sensitivity (rimb05's 16-100× dynamic range finding) → Scheme C
  as the hedge arm. The indexer-int8 arm: the top-2048 selection can shift (a token
  selected under bf16 keys is missed under int8 keys) — a *systematic* attention
  change, not a perturbation; the needle + a selection-agreement metric (the % of
  selected keys identical bf16 vs int8 at 1M — a cheap diagnostic) gate it. HBM: the
  scale storage (16 B/token/layer — 0.16 KB/token × 11 = 1.8 KB/token vs the 99 KB
  store — 1.8% overhead, trivial).

**8. Dependencies.** D20 (the oot_mla dtype support — code read, no HPU time), D18
  (the block geometry arithmetic), the A/B infrastructure (the existing same-boot
  protocol). RA-041 (the lattice HBM budget interacts at 4 HPUs), RA-047 (the 1M
  decode window is the shared measurement).

**9. Implementation instructions.** Sources: `vllm_gaudi/attention/oot_mla.py` (the
  KV write — the corpus chunk shows `latent_vec_k` concat + the paged store; the
  quant inserts here), the KV cache spec (the block page layout, the
  `cache_dtype` plumbing in `vllm_gaudi/v1/...` per the vendor fp8_inc path), the
  rimb05 reference (Tier D, `dsa.py` `LatentInt8Cache` — the group-scale scheme, the
  per-layer sensitivity data). Sequence: (1) D20 (code read); (2) Scheme A latent-
  only: the quant op (the per-group max reduce — a TPC op or an in-graph op; the Q21
  pipeline if a kernel is needed) + the dequant read + the scale sidecar in the block
  page; (3) the A/B at TP8 (the needle + the suite + the decode rate), then the TP4
  boot (the pool math recorded: the effective token capacity); (4) the indexer-int8
  arm (the selection-agreement diagnostic first — if the agreement < 99% at 1M, the
  arm is rejected before the needle); (5) Scheme C if the needle is marginal on the
  full-int8 arms; (6) Scheme B only if (2) hits a wall (the oot_mla dtype support).
  Diagnostics: the per-group scale distribution per layer (the first-layer sensitivity
  check — if layer 3's groups need the bf16 hedge, Scheme C is chosen by data, not
  speculation), the int8-vs-bf16 latent error per layer (the 0.6% reference), the
  selection-agreement %, the decode step time delta (the write tax).

**10. Validation criteria.** ADOPTED (latent-only arm) if: (i) the 1M needle at
  860,739: 100% (parity with Q12); (ii) suite ≥ 97.5% (flips attributed); (iii) I009
  long-context stability clean (the 32K non-MTP + the MTP 262K windows); (iv) decode
  bs=1 within −3% of the bf16 reference; (v) the TP4 capacity recorded (the BG-7
  answer: 1M single-request fit yes/no + the concurrent count). The indexer-int8 arm:
  additionally the selection agreement ≥ 99% at 1M AND the needle 100%. REJECTED if
  the needle fails on all int8 arms (the E4M3 fallback is then the only capacity play,
  and BG-7 reverts to the ECF8/window options per the rev-14 note).

**11. Novelty assessment.** KV quantization was rejected once (RJ-10: the E4M3
  fp8-kvcache, the abotsis same-model needle report, rev 4). This is a materially
  different hypothesis: (i) the quantization is int8 per-group with per-64-dim bf16
  scales — rimb05 measured 0.6% vs E4M3's 2.5-2.9% on the same model family (4× the
  error headroom); (ii) it is a capacity play (the decode cost is a 0-3% tax), not the
  speed play RJ-10 analyzed; (iii) the needle gate is carried over *as the arbiter* —
  the rejection condition stands, the error class changed. It is not a re-test of
  RJ-10's exact configuration.

---

## RA-047 — DSA v2 sparse decode + the unmeasured 1M decode rate (the 1M workstream)

**1. Executive summary.** The 1M context objective is *validated* (the Q12 needle at
  860,739, the 1,643 t/s deep prefill) but the **1M sustained decode rate has never
  been measured** — a gap in the record (D17). The 1M decode is structurally different
  from the 32K decode: every DSA layer's indexer scans the full context (32 index
  heads × 128 dims over up to 1M tokens = the scan cost scales linearly with context)
  and the latent KV read spans 1M tokens (the top-2048 selection reads 2,048 keys, but
  the *selection itself* touches the whole k-pool). rimb05 (Tier D) runs the GLM-5.3
  family at 1M with **DSA v2 sparse decode** (`GLM_DSA_SPARSE=1`, `GLM_DSA_V2=1`: the
  decode attention reads only the threshold-selected keys — 4,096 at their setting —
  plus a "scan stack" optimization they measured at 8-9 ms/step saved at 700K context
  with 22 indexer layers). We have 11 DSA layers → the same class of saving at ~half
  the scale [DERIVED: ~4-5 ms/step at 700K-1M]. This proposal is the 1M workstream:
  measure the 1M decode first (D17), then port the sparse-decode path if the
  indexer/scan share justifies it.

**2. Current limitation.** Unknown (the gap). Bounds [DERIVED]: at 1M the per-step
  indexer k-pool read is ~512 MB/DSA layer (32 × 128 × 2B × 1M... [careful: the
  k-pool is compressed — `index_kpool 4` = 4×128 keys per token per the exec summary's
  "4×128 compressed index keys per token" → 512 B/token/layer → 512 MB/layer at 1M]
  × 11 layers = 5.6 GB/step at TP8 → 0.7 GB/card/step ≈ 0.35 ms at 2 TB/s — **the
  indexer scan is NOT the bottleneck even at 1M** [DERIVED, the RJ-06 logic
  re-applied at 1M: the weight floor (41.9 GB/step ÷ 8 = 5.2 GB/card ≈ 2.6 ms at 2
  TB/s... the settled bs=1 device time is 27.3 ms, so the real per-card share is
  higher) dominates]. The DSA latent read at 1M: the top-2048 selection reads 2,048 ×
  512 × 2B = 2 MB/layer — trivial. **So the 1M decode may be only ~10-30% slower than
  the 32K decode** [DERIVED, wide band — D17 settles it]. If so, the DSA v2 port is
  *not* the 1M lever (the weight floor is, i.e. RA-002/RA-043), and this proposal
  closes as a measurement + a rejection. If the measured 1M decode is > 40% below the
  32K rate, the scan/selection path is the lever and the v2 port is justified.

**3. Proposed architecture.**
- **D17 (the gate, first):** one 1M boot window (the existing `serve_mtp_1m.sh`):
  the sustained decode rate at 1M context (bs=1, 2K generated tokens after the 1M
  prefill — the needle harness already reaches this state), the per-part breakdown
  (the launch-timestamp patch + the per-layer trace: the indexer scan time, the
  selection time, the attention read time, the KDA share), the HBM census (the 1M KV
  footprint at the 1M boot's block override). This converts "1M validated" into "1M
  validated + quantified".
- **If the scan/selection share ≥ 20% of the 1M step (the port):** the DSA v2 sparse
  decode from rimb05's v0.30.0 `dsa.py` (Tier D): the decode attention's key read is
  restricted to the threshold-selected keys (their 4,096 threshold vs our top-2048
  selection — the v2 path changes *how* the selection is computed incrementally
  (the "scan stack": the per-chunk partial selections accumulated, so the per-token
  scan is O(chunk) not O(context)) + the `index_kpool_always_select_tail: true`
  semantics (the tail keys are always selected — our config already has this field).
  Port surface: v0.30.0 `dsa.py` → our v0.26.0 OOT MLA backend (the medium port, the
  same class as RA-042's Scheme A port — the two can share the backend work window).

**4. Technical feasibility.** The measurement (D17) is trivially feasible (an existing
  boot script + the existing harness). The port: proven on the same model family at
  1M (rimb05, Tier D — their 1M decode numbers: 50-55 tok/s at 1M vs 70-77 at 2K,
  i.e. their 1M decode is −20-35% of the short-context rate, WITH the v2 path active —
  a useful prior: even with sparse decode, 1M costs 20-35% [Tier D, their 2-node
  PP2×TP8 regime — different from our single-node TP8, so the transfer is qualitative]).
  The v0.30.0→v0.26.0 delta is the port risk (the dsa.py surface is large; the
  cookbook's 14-patch GLM-5 port shows the version-gap effort for this model family).

**5. Expected performance impact.** D17 alone: no performance change (the record
  completes — the 1M decode rate becomes a MEASURED fact for the 4-HPU/1M planning).
  If the port is justified: the 1M decode +X% where X is the scan/selection share
  recovered (the band from their data: 20-35% of the 1M step is the context-scaling
  term; the v2 path removes most of the *scan* part, not the selection's inherent
  O(context) first pass) → **1M decode within ~10-15% of the 32K rate** [DERIVED, wide]
  vs the current unknown. At the 4-HPU 1M combination (with RA-042's capacity): the
  1M single-request production config becomes a measured, bounded-slowness config
  instead of an extrapolation.

**6. Implementation complexity.** D17: Low (one boot window). The port: Medium-High
  (weeks — the dsa.py surface, the v0.30.0→v0.26.0 delta, the A/B at 1M).

**7. Risks.** The port's accuracy: the incremental selection changes *which* keys the
  attention reads (the threshold vs the exact top-2048 — a selection-quality change;
  the gate: the 1M needle + the selection-agreement diagnostic from RA-042). The
  version delta (v0.30.0→v0.26.0): the cookbook port's 14 patches are the effort
  prior; the dsa.py-specific delta is unmeasured. If D17 shows the 1M decode is
  already within 15% of 32K (the weight-floor-dominated case), the port is rejected
  and the 1M workstream is "RA-002/RA-043 benefit 1M proportionally" — a fine outcome.

**8. Dependencies.** D17 (the gate), the 1M boot (the existing script), RA-042 (the
  shared backend work + the 4-HPU 1M capacity), the rimb05 v0.30.0 `dsa.py` as the
  reference (Tier D — read, not import, unless the port justifies a patch-level
  cherry-pick with attribution).

**9. Implementation instructions.** (1) D17: `serve_mtp_1m.sh` boot, the needle
  harness to the 1M state, the 2K decode window, the per-part breakdown (the launch-
  timestamp patch + the per-layer trace over the decode window), the HBM census;
  record as EXP-1M1. (2) The decision: the scan/selection share vs the 20% threshold.
  (3) If port: the dsa.py surface diff (v0.30.0 vs our oot_mla — the scout's report
  has the file map), the port (the sparse decode + the scan stack), the A/B at 1M
  (the rate, the needle, the selection agreement), the 4-HPU 1M combination (with
  RA-042).

**10. Validation criteria.** D17: the 1M decode rate recorded (MEASURED, the per-part
  table). The port: ADOPTED if the 1M decode improves ≥ 15% AND the needle 100% AND
  the selection agreement ≥ 99%; REJECTED if D17 shows the 1M step is already
  weight-floor-dominated (scan/selection < 20%) — the rejection is a *confirmation*,
  recorded as such.

**11. Novelty assessment.** Never measured (the 1M decode rate is a genuine gap in the
  project record — the 1M work to date is prefill + needle). The DSA v2 port is new
  (no prior project experiment touches the decode attention path — the attention work
  has been the MTP verify + the indexer config). It is explicitly gated: the
  measurement decides, and the rejection path is a designed outcome.

---

## RA-044 — Q21 kernel build + design addendum (the g2-opt-manual measurements + the GaudiInfer negative results)

**1. Executive summary.** An addendum to the in-progress Q21 (RA-002) — no new lever,
  three quality/risk items from this round's surveys that the kernel build should
  absorb: **(a) the compiler flags** the g2-opt-manual (Tier D) measured for TPC
  kernels: `--mllvm --preRA-schedule=llvutd` (a hidden LLVM option that moves the
  kernel off the TPC bandwidth cliff — their measurement: the default scheduling
  pattern hits a 5× issue-rate cliff at certain instruction mixes; the flag removes
  it) and the per-launch byte discipline (their 6-step MoE materialization: the
  routing-diversity step moves from 0.1 → 0.4 ms as the per-launch payload crosses
  the cliff — the same physics as the 16 MB tile rule already in RA-002's D4 sweep,
  applied to the *launch* dimension, not just the tile). **(b) the poison-byte
  contract** (the fp8 e4m3fn code space on Gaudi 2: bytes 0x78-0x7E/0xF8-0xFE decode
  to ±inf/NaN under the device convention; **240 is the max finite value** (the top
  exponent is non-finite in the Gaudi 2 MME convention); the CPU-side torch reads
  0x7E as 448 — a convention mismatch that bit us once in the E4M3-TABLE probe
  (11-15); the contract: the load-time assert `scale × max|w| ≤ 240`, saturation at
  240 never 448, and the subnormal/negative-zero byte rule characterized before the
  kernel ships). **(c) the measurement discipline** from GaudiInfer's negative results
  (Tier D): fusing small ops into one TPC kernel **can lose** to the compiler-
  overlapped graph-op chain (their `G2E_GATED_KERNEL`: the fused gate kernel was 0.3 ms
  *slower* per 8K verify — "the graph's small ops overlap their neighbours, the
  kernels serialise"); the Q21 kernel must be validated by **both** node count and
  device time (a node-count win that loses device time is a regression — the RA-002
  step-0 gather's −45% is the standing example), and the fused kernel should be
  structured for the **overlap law** (MME∥TPC within a single submission when there is
  no dataflow edge and the durations are comparable — the g2-opt-manual's 2-row ×
  32-row recipe anatomy, Tier D): the kernel's internal phases should expose
  MME/TPC overlap potential rather than serialize everything on TPC.

**2. Current limitation.** Q21 is in progress (the PT2 custom-op adaptation staged;
  the D4 tile sweep is the first kernel gate). The build has not yet absorbed (a)-(c)
  — they are ~2 days of the agent's time and de-risk the largest remaining kernel.

**3. Proposed architecture.** Not an architecture — three build directives: (a) add
  `--mllvm --preRA-schedule=llvutd` to the tpc-clang flags for the Q21 kernel (A/B it
  in the D4 microbench: the flag on/off, the achieved GB/s at the 25.2 MB tile — if
  the flag does nothing on our geometry, drop it and record why); (b) the poison-byte
  contract as a load-time check in the kernel's init (the assert + the saturation
  behavior documented in the kernel's header comment; the E4M3-TABLE probe's finding
  — "the kernel and stock cast agree bit-for-bit across all 256 bytes" (11-15,
  MEASURED) — is the evidence the convention is already consistent; the addendum
  makes it an *enforced* invariant); (c) the dual-metric gate in the Q21 validation
  (the node census delta AND the device-time delta, both must be wins; the overlap-
  law structure note in the kernel's design comment).

**4. Technical feasibility.** All three are measured facts from the surveys (Tier D,
  cited) or project measurements (the E4M3-TABLE probe). The llvutd flag's effect on
  our kernel is the one open question — the A/B answers it in the existing D4 window.

**5. Expected performance impact.** (a): 0-5× on the kernel's achieved GB/s if it hits
  the cliff (the g2-opt-manual's measured range for the flag; our kernel may not hit
  the cliff — the A/B decides). (b): 0 (a correctness invariant, not a speed item —
  it prevents a silent-poison failure class). (c): 0 direct (a discipline; it
  prevents the G2E_GATED_KERNEL-class regression of shipping a node-count win that
  loses device time).

**6. Implementation complexity.** Low (2 days inside the existing Q21 work).

**7. Risks.** None material. The llvutd flag could in principle change codegen in
  unexpected ways — the parity check (the kernel's output vs the reference, the
  existing Q21 test) covers it.

**8. Dependencies.** The Q21 build (the D4 microbench window).

**9. Implementation instructions.** (1) D4 microbench: add the llvutd A/B arm (the
  flag on/off at each tile size 24/8/3/1 TPC — the existing sweep matrix + 1 flag);
  (2) the load-time assert (the Q21 kernel's init: `scale × max|w| ≤ 240` per the
  fp8_143 convention; the 240 cap, the saturation behavior); (3) the design comment
  (the poison-byte contract + the overlap-law note + the dual-metric gate, in the
  kernel header); (4) the validation protocol updated: the node census + the device
  time, both reported, both must improve.

**10. Validation criteria.** (a) MEASURED (the flag's GB/s effect recorded, adopted or
  dropped); (b) the assert fires on a deliberate poison test (a synthetic 0x7E input
  → the assert/saturation, not a silent inf); (c) the protocol update committed (the
  dual-metric gate in the Q21 test plan).

**11. Novelty assessment.** New this round (the g2-opt-manual and GaudiInfer were
  surveyed 2026-11-20/21). Not a repeat — it is an addendum to the in-progress work,
  absorbing external measurements.

---

## RA-046 — Production-config env A/B sweep (the rimb05 v0.30.0 config delta)

**1. Executive summary.** Five config-level items from rimb05's production GLM-5.3
  deployment (Tier D) that differ from our serve script (V-7) and are cheap to A/B
  (config-only, same-boot, bit-exact — no arithmetic changes):
  **(1) fastmeta** (`GLM_FAST_META=1` in their stack; the block-list management moves
  from a Python loop to numpy): their measurement is up to ~8 ms/step at @16 long
  context [Tier D] — **gated by D16** (does our v0.26.0 block-list path have the
  Python hot loop? if the v0.26.0 runner already manages the block list in numpy, the
  item is a NO-OP like RA-032). **(2) `VLLM_GRAPH_RESERVED_MEM=0.2`** (theirs; the
  stock default is 0.1 per the corpus's DSV4.1 note): sizes the graph workspace vs
  the KV pool — at our GMU 0.82 + 2,048 capture, the A/B is the HBM census delta +
  the capture success (a higher reservation can fail the capture — the rev-8 OOM
  class); worth one boot if the HBM census shows the capture is near the boundary.
  **(3) `PT_HPU_QKV_SLICE_SEQ_LEN_THLD=128`** (theirs; a bridge env for QKV slicing at
  long sequences): the 1M prefill/decode path — one A/B in the D17 window (the 1M
  boot already exists). **(4) `VLLM_HPU_FSDPA_SOFTMAX_MODE=None`** (theirs; the FSDPA
  softmax mode — check our default first; if we already run None, NO-OP).
  **(5) the 1M bucket-file merge:** rimb05 curates `VLLM_BUCKETING_FROM_FILE` shape
  files for the 1M window (their 449/559 curated shapes include decode shapes up to
  8,184 blocks); our 1M boot uses the hand-tuned 128+16 file — the diff may reveal
  decode shapes our 1M boot captures at first use (the ~30 t/s first-capture cost,
  the Q14v2 figure, paid at 1M block sizes where the capture is most expensive).

**2. Current limitation.** Config drift from the only other production GLM-5-on-Gaudi
  deployment: each item is individually small (0-8 ms/step class) but the sweep is
  free (config-only) and the items are on the critical paths we already know are
  host-bound (the block-list path is per-step host work; the bucket file is the
  first-capture cost at 1M).

**3. Proposed architecture.** Five A/B arms, each config-only, each in an existing boot
  family: (1) fastmeta (after D16); (2) the graph-reserved-mem (one boot, the HBM
  census); (3) the QKV-slice-threshold (the D17 1M window); (4) the FSDPA-softmax-mode
  (a flag check first — code read, no HPU time); (5) the 1M bucket-file merge (the
  diff + one 1M boot).

**4. Technical feasibility.** All five are production-used configs on the same model
  family/ASIC (Tier D) or vendor defaults (the corpus). The v0.30.0→v0.26.0 delta
  applies to (1) (the fastmeta patch is v0.30.0-based — D16 determines whether the
  v0.26.0 path even has the hot loop; if it does, the port is the numpy-ification of
  that loop, a small patch).

**5. Expected performance impact.** (1): 0-8 ms/step at @16 (their band; D16-gated)
  → @16 258-260 → 270-295 [DERIVED if the hot loop exists]. (2): 0 on steady state
  (HBM layout only) — the value is the capture headroom (fewer capture OOMs at the
  1M block sizes) + possibly a larger KV pool at the same GMU. (3): 0-? on the 1M
  prefill/decode (the QKV slice at 1M sequences — their setting implies a long-seq
  path we may not be taking; the D17 window measures it). (4): 0 (expected NO-OP).
  (5): the 1M first-capture cost (the ~30 t/s once-per-shape stalls — at 1M block
  sizes the capture compile is minutes-class per the Q14v2 observation; removing
  unknown shapes from the first-use path is worth the 1M boot's warmup time + any
  production 1M cold-shape stalls).

**6. Implementation complexity.** Low (config-only + one small port if D16 says yes).

**7. Risks.** Negligible (all bit-exact; the only risk is the (2) capture failure at
  the higher reservation — the boot simply fails to capture and the A/B records it).

**8. Dependencies.** D16 (fastmeta gate — the code read + one py-spy window), the
  D17 1M window (items 3+5 ride it), the HBM census tooling (existing).

**9. Implementation instructions.** (1) D16: read the v0.26.0 runner's block-list
  update path (the per-step block-table management in
  `hpu_model_runner.py` — the Python loop vs the vectorized op) + one py-spy window
  at @16 (the block-list share of the host time); (2) the flag checks (4): our FSDPA
  softmax default (code read); (3) the A/B boots: (1)+(2) in one 8-HPU boot family
  (the envs are independent — 4 arms: baseline, +fastmeta, +reserved-mem, +both);
  (3)+(5) in the D17 1M window; (4) the bucket-file diff (the rimb05 file vs our
  128+16 — the shape-list comparison, no HPU time).

**10. Validation criteria.** Per item: the A/B recorded (MEASURED), adopted if ≥ 2% on
  the target metric with the suite passing, otherwise recorded as NO-OP/neutral. The
  sweep is "done" when all five have a recorded arm.

**11. Novelty assessment.** New this round (the rimb05 config survey, 2026-11-21). The
  lazy-collectives item (the sixth candidate) was checked and is a NO-OP (RJ-26 —
  already ON per rev 11). Not a repeat of any prior A/B.

---

## RA-045 — Shared-expert multi-stream overlap probe (the upstream HPU no-op)

**1. Executive summary.** vllm-gaudi's shared-expert overlap is an **explicit no-op on
HPU** (V-6, Tier D: "On Gaudi every shape is a no-op — upstream gates the multi-stream
path on `current_platform.is_cuda_alike()`"). Our model has `n_shared_experts 1` in
all 42 sparse layers (V-4): every step, every token, the shared expert's dense FFN
(gate+up+down at 4096×2048×3 ≈ 40 MB/layer, 42 layers ≈ 1.7 GB/step total, 0.21 GB/
card at TP8) runs serially with the routed-expert path. On CUDA the vLLM design runs
the shared expert on an auxiliary stream in parallel with the routed GEMM; on HPU it
is serialized. This proposal is a **gated probe**: measure the shared-expert time
share first (D19); only if it is ≥ 3% of the step does the overlap work start.

**2. Current limitation.** The shared-expert bytes at bs=1: 1.7 GB/step ≈ 0.21 GB/card
≈ 0.1 ms/card at 2 TB/s [DERIVED — trivial at bs=1]; at @16 the same 1.7 GB/step over
a 115 ms step (1.5%) [DERIVED]. **The honest prior is that this is a 1-3% item at
best** — the shared expert is small next to the 40 GB routed read. The reason to look
anyway: (i) it is on the *critical path* today (serial), so even 1-3% is free if the
overlap is cheap; (ii) the draft passes may also pay the shared expert (the MTP layer
45 has its own `mlp.gate` — D9's composition question includes it; if the draft pays
another 1.7 GB-class pass, the overlap hides it from the draft latency too); (iii) the
multi-stream question on HPU is otherwise unexplored in the project (the overlap law
— MME∥TPC within a single submission — is the known in-graph overlap; cross-stream
overlap in a captured lazy graph is the open question this probe would answer).

**3. Proposed architecture.** D19 (the gate): one profiler window (bs=1 and @16): the
shared-expert op time per step (the fused-moe runner's shared-expert branch — the op
list in the trace) → the share of the step. If < 3%: REJECT (recorded, closed). If
≥ 3%: the overlap design — the shared-expert GEMM on a second HPU stream, launched
before the routed path, synchronized at the MoE combine (the vLLM CUDA design,
transplanted); the HPU-specific questions the probe answers: (i) does a captured
lazy HPU graph support two streams with a shared workspace (GaudiInfer deliberately
avoids multi-stream — "single stream, one shared workspace" is their discipline,
Tier D — the reason is workspace aliasing; our vLLM arena model may differ), (ii)
does the cross-stream overlap actually happen (the event timeline — the MME/TPC
contention at 24 TPCs per card may serialize the streams anyway), (iii) the collective
interaction (the shared expert is pre-MoE — no collectives inside it, so the
reordering risk is lower than RA-039's draft case).

**4. Technical feasibility.** The vLLM CUDA design is the reference (the code exists
in our tree, gated off — V-6). The HPU multi-stream question is genuinely open (no
project experiment has run two streams in a captured graph — the GaudiInfer negative
space: they tried it via `G2E_PRIVATE_WORKSPACE` and chose against it, Tier D).

**5. Expected performance impact.** If D19 ≥ 3% and the overlap works: +3-5% bs=1,
+2-3% @16 [DERIVED, the share recovered]. If D19 < 3% (the expected case): 0. The
strategic value is the multi-stream answer for the project (it informs RA-039(b)'s
design choice — single-stream ordering vs a stream).

**6. Implementation complexity.** D19: Low (one window). The overlap: Medium-High
  (the multi-stream capture, the synchronization, the workspace question — weeks if
  pursued).

**7. Risks.** The multi-stream capture is a new correctness surface (the workspace
  aliasing, the event ordering, the collective interaction — the RA-039 class of
  risks at higher complexity). The expected value is low (the 1-3% prior) — the probe
  exists to *close the question cheaply*, not to deliver a win.

**8. Dependencies.** D19 (the gate), the profiler tooling (existing), sequenced after
  RA-039(b) (the single-stream ordering pattern should be established first — if
  RA-039(b) lands, the shared-expert overlap is a natural extension of the same
  ordering discipline; if RA-039(b) is rejected on multi-stream grounds, this is
  rejected with it).

**9. Implementation instructions.** (1) D19: the profiler window (bs=1 + @16, the
  shared-expert op time in the trace — the fused-moe runner's shared branch), the
  draft-pass shared-expert question (the MTP layer's MLP composition — D9's fold-in);
  (2) the decision at the 3% threshold; (3) if pursued: the two-stream capture
  (env-gated, the CUDA design's stream split), the event timeline (the overlap
  measured, not assumed), the A/B.

**10. Validation criteria.** D19: MEASURED (the share recorded). The overlap: ADOPTED
  if the event timeline shows ≥ 50% of the shared-expert time overlapping the routed
  path AND the suite passes AND bs=1 ≥ +3%; REJECTED at D19 < 3% or the overlap
  timeline < 50% (the contention case).

**11. Novelty assessment.** New this round (the V-6 no-op discovery, 2026-11-21).
  Explicitly a probe with a designed rejection — the 1-3% prior is stated, not
  hidden.

---

## Rejections (this round)

| ID | Idea | Why rejected | What would reopen it |
|---|---|---|---|
| RJ-20 | Porting Wright's C++ engine as the *development* (8-HPU) engine | Wright is single-node 4-card-native in its GLM-5.3-Flash config; the 8-HPU development targets (200 bs=1 / 800 @16) are not claimed by it (their 8-lane data is a 27B dense, not GLM-5.3-Flash at 8 cards); the MTP=4 owner directive is on our stack. The 4-HPU *production* question is RA-048 (open, owner decision) — the rejection is of the 8-HPU-development framing only. | The owner decides (a) in RA-048 AND Wright ships an 8-card GLM-5.3-Flash config with measured numbers ≥ our 8-HPU stack. |
| RJ-21 | A DSpine-style separate trained drafter for GLM-5.3-Flash (GaudiInfer's 5-layer ring-cache drafter as a model) | No trained DSpine exists for GLM-5.3-Flash (GaudiInfer's is a separate checkpoint for their 27B, Tier D); training is out of scope (no training hardware — the owner's 11-14 ruling that parked DFlash2 applies a fortiori). The drafter question on our model is the *acceptance* question (the MTP layer at k=2-8, RA-038/RA-040), not the drafter architecture. The ring-cache + target-feature pattern is recorded as the reference architecture if a future drafter is ever trained. | A trained drafter checkpoint for GLM-5.3-Flash appears (the DFlash2 class — the 11-19 feedback's B1-B5 blockers still apply: the license ruling parked it). |
| RJ-22 | GaudiInfer's host-LRU prefix cache as a standalone C++ port | Our port cannot enable prefix caching at all (the engine constraint — RA-041 addresses the *vLLM-side* fix); a parallel C++ host-LRU would be a second cache system for a feature the engine refuses. The GaudiInfer design (KV chunks + state snapshots, the pinned pool, the background pinner) is recorded as the *reference design* inside RA-041 (the snapshot-at-lattice-point decomposition is exactly their snapshot pattern). | RA-041 fails on the vLLM-side port AND a custom serving path is commissioned (i.e. the RA-048(a) decision with a Wright-class engine that needs the prefix cache — in which case it is Wright's own cache, not a port). |
| RJ-23 | The merged cross-slot attention core (GaudiInfer `G2E_ACORE_MERGED`) | Their own measurement: merging all slots' attention cores into one launch was **not faster** at 13K contexts — "per-slot K/V reads are the cost, padding to a common bucket reads more" (Tier D). Same physics as our pad-bucketing refutation (rev 14: the pad strategy generated 384 buckets, the 11-14 refutation) — the padding-to-common-bucket cost model is now doubly refuted. Do not import into the MLA verify path. | A measured case where the per-slot read cost dominates the padding cost at our shapes (i.e. very unequal context lengths in one batch — the bucketing regime the 11-14 refutation already priced). |
| RJ-24 | Changing the IOMMU configuration on our node | **Verified already safe (2026-11-21):** our node runs `iommu=pt intel_iommu=on` (cmdline) with all Gaudi devices in the identity domain (`/sys/bus/pci/devices/*/iommu_group/type`) — the 1.24.1 host-resident-PMMU panic (JChen single-card-driver, 7 captured panics, mechanism verified against the Tier B driver source `gaudi2.c prop->pmmu.host_resident = 1`) is active only under a *translated* domain. No action needed. Recorded as operational knowledge: if any future host-memory corruption appears at device acquire/context create, the IOMMU domain is the first check; the 1.24.1 bug is real in our driver tree. | A future driver upgrade changes the PMMU residency behavior, or the node's boot config drops `iommu=pt`. |
| RJ-25 | Changing the host CPU governor | **Verified already optimal (2026-11-21):** cpu0 governor = `performance` at 3.4 GHz (160 cores). The rimb05 twin-model finding (governor changes = ~70% of a twin-measured delta, Tier D) is not actionable on us — we are already at performance. | A future host migration to a different power profile (record: check the governor first on any new node — it is a 5-second check with a large potential effect). |
| RJ-26 | A/B-ing `PT_HPU_ENABLE_LAZY_COLLECTIVES` | **Already ON in our production** (rev-11 RA-032 NO-OP finding: our v0.26.0 plugin auto-sets it at import in lazy mode, `platform.py` l.432-433 server-verified; the vllm-fork `load_general_plugins` auto-set, Tier D, re-confirmed this round). rimb05's explicit `=true` is redundant for us. The 322-launch B2 structure was measured *with* it ON — no unmeasured lever. | A future bridge change makes it non-default, or a boot shows `=false` in the worker env (the check is one grep in the boot log — add it to the boot checklist). |
| RJ-27 | The GaudiInfer masked-softmax verify kernel as a standalone item | Device-side attention optimization; the project is host-bound (the settled P-HOST physics) and the verify attention is a minority of the device time. Their 1.8× vs SDPA at 131K verify width (Tier D) is real but bounded by the attention share — and the kernel is written for standard MHA scores, not MLA latents (their own note: the score layout differs). Recorded as an *optional component* of RA-043's scope (if the D15 census shows the verify attention's softmax chain is a large node share, the masked-softmax TPC op joins the WY kernel work). | D15 shows the verify attention node share ≥ 15% of the recipe (currently the MoE is 87-89%, so this is a low prior). |
| RJ-28 | The twin-model A/B methodology (rimb05's exact-dimensions random-weight model) as project infrastructure now | Genuinely useful (their layout/patch experiments run on a twin without production weights — the 4-HPU layout A/Bs would be faster) but it is new infrastructure (a twin of our 45+1 layer model, the weight bake, the harness wiring) with no current experiment that *requires* it — the same-boot A/B discipline covers every open item. Deferred, not rejected-in-principle. | An experiment series needs > 5 boots of layout variants where the production-weights boot cost dominates (the 4-HPU config space exploration, if RA-048 goes (b) and the 4-HPU tuning expands). |

---

## New research gaps (D14-D21)

| # | Question | Why it matters | Suggested method | Effort |
|---|---|---|---|---|
| D14 | Is `HPUAsyncScheduler` ACTIVE in our current MTP boot, and what exactly does the `.cpu()` at `rejection_sampler.py:273` wait on (the sampler op chain's device completion? a host parse loop?)? Does step N+1's launch begin before or after it returns? | RA-039's entire mechanism depends on it: if async is active and our device-reject's sync defeats it, the 15.6 ms post-term is removable by restructure; if async is OFF (and can be), the fix is enabling it + the two fork fixes; if the `.cpu()` is the vendor's MTP-async requirement, RA-039(b) is limited to (c) | One MTP boot: the engine config dump (`async_scheduling` value), the engine-core log (the scheduler class line), the worker env dump; then one py-spy + torch.profiler window at bs=1 timing the `.cpu()` wait composition + the next-launch timestamp (the existing launch-timestamp patch) | 1 boot (no HPU time beyond it) |
| D15 | The per-layer node census of the decode graph: the KDA chain's node count per layer (the conv/gate/state-update op list), the verify-attention softmax chain's node share, the small-op (elementwise/norm) share — the full non-MoE 11-13% broken down by class | Sizes RA-043 (the KDA WY kernel's node win), informs RA-038-A T2's ablation arms (the KDA class as the second arm), and RJ-27's gate (the verify-softmax share) | The graph_editor tooling (staged at 380303f) on the current decode recipe: the node census by op class per layer (the RA-038-A T2 arm-0 characterization) | 0.5-1 day (no HPU time) |
| D16 | Does our v0.26.0 runner's per-step block-list management have the Python-loop hot path that fastmeta removes (rimb05: up to ~8 ms/step at @16 long context, Tier D)? | RA-046(1)'s gate — if the v0.26.0 path is already vectorized, the item is a NO-OP (the RA-032 pattern) | Code read of the block-table update in `hpu_model_runner.py` (the per-step path) + one py-spy window at @16 (the block-list share of host time) | Code read (no HPU time) + 1 py-spy window |
| D17 | The sustained 1M-context decode rate (bs=1, 2K generated tokens post-1M-prefill) + the per-part breakdown (indexer scan, selection, attention read, KDA share) + the 1M KV HBM footprint | The 1M objective is validated (needle + prefill) but the decode rate has never been measured — the 4-HPU 1M planning and RA-047's gate both need it | `serve_mtp_1m.sh` boot, the needle harness to the 1M state, the 2K decode window, the launch-timestamp patch + the per-layer trace over the window, the HBM census | 1 boot window |
| D18 | The KDA state bytes per checkpoint (64×128×128 bf16 + the conv windows + the mHC state if recurrent — believed not recurrent, verify) and the per-640-token hybrid block bytes (the 11 DSA layers' latent + indexer keys at the real block geometry) | RA-041's lattice design (the N spacing from the pool math) and RA-042's capacity arithmetic (the per-token store ratio) — both are currently DERIVED from config shapes and need the measured block geometry | Source read of `kda_layer.py` (the state tensor shape) + the KV spec (the block page layout) + the arithmetic; confirm with one single-request allocation measurement at the first relevant boot | 5 min (calc) + 1 allocation read |
| D19 | The shared-expert time share of the step at bs=1 and @16 (the fused-moe runner's shared-expert branch in the trace), and whether the MTP draft passes pay the shared expert (the layer-45 MLP composition — the D9 fold-in) | RA-045's gate (the 3% threshold) + the draft-pass byte budget (the D9 refinement) | One profiler window at bs=1 + one at @16 (the existing trace tooling), the op-list read of the shared-expert branch | 2 profiler windows |
| D20 | Does our v0.26.0 OOT MLA backend (`vllm_gaudi/attention/oot_mla.py`) accept a non-bf16 cache dtype at all? (the corpus shows the bf16 write path; the NVIDIA-side `QUANTIZED_DS_MLA_CACHE_FORMATS` is a different backend) — and does v0.30.0's oot_mla have an fp8/int8 path to use as the port reference? | RA-042's Scheme A/B feasibility: if the v0.26.0 backend hardcodes bf16, Scheme A is a small backend patch and Scheme B (the vendor flag) may be blocked at our version | Code read of `oot_mla.py` (the cache-dtype plumbing) in our tree + the v0.30.0 oot_mla (the rimb05 checkout has it — the scout's report) | 1 code-read session (no HPU time) |
| D21 | The 1M boot's first-use shape captures: which decode/prefill shapes does the 1M boot capture at first use (the ~30 t/s once-per-shape stalls at 1M block sizes)? | RA-046(5)'s bucket-file merge (the rimb05 curated 1M shape list vs our 128+16) — the capture cost at 1M block sizes is the most expensive first-use cost in the project | The 1M boot's capture log (the per-shape compile events) over a mixed 1M workload (the D17 window's byproduct) | rides D17 |

---

## License compliance (this round, re-confirmed 2026-11-21)

- **abotsis/wright**: AGPL-3.0-only (verified: `COMMERCIAL.md` + `README.md` l.1713-1719).
  Cited as Tier D reference data (the 4-card GLM-5.3-Flash table, the engine
  architecture). No code import proposed (RA-048 is a decision + a spike on their
  release; the spike's license posture is the owner's decision item).
- **rimb05/glm-on-gaudi2**: no license file found in the repo (the README is the
  write-up). Cited as Tier D reference data only (the INT8 latent KV scheme, the
  k=2 chain invariants, the DSA v2, the config deltas). The vllm-gaudi v0.30.0 +
  gaudi-cookbook 1540263 it builds on are Apache-2.0 (the cookbook is our own
  upstream). No code import proposed except where the source is the cookbook itself
  (RA-041's `gdn-prefix-cache` fix, RA-039(c)'s `topk-subset-sampler` — both
  Apache-2.0, our upstream, attribution retained per the repo convention).
- **JChen-318/GaudiInfer**: no license file (the repo is the engine + its TPC
  kernels). Cited as Tier D reference data only (the WY algorithm, the deferred-
  commit design, the overlap patterns, the negative results). **No code import** —
  the RA-043 kernel is new work on our geometry using the algorithm as a template
  (the algorithm — the WY form of the delta rule — is standard numerical linear
  algebra; the specific TPC implementation is not copied).
- **JChen-318/gaudi2-single-card-driver**: no license file. Cited as Tier D
  (the 1.24.1 PMMU bug record) — no code, no performance claims adopted.
- **g2-opt-manual** (NatTuck): no license file — the standing rule applies: facts +
  short attributed phrases only, no code (the llvutd flag, the poison-byte contract,
  the 6-step materialization, the overlap law).
- **GaudiGrimoire** consultations this round: the `search` calls (MLA KV quantization,
  lazy collectives, device sampling, async scheduling), the `lookup`/corpus pins
  (the cookbook fixes at 1540263, the vllm-gaudi at 2f1aaea, the vllm-fork
  plugin auto-set, the driver source `gaudi2.c` at 770ad12e). All citations carry
  the tier + pin per the house format. No findings invented; every "Tier D" claim
  is attributed to the surveyed repo or the corpus pin above.
