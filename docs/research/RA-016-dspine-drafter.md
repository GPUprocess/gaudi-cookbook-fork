# RA-016 — DSpine drafter: the new strongest block-drafter, and its Gaudi 2 path

**Status:** PROPOSED — CONDITIONAL (no GLM-5.3-Flash DSpine draft checkpoint exists yet; the proposal is the tracking + port plan that activates on release, or on a decision to train)
**Research architect, 2026-11-09** · **Priority:** 7 (after RA-011/Q21/RA-012/RA-004; the drafter-class upgrade slot) · **Complexity:** Medium-High (HPU port, weeks) + the checkpoint dependency (external or a training campaign)
**Parent:** `BOTTLENECK_RESEARCH.md` (Section 3, RA-016) · **Family:** RA-009 (DFlash2 drafter), RA-004 (graphed draft), RA-010 (ngram probe)
**User request (2026-11-09):** "look into https://github.com/Tencent/DSpine and how to implement it by using the GaudiGrimoire MCP. It might be better than dflash2."

---

## 1. What DSpine is (verified from primary sources, 2026-11-09)

**DSpine — "Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative
Decoding"** (Tencent, arXiv 2609.36173, 2026-09-28; repo github.com/Tencent/DSpine, Apache 2.0;
project page dspine-project.github.io; training code in SpecForge PR #1038).

- A **parallel drafter**: one backbone pass drafts a whole block (b16 = 16 tokens). The novelty vs
  DFlash/DSpark/DFlash2: instead of independent per-position token selection (which produces
  inconsistent continuations and shortens the accepted prefix), **at every draft layer a gated adjacent
  injection writes each predecessor's predicted feature into its successor** — the causal conditioning
  chain unfolds over network DEPTH while all positions update in parallel.
- A **unified transfer space built from the target model's output embeddings** unifies layer-wise
  injection with predecessor-conditioned decoding; layer-wise output-embedding supervision promotes
  predicted features in shallow layers (paper's analysis of DFlash: "early positions already form
  recoverable predictions in shallow layers, and accurate adjacent predecessors help successors more
  when they enter earlier").
- Execution (SGLang 0.5.22): fused kernels + a "transition cache"; the Qwen3.8-27B path uses
  **DSpark-based dynamic verification** (arXiv 2607.05147) + **ReplaySSM** for the hybrid target's
  linear-attention state replay (`--enable-linear-replayssm-spec`, `--mamba-radix-cache-strategy
  extra_buffer`).
- **Released draft checkpoints (HF `tencent/dspine` collection, actively maintained — the 27B draft
  updated 2026-11-08, one day before this note):**
  | Checkpoint | Draft size | Block | Target | License |
  |---|---|---|---|---|
  | `tencent/Qwen3-8B-DSpine-b16` | **1B** (5-layer class) | 16 | Qwen3-8B (dense) | apache-2.0 (the 27B card is apache-2.0; the 8B card is the same collection) |
  | `tencent/Qwen3.8-27B-DSpine-b16` | **2B, 5 layers**, 8,192-token sliding window | 16 | Qwen3.8-27B (HYBRID: linear+full attention) | **apache-2.0** (verified from the model card) |
  | **GLM-5.3-Flash** | — | — | — | **DOES NOT EXIST** (checked the collection 2026-11-09) |
- **Measured acceptance (primary sources):**
  - Qwen3-8B, same training data/budget as DFlash2/DSpark/Domino (repo README table): DSpine avg
    **7.83** (T=0) vs DFlash2 7.55, DSpark 7.44, Domino 7.05, official DFlash 5.92 — **+3.7% over
    DFlash2** at equal budget; T=1: 6.98 vs 6.53 (+6.9%).
  - Qwen3.8-27B vs the OFFICIAL DFlash2-b8 checkpoint (HF card): DSpine wins every benchmark — GSM8K
    7.20 vs 5.69 (+27%), MATH-500 6.49 vs 5.49, HumanEval 5.71 vs 4.87, MBPP 6.14 vs 5.04, MT-Bench
    4.48 vs 4.11 (T=0); agent end-to-end throughput **2.71-3.84× AR vs DFlash2's 2.33-3.17× AR**
    (+15-20%). (Part of this is block 16 vs block 8; the same-budget Qwen3-8B table isolates the
    architecture gain at +3.7-6.9%.)
  - Paper abstract (vs official DFlash, Qwen3-8B T=0): 3.77 → 4.82 (**+27.8%**); SGLang serving
    **+23.3% throughput** vs DFlash on average.

**Verdict on the user's question ("might be better than dflash2"): CONFIRMED** — DSpine is the current
best-published block drafter on acceptance length, at both temperatures, in same-budget comparisons
(+3.7-6.9% over DFlash2) and against the official DFlash2 checkpoint (+8-27% acceptance, +15-20%
end-to-end). The gain is real but not a multiplier: it is a few percent of acceptance length plus the
block-16-vs-8 draft-depth difference.

## 2. Why this matters for the project (and why it is not the next boot)

Current drafter: the in-model **MTP head at k=1** (~94% per-token acceptance ≈ 1.85 tokens/step).
The queued drafter-class upgrade is **RA-009 (DFlash2)**: the incoai `GLM-5.3-Flash-DFlash2`
checkpoint exists today (block 8, top-16, rank 256, target_layer_ids [5,14,24,33,42]) — but it is
licensed **CC BY-NC-ND 4.0** (non-commercial, no derivatives; companion doc rev 5), a real constraint
for a production deployment.

DSpine changes two things if a GLM-5.3-Flash draft ever appears:
1. **The acceptance ceiling**: +3.7-6.9% over the same-budget DFlash2 (plus the block-16 depth) →
   ~3.0-3.6 tokens/step ESTIMATED for GLM-5.3-Flash (vs ~2.7-3.3 for DFlash2-b8) at bs=1.
2. **The license**: Tencent ships its drafts under Apache 2.0 — a clean license for production,
   unlike the NC-ND DFlash2 GLM checkpoint.

But the binding constraint is the **checkpoint**: there is no GLM-5.3-Flash DSpine draft, and the
drafter is a TRAINED model (5 layers, 1-2B params, trained against the target's layer features via
SpecForge). Options: (a) **track** the `tencent/dspine` collection — it is actively maintained
(27B draft released 2026-11-08; the collection's growth pattern suggests more targets; GLM-5.3-Flash
is one of the most-served open models, a plausible next target), (b) **train one** (SpecForge PR
#1038: the training code is public; a multi-week GPU campaign on domain-matched data — a project
decision, not a research task), (c) **proceed with RA-009 (DFlash2) now** and treat DSpine as the
swap-in upgrade (the HPU port machinery is shared — Section 3).

## 3. The Gaudi 2 implementation path (from the GaudiGrimoire substrate)

GaudiGrimoire has **no DSpine content** (searched 2026-11-09: "DSpine adjacent causal injection" —
only generic spec-decoding docs return). But the substrate for this exact class of drafter on this
exact model class is in the corpus, in two forks:

1. **The 1Cat-vLLM-Gaudi DFlash2 HPU path** (tier D, `vllm_gaudi/v1/spec_decode/hpu_dflash2.py` +
   `docs/features/flashinfer_gaudi.md`): "all seven drafts in one pass, runs the top-16 candidate
   selector on HPU, and verifies an eight-token block with rollback-safe convolution and GDN state
   checkpoints." The integration pattern: `propose_query_block` runs the draft model over a fixed
   [anchor + block-1] query block with explicit attn bias ("The explicit bias carries both causality
   and window alignment"), then `select_dflash2_candidates` scores the candidate graph on-device;
   `platform.py::_validate_hpu_dflash2_config` gates the feature (greedy, TP1/PP1/DP1,
   max_num_seqs ≤ 16, compact GDN, prefix caching/LoRA off, exact-shape validation).
   **DSpine's draft is architecturally SIMPLER to port than DFlash2's**: it is a plain dense 5-layer
   transformer (no dynamic convolutions, no non-causal-attention quirks — the "adjacent causal
   injection" is inside its forward, invisible to the runner) run over a 16-row block. From the
   runner's view it is the same `propose_query_block` call with `num_speculative_tokens=15`.
2. **The abotsis-vllm-gaudi KDA candidate-slot machinery** (tier D, `vllm_gaudi/models/glm5_next.py`
   `_kda_spec_decode` — the ORIGIN of our MTP port's KDA handling): "KDA over a draft chain: score all
   spec_len tokens, keep every state… Candidate slot j (indices column 1+j) holds the state after
   token j; the next step resumes from column num_accepted." The runner layout is generic
   (`[groups, bs, 1 + num_spec+1]`, `hpu_model_runner.py` part 46): canonical slot + one private
   candidate slot per spec position. **Our tree already runs this at k=1; the DSpine port extends it
   to k=15** — 16 candidate KDA states per request per KDA layer (34 layers). This is the same
   mechanism ReplaySSM provides for SGLang's linear-attention targets — the corpus shows it is a
   solved problem on Gaudi for this model (the abotsis fork ships it for glm5_next).
3. **The DFlash2 HPU path is Qwen3.8-27B-shape-qualified** (`_validate_hpu_dflash2_config`: "currently
   qualified only for the official Qwen3.8-27B-DFlash2 shape"; "Qwen3.8/Qwen3.5 hybrid targets only";
   exactly 7 speculative tokens). Both RA-009 (DFlash2-GLM) and RA-016 (DSpine-GLM) require extending
   this validation + porting the drafter model class — **the port work is shared between the two**:
   a GLM-5.3-Flash block-drafter integration (query-block proposal, the k=15/7 candidate-slot layout,
   the verify-block rollback for KDA+MLA) serves both checkpoints.

**Concrete port plan (activates on a GLM-5.3-Flash DSpine checkpoint — released or trained):**
1. **Checkpoint audit** (half a day): config.json (draft arch, block 16, transfer-space dims vs the
   target's embedding dim), weight mass (~1-2B → 2-4 GB BF16 per rank at TP8, or 8-16 GB at TP4 —
   fits with wide margin), license (expect apache-2.0 per the collection pattern).
2. **Drafter model class** (1-2 weeks): port the 5-layer DSpine draft to the vllm-gaudi model
   registry (dense transformer + the gated adjacent-injection layers + the transfer-space projection;
   the SGLang reference is `sglang_dspine/models/dspine.py`; the draft forward is ordinary ops —
   attention over [context window (8192); block], per-layer injection, final norm, target lm_head
   rows). Run it as its own HPU graph per bucket shape (the RA-004/abotsis draft-graph pattern:
   `wrap_in_hpu_graph(core, disable_tensor_cache=True, dry_run=False)`, metadata as graph inputs).
3. **Proposer integration** (1 week): an `HpuDspineProposer` following `hpu_dflash2.py`'s
   `propose_query_block` (16-row query block, explicit attn bias, device-side selection); the
   `use_dflash()`-class branch in `propose_draft_token_ids`; `num_speculative_tokens=15`.
4. **Candidate slots** (2-4 days): extend the KDA spec-decode slot layout to k=15 (the abotsis
   `_kda_spec_decode` is already generic in `spec_len`); size the state pool (16 slots × 34 KDA
   layers × max_num_seqs — re-run the pool math that the 1Cat validation enforces for the 8-slot
   case: "each active request reserves eight rollback checkpoints per GDN layer" → 16 slots roughly
   DOUBLES that reserve; the 125 GB host / 96 GB HPU budgets must be re-checked at the target
   max_num_seqs — expect the same ≤16-seq constraint, possibly tighter); the MTP k=1 path stays as
   the fallback arm.
5. **Validation extension** (days): the `_validate_hpu_dflash2_config`-class gate for the DSpine
   shape (greedy first — the DFlash2 HPU qualification scope; sampling later).
6. **A/B** (the same-boot protocol): MTP-k1 (arm 0) vs DFlash2-GLM (arm 1, RA-009) vs DSpine-GLM
   (arm 2, if both checkpoints exist): bs=1 + @16 + the accuracy suite; acceptance length on the
   bench prompt set (the direct metric); the KDA state-pool memory watch.

**If no GLM checkpoint appears:** the same port work can be amortized on RA-009's DFlash2-GLM
checkpoint (block 8, k=7 slots); DSpine then becomes a checkpoint swap + the slot extension
(k=7 → k=15).

## 4. Expected performance impact (all ESTIMATED)

Reference points: current MTP k=1 = 1.85 tokens/step (MEASURED, 94% per-token acceptance); DFlash2-b8
GLM acceptance ≈ 2.7-3.3 tokens/step (DERIVED from the inco blog's 2.7-3.4×-vs-AR class +
project-maya's 148.6 t/s on consumer HW vs its MTP baseline); DSpine-b16 = DFlash2 acceptance ×
(1.04-1.07) at equal budget, or ×(1.08-1.27) vs the official b8 checkpoint (MEASURED on Qwen; the
transfer to GLM is the ESTIMATED part).

| Metric | MTP k=1 (now) | DFlash2-GLM port (RA-009) | DSpine-GLM port (RA-016) |
|---|---|---|---|
| Tokens/step bs=1 | 1.85 (MEASURED) | ~2.7-3.3 (EST) | ~3.0-3.6 (EST) |
| Draft work/step | the MTP head, eager (7.5-10.6 ms host, D1) | 5-layer draft, 8-row block, graphed (EST ~3-8 ms device on 8 HPU) | 5-layer dense draft, 16-row block, graphed (EST ~5-14 ms device on 8 HPU — ~2× the rows, plain dense ops) |
| Verify work/step | 2 rows (bs=1) | 8 rows | 16 rows (the MoE weight read is unchanged — weight-bound; attention/compute ~2× the 8-row case) |
| bs=1 t/s (8 HPU, pre-RA-011 stack) | 42.4 (MEASURED) | ~65-80 (EST) | ~75-95 (EST) — **+10-25% over the DFlash2 port** |
| bs=1 t/s (8 HPU, post-RA-011+Q21+RA-012 stack, ~18 ms/step base) | — | ~110-140 (EST) | ~125-165 (EST) |
| @16 t/s (8 HPU) | 257 (MEASURED) | ~450-560 (EST) | ~480-600 (EST) — the +4-7% acceptance is nearly pure at high batch (draft cost amortized over 16 requests) |
| KDA candidate-state memory | k=1: 2 slots/layer/req | k=7: 8 slots/layer/req | **k=15: 16 slots/layer/req — ~2× the DFlash2 reserve** (the 1Cat validation's max_num_seqs≤16 bound must be re-derived; possible tightening to ≤8-12) |

**Conservative:** DSpine ≈ DFlash2 + 4% acceptance, draft cost +60% (block 16) → +5-10% end-to-end
over the DFlash2 port.
**Optimistic:** the Qwen-27B-class gain transfers (acceptance +15-20% vs the b8 checkpoint) →
+15-25% over the DFlash2 port; at bs=1 that is the difference between ~110 and ~130 t/s on the
post-lever stack.
**Worst case (the real risk):** no GLM checkpoint appears within the project horizon AND training a
draft is not accepted → RA-016 stays dormant and RA-009 (DFlash2, NC-ND license caveat) is the
drafter-class upgrade. The tracking cost is near-zero (watch the HF collection).

## 5. Risks

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | No GLM-5.3-Flash DSpine checkpoint (the whole proposal is conditional) | scope | Track `tencent/dspine` (active collection); the training route (SpecForge PR #1038) is the fallback project decision; RA-009 is unaffected (shared port machinery) |
| R2 | The 16-candidate KDA slot pool exceeds the memory budgets at the target max_num_seqs (2× the 8-slot reserve) | memory | Re-derive the pool math at port time (the 1Cat validation formula is in the corpus); cap max_num_seqs if needed (the DFlash2 path already runs ≤16); the state pool is host/HBM — the 125 GB / 96 GB budgets have measured headroom (RA-001's 0.7-0.9 GB/worker RSS precedent) |
| R3 | The Qwen acceptance gains do not transfer to a MoE hybrid target (GLM-5.3-Flash is harder to draft than dense Qwen — our MTP k=1 is at 94%, but DFlash2's GLM numbers are the weak link in the chain) | performance | The A/B measures it directly (acceptance length on our bench set); the DFlash2-GLM arm (RA-009) is the control — if DFlash2 itself underperforms on GLM, DSpine does too, and the drafter-class question reopens at the data, not the architecture |
| R4 | The draft's 8,192-token sliding window needs its own attention cache on the HPU (the DFlash2 HPU path adds a "draft attention cache" — the 1Cat doc warns an undersized KV budget "silently queues otherwise concurrent requests") | operational | Budget the draft cache explicitly in the port (the 1Cat warning is the precedent); the A/B includes the concurrency-under-test check |
| R5 | Greedy-only qualification scope (the DFlash2 HPU precedent) | compatibility | Same scope as RA-009 (greedy first; sampling later) — consistent with the project's current operating mode (byte-identical greedy) |
| R6 | The "transition cache" + fused kernels are CUDA (SGLang) — no HPU equivalents exist | implementation | The HPU port uses the ordinary vllm-gaudi ops (the draft is a plain dense transformer — no exotic kernels needed; the injection is a gated residual write, a pointwise+matmul class of op). The fused kernels are a performance optimization, not a correctness dependency |

## 6. Dependencies

- **Checkpoint** (external): `tencent/dspine` collection release for GLM-5.3-Flash, OR a SpecForge
  training campaign (project decision; weeks; GPU budget).
- **RA-009's port machinery** (shared): the GLM block-drafter integration (query-block proposal,
  candidate slots, verify rollback) — doing RA-009 first makes RA-016 a checkpoint swap + the
  k=7→k=15 slot extension.
- **RA-004** (the draft-graph pattern): the draft runs as its own HPU graph (the abotsis
  `VLLM_GLM_MTP_DRAFT_GRAPH` precedent is in-tree).
- **RA-011** composes (the async host hides the draft dispatch either way).

## 7. Novelty assessment

- **Not attempted** in the project (no DSpine entry in TASKS/EXPERIMENTS/DECISIONS; GaudiGrimoire
  search confirms no DSpine content in the corpus — it postdates the corpus's Gaudi spec-decoding
  coverage).
- **Closest prior work:** RA-009 (DFlash2 — the same drafter CLASS, weaker measured acceptance,
  NC-ND-licensed GLM checkpoint); RA-004 (the draft-graph mechanism — reused, not repeated); the
  1Cat DFlash2 HPU path (the port template — corpus tier D).
- **Why it deserves a proposal now (despite the conditionality):** (a) the user asked, and the answer
  is "yes, better than DFlash2 — by 4-7% same-budget, 8-27% vs the official b8 checkpoint"; (b) the
  tracking cost is near-zero and the collection is actively growing (27B draft released 2026-11-08);
  (c) the port machinery is shared with RA-009, so the proposal costs nothing until triggered;
  (d) the Apache-2.0 draft license is a genuine production advantage over the NC-ND DFlash2 GLM
  checkpoint — if Tencent ships a GLM draft, it may be the RIGHT drafter for the production
  deployment while DFlash2 is the dev-environment stopgap.

## 8. Sources (all accessed 2026-11-09)

- github.com/Tencent/DSpine (README: mechanism, same-budget acceptance table, install/usage, news —
  2026-11 Qwen3.8-27B + MLX; 2026-10 training code in SpecForge, Qwen3-8B weights, paper).
- arXiv 2609.36173 (abstract: adjacent causal injection, unified transfer space from the target's
  output embeddings, layer-wise supervision, fused kernels + transition cache, +27.8% acceptance and
  +23.3% serving throughput vs DFlash on Qwen3-8B T=0).
- huggingface.co/tencent/Qwen3.8-27B-DSpine-b16 (card: 5-layer 2B draft, block 16, 8,192 window,
  **license apache-2.0**, acceptance tables vs the official DFlash2 checkpoint, agent end-to-end
  throughput tables, the DSpark+ReplaySSM hybrid-target path, usage flags).
- huggingface.co/collections/tencent/dspine (the two released checkpoints; NO GLM-5.3-Flash entry;
  collection updated 2026-11-08).
- GaudiGrimoire (tier D): 1Cat-vLLM-Gaudi `hpu_dflash2.py` (the HpuDFlash2Proposer, propose_query_block,
  the regionally-compiled draft methods), `docs/features/flashinfer_gaudi.md` (the DFlash2 HPU
  qualification scope + the draft-cache budget warning), `platform.py::_validate_hpu_dflash2_config`
  (the shape gate, the 8-rollback-checkpoints-per-GDN-layer memory rule); abotsis-vllm-gaudi
  `vllm_gaudi/models/glm5_next.py::_kda_spec_decode` + `_kda_load_slots` (the KDA candidate-slot
  mechanism for glm5_next — the origin of our MTP port) and `hpu_model_runner.py` part 46 (the generic
  `[groups, bs, 1+num_spec+1]` slot layout); abotsis `docs/features/glm53_flash.md` (the
  `VLLM_GLM_MTP_DRAFT_GRAPH` draft-graph precedent).
- Project data: D1 (the eager draft dispatch cost), the MTP k=1 acceptance (94%, 1.85 tokens/step),
  companion doc rev 5 (the DFlash2 GLM checkpoint facts + the CC BY-NC-ND 4.0 license + the inco
  acceptance tables), project-maya (DFlash2 in production on this model, 148.6 t/s consumer-HW).
