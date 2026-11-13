# RA-034 / RA-035 — Round-5: the bridge compile/dispatch track + the gap catalog (2026-11-11)

**Status:** PROPOSED (research architect, round 5) · **Parent:** `BOTTLENECK_RESEARCH.md` (rev 1.7.1) · **Handoff:** `EXTERNAL_RESEARCH_PROPOSALS.md` rev 11
**Round-5 scope (per owner directive 2026-11-11):** the implementation agent is profiling the
workload (flame-graph gaps) and the pytorch-bridge (graph compile and/or dispatch latency). This
note supplies: (1) the **gap catalog** — the ranked predicted gap classes with their expected
flame-graph signatures, causes, and fix pointers, to match against the actual flame graph;
(2) the **dispatch-latency theory** — what the 322-launch structure actually is (corrected:
both "obvious" launch levers are already ON in production), the boundary-census protocol that
decomposes it, and the per-class fix pointers; (3) the **graph-compile theory** — the 25-min
boot cost structure and the cadence levers; (4) two new proposals (RA-034 the launch-boundary
program, RA-035 the dense-linear dequant-fusion question); (5) a live-env discrepancy flag
(the current boot's `PT_HPU_LAZY_ACC_PAR_MODE=0` vs the pinned `=1`).

---

## 1. The dispatch-latency correction (RA-032 reclassified)

**The round-4 record said RA-032 (`PT_HPU_ENABLE_LAZY_COLLECTIVES=true`) was an untried env
lever (322 → ~162 launches estimated). That is wrong — the flag is ALREADY ON in production:**

1. **Tier B (our v0.26.0 plugin source, server-verified):** `vllm_gaudi/platform.py`
   `set_torch_compile()` (import/registration time): in lazy mode,
   `if os.environ.get('PT_HPU_ENABLE_LAZY_COLLECTIVES') is None: os.environ[...] = 'true'`
   (l.432-433) — with the comment "multi-HPU inference with HPUGraphs (lazy-only) requires
   enabling lazy collectives". Same code in the corpus's newer vllm-gaudi (the GLM-5.3-Flash
   recipe doc: "`PT_HPU_WEIGHT_SHARING=0`, `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` — **auto (all
   archs, if unset) — required for multi-rank HPU-graph serving; set by the plugin at import**").
2. **MEASURED (live worker env, boot 2026-11-11 02:52, `/proc/<worker>/environ`):**
   `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` ✓, `GLM5_NEXT_DECODE_LAYER_GROUP=45` ✓,
   `PT_HPU_LAZY_MODE=1` ✓, `PT_HPU_LAZY_ACC_PAR_MODE=0` (see §5), `GLM_EARLY_DRAFT=1`,
   `GLM_ROUTER_FAST=1`, `GLM_SPEC_DEVICE_REJECT=1` (the Q18 levers all on),
   `VLLM_MOE_CHUNK`/`VLLM_MOE_TOKEN_BOUNDARY` unset (the RA-030 no-op, as corrected),
   `VLLM_GLM_SKIP_LAYER_MARKSTEP` unset, `PT_HPU_RECIPE_CACHE_CONFIG` unset (RA-031 not yet).
3. **The implementation agent independently verified the same** (RESEARCH-001 addendum 8,
   commit 3ccd9af: "PT_HPU_WEIGHT_SHARING=0 and PT_HPU_ENABLE_LAZY_COLLECTIVES=true are
   auto-pinned by our plugin (multi-rank graph serving matches the reference recipe)").

**Consequences:**

- **RA-032 is a NO-OP as an A/B arm** (setting "true" when the plugin already set it changes
  nothing). Remove it from the first-boot A/B family; the family is the RA-034 census boot +
  RA-031 (recipe cache) + the RA-030 confirmation arm.
- **The 322-launch structure (B2) was measured WITH lazy collectives ON** — so the per-layer
  recipe boundaries are NOT (only) "collectives ending the recipe". The boundary census (§2.2)
  is the decisive measurement: what ACTUALLY cuts the 322 recipes, per op class.
- **`GLM5_NEXT_DECODE_LAYER_GROUP=45` is also already ON** (port `model.py` l.61-87,
  l.363-372 — the cookbook mechanism, bit-identical, in every serve script). The per-layer
  `mark_step` hook is already bypassed on pure decode steps (direct `layer.forward()` calls,
  one mark per 45-layer stack). The Muse measured value of this lever (+15% 1-stream, 34.5 →
  39.7 tok/s same-pod A/B) and the reference-tree value (+10%, 45 of 176 launches removed,
  bit-identical greedy parity 8/8 — the runner patch comment, `hpu_model_runner.py` l.5065-5080)
  are ALREADY BANKED in our 42.4 baseline. The remaining experimental knob in this family,
  `VLLM_GLM_SKIP_LAYER_MARKSTEP=1` (removes the hook entirely), has ~0 marginal decode effect
  (the hook already doesn't fire on decode steps) — its value is prompt/mixed steps only
  (bounded host run-ahead + prompt memory peak — a prefill concern, deferred).

**What this means for the B2 host floor (~14 ms/step, 93.3% of `execute_model` host time in
`graphs.py:75`, D1):** the fixable surface is (a) the LAUNCH COUNT (the census tells us which
class to attack — §2.3), (b) the PER-LAUNCH COST (input hashing scales with the input-tensor
count per recipe — the Q21 fused op attacks this from both sides), and (c) the OVERLAP
(RA-011-class async replay + T6 — the host runs ahead of the device instead of serializing;
see §3 gap G1). None of (a)/(b)/(c) is a new env flag — the env-flag surface for dispatch is
essentially exhausted on our tree.

---

## 2. RA-034 (new) — The decode launch-boundary census + the per-class program

**ID:** RA-034 (round 5) · **Complexity:** LOW (the census: no new HPU time — the RECIPE-DIFF
GC-trace logs are archived; one optional confirmation boot) · **Status:** PROPOSED

### 2.1 The current structural model (what is known)

Per B2 (agent, VERIFIED class) + the live env + the corpus:

- The decode step = **~322 recipe replays**, each a separate HPU graph launch at ~40-74 µs
  host (the cookbook anatomy: 322 × 74 µs ≈ 24 ms of a 39 ms step on the same model family;
  our D1: ~14 ms of a ~24 ms host step, 93.3% in `HPUGraph.replay` internals).
- `calculate_num_chunks` (tier B, §RA-030 correction) shows the MoE op itself does NOT add
  per-chunk launches at decode (the chunk geometry organizes the activation GEMMs, one op
  call per layer).
- The B2 model (agent l.220): "the all-reduce operations force `mark_step()` boundaries,
  splitting the lazy graph into ~162 segments (one per layer per TP boundary). Each segment
  may contain multiple recipes." — i.e., ~162 AR-driven segments containing ~322 recipes.
  **This is a MODEL, not a boundary census** — with lazy collectives ON (verified), the
  question "do the ARs still force boundaries, and what ELSE cuts recipes" is OPEN.
- Candidate boundary sources (hypothesis list for the census):
  1. **AR/collective boundaries** — 2 ARs × 45 layers = 90 (intrinsic to TP; even lazy
     collectives are cross-rank sync points — VERIFY whether they cut recipes in our stack).
  2. **Dequant-then-GEMM chains** (agent l.182, UNVERIFIED): `apply_block_fp8_linear_hpu`
     (`extension/ops.py` l.940-960) dequants EVERY block-FP8 linear weight to BF16
     (`dequant_block_fp8_weight_naive`, l.868-915: cast + broadcast-mul) then runs a BF16
     `F.linear` — per step on the attention QKV/O projections, dense layers, shared experts,
     the MTP module (~45×2+ linears). If the compiler does NOT fuse the cast+mul into the
     GEMM's weight path, each such linear = extra recipe nodes/boundaries AND doubled weight
     traffic (FP8 read + BF16 write + BF16 read) — **RA-035, §4**.
  3. **KDA layer internals** — the recurrent state update (34 layers): state-dependent ops
     (the scan/decay chain) may carry internal boundaries.
  4. **MTP seams** — the draft/verify split + the seam discipline (mark_step+sync at both
     eager-draft seams, deployed in the RECIPE-DIFF fix, `mtp.py` l.385-541): in the adopted
     config the draft is EAGER — its traversal runs outside the graph and adds its own
     boundary structure (B5: ~7.5 ms/step host+device at bs=1).
  5. **dtype/dynamic-shape boundaries** — the `refine_dynamic_shapes` class, the bf16/fp32
     cast sites (the mHC prep residual cast, the 184-MiB MEMLOG class).
  6. **The one group mark** (G=45) + the final-layer mark + the MTP block marks.

### 2.2 The census protocol (no new HPU time — the archived GC logs)

The RECIPE-DIFF boots archived `LOG_LEVEL_GC=0` traces (commit 9310884: "the hash-content
files give the op sequences"). For the DECODE-shape capture (not the 2,944 prefill one):

1. From the GC trace of ONE decode step (bs=1 k=1 shape), list every recipe boundary and tag
   it with the op class at the boundary (collective / mark_step / sync / cast / dynamic-shape
   / op-type transition).
2. Aggregate: boundaries per layer × class. Expected total ≈ 322/45 ≈ 7.2/layer; the census
   says which class owns which fraction.
3. Cross-check against the 162-segment model: if ARs own ~2/layer (90 total), the remaining
   ~232/45 ≈ 5/layer are the dequant/KDA/MTP/dtype classes — the fix targets follow.
4. (Optional, one boot) confirm on the CURRENT live config (the archived logs are from the
   RECIPE-DIFF boot — same production env class, but the census should be re-run if the
   agent's profiling boot changes the recipe structure).

**Deliverable:** the per-class boundary table (45 layers × classes) — the single number set
that tells the owner which B2 attack has the biggest surface.

### 2.3 The per-class fix pointers (what the census result maps to)

| Boundary class | Fix pointer | Gate |
|---|---|---|
| AR (if it cuts recipes) | intrinsic to TP8 — the only attacks: the one-shot AR / `gather_sum` AR order (MEASURED: no decode cost, but flips R003), or reducing AR COUNT (the EP layout changes which ranks AR — Q17/RA-025 coupling) | T7 |
| Dequant-then-GEMM (RA-035) | the compiler-fusion verification (the census + the GC op stream shows if the cast+mul survive as separate nodes); if not fused: the fused block-FP8 linear kernel (the Q21 analog for the ~90 non-MoE linears — one op, in-register E4M3→bf16, one GEMM) | none (bit-identical target) |
| KDA internals | the KDA decode kernel diff vs the cookbook's (RA-024 re-scoped — their `VLLM_KDA_SCAN=parallel` + the decay-dot MME fix are the reference); the T3 conv-chain fusion (round 2: ~1,190 small ops/step at spec_len=5) | the T7 class if it touches the residual stream |
| MTP seams / eager draft | the T1 chain graph (RA-026 — the draft IN the verify graph, ~0 draft cost; the seam count drops to 1) | T7 (the captured-draft R003 class — known and documented) |
| dtype/dynamic-shape | the mHC-prep cast elimination (the RA-019 mHC kernel family — deferred prefill value, but the cast sites are decode too); `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` check (the vendor default 0 — confirm ours) | none expected |
| marks | already minimized (G=45 + the seam discipline) | — |

**The expected-value order (before the census — the census re-ranks):** (1) the MTP draft
graphing (RA-026: removes the entire B5 boundary family + the 7.5 ms; +20-40% class, T7-gated),
(2) the dequant-fusion question (RA-035: if UNFUSED, the ~90 linear dequant chains are a
large boundary + byte surface; the fused linear kernel is a Q21-class win on the non-MoE
path), (3) the AR class (T7-gated, smaller surface), (4) KDA (the T3 family).

---

## 3. The gap catalog (for the flame-graph match)

Ranked predicted gap classes in a decode step (bs=1 k=1 adopted config: step ≈ 45 ms =
~24 ms host + ~21 ms device, ~zero overlap — D1). For each: the expected flame-graph
signature, the size, the cause, and the fix pointer. The agent's job: match each observed
gap to a class; any gap that matches NONE is a new finding to report back.

| # | Gap class | Size (MEASURED unless noted) | Flame-graph signature | Cause | Fix pointer |
|---|---|---|---|---|---|
| G1 | **Replay internals / launch storm** (host) | ~14 ms/step (93.3% of execute_model host, D1 py-spy 1458 samples) | a dense run of `torch/hpu/graphs.py:75 HPUGraph.replay` frames — per-recipe `input_hash` + `copy_to` staging + launch; staging ~1.6%, the LOOP dominates | 322 recipe launches × per-launch host cost; the host is SYNCHRONOUS (asynchronous=False) → zero overlap with the device (AIP 0-24%) | RA-034 census (which class to cut) + Q21 (fewer ops → fewer recipes + fewer inputs to hash) + RA-011/T6 (the overlap — the async-D2H design, §3.1) |
| G2 | **Eager draft traversal** (host+device) | ~7.5 ms/step bs=1, ~10.6 ms @16 (B5, MEASURED) | the MTP draft forward frames BETWEEN the verify replay and the next step — the full 46-layer eager traversal (no graph in the adopted config) | the draft graphs are T7-gated (the R003 captured-draft class); the adopted config runs the draft eager | RA-026 (the T1 chain graph — the draft inside the verify graph; ~0 draft cost at bs=1; the 59.5 t/s projection, calibrated by MTP4-T0 2.68 tok/step) — T7-gated |
| G3 | **Step boundary / device-idle host work** (host) | ~5.5 ms (cookbook G27, the same model class; OURS: the host 24 ms is serialized so the whole tail is critical — DERIVED larger) | the sampler/meta/output-D2H frames after the last replay, while the device is idle | the sampler + meta update + the output copy + the next-step metadata build cannot overlap the device (sync replay) | T6 (the async-D2H output staging — the end-state overlap), the Q18 levers (already adopted: router fast + early draft + device rejection removed the 2 sync points) |
| G4 | **AR waits** (device) | ~3.2 ms/step (162 ARs, the B4 class; small-message AR is 20-30× worse per the corpus 8-card study) | device idle bubbles aligned to the per-layer AR boundaries; the HCCL frames on the host | the per-layer TP all-reduce latency (the 2,400-token-class payload at decode is small-message regime) | the census (§2.3) — if ARs cut recipes, the one-shot/gather_sum AR (T7-gated); the EP-layout coupling (RA-025) |
| G5 | **MoE weight streaming** (device — a FLOOR, not a gap) | 15.5 ms/step = 38.1 GB (all 36 local experts × 25.2 MB × 42 layers, MEASURED to <1%) | the MoE op frames dominate the device timeline; the TPC/MME busy, no idle | the stock op reads EVERY local expert per call regardless of routing (97% dead reads at bs=1 — the ~2.2 routed experts/layer) | Q21/RA-025 (IN PROGRESS — 39cfd76: the single-expert GEMV core exists, 11-16 µs un-pipelined; the E4M3 NaN calibration 19,519/9,291/0 is the next gate; the routing-aware version → 2.3-5.4 ms) |
| G6 | **KDA/DSA attention** (device) | ~3.5 ms/step (the D1 class) | the per-layer attention frames (11 DSA-MLA + 34 KDA) | the per-layer MLA/KDA decode (the QKV/O GEMVs + the KDA state update) | the RA-024 re-scoped diff (the cookbook's KDA/MLA decode kernels are the reference — the per-layer device time once the census/trace exists) |
| G7 | **Per-row worker overhead** (host) | ~11% of the sampler/meta (B3, DERIVED) | the per-seq metadata/sampling frames scaling with batch rows | the per-request processing in the v0.26.0 runner (the I015 scope finding: the per-request prefill tuples + separate decode/spec-decode forward paths — the reference's single `prepare()` point does not exist in our tree) | the I015 integration (the agent's own refined scope: ~1 day runner work, sequenced after Stage-1/A014) |
| G8 | **@16 device gap** (device, 8-stream class) | 53 ms MEASURED vs 24.5 ms roofline — **the 28 ms gap is the unknown** (D1-16) | the device timeline at 16 streams: per-layer waits that don't appear at bs=1 | hypothesis: the EP imbalance wait (the busiest rank's expert load) + the per-layer dispatch + the AR payload scaling (×16) — UNVERIFIED | **the BG-1 8-rank trace** (the cookbook method: trace ALL ranks, the slowest-rank critical path per collective; our in-worker torch.profiler segfaults → the vendor HPU profiler or mark-step ranges + hl-smi sampling) — THE 800-feasibility decider |

**The prefill gaps (deferred per the owner, but in the catalog for completeness):** the
inter-step gaps 7.5-10.4 ms (D1-3 MEASURED — negligible); the eager 2,944-row prefill 1.35 s
per call (the OOM class — the >2,048-row band); the bucket-dip band (FIXED in the COMPOSE
profile: 5,555-7,731 t/s); the ACC_PAR busy-wait core (the agent's prefill profiling finding
— see §5).

### 3.1 The async-replay implementation sensitivity (the RA-011/T6 design constraint)

The corpus (rev-5 GaudiGrimoire log) shows the reference fork's `VLLM_HPU_GRAPH_ASYNC_REPLAY`
naive implementation MEASURED A REGRESSION ("double-digit percent" single-stream loss; the
replay-thread launches ~10% slower, contention in `sample_tokens`) — while the same idea
measured +29% in our RA-011 (rejected on R003, the in-flight consumption of the graph's
cached output tensors — the R3 risk). The design constraint that separates the two: **the
async path must not let the host consume the graph's cached output tensors while the next
replay is in flight, and the output copy must be the async-D2H (the T6 redesign)** — the
vendor scopes the flag "along with async D2H to copy outputs". The reference's later
"bit-identical" claim (their addendum 10) is consistent with the T6 design (the copy
isolates the host from the cached storage). Any RA-011 re-measurement on the current stack
must be the T6-shaped implementation, not the naive async flag.

---

## 4. RA-035 (new) — The dense-linear dequant-fusion question (the Q21 analog for the non-MoE path)

**ID:** RA-035 (round 5) · **Complexity:** the verification is LOW (the census + the GC op
stream, no HPU beyond the archived logs); the kernel (if needed) is HIGH (weeks, Q21-class) ·
**Status:** PROPOSED

The agent's own finding (B2 l.182, UNVERIFIED): `apply_block_fp8_linear_hpu`
(`vllm_gaudi/extension/ops.py` l.940-960) dequants the FULL weight to BF16
(`dequant_block_fp8_weight_naive` l.868-915: a cast + a broadcast-mul chain) and then runs a
BF16 `F.linear` — per step on the attention QKV/O projections, the dense layers, the shared
experts, and the MTP module (~45×2+ linears, ~180 dequant+GEMM pairs/step).

**The two coupled questions:**

1. **Fusion:** does the Synapse recipe compiler absorb the cast+mul into the GEMM's weight
   DMA path? **The census answers it** (§2.2): if the cast+mul survive as separate recipe
   nodes (or force boundaries), the compiler does NOT fuse them. The GC-trace op stream from
   the RECIPE-DIFF archived logs (no HPU) is the instrument — look for the dequant cast/mul
   node classes between the weight DMA and the GEMM nodes of one attention linear.
2. **Bytes (if unfused):** the non-MoE weight traffic DOUBLES — FP8 read + BF16 materialized
   write + BF16 GEMM read. The non-MoE weights per card: the attention QKV/O (45 layers ×
   ~17 MB-class per card) + dense + shared experts + MTP ≈ ~1-2 GB/card/step at the FP8
   class → ~2-4 GB extra at the BF16-materialized class → **~1-1.6 ms/step of the 21 ms
   device time (ESTIMATED, ±50%)** — a real but secondary floor next to G5 (15.5 ms).

**The fix (if unfused):** the fused block-FP8 linear kernel — the EXACT Q21 design (in-register
E4M3→bf16 cast, per-lane fp32 scale, one GEMM) applied to the ~90 non-MoE linears: one op per
linear (vs dequant-cast + mul + GEMM), the weight read once (FP8, no BF16 materialization),
the launch count cut (the G1 surface). The 39cfd76 GEMV core is the starting point (it is
shape-generic: [4096,2096]-class attention projections are the same [N,K] GEMV family).
**Accuracy:** the target is BIT-IDENTICAL to the current dequant-then-GEMM path (the same
E4M3 semantics — the 19,519/9,291/0 NaN calibration from 39cfd76 is the shared correctness
gate for BOTH the MoE and the linear kernel).

**Sequencing:** after the census (which sizes the surface) and after the Q21 MoE kernel
passes the micro-benchmark (the shared kernel core + the calibration). Not before — the Q21
MoE surface (15.5 ms + ~50 ops/layer-class) dwarfs the linear surface until measured.

---

## 5. The live-env discrepancy flag (the ACC_PAR mismatch)

**OBSERVED:** the current boot (2026-11-11 02:52, the agent's profiling boot — the live
`vllm serve` command matches the COMPOSE-class config: KV 1280, threshold 8192, capture
2048, ns32) has `PT_HPU_LAZY_ACC_PAR_MODE=0` in the worker env, while the pinned serve
scripts (serve_mtp.sh, serve_control_window.sh, + the diag/e015 family, per commit 3ccd9af)
export `=1` ("ACC_PAR_MODE=1 matches the bridge default but is pinned explicitly for a
reproducible env").

**What the flag does (tier B, `pytorch_helpers/habana_helpers/kernels_accumulation.cpp`):**
`AccThread::IsAccThreadEnabled() = (PT_HPU_LAZY_ACC_PAR_MODE != 0) && (PT_HPU_LAZY_MODE == 1)`
— it enables the bridge's ACCUMULATION THREAD POOL: the host-side op-accumulation work (the
lazy executor building the op stream between mark_steps) runs on a dedicated thread pool
instead of the main thread. The agent's addendum 8 identifies it as "the busy-wait core our
prefill profiling identified" — i.e., their prefill profiling already localized host time in
the accumulation phase.

**The action (for the agent):** (a) confirm which launcher started the 02:52 boot (if a
manual profiling command, add the pin); (b) if the profiling numbers are meant to represent
the pinned recipe, re-boot with `=1` before drawing conclusions — the ACC_PAR=0 vs 1 A/B is
exactly the dispatch-latency surface the owner asked to investigate (the accumulation thread
parallelizes the host work that G1/G3 serialize). Expectation (ESTIMATED, ±50%): if the
accumulation phase is a meaningful fraction of the 24 ms host, =1 moves several ms; the
reduction-tree/numerics risk is LOW (host-side scheduling, not device op order) but the R003
canary runs anyway (it is the only instrument that catches the timing-sensitive in-flight
classes).

---

## 6. The graph-compile theory (the 25-min boot)

**The cost structure (MEASURED + corpus):**

- 277 graphs per adopted boot (120 prompt + 157 decode, T12-MAP) ≈ 25 min — **the capture
  dominates the boot** (T12-TEST: the naive 215-graph pin boot took 29 min — "the capture
  cost is the PER-GRAPH COMPILE (ctx-size-dependent), not the graph count").
- The per-graph compile has two stages (the two-level compile, G25 + the RECIPE-DIFF data):
  (1) the **GC (graph-compile) stage** — the op-level compilation + the cast-fusion decisions
  (the 2,944-row session: 1,402,016 nodes, 136,358 removeOrMergeCasts; the eager arm: 444K
  nodes — the captured path compiles ~3.2× more because the draft's ops leak in, now fixed by
  the seam isolation to a 26,339-node separate session); (2) the **HPU-graph capture** (the
  recipe-level, the device-side). gaudi-benchmark Tip 6 (corpus, tier D): "first iteration
  7.9-9.2 s compiled vs ~0.13 s steady state, every rank pays it independently" — the GC stage
  is the ~8 s/graph-class cost, the capture the rest.
- **The recipe cache (`PT_HPU_RECIPE_CACHE_CONFIG`) caches the GC stage across boots** (the
  MiniMax/Muse production recipes use it; the per-rank dir appending is automatic in
  inference mode — the vendor source, G31; the dump-DEBUG path is the one blocked by the
  multi-rank hash-content write race — the PRODUCTION cache mode is not blocked).
- The MUSE decode-graph cap (`MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS=5408`): "decode batches
  above N blocks run without an HPU graph" + "each captured decode graph keeps ~0.5 MiB per
  block" (the HBM retention fact) — the cap trades the graph for the eager path on large
  decode batches; our ns32/KV1280 config is far below the cap class (the cap matters at
  the 1M/long-ctx class — note for the Q10/1M work).

**The cadence levers (for the decode program — prefill deferred):**

1. **RA-031 (the recipe cache + the corrected pin)** — the production-mode cache (NOT the
   blocked dump-debug mode): `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,False,131072` in the serve
   env. The first boot pays the full compile; every subsequent boot with the same graph
   signatures (the decode buckets are stable across the decode experiments) skips the GC
   stage → **boot 25 → ~8-12 min (ESTIMATED, 2-3×)**. The pin file (the corrected T12 form —
   INCLUDING the MTP seed buckets) reduces the DISTINCT shapes (277 → ~170) and adds
   determinism.
2. **The decode-only boot (NEW, round 5):** the decode A/B program needs only the 157 decode
   buckets — the 120 prompt buckets are prefill (deferred). A decode-focused boot could skip
   the prompt capture: (a) `VLLM_SKIP_WARMUP=true` (the cookbook pod spec lists it: "trade-off
   between start time and live compiles") — the decode shapes compile LIVE on first use
   (~8 s/shape, once, absorbed by the ×6 warm bench), or (b) a prompt-bucket cap env if the
   tree has one (VERIFY — the bucketing knobs `VLLM_PROMPT_*` exist; the skip-warmup is the
   documented one). Expectation (ESTIMATED): boot → ~12-15 min without the cache, ~5-8 min
   with the cache. **One-line verify + one boot — the biggest single cadence win for the
   decode program.**
3. **The regional compilation** (the agent's T12-TEST candidate — "the boot-time work
   requires a compile-time lever (the regional compilation or the recipe cache)"): the corpus
   has no vendor "regional compile" flag at our release (the GC compiles per captured graph;
   the only stage-skip is the recipe cache) — **treat the recipe cache AS the compile-time
   lever; regional compilation is not available at our v0.26.0/1.24.1 (flagged as a
   vendor-request candidate, not a project item).**

**The interaction with the dispatch track:** the recipe cache changes NOTHING at steady-state
serving (it is a boot-time-only lever) — it composes freely with RA-034/RA-035 (no numerics
change: the GC-stage reuse is the same compiled graph; the canary confirms).

---

## 7. Updated do-first order (round 5 — the bridge/dispatch focus; supersedes the rev-10 list where conflicting)

1. **The agent's profiling (in flight):** match the flame-graph gaps to the catalog (§3);
   the BG-1 8-rank @16 trace (G8 — the 28 ms gap, the 800 decider) with the vendor HPU
   profiler or the mark-step + hl-smi method (the torch.profiler segfaults). **Flag: the
   02:52 boot runs ACC_PAR=0 vs the pinned =1 (§5) — confirm the launcher before trusting
   the dispatch numbers.**
2. **RA-034 census (no new HPU time):** the per-class boundary table from the archived
   RECIPE-DIFF GC logs (§2.2) — re-ranks the B2 attack surface; the RA-035 fusion question
   is answered by the same logs (the dequant cast/mul node check, §4).
3. **RA-031 + the decode-only boot (one boot family):** the production-mode recipe cache +
   the corrected pin (with the MTP seed buckets) + the `VLLM_SKIP_WARMUP`/prompt-skip verify
   → the 3-5× cadence for the rest of the program (the Q21 A/B, the RA-026 shape family, the
   ACC_PAR A/B all pay the boot).
4. **The ACC_PAR A/B (one boot, in the family above):** =1 (pinned) vs =0 — the accumulation
   thread pool; the dispatch surface the owner named; the canary runs (the timing-sensitive
   class).
5. **Q21/RA-025 continues (the critical path):** the E4M3 NaN calibration (19,519/9,291/0 →
   the CPU-fn reference) is the correctness gate; then the per-route micro-benchmark
   (8 calls × 30 replays in-graph, per-route cost per slice width); the 39cfd76 core's
   pipelining is the ~4× latency target (0.5-0.75 → ~2-3 TB/s class).
6. **RA-026 (the MTP depth / the T1 chain graph):** after the Q21 micro-benchmark — the
   G2 gap (7.5 ms eager draft) is the biggest SINGLE gap class in the catalog; the 2.68
   tok/step is calibrated; T7-gated.
7. **RA-027 (the T7 evidence — parallel, to the owner):** the 10× canary + the GC-trace op
   diff — the unlock for the draft graphs (G2), the async (G1/G3), the AR order (G4).
8. **RA-035 (the fused block-FP8 linear kernel):** after the census sizes it + after the Q21
   core is calibrated (the shared kernel).
9. **RA-028 (the 4-HPU program + the feasibility report — to the owner):** unchanged (the
   200 bs=1 = full stack 120-170 est; the 800 @16 = 8 HPUs only, the BG-1 trace is the
   decider; 4 HPUs infeasible for the 800).
10. **The rest continues:** RA-011/T6 (the async-D2H design — §3.1 is the design constraint),
    RA-015 (with the 1M/Q10), RA-017 Phase 0, the ops standing items.
11. **Deferred (prefill, per the owner):** RA-022, RA-033 (the chunk-size knob — RA-030's
    prefill-class residual), RA-019 (the mHC kernel — its cast sites are in the RA-035
    census scope), the MUSE decode-graph cap (the 1M class).

## 8. Evidence index (round 5)

**Live/env (MEASURED, 2026-11-11):** the live worker env (boot 02:52) —
`PT_HPU_ENABLE_LAZY_COLLECTIVES=true` (the plugin auto-set, verified),
`GLM5_NEXT_DECODE_LAYER_GROUP=45` (on), `PT_HPU_LAZY_ACC_PAR_MODE=0` (vs the pinned =1),
the Q18 levers on, the RA-030 envs unset, `VLLM_GLM_SKIP_LAYER_MARKSTEP`/`PT_HPU_RECIPE_CACHE_CONFIG`
unset. The live `vllm serve` command (the COMPOSE-class config).
**Tier B (server-verified + corpus):** `vllm_gaudi/platform.py` l.415-433 (the
WEIGHT_SHARING/LAZY_COLLECTIVES auto-set, the GAUDISW-248809/249135 comments);
`hpu_model_runner.py` l.5065-5080 (the `[glm5_next] per-DecoderLayer mark_step hook` patch:
"45 of 176 launches", "+10% single-stream on the reference tree, bit-identical greedy 8/8",
the `VLLM_GLM_SKIP_LAYER_MARKSTEP` experimental gate) + l.745-762 (`modify_model_layers`);
`hpu_ops/backend/mixture_of_experts_gen.cpp` l.120-240 (the chunk-size invariance,
§RA-030); `pytorch_helpers/habana_helpers/kernels_accumulation.cpp` (`AccThread::
IsAccThreadEnabled`); the vllm-gaudi docs (`env_variables.md` — the "must set" vs "auto"
split; the GLM-5.3-Flash recipe — the auto-set list; the pod spec — `VLLM_SKIP_WARMUP`, the
`PT_HPU_RECIPE_CACHE_CONFIG` format).
**Tier D (corpus, the cookbook):** `ports/muse-spark-open/muse_spark_hpu/model.py` (the
`MUSE_SPARK_DECODE_LAYER_GROUP` mechanism: "5 cached recipes per layer at 25-170 µs host
each", "G=62 drops 62 of the 313 launches", +15% measured 34.5→39.7 same-pod A/B) +
`long_context.py` (`MUSE_SPARK_DECODE_GRAPH_MAX_BLOCKS`, the 0.5 MiB/block retention) +
`ports/glm-5-3-flash/glm5_next_hpu/model.py` (`GLM5_NEXT_DECODE_LAYER_GROUP`, "G=45 gives
one mark for the whole stack", "bit-identical"); the mxfp4 step anatomy (322 × 74 µs).
**Project (MEASURED):** D1 (the 93.3% replay internals, 14/24 ms), D1-16 (the 53 ms device,
the 28 ms gap), B2/B4/B5 (the agent's current_bottleneck l.34-40, l.210-258), the RECIPE-DIFF
GC logs (the archived op sequences), T12-MAP/TEST (277 graphs/25 min; the per-graph-compile
lesson), MTP4-T0 (2.68 tok/step), 39cfd76 (the GEMV core + the NaN calibration), the Q18
levers, RA-001 (+10.7% tensor cache).
**DERIVED:** the G1-G8 sizes (from the MEASURED inputs above); the RA-035 byte estimate
(1-1.6 ms ±50%); the boot-cadence estimates (the 2-3× / 3-5× classes); the ACC_PAR
expectation (±50%).

---

## 9. Addendum (round 5b) — The Q21 E4M3 NaN calibration recipe (the 19,519 / 9,291 / 0 gap, resolved)

**The three-way gap from 39cfd76:** the single-expert GEMV core measured on a real [4096,2048]
expert weight: the stock cast path = 19,519 NaNs ("fnuz-interp"), the kernel's
`v_f8_ld_tnsr_b + v_convert_f8_to_bf16_b` conversion = 9,291 NaNs, the CPU float()/e4m3fn
reference = 0 NaNs. The calibration target and the fix are now determined from the corpus:

### 9.1 The settled device-FP8 semantics (MEASURED on HPU — `g2-opt-manual` 03-porting.md §3.4.1)

| | device fp8 | checkpoint fp8 (OCP) |
|---|---|---|
| code assignment | e4m3fn, bias 7 | e4m3fn, bias 7 |
| max finite | **240 (`0x77`)** | **448 (`0x7E`)** |
| the `E=15` bin (bytes 0x78-0x7F / 0xF8-0xFF) | **reserved: `M==0` (0x78/0xF8) → ±inf, `M!=0` → NaN** | finite 256…448, plus NaN at 0x7F/0xFF |

Decisive facts (all MEASURED in that source): (1) `float8_e4m3fnuz` is NOT a supported dtype on
HPU — a fnuz-declared decode is rejected by the runtime; there is no fnuz code path on the
device; (2) the device's own encoder (`cast_to_fp8_v2`) emits bias-7 codes (1.0→0x38, 240→0x77)
— storage and decode agree on e4m3fn; (3) the device decoder (`cast_from_fp8`) matches the CPU
e4m3fn table **242/256** byte-for-byte (the 14 differing codes = the 0x78-0x7E/0xF8-0xFE top
bin) and the fnuz table 1/256; (4) the rule is engine-independent (TPC intrinsic and the
runtime cast agree).

**Reading the three counts against this:** the checkpoint is RAW OCP e4m3fn — it contains top-
bin bytes (values 256-448), which the TPC device decoder maps to inf/NaN: **the kernel's 9,291
= N(0x79-0x7E) + N(0xF9-0xFE)** (the device-NaN subset; if inf was counted in, add
N(0x78/0xF8)). The CPU reference = 0 NaN confirms N(0x7F/0xFF) = 0 (the quantizer emits no OCP
NaNs). **The 1Cat reference core (`block_fp8_dequant_gaudi2.c`, the "model contract") does NOT
handle the top bin — it is correct only for pre-converted weights (all |v| ≤ 240; the standard
conversion multiplies weights ×0.5 and scales ×2.0).** Our checkpoint is not pre-converted, so
the 1Cat core as-is produces the 9,291 NaNs — "follow the 1Cat core" is insufficient for our
data; the missing piece is in the project's own DSV4.1 port.

**The stock's 19,519:** a second convention (the agent's "fnuz-interp" hypothesis). The
decomposition comes from one byte histogram (§9.3) — under the fnuz reading the poison set is
{0x7E, 0xFE, 0x80} (fnuz: 0x7E = NaN, 0x80 = NaN-where-fn-has-negative-zero), so
19,519 ≈ N(0x7E/0xFE) + N(0x80) ± the 0x7F/0xFF class — the histogram settles it exactly.

**The production path is OCP-exact (inferred, strong):** the production MoE op consumes the raw
checkpoint and the suite is 100% — if the stock GEMM path used the device-decoder semantics,
the 9,291 NaNs/expert would poison the routed outputs catastrophically. The MME FP8 GEMM path
therefore implements full OCP e4m3fn (256-448 finite) — different from the TPC v_convert
decoder. **The Q21 kernel's parity target = the production semantics = the OCP e4m3fn CPU
reference (0 NaNs, exact 256-448 values) — NOT the device decoder, NOT the 1Cat-core-as-is.**

### 9.2 The kernel fix — the DSV4.1 top-bin remap (in the project's own tree)

The DSV4.1 port's dequant kernel (`csrc/deepseek_v4/include/deepseek_v4_dequant_gather_bf16.h`,
in the corpus; the DSV4.1 port is in our fork) already solves exactly this: for the top-bin
codes (120..126 / 248..254, the finite 256..448), **subtract 8 from the raw byte** (moves it to
the E=14 bin, preserving the mantissa and the sign: 0x78-0x7E → 0x70-0x76 = 128..224, and the
signed class 0xF8-0xFE → 0xF0-0xF6 = -128..-224), **convert with the standard bias-7 path, and
multiply by 2.0** → exactly 256..448 / -256..-448. Applied per-lane on the raw bytes before the
`v_convert_f8_to_bf16` step (the DSV4.1 code shows the exact TPC idiom: `v_u8_sub_vb(raw, 8, …,
extended, 0)` on the magnitude-selected lanes + the ×2 on the converted result). This covers
ALL 14 poison bytes (including the 0x78/0xF8 inf-mapped ones) and leaves the 242 in-range bytes
untouched → **bit-exact vs the OCP CPU reference, 0 NaNs.** The 0x7F/0xFF class (OCP NaN,
absent from the data — verify) can map to 0.

### 9.3 The calibration protocol (for the agent — no HPU for step 1)

1. **The byte histogram (CPU, no HPU):** 256-bin count over the [4096,2048] probe weight
   (int8 view). Verify: (a) N(0x7F/0xFF) = 0 (no OCP NaNs); (b) N(0x79-0x7E)+N(0xF9-0xFE)
   [= +N(0x78/0xF8) if inf was counted] = 9,291 (confirms the device-decoder reading of the
   kernel count); (c) decompose the stock's 19,519 by convention (fnuz poison {0x7E,0xFE,0x80}
   vs other — the histogram says which).
2. **The kernel patch:** the §9.2 remap in the 39cfd76 core (the magnitude-lane selection from
   the DSV4.1 idiom; the ×2 folded into the scale multiply — the scale is already fp32-then-
   bf16-rounded per the model contract, so ×2 can be applied to the bf16 scale with NO extra
   rounding: 2.0 is exact in bf16 — the per-block scale×2 is bit-exact, the value remap is
   bit-exact → the whole dequant is bit-exact vs the CPU reference).
3. **The parity gate:** kernel dequant output vs the 1Cat-form CPU reference
   (`weight.to(bf16)` elementwise + `scale.to(bf16)` BEFORE the multiply — the model contract)
   = **0 mismatches, 0 NaNs** on the probe weight (all 8.4M elements). Then the three-way
   re-run: stock / kernel / CPU — the kernel must read 0.
4. **Generalize before the rollout:** re-run the histogram on 2-3 more experts (different
   layers) — if any expert has N(0x7F/0xFF) > 0 or a different top-bin count class, the
   histogram is part of the rollout gate (one CPU call per expert).

**Why this unblocks the Q21 critical path:** the NaN gap was the 39cfd76 commit's named next
step ("the calibration is the next step, the deployment mechanism is proven"). With the target
defined (OCP-exact, the production semantics) and the fix identified (the in-tree DSV4.1 remap,
bit-exact, no checkpoint change, no scale-rounding change), the calibration is a one-kernel-
patch + one-CPU-histogram task — after which the per-route micro-benchmark (the RA-025
viability gate) is the only thing between the core and the rollout.

---

## 10. Addendum (round 5c) — The Q21 per-route micro-benchmark protocol (executable spec)

**Purpose:** the rollout viability gate for RA-002/RA-025 (the routing-aware block-FP8 GEMV kernel,
39cfd76). Measures the per-route (per-expert) DEVICE cost of the new kernel vs the stock path,
in-graph (no host dispatch), per slice width — the numbers that decide the 42-layer rollout and
feed the EP4×TP2 re-test and the bs=1 projection. Gates in order: §9 calibration → this bench →
rollout.

### 10.1 Prerequisites

- **G0 (correctness, the §9 recipe):** the E4M3 top-bin remap patch in the 39cfd76 core; the
  parity vs the OCP CPU reference = 0 mismatches / 0 NaNs over all 8.4M elements of the probe
  weight; the three-way re-run reads 0 for the kernel. **Latency numbers are only meaningful once
  the values are right.**
- **G0b (the bench weights, CPU-side, free):** the bench uses 8 DIFFERENT real experts from the
  checkpoint (expert IDs recorded). Run the §9.3 step-4 byte histogram on all 8 (CPU, no HPU) —
  the rollout's histogram gate and the bench setup are the same work. Verify all 8:
  N(0x7F/0xFF)=0 and the top-bin counts (the remap must be validated on bench data, not just the
  probe expert).

### 10.2 Setup

- 1 HPU (rank 0), no other workload, the 39cfd76 probe environment: `PT_HPU_LAZY_MODE=1`,
  `GC_KERNEL_PATH=<tpc lib>` (load after the HPU init), `PT_HPU_LAZY_ACC_PAR_MODE=0` (the
  custom-op-on-views requirement — the Q21 playbook; record the D7 outcome if 1 is also tested).
- **Weights:** 8 real experts, each = w13 [4096,4096] E4M3 + w2 [4096,2048] E4M3 + the fp32
  128×128 block scales (the exact checkpoint tensors — no synthetic data; the top-bin bytes are
  the whole point of the calibration).
- **x:** fixed [1,2048] bf16, fixed seed (the latency arms are data-independent at E4M3 — the
  remap's lane selection is per-lane uniform). The PARITY arm (§10.5) uses the real logged
  activation from one decode step.
- Record: the exact env, the GC-trace node count of the bench graph (a sanity check on the
  recipe size), the expert IDs, the histogram summary.

### 10.3 The graph design — 8 calls × 30 replays

For each (arm, slice width) build ONE small lazy graph containing **8 sequential op calls**:

- **Arm A (the kernel):** 8 × `block_fp8_gemv_glm` — call i uses expert i's weight slice, the
  shared x; the bf16 output is STORED to HBM (the lane-inserted vector store is real).
  **Dead-store check (once):** read back one output and verify non-zero — if the compiler
  eliminates the store, the bench is invalid.
- **Arm B (the stock reference):** the exact PRODUCTION op sequence for the same slice (the
  dequant op + the GEMM op as the compiler emits them for the MoE linear at that shape — NOT a
  reimplementation). Mandatory at FULL width (it is the parity reference AND the harness
  calibration); optional at half/quarter (its per-slice cost is the RA-035 non-MoE-linear data).

**Slice widths:** full [4096,2048] (8.4 MB — the 39cfd76 probe shape), half [2048,2048]
(4.2 MB), quarter [1024,2048] (2.1 MB). The full EXPERT is 3× the full slice (25.2 MB).

**Why 8 sequential calls:** (i) in-graph timing = device time only — the B2 launch overhead is
excluded by construction (this bench measures the kernel, not the 14 ms dispatch storm); (ii) a
1-op graph is degenerate (compiler specialization/sinking; the recipe boundary isn't amortized);
(iii) 8 DIFFERENT experts prevents any compiler op-dedup and matches the real routing diversity;
(iv) the recipe stays tiny → capture/compile cost negligible (T12-TEST, §11 below: the per-graph
COMPILE is the boot cost — the bench graph must not eat the boot budget).

**30 replays:** replay the graph 30×. **Report replay 1 separately** (the first-replay effect —
the D1 localization: per-launch input-hash/staging setup on the first replay of a shape), then
min/median/p90/max over replays 2-30. **Per-call cost = median(replay 2-30 wall) ÷ 8** (the
calls are sequential in the recipe — no overlap). HPU time per arm ≈ 30 × 8 × ~16 µs ≈ 4 ms —
negligible; the whole bench (6 arm×width configs + Arm B) fits in one boot.

### 10.4 The linearity check (the issue-bound test)

- t(half)/t(full) and t(quarter)/t(full) for Arm A.
  - **~0.5 / ~0.25 → bandwidth-bound:** the full-expert cost projects directly
    (t(25.2 MB expert) ≈ 3 × t(full slice) × (the w13/w2 shape factors)); the rollout gate is
    the absolute number (§10.6).
  - **Deviation upward (e.g. quarter = 0.4 of full) → a fixed per-call overhead (issue-bound —
    the Strata-audit class: 5-tile TPC GEMV at 237 GB/s = 10% of peak):** the D4 tile sweep +
    the pipelined-load variant are MANDATORY before rollout (the 39cfd76 commit names it:
    "the software pipelining is the ~4× target" — 11-16 µs un-pipelined [4096,2048] → the
    ~3-4 µs class).
- **D4 tile sweep (Phase 2, gated by the linearity result):** full width, TPCs-per-expert ∈
  {24, 8, 3, 1} (1.05 / 3.15 / 8.4 / 25.2 MB tiles per TPC) — find the issue-bound knee; report
  GB/s per config. (This is the existing D4 item; this section is its executable form.)
- **Harness calibration (Arm B, full width):** the stock sequence's achieved GB/s must land in
  the stock MME class (the 1.696/1.945/2.147 TB/s envelope, B10's 1.94 at the 16.8 MB tile). If
  Arm B reads far below that, the harness is broken (graph/recipe overhead), not the kernel —
  fix the harness before trusting Arm A.

### 10.5 The parity gate (the cookbook's method)

The full expert computation (w13 GEMV → clamped SwiGLU, swiglu_limit 10.0 from the fp8 quant
config → w2 GEMV) at T=1, same inputs, Arm A vs Arm B:

1. **NaN/Inf count = 0** in both arms (Arm A's = the G0 re-run on the bench experts).
2. **The diff metrics:** max abs diff, max rel diff (non-zero elements), max ULP, cosine
   similarity.
3. **The FP8-contract tolerance (NOT bit-exact):** the kernel's rounding = the scale rounded to
   BF16 before the multiply (the model contract) + the fp32 accumulation. The diff distribution
   must be EXPLAINED by those two: inspect the top-10 diff elements against the scale-rounding
   arithmetic. **A structural diff is a bug, not rounding:** any row off by a scale-factor
   multiple, a transposed slice (the innermost-first ws[0]=K/ws[1]=N glue invariant — the
   39cfd76 fix), or a missing 128-block = geometry/routing error → fail regardless of the
   aggregate metrics.
4. **VERIFY mode clean** on the op (the project's VERIFY harness).
5. The parity covers the FULL [4096,2048] output (all 4096 rows) at full width — the slice
   parity (half/quarter) is implied by the row-independence if the full passes, but verify the
   quarter once (it exercises the partial-store path, the 1Cat core's `second` flag).

### 10.6 The verdict (the rollout gate — RA-002 §10 numbers)

- **GATE-1 (parity):** pass as §10.5.
- **GATE-2 (per-route, the full 25.2 MB expert at T=1):**
  - **PASS: ≤ 16.8 µs (≥ 1.5 TB/s effective, ≥ 70% of B10's 1.94)** → single-layer integration →
    the 42-layer rollout + the parity A/B.
  - **MARGINAL: 16.8-40 µs (1.5 → 0.63 TB/s)** → project the bs=1 tok/s (§10.7), present to the
    owner WITH the pipelining plan (the ~4× target) before the rollout decision.
  - **FAIL: ≥ 40 µs (< 0.63 TB/s)** → the issue-bound regime: the D4 knee + the pipelined-load
    variant (the Strata GEMV_PIPE class — column-pair loads issued ahead) are mandatory; the
    4-HPU decision (RA-005) proceeds with the ~50-60 tok/s projection (RA-002 §10's failure
    mode, recorded, not hidden).
- **GATE-3 (end-to-end, after the rollout):** bs=1 median ≥ 80 tok/s (2×) with the adopted
  levers + RA-001; acceptance held; the suite ≤ 0.5 pp same-boot; @16 aggregate not regressed;
  the T > 32 fall-through to the stock op is clean (the kernel's fail-closed shape caps).

### 10.7 The bs=1 projection (report with every gate result)

MoE term = per-route µs × (routed experts per rank per step). The count: the D5 EP histogram /
the 91d6192 router dump at k=1 (measured); the byte-model fallback: 3.2 GB (k=0) - 15.9 GB
(k=4) per card per step ÷ 25.2 MB = 127-630 expert-reads/rank/step. Step time = current 26.1 ms
− the stock MoE term (15.5 ms slowest rank / 9.6 ms mean — the step-anatomy study; the slowest
rank is the binding one at bs=1, the EP imbalance) + the new MoE term. Report the tok/s at the
MEASURED per-route cost, plus the pipelined projection (× the measured un-pipelined/pipelined
ratio from the D4/pipelining runs).

### 10.8 The report format (the agent's deliverable)

- The table: (arm, W, TPCs) → {replay-1 µs/call, min/median/p90/max µs/call (replays 2-30),
  effective GB/s, NaN count} + the Arm A parity metrics.
- The linearity ratios + the D4 knee + the GATE-1/GATE-2 verdict + the §10.7 projection.
- The env record + the GC node count + the expert IDs + the histogram summary.
- **Anti-patterns (invalidate the bench if present):** dead-store elimination (no readback
  check); 8 calls on the SAME expert tensor (dedup risk + wrong model); out-of-graph timing
  (host dispatch pollutes the number — that is the B2 measurement, a different bench); replay 1
  in the median; a reimplemented stock path instead of the exact production op sequence (the
  harness must measure both arms identically).

---

## 11. Audit note (round 5d) — T12-TEST (a71ff64): the pinned-bucket refutation CONFIRMS the per-graph-compile theory

The implementation agent executed the T1.2 pinned-bucket experiment (a71ff64, 2026-11-11 00:58).
Findings vs this document's graph-compile cost theory:

1. **CONFIRMED — the capture cost is the per-graph COMPILE, not the graph count.** The pinned
   boot (the decode bs {1,4,16,32} × ctx subset) generated 215 graphs (the 49 file rows + 46
   spec-decode seed buckets + the 120 prompt) in **~29 min vs the adopted 277-graph boot's ~25
   min — SLOWER with FEWER graphs**. Per-graph: 8.1 s vs 5.4 s. The naive "fewer buckets →
   faster boot" (T1.2) is REFUTED, exactly as the theory predicted (the boot stage that scales
   is the GC compile per graph, 7.9-9.2 s class, not the capture count).
2. **NUANCE ADDED (theory refinement): the per-graph compile cost is SHAPE-REGULARITY
   DEPENDENT.** The sparse pinned ctx rows compile ~50% slower per graph (8.1 vs 5.4 s) than the
   dense exponential bucketing's regular ctx series. Interpretation: the compiler's
   shape-specialization / intermediate-compile-artifact reuse works better on the regular series
   (or the irregular sparse-ctx attention shapes do more per-graph work) — either way, the
   boot-time lever is the COMPILE-TIME state sharing (the recipe cache / regional compilation —
   the agent's own conclusion: "the boot-time work requires the compile-time lever (the regional
   compilation or the recipe cache), not the bucket reduction"), and the RA-031 (bs,k)-pinned
   file design must account for BOTH the graph count AND the shape regularity. The production
   stays on the exponential bucketing (the agent's decision — agreed: the −5% decode cost below
   plus the slower boot makes the naive pinning strictly worse).
3. **MEASURED COST OF PINNING: the decode 40.3 t/s (−5% vs the adopted ~42.4)** — the pinned
   ctx rows pad the active ctx more coarsely than the exponential series (the padding work is
   real device time). Any future pinned-file design (RA-031's (bs,k) file) must dense-coverage
   the observed ctx band, not just the bs grid.
4. **THE +46 SPEC-DECODE SEED BUCKETS** (49 file rows → 95 decode graphs): the MTP spec-decode
   path adds seed buckets per shape on top of the file rows — a design constraint for the RA-031
   file (the file's row count is not the graph count; the expansion factor ≈ 1.9× on decode).
5. **THE R003 NULL AS PREDICTED:** the suite 97.5% (the captured-draft flip is unrelated to
   bucketing — it is the inherent recipe-compilation drift, per the NONRECIP verdict). No
   surprise; the R003 unlock remains the T7 policy (owner) or the compiler RE.
6. **39cfd76 (the Q21 kernel, 02:48) is the newest commit** — the agent moved to the critical
   path (the kernel core executes; the calibration is next per §9; the micro-benchmark per §10).
   The agent's session is now HPU-heavy (the kernel bench); the no-HPU items (the §9.3 step-1
   histogram, the RA-034 census on the archived GC logs) can run in parallel without contending
   for the card.
