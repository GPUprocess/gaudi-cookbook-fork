# RA-013 — Expert placement by routing histogram (EP load balancing + Q21 byte-model input)

**Status:** PROPOSED (research architect, 2026-11-09) · **Priority:** 5 (after RA-011/Q21/RA-012; the histogram itself should start EARLY — it feeds the Q21 kernel design) · **Complexity:** Low-Medium
**Parent:** `BOTTLENECK_RESEARCH.md` (Section 3, RA-013) · **Targets:** B5 (EP imbalance / slowest-rank tail), the Q21 byte model (the distinct-expert distribution under real skew)

---

## 1. Executive summary

The expert→rank mapping is the checkpoint's shard order (rank r holds experts [36r, 36r+36)). The router's
per-expert popularity is non-uniform (the load-balancing loss keeps it CLOSE to uniform — not uniform).
Two consequences: (i) per-rank per-step token load is skewed → the slowest-rank GEMV tail (B5; the mxfp4
study measured slowest rank 19.5 vs mean 9.6 ms — re-measure on this tree per BG-1); (ii) the Q21
routed-only byte model (uniform balls-in-bins, correction C1) is an idealization — a skewed distribution
hits FEWER distinct experts than uniform, so Q21's @16 saving is LARGER than the 4-6.4 ms estimated, and
the bs=1 hot set is narrower than 8 uniform draws. The fix: compute the per-layer, per-expert route
histogram from representative traffic; reassign experts to ranks by interleaving the popularity-sorted
list (rank 0 gets the hottest, rank 1 the next-hottest, …). A PURE PERMUTATION of identical weights:
zero accuracy impact, zero byte change, a load-time change only. Precedent: project-maya's GLM port
allocates "VRAM slots per layer by routing share (`expert_counts.txt`)" and measured 35.5→32.9 ms/token
from the popularity-proportional allocation (tier D).

## 2. Current limitation (evidence)

| Fact | Value | Source |
|---|---|---|
| Expert→rank map | checkpoint shard order: rank r ↔ local experts [36r, 36r+36) (288 global / 8 ranks / 36 local) | the architecture in the diagnosis (the expert-shard map) |
| The imbalance (mxfp4 study — DIFFERENT precision/model) | slowest-rank MoE 19.5 ms vs mean 9.6 ms at high batch (the ~2× tail) | the diagnosis B5 (carried from the mxfp4 step-anatomy study — N1: re-measure per BG-1) |
| The router | a [4096→288] GEMM + top-8 (k=8) + the load-balancing auxiliary loss (the checkpoint is trained near-uniform, not uniform) | the model architecture; the port source (`model.py`) |
| The balls-in-bins math (uniform idealization) | bs=16/k=1: 32 verify rows × 8 = 256 global routes → ~32/rank into 36 local → E[distinct] = 36(1−e^(−32/36)) ≈ 21/36 = 59% (C1); a SKEWED distribution concentrates routes → E[distinct] < 59% (Zipf-like) | DERIVED (uniform); the skew is measured by this proposal (BG-2) |
| The placement precedent | project-maya GLM port: per-expert routing profile (`data/expert-profile-glm.bin`), "VRAM slots per layer by routing share (`expert_counts.txt`)" → 35.5→32.9 ms/token on 2×consumer GPUs | GaudiGrimoire tier D (github.com/mw00/project-maya, README + the GLM spec bench README) |
| The layout refutation context | Q17 (EP4×TP2) was refuted on the STOCK op — a LAYOUT change; placement is orthogonal (it helps under any layout) and is not what Q17 tested | project data (TASKS.md/EXPERIMENTS.md) |

**The gap:** no experiment in the project has re-sharded the expert→rank assignment (checked
EXPERIMENTS.md — no expert-placement entry), and the routing histogram has never been computed (BG-2
is new). The Q21 kernel (in progress) is being designed against the UNIFORM idealization — the measured
skew changes its byte model and possibly its expert-index-table sizing.

## 3. Proposed architecture

### Step 1 — the routing histogram (BG-2; no HPU time required)

Two options (either suffices; the CPU route is simpler):
- **Option A (CPU router pass):** load the router weights (the per-layer [4096→288] GEMM + the router
  norm — small, MB-scale) into CPU torch; run the bench prompt set (the standard suite prompts + a
  serving sample) through the model's ACTUAL hidden states — practically: capture the per-layer
  pre-router hidden states from ONE short serving run (a debug dump hook, or reuse a bench run with
  `GLM_ROUTER_FAST=1` + a dump flag), then the top-8 selection on CPU. Per layer: a histogram over the
  routes (288 bins). 45 layers × N prompts. CPU time: minutes (the top-8 over [rows, 288] is trivial).
- **Option B (on-device hook):** extend the router fast path (`GLM_ROUTER_FAST` — the top-8 is already
  computed on-device for the port) with a debug accumulator: per-layer per-expert route counts into a
  pinned CPU tensor over N=1,000 steps, dumped to .npz at the end. A small change; the cost is ~0
  (a scatter-add of 8 indices/layer/row).

Output: `routing_histogram.npz` — per layer (45), per global expert (288), route count. From it:
the per-layer popularity ranking, the skew index (e.g., the Gini / the top-10 share), the E[distinct]
distribution at bs=1/16/32 under the MEASURED distribution (sampling routes from the per-expert
probabilities, not uniform), and the bs=1 hot set (the experts that carry ≥X% of routes).

### Step 2 — the re-sharded load map

Per layer: sort the 36 local candidates… precisely: sort the 288 global experts by popularity (the
per-layer ranking); interleave to ranks: rank r gets positions r, r+8, r+16, … (rank 0: the #1, #9,
#17, #25, … hottest; rank 7: #8, #16, #24, …). The expected per-rank route load becomes
equal by construction (the standard round-robin-over-sorted load-balancing argument: the variance of
the per-rank route sum is minimized by interleaving the descending order). Record the map as a fixed
table (pinned from the histogram — a deterministic artifact committed to the experiment record, not
re-derived per boot).

The change: at expert-shard load time, rank r loads the experts at the interleaved positions (a
permutation of the existing shard-load code — the safetensors→rank assignment; no kernel, no
collective, no graph change; the weights land at the same per-rank HBM addresses with permuted
content). The Q21 kernel's per-rank expert index table uses the new map automatically (the kernel reads
routing indices, not shard order — the index table is built from the map at load).

### Step 3 — the A/B

Arms (same-boot protocol): arm 0 = current (shard-order) map; arm 1 = interleaved map (from the
committed table). Metrics: bs=1 / @16 / @32 bench t/s + the per-rank MoE kernel times from the trace
(BG-1's per-rank data is the before; the after is the A/B arm's trace) — the slowest/mean ratio is the
direct B5 metric.

## 4. Technical feasibility

- **Pure data placement:** identical weights, permuted assignment. No math change → the correctness
  risk is a buggy load map, not numerics (the parity gate should pass trivially — if it doesn't, the
  map is wrong, and that is a load bug to fix, not an accuracy question).
- **The load-balance argument is standard:** interleaving the popularity-sorted list equalizes the
  expected per-rank route count (the variance-minimizing assignment for a fixed per-rank capacity of 36
  slots). The residual skew (the within-position popularity spread) is what bounds the gain — with a
  near-uniform distribution (the load-balancing loss) the gain is modest but certain in sign.
- **The precedent measured the effect on this model class:** project-maya's popularity-proportional
  slot allocation: 35.5→32.9 ms/token (−7%) on their consumer-GPU engine (tier D). Their regime
  (all-experts-in-VRAM, per-layer slot sizing) differs from ours (fixed 36/rank, HBM-resident), but the
  mechanism (popularity → placement) is the same and the direction is confirmed.
- **No Gaudi-specific obstruction:** the EP all-reduce/communication pattern is unchanged (each rank
  still holds 36 experts and participates in the same collectives); only WHICH experts are local
  changes. The per-rank HBM footprint is unchanged (36 × the same expert size).
- **Layout-independent:** works at TP8+EP8 (current) AND at TP4+EP4 (the 4-HPU final deployment — 72
  local experts/rank, the same interleaving argument at 72 slots). The map is per-layout (recompute the
  interleave for the slot count) — trivial.

**Unsupported assumptions:** (A1) the B5 magnitude on THIS FP8 tree is similar to the mxfp4 study's
(2× mean tail) — BG-1's per-rank kernel times decide it; if our tree is already balanced (the FP8
routing may differ from mxfp4's), the gain is small and the value is the Q21 byte-model input alone.
(A2) the bench prompt set + serving sample is representative of the production traffic distribution
(the histogram is only as good as its sample — record the sample composition in the experiment).

## 5. Expected performance impact (all ESTIMATED)

| Metric | Current | After (ESTIMATED) | Notes |
|---|---|---|---|
| Slowest/mean per-rank MoE kernel time @16 | ~2× (the mxfp4 figure; re-measure per BG-1) | ~1.2-1.5× | the interleaving residual (within-position spread) |
| bs=16 step (8 HPU) | 114 ms (pre-RA-011) | −1 to −3 ms | the B5 tail is a fraction of the step; larger after RA-011 (the device term becomes the ceiling — a −X ms device tail is then a −X ms step) |
| bs=16-32 step (4 HPU) | the 2× per-card weight regime | −2 to −5 ms | the larger per-rank expert count (72) makes the imbalance proportionally larger |
| bs=1 | 45.1 ms | ≤ −1 ms | the hot-set effect (narrower distinct set → Q21's bs=1 floor is even lower than C2's 0.86-1.26 ms) |
| The Q21 byte model | uniform E[distinct] = 21/36 @16 (59%, C1) | the MEASURED distribution — expected 5-20% below uniform (typical router skew) | the Q21 @16 saving rises from ~4-6.4 ms toward ~5-8 ms (8 HPU) / proportionally more at 4 HPU |

- **Conservative:** −1 to −2 ms at @16 + the byte-model input (which improves the Q21 design either
  way).
- **Optimistic:** −4 to −5 ms at @16-32 (if BG-1 confirms a 2×-class tail on our tree) + the tighter
  Q21 model.

## 6. Risks

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | A buggy load map (an expert duplicated/missing across ranks) | correctness | The parity gate (greedy identical — the permutation is exact; any flip = the map is wrong); a load-time assertion: each of the 288 global experts appears on exactly one rank (a checksum of the per-rank expert-ID table vs the full set) |
| R2 | The histogram sample is unrepresentative (code-heavy bench prompts vs a mixed production mix) | validity | Record the sample composition; if production traffic is known to differ, re-run the histogram on a production sample (the Option-B hook makes this a soak, not a new experiment) |
| R3 | The map is boot-pinned but the Q21 kernel's index table is built from a STALE map (a config drift between the load map and the kernel table) | correctness | Single source of truth: the committed table file feeds BOTH the shard load and the Q21 index-table build (one artifact, two consumers); the experiment record pins the table hash |
| R4 | The gain is small (A1 — the FP8 tree is already balanced) | scope | The histogram (BG-2) has standalone value (the Q21 byte model); the A/B is cheap (1 boot, a load-time permutation); a null result closes the item with the measurement recorded |
| R5 | The 4-HPU re-derivation (72 slots) | correctness | The interleave is a function of (ranking, slot count) — unit-testable on CPU against the checksum assertion before any HPU boot |

## 7. Dependencies

- **BG-1** (the per-rank MoE kernel times on our tree) — sizes the B5 effect; OR the histogram alone
  (independent, no HPU time) if the A/B is deferred.
- The bench prompt set + a serving sample (the histogram input).
- Composes with Q21 (the byte model + the index table — R3's single-source-of-truth rule), RA-011
  (orthogonal; the B5 tail matters more AFTER RA-011 because the device becomes the ceiling), and the
  4-HPU layout (the map is re-derived at 72 slots — the function is the same).
- No kernel work. No graph change. No collective change.

## 8. Implementation instructions

**Step 1 — the histogram (start NOW; no HPU time):**
1. Option A (preferred for speed): add a debug dump flag to the router fast path (`GLM_ROUTER_FAST`):
   during one bench run, dump the per-layer pre-router hidden states + compute the top-8 on CPU over
   the bench prompt set; accumulate the per-layer 288-bin histograms. (Alternatively Option B: the
   on-device scatter-add accumulator over N=1,000 steps — the soak-friendly variant.)
2. Output: `routing_histogram.npz` (per layer, per global expert, count) + the derived artifacts:
   per-layer popularity ranking, the skew index, the E[distinct] samples at bs=1/16/32 (sampled from
   the per-expert probabilities), the bs=1 hot set. Commit to the experiment record (BG-2 closed).
   **Feed the E[distinct] distribution to the Q21 design immediately** (the byte model update — C1's
   uniform idealization is replaced by the measured distribution).

**Step 2 — the re-sharded map:**
3. The interleave function: `(ranking, slot_count) → per-rank expert list`; unit-test on CPU: the
   checksum assertion (288 unique, 36/rank), the expected per-rank load vs the uniform baseline (the
   variance reduction is the unit-level proof).
4. Wire the map into the expert-shard load (env-gated: `GLM_EP_PLACEMENT=<table-file>`; default = the
   shard order — fail-safe). The Q21 index-table build reads the SAME table file (R3).

**Step 3 — the A/B (1 boot family):**
5. Arm 0 = current map; arm 1 = interleaved map. Same-boot protocol. bs=1 / @16 / @32 + the per-rank
   MoE kernel times from the trace (the before/after of the slowest/mean ratio). Parity gate (R1).
6. Record EXP-RA013: the histogram + the table hash + the A/B numbers + the slowest/mean ratio
   before/after.

**Diagnostics to capture:** the per-rank kernel-time distribution (the histogram of rank times, not
just the max/mean), the per-layer load variance before/after (the mechanism proof), the E[distinct]
distribution (the Q21 input).

## 9. Validation criteria

1. **Parity:** greedy bit-identical vs arm 0 (the permutation is exact — any flip = a load-map bug,
   R1).
2. **The load-balance mechanism:** the per-layer per-rank route-load variance drops (the unit-level
   proof) AND the per-rank kernel-time slowest/mean ratio drops at @16 (target: from the BG-1
   baseline toward ≤1.5×).
3. **Performance:** @16 bench ≥ arm 0 (no regression) with a target of +1 to +3%; bs=1 unchanged
   within noise.
4. **The Q21 input:** the E[distinct] distribution is recorded (BG-2 closed) regardless of the bench
   outcome — this is a validation success even if the A/B is null (R4).
5. **The 4-HPU map (when the 4-HPU config is next booted):** the 72-slot interleave passes the
   checksum assertion + the parity gate (no separate bench needed until the 4-HPU work starts).

## 10. Novelty assessment

- **Not previously attempted:** no expert-placement experiment in the project (checked EXPERIMENTS.md —
  no entry); the routing histogram has never been computed (BG-2 is new).
- **Closest prior work and the difference:**
  - Q17 (EP4×TP2, refuted): a parallelism LAYOUT change (TP/EP split), tested on the stock op — it
    changed how the ranks cooperate, not which experts are local. Placement is orthogonal and would
    help under ANY layout; Q17's refutation does not touch it.
  - B5's mxfp4 measurement: the effect is documented (2× tail) but never attacked — and it is from a
    different precision/model (N1).
  - project-maya's slot allocation: the precedent is for a consumer-GPU tiering engine (a different
    regime — all-VRAM vs HBM-resident); what is transferable is the POPULARITY→PLACEMENT mechanism +
    the measured direction, not their numbers.
- **Why it deserves investigation now:** (a) the histogram (BG-2) is no-HPU-time and feeds the in-
  progress Q21 kernel design TODAY (the byte model under real skew — C1 shows the uniform idealization
  understates Q21's value by 5-20%); (b) the A/B is a load-time permutation (1 boot, days); (c) it is
  the only item that attacks B5 (the high-batch slowest-rank tail) — which becomes the BINDING term
  after RA-011 (the device tail under the host ceiling); (d) zero accuracy risk (pure permutation).
