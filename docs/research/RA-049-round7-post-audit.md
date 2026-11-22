# RA-049..RA-052 — Round 7 (2026-11-22): the post-audit round

Companion detail to `EXTERNAL_RESEARCH_PROPOSALS.md` revision 16 (the master doc carries
the pointers; this file carries the full Section-8 specifications).

**Context for the round:** the owner's two directives of 2026-11-22 (FP8 KV cache
prohibited ever; the KV cache stays at BF16 in every configuration — RJ-29, RA-042
WITHDRAWN) + the full audit of the 11-21 → 11-22 implementation arc (23/23 proposals
accounted; the production is now k=1 + T1 + Q01v2 at bs1 51.4-53.9 / @16 181.3 bf16; W7
measured; RA-039 refuted by the vendor gate; the 4,173-graph census banked). The new
proposals below target the remaining path to 200 tok/s bs=1 and 800 tok/s @16.

Evidence labels: MEASURED (project or cited third-party run), VENDOR (Tier B corpus),
DERIVED (theoretical computation from measured inputs — shown). GaudiGrimoire citations
in [GG: ...].

---

## 0. Execution order (the do-first sequence for the implementation agent)

1. **Route to the owner** (no HPU time): the RA-049 drafter decision (the acceptance wall
   makes the drafter the critical path to both targets) + the T7 tie-aware gate policy
   (the agent's recorded design).
2. **RA-051** — the @16 T1-package attribution A/B matrix (3-5 boots; the first HPU
   action; the cheapest large win remaining; +40-52% @16 in the best case).
3. **RA-052 step 1** — D22 (no HPU time, 0.5-1 day): the per-step node-cost model over
   the banked census; then the FX-pass skeleton + the CPU contract test; then the boot
   (weeks-class overall).
4. **RA-050** — the top_p-nucleus-bounded sampler (1-2 weeks, parallel with 3; the CPU
   distribution test first; one A/B boot at the end).
5. **RA-041 / RA-043** per the agent's staged plans (with the rev-16 RA-041 constant
   corrections applied first).
6. **D17 1M boot window** (RA-047's gate).
7. **D16 re-queue** (the batched-shape CPU test, no HPU time first).

---

## RA-049 — The acceptance wall: the target arithmetic under the measured W7 ceiling
### (the drafter is the critical path — owner decision item)

**Status:** PROPOSED (owner decision) · **Complexity:** none (framing + decision
package) · **Effort:** the decision; the drafter options are High (weeks-class) if
authorized.

### 1. Executive summary

The 11-22 measurement of the graphed MTP=4 chain's acceptance (AL(4) = 1.87 tok/step =
the k=1 band; temperature- and prompt-type-invariant) confirms W7: the single published
MTP head does not convert speculation depth into acceptance. The arithmetic below shows
both project targets (200 tok/s bs=1, 800 tok/s @16) are **drafter-gated** under the
measured acceptance ceiling: at AL ≈ 1.88 the required step times (9.4 ms bs1 / 37.6 ms
@16) are 3.7-4.4× below what any known bf16 lever stack can deliver (the stack's best
case: ~22-28 ms bs1 / ~105-125 ms @16). At a drafter-class acceptance (AL ≥ 4.5-4.8,
the DFlash2-class number) the required step times (≤24 ms bs1 / ≤104 ms @16) are inside
the lever stack's best case. **The drafter is therefore the critical path to both
targets; the bf16 lever stack (RA-051/052/050, RA-043/041/047) is the substrate the
drafter needs (the step-cost term) and buys the intermediate milestones (bs1 ~67-90,
@16 ~240-350) either way.** This is an owner decision: which drafter route to authorize,
given the owner's existing rulings (no training hardware; DFlash2 parked on license).

### 2. Current limitation (MEASURED, 11-22)

- AL(k=1) = 1.65-1.95 tok/step (the counter-delta band, prompt-dependent).
- AL(4) on the graphed chain = **1.87** tok/step (4000 generated / 2143 draft-steps,
  fixed 8×500-token workload) — inside the k=1 band; the acceptance program: temperature
  0.0-1.0 → 44.2-49.1 tok/s (invariant); math/code/prose/summary/translation/extraction
  → 49.6-52.5 tok/s (±3%, no prompt class collapses acceptance); the chain is −6% at bs1
  vs k=1 (50.5 vs 53.9 tok/s) → PARKED, k=1 is production.
- The k=1 production step ≈ **34.9 ms** (1.88 tok/step ÷ 53.9 tok/s). The chain step
  ≈ 37 ms (the agent's decomposition: 4 sequential draft replays ~3 ms each + the ~25 ms
  verify). The @16 production step (bf16, 181.3 tok/s, 30.08 tok/step at AL 1.88) ≈
  **166 ms**. The fp8-KV data point (prohibited, record only): the same config at
  346.0 tok/s = 86.8 ms/step.
- W7 mechanism (the agent's, consistent with the literature on MTP modules): the head at
  depth j is conditioned on its own draft tokens 1..j-1; once the first draft is wrong
  the conditioning is wrong, so the marginal acceptance beyond the first position is
  ~0 (1.87 ≈ 1.88: the extra 3 passes add nothing). The head sees one distribution; the
  deeper drafts re-propose what the head already missed.

### 3. Proposed change (none to the implementation — the decision package)

The targets require a drafter with AL ≥ ~4.5-4.8 (block-drafter class: one pass drafts
the whole verify block). The options, with their blockers as of this audit:

| Option | Mechanism | Blockers (11-22 state) | DERIVED payoff |
|---|---|---|---|
| (a) DFlash2 HPU port | incoai's 5-layer sliding-window block-drafter + top-16 path selector, one pass drafts the 8-token block, conditioned on target layers [5,14,24,33,42] | **B1:** the 1Cat `flashinfer_gaudi` compiled GDN op is absent on our server (the HPU port needs the package; the package builds from source, Apache-2.0 — buildable); **B5:** CC BY-NC-ND 4.0 (the owner's 11-19 ruling parked it: "no training hardware: prefill met — decode focus" — the ruling deprioritized, it did not prohibit the checkpoint) | inco's Qwen evidence: acceptance 4.80 vs 4.28 native MTP (Tier D); 1Cat v1 HPU implementation precedent + the Strata port exist (Tier D) |
| (b) DFlash2 on a separate CUDA GPU | the drafter runs natively on CUDA (standard sliding-window attention — NO GDN op needed; B1 sidestepped); the draft tokens + target features cross PCIe (CPU-mediated sync ~1-3 ms; no CUDA↔Synapse interop — the host is the bridge); the verify stays on HPU | the cross-device sync on the critical path (~1-3 ms/step); B5 still applies (the checkpoint is the license object regardless of device); a second device + driver stack on the node | bs1: 4.8 tok/step ÷ (34.9 + ~5-7 ms drafter+sync) ≈ **120-160 tok/s** (2.2-3.0× the current 53.9); @16: the 128-row verify (16 seqs × 8 rows) is the cost term — the 80-row shape already depressed the k=4 chain @16 (the capture-depressed 133.5 first bench) — **600-800 tok/s, borderline** |
| (c) Self-trained DSpline-class feature drafter (RTX 3090, 24 GB) | a small drafter (5-18 class, the GaudiInfer reference architecture — ring-cache + target-feature conditioning, Tier D) trained on saved target hidden-state features from the HPU forward passes (the DSpline pattern); runs on the 3090 like (b) or, if small enough (a 1.2B-class drafter ≈ 2.4 GB bf16), could in principle run HPU-side later | **adding the 3090 = adding training hardware — the owner's 11-14 ruling ("no training hardware") is the constraint; this is the owner's call** (it is a ~$1,500-2,000 used card, the cheapest way to get drafter-training capability); training time: days-class on one 3090 for a 5-18B drafter at our batch | own weights (sidesteps B5 entirely); acceptance: the DSpline pattern targets 4+ AL on the class (the GaudiInfer reference measured its drafter's acceptance on their 27B — Tier D, model-dependent ±); the feature-saved training data is free (a dump hook on the HPU side, no extra forward cost) |
| (d) Head-reuse / no-draft-forward | replace the MTP layer's forward with an approximation (reuse the target's last-layer features via a light head; skip the draft's KDA/MoE traversal) | untrained → the approximation's acceptance is unknown and the accuracy risk is unbounded (a draft from the wrong distribution is worse than no draft: it burns verify rows for rejected tokens); the agent recorded it as a candidate but there is no concrete design | rejected as a proposal; recorded so it is not re-derived from scratch |

### 4. The target arithmetic (DERIVED — the computation is shown)

Notation: AL = accepted tokens per step (incl. the bonus); step = ms per decode step.

**bs=1 (target 200 tok/s):**
- at AL 1.88: step ≤ 1.88/200 s = **9.4 ms** — vs the current 34.9 ms (3.7× below).
  The bf16 lever stack's bs1 effect: RA-052 (host replay floor 25-45 ms → 8-15 ms;
  device MoE −5-15 ms at bs1) + RA-050 (−0.2-0.5 ms) → best case ~22-28 ms →
  1.88/0.022-0.028 = **67-85 tok/s**. (The stack cannot reach 200 at AL 1.88 — no known
  lever removes a factor of 3.7 from a 34.9 ms step; the host floor after the node
  collapse is the wall: 8-15 ms + the device ~10-13 ms.)
- at AL 4.8 (drafter): step ≤ 4.8/200 = **24 ms**. The lever-stack-best-case step
  22-28 ms + the drafter's marginal pass cost (a block-drafter adds ~1 pass: ~3-7 ms
  on the GPU-resident variants (b)/(c); the HPU-resident (a) adds ~0.8-6.4 ms per the
  D9-class draft-pass bytes) → 25-35 ms → **137-192 tok/s at AL 4.8**; at AL 5.2
  (the DFlash2 headroom): 149-208 tok/s. **200 bs=1 needs AL ≥ ~4.9-5.2 AND the lever
  stack at its best case** — probability per the rev-5 physics: ~15-25% (the host ≤
  10-13 ms term is the binding constraint; the drafter must be the cheap single-pass
  class, not a 4-pass chain).
- at AL 2.68 (the eager-k=4 probe's historical number, MTP4-T0): step ≤ 13.4 ms —
  below even the stack's best case at bs1. **Even the optimistic MTP-depth number does
  not reach 200 bs=1** — the ceiling is structural, not just the graphed chain's defect.

**@16 (target 800 tok/s; 16 seqs × AL tok/step per step):**
- at AL 1.88 (30.08 tok/step): step ≤ 30.08/800 = **37.6 ms** — vs the 166 ms bf16
  production step (4.4× below). The bf16 lever stack's @16 effect: RA-051 (181 →
  255-277, i.e. 166 → 109-118 ms) + RA-052 (−10-25 ms device+host) + RA-050 (−2-3 ms)
  → **~105-125 ms → 240-285 tok/s**. (The fp8 data point — 86.8 ms/step — shows the
  batched-shape device term is KV/attention-sensitive; with the KV locked to bf16 the
  MoE collapse (RA-052) is the remaining device lever for that term.)
- at AL 4.8 (76.8 tok/step): step ≤ 76.8/800 = **104 ms** — the lever stack lands
  105-125 ms → **615-731 tok/s**; at AL 5.2 (83.2 tok/step): 666-792 tok/s; the
  GPU-resident drafter (b)/(c) trades the HPU draft cost for the sync (~1-3 ms) + the
  wider verify (the 128-row shape) — net: **700-850 tok/s, i.e. 800 @16 is reachable at
  AL ≥ ~4.9-5.2 with the stack, or AL 4.8 with ~20 ms of further step reduction.**

**The intermediate milestones (what the stack buys without a drafter):** bs1 ~67-90
(current 53.9 → +24-67%), @16 ~240-350 (current 181.3 → +32-93%). These are real and
worth banked, but they do not meet the stated targets.

### 5. Expected performance impact

- Target metrics: 200 bs=1 / 800 @16.
- Best case (drafter authorized + stack at best case + drafter AL ≥ 4.9-5.2): both
  targets met (200 bs=1 at ~15-25% probability per the rev-5 physics; 800 @16 at
  ~40-55% — the @16 step budget is looser).
- Conservative (drafter AL 4.5-4.8 + stack median): bs1 ~150-190, @16 ~700-800.
- Worst (no drafter, owner declines; stack median): bs1 ~75-85, @16 ~250-285 — the
  targets are formally unmet; the intermediate milestones are banked.

### 6. Risks

- **Acceptance risk of the drafter itself:** a drafter from a different distribution
  than the target has acceptance model-dependent (inco's 4.80 is on Qwen, not
  GLM-5.3-Flash); the A8 (acceptance on our suite) gate is the go/no-go before any
  production cutover — the same gate the MTP4-T0 probe used (calibrated: the probe's
  2.68 vs the graphed 1.87 shows the workload sensitivity, so the drafter's A8 must be
  measured on OUR prompts, not borrowed).
- **Cross-device sync (options b/c):** the ~1-3 ms CPU-mediated round trip is on the
  critical path; the CUDA↔Synapse interop is a non-supported path (the host is the
  bridge — the D2H/H2D copies are the cost, not the bandwidth: the draft tokens are
  16×2 B, the target features are 5 layers × 8 KB = 40 KB/token → ~640 KB/batch at
  bs16 — 27 µs at 24 GB/s; the latency, not the bandwidth, is the cost).
- **License (option a/b):** CC BY-NC-ND 4.0 — non-commercial; the owner's ruling stands
  as the decision input (research evaluation unblocked; commercial deployment needs a
  license from inco.ai).
- **Hardware (option c):** the 3090 on an 8×350 W HPU node — power/thermal headroom is
  a chassis question (a minor point); the training data dump (the target features) is a
  one-time HPU-side hook with no serving cost.
- **Numerics (all drafter options):** the draft distribution changes the acceptance
  walk — the R003-class knife-edge sensitivity applies to the verify, not the draft
  (the draft never changes content — the verify is the target model; the T1 invariance
  package covers the verify's numerics).

### 7. Dependencies

- The bf16 lever stack (RA-051/052/050) — the step-cost substrate (a drafter at AL 4.8
  on a 166 ms @16 step yields only 462 tok/s; the stack must land first or in parallel).
- For (a): the 1Cat package build (Apache-2.0 source) + the v1 HPU wiring (the B1-B5
  blocker list from the 11-19 feedback) + the KDA rollback path (the draft's
  accept/reject must restore the KDA state — the RA-043 deferred-commit design is the
  substrate).
- For (b)/(c): the 3090 (owner call for (c)); the feature-dump hook (the target
  hidden-state capture at layers [5,14,24,33,42], one-time, days of data).
- The T7 gate policy (owner decision item) — a new drafter is a new numerics surface;
  the tie-aware gate is the right acceptance instrument for it.

### 8. Validation criteria (for the implementation agent, once a route is authorized)

1. **A8 gate (the go/no-go):** the drafter's acceptance on our prompt suite (the 8×500
   workload + the 6 prompt classes of the acceptance program), at bs1 and @16, greedy
   and top_p 0.95 — the bar: AL ≥ 4.5 at bs1 (the 200 bs=1 math needs ≥ 4.9-5.2 with
   the stack; 4.5 is the @16 floor).
2. The parity gate: 1/8 on the 8-identical-prompt probe under the drafter config (the
   draft changes the verify rows but NOT the content — the target model's distribution
   is what's sampled; the T1 package stays on).
3. The golden re-baseline under the drafter config (the verify-row geometry changes the
   recipe — the golden reference is shape-dependent).
4. The warm bench: bs1 + @16 vs the production (the 53.9 / 181.3 baseline), after the
   lever stack is banked (the drafter's gain is measured on the substrate, not on the
   old stack).
5. The accuracy suite (40 items × 5 reps, ≤ 0.5 pp) — the drafter changes which tokens
   get verified, not the target's distribution, so the expectation is 100.0% on greedy
   (the suite is greedy) — any regression is a verify-path bug, not a drafter property.

### 9. Novelty assessment

Not a retry of any closed experiment: RA-004 (the MTP k=4 theories) and RA-040 (the
graphed k=2 chain) are closed by the W7 measurement — they are MTP-depth proposals, and
W7 is exactly the statement that MTP depth does not convert into acceptance on this
model. The block-drafter class (one pass drafts the whole block, AL ≥ 4.5) is a
different mechanism (a different distribution generator, not a deeper autoregressive
chain). RA-009 (DFlash2) is the same class — this round re-opens it as the decision
item it was always, with the measured arithmetic that makes it the critical path (the
11-19 feedback parked it on "no training hardware: prefill met — decode focus"; the
decode focus is now precisely the drafter). The 3090 option is new (it is the
implementation of the owner's own earlier question, "is it worth adding an extra GPU" —
answered "no for the MTP model" on 11-21; the drafter is the scenario that answer was
scoped to exclude, and the measured AL wall now makes it the decision the owner should
make with the full arithmetic).

---

## RA-050 — The top_p-nucleus-bounded gumbel sampler (the census's dominant TPC guid
### class, made exact)

**Status:** PROPOSED · **Complexity:** Medium (a new HPU sampler op + the CPU
distribution test + one A/B boot) · **Effort:** 1-2 weeks.

### 1. Executive summary

The decode sample (temperature 1.0, top_p 0.95, no top_k) is a **gumbel-max over the
full 151,936-token vocabulary**: an Exp(1) draw per token per row, sampled =
argmax_i (p_i / E_i). The draw is materialized inside the captured graph — the census's
`philox_random_uniform_bf16` (124-248 nodes in each of the 16 sampler graphs, 1,992
total — the dominant TPC guid class within those graphs). **top_p assigns exactly zero
probability to tokens outside the nucleus** (the nucleus is the smallest prefix with
cumulative mass ≥ p, renormalized). Tokens with p = 0 can never win a gumbel race
(0/E = 0 for any finite E). Therefore **bounding the draw to the nucleus — a fixed
K_max = 8192 at capture — preserves the sampling distribution EXACTLY** for every row
whose true nucleus is ≤ K_max (greedy is untouched by construction; the residual shift
for the measure-zero rows with nucleus > K_max is bounded by 1 − cumsum(8192) ≈ 0 at
p = 0.95). The fix removes ~93-97% of the sampler's philox nodes and ~95% of the
full-vocab race device work, and it removes the lazy-mode RNG fusion hazard class (the
documented mis-execution genre). The agent's ported `topk-subset-sampler` (the upstream
cookbook fix) is INERT for this model (it engages only when a row has top_k; our
generation_config has none) — this proposal is its designed top_p-only successor, which
the agent recorded as "the candidate for the next round" without a design. This is that
design.

### 2. Current limitation (MEASURED)

- The census (4,173 post-graphs, banked 11-22): `philox_random_uniform_bf16` = 124-248
  nodes in each of the 16 sampler graphs (median 124; max 248; total 1,992); the
  D15-session "40% of TPC" is the within-sampler-graph share (~310 TPC nodes each).
- The topk-subset-sampler A/B (11-22 05:25-06:00): the fix applied to both trees
  (`vllm/v1/sample/ops/topk_topp_sampler.py` + `vllm_gaudi/v1/worker/hpu_input_batch.py`,
  the pinned upstream 1540263), gate-clean (golden 12/12 identical, tools 8/8), the
  fresh census delta **ZERO** — root cause: `generation_config.json` has NO `top_k`
  (temperature 1.0, top_p 0.95 only); the fix requires k non-None, so the HPU subset
  path never engages; "the philox block comes from the top_p full-vocab path the fix
  does not address."
- The mechanism (corpus-verified, the master doc §5.1): the main non-greedy path's
  `random_sample` draws `exponential_()` over the full `[batch, vocab]` probs (the
  vendor test mocks exactly this: `q.exponential_()` over the full tensor); the abotsis
  fork's non-greedy spec path (`_random_path_device`, Tier D, commit 919ec98) computes
  the per-row nucleus IN-GRAPH (one descending sort; `n_keep = min(k, (cum - probs) <
  top_p).sum()`) and documents the RNG hazard: "fusing anything downstream of the RNG
  into a lazy graph mis-executed here (see `sample_recovered_tokens`)".

### 3. Proposed architecture

A new HPU op `top_p_nucleus_sample` (or a modification of the existing sampler's HPU
path), in-graph, static shape:

1. **Nucleus (reuse):** the per-row sort + cumsum + cutoff that the current recipe
   already computes for the top_p masking (the abotsis `_random_path_device` shows it
   in-graph; our recipe has the equivalent — verify in the dump JSONs, the
   `logical_slice`/`sort` chain in the 16 sampler graphs).
2. **Clamp:** `K = min(n_keep, K_max)`, K_max = 8192 (1/18.5 of the vocab).
3. **Bounded draw:** `E ~ Exp(1)` for `[rows, K_max]` (one philox node per value-chunk —
   the chunk size is the compiler's; the node count scales with the drawn values, so
   124-248 → ~7-20).
4. **Bounded race:** `sample = argmax_i (p_masked_i / E_i)` over the K_max slots; slots
   beyond the row's n_keep get p_masked = 0 (the existing masking) → 0/E = 0 → never
   win. The full-vocab division + argmax over `[rows, 151,936]` fp32 (19.5 MB at 32
   rows) becomes `[rows, 8192]` (1.0 MB) — ~18.5× less.
5. **Return** the token id (the host sees the same interface as today).

No host sync, no new D2H, no conditional graph branch (the clamp is data-dependent but
shape-static — K_max is the capture constant). The greedy rows (temperature 0/-1
sentinel) keep the current argmax-only path (no draw at all — unchanged).

**Distribution argument (exactness):** gumbel-max: if E_i = Exp(1) i.i.d., then
argmax_i (p_i/E_i) ~ Categorical(p) — this holds over ANY support containing the full
support of p. The top_p-constrained p has support = the nucleus (p_i = 0 outside by
construction of the masking the recipe already applies). For rows with nucleus size
≤ K_max: the bounded race's support = the same support → **identical distribution,
exactly, not approximately**. For rows with nucleus > K_max (at p = 0.95: a row whose
top-8192 tokens carry < 95% of the mass — effectively measure-zero for a 151,936-vocab
language model at temperature 1.0; it would require > 8192 tokens sharing the residual
5%): the tokens between K_max and the true cutoff lose their mass → the shift is
bounded by the residual mass 1 − cumsum(K_max) < 5% (and typically ≪ 1%). If the owner
wants the shift literally impossible, K_max = 16384 (9.3× reduction, the residual
< 2.5%) — the node savings are still 9×.

**Why K_max, not per-row variable K:** the graph is captured at a static shape; the
draw count is a tensor dimension. A per-row variable draw would need a data-dependent
graph (not capturable) or a per-row masked draw over a fixed max (which is exactly the
K_max design). The wasted draws for short-nucleus rows (K_max vs n_keep ≈ 100-2000)
are cheap: the philox throughput is the cost, and K_max = 8192 is 18.5× below the
status quo regardless.

### 4. Technical feasibility

- **In-graph static shape:** the nucleus computation is already in-graph (the current
  recipe masks with it — the sort + cumsum are in the 16 sampler graphs per the
  census); the bounded draw + race are standard ops (the philox draw, the masked div,
  the argmax) — all TPC/MME, no new engine class.
- **The RNG hazard class:** the abotsis note says downstream-of-RNG fusion mis-executed
  — the design keeps the race (div + argmax) in the same lazy segment as the draw, on
  a private fp32 copy (their documented safe pattern: "Everything runs in one lazy
  segment bracketed by mark_step()s, on a private fp32 copy of the logits (no in-place
  op touches the sampler's tensor)"). The CPU contract test (the distribution
  equivalence, item 8) is the guard.
- **The reference plumbing exists:** the abotsis fork (Tier D) has the in-graph
  nucleus; the upstream topk-subset-sampler (ported, in our tree) has the HPU subset
  path structure (the `hpu_input_batch.py` half of the port) — the top_p-only op is
  the same pattern with the top_k condition replaced by the top_p cutoff.
- **Unsupported assumptions:** (i) that our v0.26.0 recipe's top_p masking is the same
  sort+cumsum pattern (verify in the dump JSONs — if our recipe masks by a different
  mechanism, the nucleus step is written fresh — 1-2 days); (ii) that the philox node
  count scales linearly with the drawn values (the census shows 124-248 across the
  16 shapes — the shape scaling is rows × vocab; the K_max scaling should follow —
  the fresh census after the fix is the measurement).

### 5. Expected performance impact (DERIVED)

- Philox nodes per step: 124-248 → ~7-20 (×8192/151,936 = ×0.054) — **−93-97%**.
- Host replay: 124-248 × ~1.4 µs = 0.17-0.35 ms → 0.01-0.03 ms — saves **0.15-0.3 ms/
  step** (bs1: +0.4-0.9% on a 34.9 ms step; @16: +0.1-0.2% on a 166 ms step — the
  node count is shape-static, not row-scaled).
- Device: the full-vocab race `[rows, 151,936]` fp32 div + argmax → `[rows, 8192]` —
  ~18.5× less work at the batched shapes; estimated **0.5-2 ms at @16** (the rows
  scale: 32 verify rows × 19.5 MB → 1.0 MB; the TPC/argmax throughput on the current
  class is not measured — D22 prices it), < 0.1 ms at bs1 (2 rows).
- **Total: bs1 +0.5-2% (53.9 → 54-55), @16 +2-5% (181.3 → 185-190).** Modest — the
  honest number. The non-performance values: (i) the EXACT distribution (no accuracy
  risk class — the greedy golden is untouched, the sampling shift is bounded and
  measure-zero); (ii) the removal of the lazy-mode RNG fusion hazard class (the
  documented mis-execution genre — a standing stability win for every future sampler
  change); (iii) the smaller sampler graphs (faster capture/compile of the 16 shapes;
  the lazy-mode first-use cost).
- Best case: the D22 cost model shows the philox device time is larger than the
  estimate (the 248-node shapes at the 32-row buckets) → @16 up to +6-8%.
- Worst case: the node savings are as estimated but the device race was never the cost
  (the sampler graph's other ~100-190 non-philox nodes dominate) → +0.5-1% total.

### 6. Implementation complexity

Medium. The op is 300-500 lines (the nucleus clamp + the bounded draw + the race + the
greedy passthrough), the CPU test is 100-200 lines, the graph contract test reuses the
existing sampler test scaffolding (`tests/unit_tests/sampler/test_hpu_sampler.py`
pattern). The A/B is one boot (the env-gated op, the fresh census, the gates, the
warm bench). No model changes, no recipe changes, no new capture shapes (the op is
inside the existing sampler graphs — the node count drops, the shape count is
unchanged).

### 7. Risks

- **Correctness:** the distribution shift for nucleus > K_max rows (bounded by the
  residual mass, measure-zero at K_max = 8192; the CPU test measures the actual TV
  distance on our real logits — if it exceeds 1e-3 on any suite prompt, K_max →
  16384). The greedy path is untouched by construction (no draw).
- **Numerics:** the race on a private fp32 copy (the abotsis pattern) — no in-place on
  the sampler's tensor; the golden 12/12 (greedy) is the hard gate.
- **Lazy-mode hazard:** the mis-execution genre (four prior sampler bugs in the abotsis
  history + the T3 strided-slice refutation) — the contract test + the gate suite are
  the mitigation; any gate failure reverts (the op is env-gated, default OFF).
- **Memory:** the draw buffer shrinks (K_max × rows × 4 B fp32 = 1 MB at 32 rows vs
  19.5 MB) — negative memory impact.
- **Compatibility:** the op is in-tree (the vllm/v1/sample/ops path + the
  vllm_gaudi worker half, the same two files the topk port touched) — no vendor
  surface beyond what the port already validated.

### 8. Implementation instructions

1. **Verify the in-graph nucleus (no HPU time, 0.5 day):** read the 16 sampler-graph
   JSONs (the `/data/glm53/graphs/.graph_dumps` dot-dir, the `philox`-containing
   files) — confirm the sort + cumsum + cutoff chain is present (the
   `logical_slice`/`sort`/`cumsum` guid class) and where the masking feeds the race.
2. **The CPU distribution test FIRST (the gate):** a CPU reference of the full-vocab
   gumbel-max (the current semantics: sort, top_p cutoff, renormalize, Exp(1) race
   over the full vocab) vs the bounded version (the same with the draw over K_max
   slots, the masked slots p=0) — 100k samples each, on (a) synthetic logits
   (geometric, flat, peaked), (b) real logits dumped from one production step (the
   6 prompt classes). The bar: total-variation distance < 1e-3 on all; record the
   nucleus-size histogram (the fraction of rows with n_keep > 8192 — expected ~0).
3. **The op (1 week):** `top_p_nucleus_sample` per §3, env-gated
   (`VLLM_HPU_TOPP_NUCLEUS_SAMPLE=1`, default OFF), the greedy passthrough, the
   private-fp32-copy race, the K_max constant (8192, the 16384 fallback).
4. **The HPU graph contract test (2-3 days):** the op in a minimal captured graph —
   the output token distribution over 10k steps matches the CPU reference within the
   TV bar; the philox node count in the dump is ≤ 20 for the op's graph.
5. **The A/B boot (1 boot):** the env on, the fresh `DUMP_POST_GRAPHS` census (the
   philox delta), the golden 12/12 (greedy — must be identical), the tools 8/8, the
   smoke 391, the warm bench bs1/@8/@16 vs the production. Any gate failure → revert
   (default OFF).
6. **What to benchmark:** bs1 (expect +0.5-2%), @16 (expect +2-5%), the per-shape
   philox node count (expect 124-248 → ≤ 20), the sampler-graph capture time (the
   first-use cost on the 16 shapes).

### 9. Validation criteria

1. The CPU TV distance < 1e-3 on all test distributions (item 2) — **the exactness
   proof, the primary gate**.
2. The fresh census: the philox nodes per sampler graph ≤ 20 (the 124-248 baseline).
3. The golden 12/12 token-identical (the greedy path untouched — a regression here is
   a bug in the op, not an expected difference).
4. The tools 8/8 + the smoke 391.
5. The warm bench: bs1 ≥ 54 (the +0.5% floor — the honest bar is "not a regression";
   the +2% is the bonus), @16 ≥ 185.
6. The soak: the 16-stream 30-min stability (the RNG hazard class is a stability
   risk — the soak is the regression guard).

### 10. Novelty assessment

Not a retry: the upstream `topk-subset-sampler` was ported, gate-tested, and found
INERT (measured 11-22) — it targets the top_k path; our model has no top_k. The
top_p-nucleus bound is the different mechanism (the draw domain, not the sort
shortcut) and the agent explicitly recorded it as "the candidate for the next round —
would need a new fix" without a design. This is that design, with the exactness proof
(the gumbel-max-over-the-support argument) that makes it accuracy-risk-free in a way
the approximate subset samplers are not.

---

## RA-051 — The @16 T1-package attribution: recover 181.3 → 276.6+ in bf16

**Status:** PROPOSED · **Complexity:** Low (env/flag A/B, 3-5 boots) · **Effort:**
3-5 boots + the per-arm gate pass.

### 1. Executive summary

The production config (k=1 + T1 + Q01v2, bf16 KV) measures **181.3 tok/s @16** — 34%
below the pre-T1 adopted config's warm record (**276.6**, same bench, same day) and 47%
below the same production config with the now-prohibited fp8 KV (**346.0**). The T1
package (the RA-018 invariance fix) was adopted as a monolith for the determinism
contract (the R003 knife-edge + the 7-distinct-prompt parity failure), but **no
component has been priced at the batched shapes**: T1a (the order-invariant
`gather_sum` all-reduce), T1b (the pinned decode buckets `MIN/STEP/MAX=16`), T1c (the
fp32 draft cores), and the Q01v2 draft graph are four candidates for the −34%, and the
fp8 removal made the @16 gap the project's largest unattributed performance loss. A
3-5 boot A/B matrix attributes it; the expected recovery (the {16,32,48} bucket-set
extension, or the shape-conditional draft graph) is **+40-52% @16 in bf16** — the
cheapest large win remaining in the queue, and it is independent of the kernel work
(RA-052) and the drafter decision (RA-049).

### 2. Current limitation (MEASURED, 11-22 — the same day, the same bench script)

| Config (all bf16 KV unless noted, k=1) | bs1 | @8 | @16 | Time |
|---|---|---|---|---|
| pre-T1 adopted (k=1 + RA-011 async replay) | 50.6-51.4 | 282.7 | **276.6** | 03:50-04:02 (warm) |
| T1 + Q01v2 (production) | 51.4-53.9 | 255.5 | **181.3** | 08:48 (warm) |
| T1 + Q01v2 + fp8_inc KV (prohibited, record only) | 47.5-49.1 | 254.8 | **346.0** | 10:40 (warm) |
| T1 + Q01v2 + fp8 + chain k=4 (prohibited KV, parked chain) | 52.1 | 256.5 | **346.7** | 12:04 (warm) |

The deltas: production vs pre-T1 = **−34% @16** (flat-to-up at bs1); production vs
production+fp8 = **−47% @16** (the fp8 lever, prohibited). The T1 package's own @16
cost is the unattributed 276.6 → 181.3 gap (the fp8 boot shows the config CAN run at
346 — the batched shapes are not structurally broken by T1/Q01v2; something in the
bf16 path costs 95 ms/step at @16).

### 3. Proposed architecture (the A/B matrix)

The T1 package components and their @16 candidate mechanisms:

- **T1a** — `GLM_ALLREDUCE_GATHER_SUM=1`: replaces the HCCL all-reduce with a
  gather-then-sum (the order-invariant variant, the 16 MiB cap + the fp32 sum).
  Candidate cost: the gather-all-then-sum is a different data-movement pattern than
  the ring/recursive AR — at the 32-row verify shapes (16 seqs × 2 rows) the per-rank
  gather width is 32 × hidden × 2 B = 3.3 MB per AR × 162 ARs/step — the extra
  movement + the sequential sum could cost the batched shape (the bs1 shapes are
  small — consistent with the bs1 flat-to-up).
- **T1b** — `VLLM_DECODE_BS_BUCKET_MIN/STEP/MAX=16`: pins the decode bucket ladder to
  16. The @16 verify shape is 32 rows — the pinned ladder changes which captured
  graph the @16 step replays (the 16-row bucket with padding? the 32-row shape off-
  ladder? the "free ladder" vs "pinned" recipe difference) — the Q01v2 boot note
  says exactly this: "the 16-row pinned bucket changes the @16 recipe from the free
  ladder."
- **T1c** — `VLLM_GLM_MTP_DRAFT_CORE_FP32=1`: the fp32 draft cores (the draft-only
  numerics fix) — small expected cost (the draft is ~3 ms of the step).
- **Q01v2** — `VLLM_GLM_MTP_DRAFT_GRAPH=1`: the graph-captured draft (adds capture
  shapes + a draft replay per step) — the boot note lists it as a candidate too.

**The matrix (5 boots, same warm protocol — the 22-min warm ramp documented; bench
after the ramp):**

| Arm | T1a | T1b | T1c | Q01v2 | Measures |
|---|---|---|---|---|---|
| B0 | on | on | on | on | baseline (expect ~181.3) |
| B1 | **off** | on | on | on | the T1a cost at @16 |
| B2 | on | **off** (free ladder) | on | on | the T1b cost at @16 |
| B3 | on | on | on | **off** (eager draft) | the Q01v2 cost at @16 |
| B4 | the intersection of the two causal arms | | | | the combined recovery |

T1c is not a separate arm (the small expected cost; it stays on in all arms — if the
matrix shows no arm recovers the gap, T1c is the 6th boot, last in the queue).

**Per-arm gates (the determinism contract is the invariant — every arm re-verifies):**
the parity probe (8 identical prompts → 1 distinct text), R003 (5/5 "5"), the tools
8/8, the smoke 391, and the golden (re-captured under B1 — the standard AR changes
the ULP pattern, so B1's golden is its own reference; the T7 tie-aware gate is the
right instrument if R003 flips under B1 — the owner decision item is pending, so the
conservative reading applies: a knife-edge flip with the measured top-2 gap reported
is documented, not auto-failed).

### 4. Technical feasibility

- All four components are env-gated (the T1 package + Q01v2 were booted by env
  changes only — the matrix is env-only, no code changes).
- The warm protocol is documented (the 30-60 min ramp; the 22-min third-bench-point
  convention) — the A/B is comparable by construction (same day, same script, same
  ramp discipline).
- **Unsupported assumptions:** (i) that the 276.6 baseline is still reproducible
  (the pre-T1 config is a rollback target — `serve_mtp.sh` — the B-arm that reverts
  furthest is the control; if the 276.6 does not reproduce, the gap is boot-variance,
  not T1 — the matrix still runs, the baseline is the re-measured control); (ii) that
  the bucket pinning is the lever (the boot note's attribution is the agent's
  hypothesis, not a measurement — the matrix tests it).

### 5. Expected performance impact (DERIVED)

- **Case (i) — T1b causal (the pinned bucket changes the @16 recipe):** the fix is
  the bucket-set extension `{16, 32, 48}` (keep per-shape pinning for determinism,
  add the 32-row verify shape so the @16 step replays its natural shape) — the
  @16 recovers to the free-ladder class: **181.3 → 255-277 (+40-52%)**. bs1:
  unchanged (the bs1 shape is the 16-row bucket — the pinning is exactly right for
  it; the parity is preserved per-shape).
- **Case (ii) — T1a causal (the gather_sum costs the batched AR):** the options:
  (a) accept the cost (the determinism contract is the reason T1 exists — the 7-
  distinct failure mode returns under the standard AR at the batched shapes — that
  is the measured risk), (b) design a cheaper order-invariant AR for the batched
  shapes (the fp32 ring? the vendor's 2-phase gather in a different layout? — a
  follow-up design item, days-class), (c) shape-conditional AR (gather_sum for bs ≤ 8
  where the determinism failures were observed, standard AR at ≥ 16 — the parity
  probe is 8-concurrent, so the failure mode is at the 8-16 row class; the 32-row
  shape's determinism under the standard AR is untested — the probe at 16 concurrent
  identical prompts is the test). Recovery: **181.3 → 220-260 (+21-43%)** in (c).
- **Case (iii) — Q01v2 causal at @16 (the draft graph's replay/capture cost at the
  batched shapes):** the shape-conditional draft graph (graphed draft for bs ≤ 8 —
  where it won +20% and fixed nothing at the batched shapes; eager draft for ≥ 16 —
  the pre-Q01v2 behavior at @16). Recovery: **181.3 → 240-277 (+32-53%)** (the
  pre-Q01v2 @16 under T1 is unmeasured — B3 measures it).
- **Mixed (the honest prior):** two of the four contribute (T1b + Q01v2 is the boot
  note's own pairing) — the combined recovery **181.3 → 250-280 (+38-54%)**.
- bs1: every arm re-verifies 51.4-53.9 (the determinism + the bs1 performance are the
  invariants; an arm that loses the bs1 band is a failed arm regardless of the @16
  gain).

### 6. Implementation complexity

Low. Env-only boots (the existing boot-script family: `boot_t1q.sh` is B0; B1-B4 are
one-flag variants — 5 script copies, 5 boots). The per-arm gate pass is the standard
suite (the parity probe, R003, the tools, the smoke, the golden where the numerics
change). The bucket-set extension (case i's fix) is a one-line env change
(`VLLM_DECODE_BS_BUCKET_MAX=48` or the explicit set, per the in-tree bucketing
mechanism — verify the env's set syntax in `features.py`). The shape-conditional
draft graph (case iii's fix) is a small runner change (the draft-graph dispatch on
the batch size — the env + the condition, ~50 lines).

### 7. Risks

- **Determinism regression (the primary risk):** any arm that loosens T1 (B1 the
  standard AR, B2 the free ladder) can re-open the 7-distinct-prompt failure mode or
  the R003 flip — the per-arm parity + R003 gates are the guard; a failed arm reverts
  (the production is env-restorable in one boot). The T7 policy (if the owner adopts
  it) is the right instrument for the knife-edge class; until then, the conservative
  reading (the flip is documented with the top-2 gap, not auto-failed) applies.
- **Boot variance:** the 22-min warm ramp + the 3-bench-point convention control it;
  the B0 re-measure (the first boot) calibrates the day's variance.
- **No memory or stability risk:** env-only, no new shapes (the {16,32,48} set adds
  at most the 32-row/48-row decode shapes — already captured in the free-ladder
  boots), no model changes.
- **The fp8 confound:** the 346.0 data point is fp8 — it bounds the gap's ceiling
  (the config CAN run fast) but is not an A/B arm (prohibited). The matrix's control
  is the re-measured pre-T1 config, not the fp8 number.

### 8. Implementation instructions

1. **B0 first (1 boot):** re-measure the production @16 (the warm protocol: the
   third bench point, ~22 min post-restore). Expect ~181.3 ± the day's variance.
   This is the day's calibration.
2. **B1 (1 boot):** `GLM_ALLREDUCE_GATHER_SUM` off (the env unset → the HCCL
   default), everything else as B0. The gates (the parity, R003, the tools, the
   smoke) + the golden re-capture (the numerics change) + the warm bench bs1/@8/@16.
3. **B2 (1 boot):** the bucket pin off (`VLLM_DECODE_BS_BUCKET_MIN/STEP/MAX` unset →
   the free ladder), everything else as B0. The gates + the warm bench.
4. **B3 (1 boot):** `VLLM_GLM_MTP_DRAFT_GRAPH` off (the eager draft), everything
   else as B0. The gates + the warm bench.
5. **B4 (1 boot):** the intersection of the two causal arms from B1-B3 (e.g. if B2
   and B3 both recover part of the gap: both off). The gates + the warm bench.
6. **The fix per the decision rules** (§5): the bucket-set extension (case i) / the
   shape-conditional AR design or the 16-concurrent parity probe (case ii) / the
   shape-conditional draft graph (case iii). The fixed config gets the full gate
   suite + the soak + the warm bench, and becomes the new production.
7. **What to benchmark:** @16 (the target: 181.3 → ≥ 250), @8 (the 255.5-282.7 band
   — a regression here is a signal), bs1 (the 51.4-53.9 invariant), the parity probe
   per arm (the 1/8 invariant), R003 per arm (the 5/5 invariant).

### 9. Validation criteria

1. The causal arm(s) identified: the @16 delta between B0 and each single-component
   arm, with the day's variance (B0's std over 3 bench points) — the component with
   the largest @16 delta is causal (the ≥ 30% of the 95 ms gap threshold).
2. The fix recovers ≥ +40% @16 (181.3 → ≥ 254) in the best-case diagnosis, or the
   partial recovery is measured and the residual is documented with the component
   attribution (the honest bar if the gap is multi-causal).
3. Every fixed-config arm: the parity 1/8, R003 5/5, the golden (its own reference
   where the numerics change), the tools 8/8, the smoke 391, the bs1 band 51.4-53.9.
4. The 30-min 16-stream soak on the fixed config (the stability guard).

### 10. Novelty assessment

Not a retry: the T1 package was adopted as a monolith (the 08:10 validation booted
the full package; the Q01v2 re-adoption booted the package + the draft graph) — no
component has been measured in isolation at the @16 shapes. The 181.3 vs 276.6 vs
346.0 data all exist in EXPERIMENTS.md but have not been attributed (the boot note
names the bucket + draft-graph candidates as a hypothesis: "the draft-graph capture
adds shapes; the 16-row pinned bucket changes the @16 recipe from the free ladder" —
hypothesis, not measurement). The matrix is the attribution; the fp8 removal (the
owner directive) is what makes the @16 gap the project's top unattributed loss — the
material change that makes this worth the 3-5 boots now.

---

## RA-052 — The MoE node collapse: the Q21 kernel as the first FX-pass target,
### census-sequenced (the largest structural win)

**Status:** PROPOSED · **Complexity:** High (weeks-class: the FX pass + the native op
integration + the gate stack) · **Effort:** 2-4 weeks (the Q21 kernel core is already
compiled + gated in our variant — 11-19; the FX-pass skeleton is ~1 day per the
agent's RA-044 addendum; the integration + gates are the bulk).

### 1. Executive summary

The banked census (4,173 post-graphs, re-analyzed this round) shows the decode
recipe's node mass is dominated by the **MoE region and its plumbing**: the
HabanaFusedOpLazy MoE subgraphs at ~3,287-3,352 nodes each (multiple per decode
shape), the class totals `moe_multiplexer` 37,408 / `gemm` 34,689 / `batch_gemm`
32,422 / `reshape` 109,321 / `logical_slice` 76,781 / `expand_dims` 26,313 /
`concat` 22,704 / `split` 19,972 (across the whole corpus — the decode shapes are
the densest consumers). The per-step host replay cost scales with the executed node
count (the rev-14 model: 20.5-26.4k nodes × ~1.4 µs/node = the 25-45 ms host floor);
the MoE region is 87-89% of the recipe nodes. **The corpus-verified 1Cat FX-pass
pattern** (`flashinfer_projection_fusion.py` `fuse_projection_graph`, commit e2266e0e:
graph-node matching against a fail-closed qualified-shape allowlist → native-op
replacement → meta-preserving getitem rewiring → DCE of the obsolete storage update
→ `graph.lint()` + recompile; registered at `OptimizationPassPlacement.PRE_
PLACEMENT` with the explicit no-default-promotion discipline) is a production-proven
instrument for exactly this: match the dequant + multiplexer + batch_gemm cluster in
the MoE region, replace it with the native grouped op (the Q21 kernel — the 1Cat
`block_fp8_dequant` GEMV core, compiled + gated in our variant), and let the Synapse
fuser absorb the remainder. The value is the **node count** (the host replay floor
25-45 ms → 8-15 ms) AND the routed-read device benefit (the 5.4-12.5× expert-byte
cut at bs ≤ 4; the @16 verify read 41.9 GB/step → routed ~25-30 GB) — the dual-metric
discipline from the GaudiInfer negative results (the fused small kernels LOST 0.3 ms
to the compiler-overlapped graph ops — the kernel must win node count AND device
time). This adopts and sequences the agent's own RA-044 design addendum (11-22 15:20)
with the census's target-selection data (D22) that was missing when the addendum was
written.

### 2. Current limitation (MEASURED — the census + the rev-14 host model)

- The census (this round's re-analysis, all 4,173 banked graphs): the per-graph node
  leaders are the MoE lazy subgraphs (~3,287-3,352 nodes each; 8+ copies of the
  largest shape in the corpus); the node-class totals (whole corpus): reshape
  109,321; logical_slice 76,781; memcpy_dma 44,687; moe_multiplexer 37,408; gemm
  34,689; batch_gemm 32,422; static_reshape 28,757; expand_dims 26,313; concat
  22,704; split 19,972; add_fwd_f32 11,728; transpose ~22,000; engines TPC 147,399 /
  MME 67,111 / DMA 55,178 / unlabeled 341,955.
- The rev-14 host-floor model (the 20.5-26.4k-node decode recipe × ~1.4 µs/node =
  25-45 ms bs1; the 322 replays + ≥ 162 AR boundaries on the multi-row step) — the
  node count is the host term's driver, and the MoE region is 87-89% of the nodes.
- The Q21 state (11-19 → 11-22): the kernel core is compiled + gated (the
  `GLM_Q21_MOE=1` fail-closed gate, the decode cap 32 tokens, the EP -1-sentinel
  local remap, the two-pass branch at the moe_op site); BG-1 closed it as a *traffic*
  lever at @16 (the all-experts-read is not the @16 device bottleneck — the 28 ms
  unknown is); the RA-044 addendum reframes it as the *node-count* lever (the
  grouped-MoE GEMM region collapse 1,496 batch_gemm MME + 1,002 multiplexer TPC
  nodes → ~42-150); the build skeleton is staged at `/data/glm53/q21_ab` (the
  llvutd A/B arms, the poison-byte contract measured — the device e4m3fn clamps max
  finite at 240, the load-time assert `scale·max|w| ≤ 240` —, the dual-metric
  template).

### 3. Proposed architecture

**The instrument (the 1Cat FX-pass pattern, corpus-verified):** an FX-level
optimization pass registered at `PRE_PLACEMENT` (after Bridge's layout/view
rewrites — the 1Cat pattern's explicit constraint, because re-executing the whole
graph with FakeTensor propagation is invalid on the Bridge-rewritten program):

1. **The match:** a graph-node pattern predicate over the MoE region of the decode
   layer subgraphs — the dequant (the `cast_to_fp8`/scale chain or the in-GEMM
   dequant), the `moe_multiplexer` (the token→expert gather/scatter), the
   `batch_gemm`/`gemm` pair (the up/gate + down projections) — with the fail-closed
   shape qualification (only the qualified decode-bucket shapes: the 16 verify-row
   class and the 32-row class, the hidden/expert dimensions from the config —
   "No ordinary-scale promotion", the 1Cat discipline: the match fires only on the
   shapes that were parity-qualified).
2. **The replacement:** the native grouped op — the Q21 kernel (the
   `block_fp8_dequant` GEMV core wired as the two-pass branch, the routed-read
   variant) exposed as the custom op (the PT2 adaptation staged for the v0.26.0
   registrar collision — the B7-resolved load-context pattern).
3. **The rewire:** the meta-preserving getitem rewiring (the fused op's outputs
   replace the cluster's outputs; the `meta["val"]` / `placement` / `output_*`
   fields are carried over — the 1Cat pattern's `_fresh_output_value` + the
   `node.meta = dict(source.meta)` discipline; the `hpu_cluster` placement).
4. **The DCE:** `graph.erase_node` for the obsolete storage updates (the 1Cat
   documented hazard: FX DCE deliberately retains impure nodes even after their last
   use disappears — the private-owner proof permits removal of the obsolete `add_`
   storage update; leaving it can mutate the new native producer's input before that
   producer executes) + `graph.eliminate_dead_code()`.
5. **The lint + recompile:** `graph.lint()` + `graph_module.recompile()` (only the
   new native nodes receive fresh non-aliasing values + complete placement metadata
   — the 1Cat constraint against whole-graph re-execution).

**The sequence (per the agent's RA-044 addendum + the census):**

- **Step 1 — D22 (no HPU time, 0.5-1 day):** the per-step node-cost model over the
  banked JSONs — for each decode shape, the executed subgraph list (the
  `breakpoint_after_graph` + `exec_order_idx` fields join the 4,173 graphs to the
  recipe execution order), the per-class node count × the JSON's cost fields (the
  per-node duration) → the ranking "engine × count × mean duration" (the 1Cat
  dead-ends list's rule — the raw count is NOT a cost model). The output: the first
  target (expected: the dequant + multiplexer + batch_gemm cluster of the MoE
  subgraphs — confirm, do not assume) + the re-priced host-floor model (the rev-14
  20.5-26.4k × 1.4 µs against the real executed sum).
- **Step 2 — the FX-pass skeleton (~1 day, per the addendum):** the match predicate
  + the native-op call + the meta-preserving rewire + the DCE + the lint, env-gated
  (`VLLM_HPU_MOE_FX_FUSION=1`, default OFF), the fail-closed shape qualification
  (the qualified decode-bucket shapes only).
- **Step 3 — the CPU contract test (2-3 days):** the graph target before/after (the
  FX graph's op sequence + the tensor values) — **bit-exact** (the T3 precedent: the
  closed-form rewrite's CPU parity was fp32 exact / bf16 ≤ 1 ULP — the bar), on the
  qualified shapes with the real checkpoint weights (the Q21 gate's s11/s13 compare
  governs — the s14/s15 probes are void per the 11-19 platform facts).
- **Step 4 — the boot (1-2 weeks including the gates):** the env on, the fresh
  `DUMP_POST_GRAPHS` census (the node delta — the primary metric), the dual-metric
  device time (the per-shape device span before/after — the secondary metric; the
  GaudiInfer negative result is the warning: a node-count win that loses device time
  is a loss), the R003 + the parity + the golden (the numerics regime — the Q21
  requant noise compounds per the documented EXP-FINDING; bit-exact is the
  expectation, the T7 tie-aware gate is the acceptance instrument if the owner
  adopts it), the warm bench bs1/@16, the soak.

### 4. Technical feasibility

- **The pattern is production-proven** (the 1Cat fork runs it on a decode projection
  — the corpus shows the full implementation + the hardware test with the storage-
  reuse audit + the "one additional residual producer" edge case handled). The MoE
  region is a larger cluster (the multiplexer + the GEMM pair vs the norm + cast +
  GEMM) — the match predicate is more complex, but the mechanism (match → replace →
  rewire → DCE → lint) is identical.
- **The native op exists** (the Q21 kernel core, compiled + gated in our variant —
  the two-pass branch at the moe_op site is the in-runner wiring; the FX pass is the
  compile-time alternative/complement: the FX pass collapses the graph structure,
  the in-runner branch swaps the op — the two can compose: the FX pass first (the
  structure), the in-runner branch as the fallback for shapes the pass does not
  qualify).
- **The fuser absorbs the remainder** (the 1Cat pattern's design: the FX pass
  replaces the cluster with the native op, and the Synapse fuser collapses what's
  left — the reshape/slice/concat plumbing that feeds the cluster is the fuser's
  domain; the dead-ends list's "rank by engine × count × mean duration" is the
  guard against chasing the plumbing that the fuser already handles).
- **Unsupported assumptions:** (i) that the Q21 kernel's grouped form covers the
  MoE cluster's exact op semantics (the two-pass branch's EP remap + the decode cap
  32 tokens — the 32-row verify shape is at the cap; the 16-row class is inside;
  the kernel's per-route target (beat 1.5 TB/s, the D4 tile sweep) is the device-
  metric gate); (ii) that the PRE_PLACEMENT registration point is available in our
  v0.26.0's `habana_frameworks.torch.dynamo.compile_backend.passes` (the 1Cat
  pattern is from a newer tree — verify the API exists in ours; the alternative is
  the compile-backend hook the 11-19 Q21 wiring used); (iii) that the lazy-mode
  storage-reuse hazards (the Bridge may reuse a private sum's storage — the 1Cat
  hardware test's edge case) do not bite the MoE cluster (the contract test + the
  gate suite are the mitigation; the T3 strided-slice refutation is the standing
  warning that the device access pattern must be checked, not assumed).

### 5. Expected performance impact (DERIVED — wide uncertainty, the dual metric)

- **Node count (the primary metric):** the MoE region collapse per the RA-044
  estimate (1,496 batch_gemm MME + 1,002 multiplexer TPC → ~42-150 per region) +
  the fuser absorbing the feeding plumbing (the reshape/slice/concat/expand class,
  a fraction of the 200K+ corpus total is MoE-feeding) → the recipe 20.5-26.4k →
  **~6-10k nodes** (best/mid case; the floor is the non-MoE 11-13% + the MoE
  remainder).
- **Host replay (from the node count):** 25-45 ms → **8-15 ms** (×1.4 µs/node) —
  the bs1 step 34.9 → **~22-28 ms** → 1.88/0.022-0.028 = **67-85 tok/s** (from
  53.9: +24-58%); the @16 multi-row host term (the ~90-115 ms of the 166 ms step —
  the D1 model) scales with the node count the same way → **−15-30 ms** → the @16
  step 166 → ~115-145 ms (before RA-051's recovery: the two compose — RA-051 fixes
  the T1-package cost, RA-052 fixes the node floor; the combined @16 best case is
  the RA-049 stack estimate ~105-125 ms).
- **Device (the secondary metric):** the routed-read benefit (the 5.4-12.5× expert-
  byte cut at bs ≤ 4 — the 41.1 GB/card/step all-experts read → ~3.3-7.6 GB; at
  1.945 TB/s that is the 15.5 ms → 1.7-3.9 ms device term at bs1) + the @16 verify
  read (41.9 GB/step → routed ~25-30 GB per the 69-89% local-expert hit at 16
  active) — **bs1 device −5-15 ms, @16 device −8-20 ms** (the fp8 data point — the
  batched-shape device term dropped 80 ms with the KV/attention path halved —
  bounds the uncertainty: the MoE read is the remaining bf16 device lever for that
  term).
- **Total (the honest range):** bs1 53.9 → **90-140 tok/s** (the host −15-30 ms +
  the device −5-15 ms on a 34.9 ms step, wide uncertainty per the dual-metric
  discipline); @16 181.3 → **260-380 tok/s** (before/with RA-051's recovery — the
  combined stack is the RA-049 estimate).
- Best case: the D22 cost model confirms the MoE region is ≥ 80% of the executed
  node-time AND the kernel beats the stock op on device time at the 16/32-row
  shapes → bs1 ~140, @16 ~380 (the fp8-class @16 in bf16 — the 346.0 data point
  shows the shape can run that fast).
- Worst case: the kernel wins nodes but loses device time at the batched shapes (the
  GaudiInfer negative result) → the env stays OFF at the batched shapes (the shape-
  conditional enablement: the pass qualifies the 16-row class only) → bs1 ~90-110,
  @16 ~200-240.

### 6. Implementation complexity

High. The FX-pass skeleton is ~1 day (the 1Cat pattern is the template — the match
predicate is the new work: the MoE cluster's node topology vs the 1Cat projection's);
the CPU contract test is 2-3 days (the bit-exact bar on the real weights); the boot +
gate stack is 1-2 weeks (the fresh census, the dual metric, the R003/parity/golden,
the warm bench, the soak, the revert path). The Q21 kernel core is done (compiled +
gated) — the remaining kernel work is the grouped-form coverage check (the per-route
target, the D4 tile sweep with the llvutd arms — the staged skeleton). Total: 2-4
weeks to the first measured A/B.

### 7. Risks

- **The R003 numerics regime (the primary risk):** the Q21 requant noise compounds
  through the 42 MoE layers (the documented EXP-FINDING: the Q21 3-session failure
  class was the requant noise — resolved for the kernel's own parity, but the
  in-recipe compounding at the near-ties is the R003 class). Bit-exact is the
  expectation (the two-pass branch's parity gate passed 1.215e-4 / 3.62e-8 at the
  12-stage gate); if the recipe-level parity shows the ULP drift (the T3 class),
  the T7 tie-aware gate is the acceptance instrument (the owner decision item is
  pending — until then, the conservative reading: a knife-edge flip with the
  measured top-2 gap is documented, the adoption decision goes to the owner).
- **The lazy-mode hazard class:** four prior sampler bugs + the T3 strided-slice
  refutation + the 1Cat storage-reuse edge case — the dual-metric discipline (the
  kernel must win node count AND device time) + the contract test + the gate suite
  are the mitigation; any gate failure reverts (the env is default OFF).
- **The native grouped-op feasibility:** the vendor grouped form (the stock
  `batch_gemm` semantics) vs the custom TPC kernel — both precedents exist in the
  corpus (the 1Cat native `add_rmsnorm_quant_op` is the custom-op precedent; the
  stock grouped GEMM is the vendor precedent) — the D4 tile sweep + the per-route
  target are the go/no-go (a kernel that loses the device metric at the batched
  shapes is shape-conditioned OFF, not forced).
- **Capture overhead:** the pass changes the graph structure → the recompile of the
  qualified shapes (the per-graph compile ~5.4-8.1 s — the 12-16 decode shapes
  recompile once; the warmup cost is bounded, the disk recipe cache (RA-006) is the
  mitigation if it matters).
- **Memory:** the native op's workspace (the Q21 kernel's tile buffers — the staged
  contract) vs the cluster's intermediate tensors — expected negative (the cluster
  materializes the dequantized weights; the kernel dequants in the load pipe) —
  verify in the HBM census at the A/B boot.

### 8. Implementation instructions

1. **D22 first (no HPU time, 0.5-1 day):** the per-step node-cost model over the
   banked JSONs (`/data/glm53/graphs/.graph_dumps`) — the join (the
   `breakpoint_after_graph` + `exec_order_idx` fields), the per-class ranking (the
   engine × count × mean duration), the first-target selection (expected the MoE
   cluster — confirm), the host-floor re-pricing. The output is the target spec for
   step 2 (the exact node topology to match, the expected node delta, the expected
   host savings).
2. **The FX-pass skeleton (~1 day, per the staged RA-044 addendum pattern):** the
   1Cat `fuse_projection_graph` as the template — the match predicate over the MoE
   cluster (the D22-confirmed topology), the native Q21 op call (the two-pass
   branch's op, exposed as the custom op), the meta-preserving rewire, the DCE
   (the obsolete storage updates — the private-owner proof per the 1Cat hazard
   note), the lint + recompile, the env gate + the fail-closed shape qualification.
3. **The CPU contract test (2-3 days):** the before/after graph target (the op
   sequence + the values) on the qualified shapes with the real checkpoint weights —
   bit-exact (the fp32 exact / bf16 ≤ 1 ULP bar, the T3 precedent; the Q21 gate's
   s11/s13 compare governs).
4. **The D4 tile sweep + the llvutd A/B (the staged skeleton at
   `/data/glm53/q21_ab`):** the kernel's per-route target (beat 1.5 TB/s to clear
   the 200 t/s roofline with margin — the rev-5 number; the 24/8/3/1 TPCs-per-
   expert tiles) + the bandwidth-cliff flag A/B — the device-metric input for the
   dual metric.
5. **The A/B boot (the gates):** the env on (the qualified shapes), the fresh census
   (the node delta — primary), the dual-metric device time (secondary), the R003 +
   the parity + the golden (the numerics regime), the warm bench bs1/@16, the
   30-min soak. Any gate failure → the env OFF (the revert is one flag).
6. **The shape-conditional enablement (if the worst case bites):** the pass
   qualifies the 16-row class only (the bs1/low-batch shapes where the routed-read
   benefit is largest and the kernel's per-route target is met) — the 32-row class
   stays on the stock path until the kernel wins its device metric there.
7. **What to benchmark:** the per-shape node count (the census delta — the primary
   metric: expect the MoE cluster 2.5-3.3k → ~42-150 per region), the device span
   per shape (the dual metric — the no-regression bar), bs1 (expect +24-58%), @16
   (expect +43-109% before RA-051; the combined stack is the RA-049 estimate), the
   warmup time (the recompile cost), the HBM census (the workspace delta).

### 9. Validation criteria

1. **The node delta (the primary metric):** the fresh census shows the MoE cluster
   collapsed per the D22 estimate (the 1,496 + 1,002 → ~42-150 per region, the
   recipe total 20.5-26.4k → ~6-10k) — a node delta < 50% of the estimate is a
   failed match (the predicate is too narrow — widen it, or the cluster topology
   differs from the D22 spec — re-derive from the JSONs).
2. **The dual metric (the no-regression bar):** the device span per qualified shape
   does not regress > 5% (the GaudiInfer warning: the node-count win must not buy
   device-time loss; a > 5% device regression at any qualified shape → that shape
   is shape-conditioned OFF).
3. **The numerics:** bit-exact (the contract test) + the R003 5/5 + the parity 1/8 +
   the golden (its own reference if the ULP pattern shifts — the T7 instrument if
   the owner adopts the policy; until then the conservative reading).
4. **The performance:** bs1 ≥ 90 (the +67% floor of the estimate range — the
   honest bar is the host-floor model: the node delta × 1.4 µs, measured); @16 ≥
   260 (before RA-051's recovery).
5. **The stability:** the 30-min soak (the lazy-mode hazard class is a stability
   risk — the churn determinism + the 0-error bar).

### 10. Novelty assessment

The Q21 kernel (RA-002) is IN PROGRESS as a *byte/traffic* lever — BG-1 closed that
framing at @16 (the all-experts read is not the @16 device bottleneck). The
**node-count reframing** (the host replay floor as the value, the FX pass as the
instrument) is the materially different hypothesis: the same kernel, a different
value proposition (the 25-45 ms host floor → 8-15 ms, not the 15.5 ms device floor
→ 1.7-3.9 ms), and a different instrument (the compile-time FX pass, not the in-
runner op swap alone). The agent's own RA-044 addendum (11-22 15:20) made the
reframing ("the per-node execution and the recipe-count host term as the value, not
the bytes") and named the instrument (the 1Cat FX pass, "NOT the Synapse fuser
knobs") — this proposal adds what was missing: the census's target-selection data
(D22 — the addendum said "the census rank (the banked JSONs)" as step (a); this
round's re-analysis provides the class totals + the per-graph leaders + the D22
spec for the engine × count × mean duration ranking), the exactness/numerics regime
(the bit-exact bar + the T7 instrument), and the dual-metric validation design. It
is the continuation of the in-progress work, not a retry of a failed experiment —
the only failure in the class is the GaudiInfer negative result (the fused small
kernels lost to the overlapped graph ops), which the dual-metric discipline is
designed to catch before adoption.

---

## Appendix — the census re-analysis method (for reproducibility)

The banked dump: `/data/glm53/graphs/.graph_dumps/` (4,173 post-graph JSONs; the
`DUMP_POST_GRAPHS` tool — corpus-verified, docs 01-basics §1.11, Tier B; the dot-
dir gotcha: the dumps land in a dot-directory). Each JSON: `graphs[]` with `name`,
`nodes[]` (the per-node `type`/`guid` + `engine` fields), `tensors[]`,
`breakpoint_after_graph`, `recipe_debug_id`, `workspace_size`. The analysis
(this round, read-only): per-graph node counts by guid class + engine; the
philox-containing graphs (16; 124-248 nodes each); the whole-corpus class totals
(§2 of RA-052); the per-graph leaders (the MoE lazy subgraphs, ~3,287-3,352 nodes).
The D22 extension (the cost model): the JSONs' per-node cost fields (the duration
data) × the per-shape executed-subgraph join (the `breakpoint_after_graph` +
`exec_order_idx`) — the 1Cat dead-ends list's ranking rule (engine × count × mean
duration), not the raw count.
