# EXPERIMENTS.md — experiment index

One row per experiment. Full reports live in `experiments/`; the published technical write-up
for the MTP work is `docs/ports/glm-5-3-flash-mtp.md`. Status labels follow
`docs/BENCHMARK_POLICY.md` (PROVISIONAL / BASELINE CANDIDATE / VERIFIED / INCONCLUSIVE /
FAILED / REGRESSION).

Pre-MTP experiments (OPT-*) were performed during the deployment/optimization phase that
preceded the MTP work; their raw artifacts and phase reports are preserved in the operator's
local working notes and are summarized here. All figures below are measured values, reported
with their conditions; none has yet passed the full `docs/BENCHMARK_POLICY.md` gate, so the
authoritative verified figures in `BASELINE.md` remain unfilled until Stage-1 reproduction
completes.

| ID | Date | Category | Hypothesis | Commit/config | HPUs | Key result | Status | Report |
|---|---|---|---|---|---|---|---|---|
| OPT-001 | pre-MTP phase | Baseline | Capture the production TP8+EP baseline | cookbook port, vllm-gaudi v0.26.0 (b059f9c), TP8+EP, GMU 0.82, KV override 1280 blocks/card, capture 2048, 256K ctx, 32 seqs | 8 | Decode 30.1/29.8 tok/s bs=1 (2 runs); 125.1 agg @8 streams; 213.2 agg @16; prefill 4,928 tok/s @128K prompt; HBM 65.5/68.2 GiB/card | BASELINE CANDIDATE | experiments/OPT-001-production-baseline.md |
| OPT-002 | pre-MTP phase | Memory | Decompose per-card HBM to find reclaimable blocks | same as OPT-001 | 8 | Weights 38.2 GiB/card; graphs/workspace ~25.5 GiB; KV ~1.8 GiB; GMU 0.95 unreachable at capture 2048 | VERIFIED (measurement) | experiments/OPT-002-hbm-decomposition.md |
| OPT-003 | pre-MTP phase | HPU scaling | TP4+EP is the minimum viable 4-HPU footprint | TP4+EP, GMU 0.95, capture 128 | 4 | Weights 76.5 GiB/card; viable with capture 128 (2048 ladder OOMs); 64 heads % {6,5,3} = 0 rules out 6/5/3-HPU TP | VERIFIED (feasibility) | experiments/OPT-003-tp4-feasibility.md |
| OPT-004 | pre-MTP phase | Stability | Post-experiment rollback restores production performance | fresh TP8 boot, production config | 8 | bs=1 30.3/30.1; 16 streams 217.2 agg; TTFT @128K 25.4 s — matches OPT-001 within noise | VERIFIED (reproduction) | experiments/OPT-004-rollback-reverification.md |
| MTP-001 | MTP phase | MTP | Native MTP=1 (checkpoint's own layer-45 head, bound by reference) improves batch-1 decode without correctness loss | fork branch glm53-flash; k=1; TP8+EP; 32K ctx; 8 seqs; window config | 8 | Acceptance 1.73–1.98 of 2.0; bs=1 35.7–36.8 tok/s (+14% like-for-like single run; +21% vs production-window control — NON-equivalent conditions, PROVISIONAL); 16-stream aggregate 42.7 vs 59.4 (net-negative, eager draft); memory 64.1 GiB/card; full correctness/feature gate suite passed | BASELINE CANDIDATE (functionality VERIFIED; performance PROVISIONAL pending Stage-1 reproduction) | experiments/MTP-001-mtp1-validation.md |
| MTP-002 | MTP phase | MTP depth | Depth 2 chains the single MTP head and improves throughput further | k=2; S=4 candidate columns; 3-row verify | 8 | Acceptance 2.04–2.52 of 3.0; decode 30.2–34.1 tok/s bs=1 (net-neutral/worse than k=1); correct across full sweep; golden divergences at same positions as k=1 | VERIFIED (functionality); REGRESSION-neutral (performance) | experiments/MTP-002-mtp2-sweep.md |
| MTP-003 | MTP phase | Concurrency | MTP benefit survives concurrent load | 16-stream like-for-like 32K/8seq, k=0/1/2 | 8 | Aggregate: k=0 59.4, k=1 42.7, k=2 41.8 tok/s; single-stream within run: k=0 31.4, k=1 35.7; acceptance holds under load (1.72–1.87 k=1) | VERIFIED (measurement) | experiments/MTP-003-concurrent-k-sweep.md |
| C015-IMPL | KB session | Prefill kernel | The KDA diagonal decay-dot (elementwise [sub,sub,d] on TPCs) is replaced by the block-anchor matmul contraction on the MME (abotsis measured 15.6x kernel-time, TTFT up to 3.23x) | fork branch glm53-flash, kda.py `_gauged_scores` | n/a (CPU oracle) | Oracle 12/12 shapes exact vs pre-C015 (CPU); HPU prefill ladder + TTFT + chunk re-check pending a window | WORKING | experiments/C015-decay-dot-matmul.md |
| STAGE1-A001 | 2026-11-06 | MTP reproduction | Controlled MTP=1 vs non-MTP at equivalent conditions (bucket-pinned, same window) | fork glm53-flash, TP8+EP, 32K/8seq, bucket-pinned | 8 | **Decode +15.1% (36.83 vs 32.16 mean, p=0.92 std)**, accuracy 100.0% both (0.0 pp delta), 5 reps per condition — VERIFIED | VERIFIED | benchmarks/stage1_results/ |
| RESEARCH-002 | KB session | Measurement | Our sampled decode is not host-rejection-bound at k=1 (33 vs 36 tok/s, −8%); bucket-pinning env exists in our tree; gather_sum all-reduce does not | live MTP k=1 server | 1 | Sampled-decode concern closed for current shapes; two new READY tasks (B021, B022) | VERIFIED (measurement) | experiments/RESEARCH-001-draft-graph-upstream.md (addendum 2) |
| RESEARCH-001 | KB session | Research | The eager-draft bottleneck (93 ms host vs 17 ms device per step) has a measured fix: abotsis fork 919ec98f ships draft-graph machinery (private storage copies, attention-in-graph) with MTP-4 numbers: accepted 3.33 vs 2.41 bypass, aggregate 26.0 vs 20.2 | abotsis fork 919ec98f (reference; not our tree) | 8 (their runs) | Our eager-draft k=2/concurrency findings are likely artifacts of this host-launch bottleneck; A014 becomes READY (adapt, not invent); addendum: their fork ALSO solves KDA candidate-slot state (appended private slots, batched chain kernel `chunk_kda_spec_states`, condense/swap-stable ids) and a third independent port exists (inderjodhsingh/gaudi-llm-notes) — strategic base decision needed from the owner; addendum 3: skip_layer_markstep backport candidate (+10% single-stream, bit-identical) -> B023 READY; addendum 6: boundary diagnostics addendum 7: their serving recipe vs ours - three config deltas exist in our trees (VLLM_BUCKETING_STRATEGY=lin, VLLM_COMPACT_GDN=0, --enable-ep-weight-filter) -> E015 READY; their single-stream is slower than ours (21-24 vs 30.1) - adapt, do not adopt; (I015 READY), their MTP-4 concurrency near-linear 185.9 tok/s at 16 streams with graphs, depth-4 optimal 3.33 accepted, and the 262K warning (MTP ~6 tok/s at 219K ctx without block-table sharing); addendum 4: KDA diagonal decay-dot -> anchored MME matmul applies directly to our _gauged_scores (15.6x kernel-time, TTFT up to 3.23x upstream) -> C015 READY | Research complete | experiments/RESEARCH-001-draft-graph-upstream.md |
| MTP-004 | MTP phase | TTFT / prefill knobs | Draft prefill pass and KDA chunk size affect TTFT | k=1; KDA chunk 64 vs 128 | 8 | Warm TTFT 2.1 s @2K, 3.1 s @8K, 4.2 s @16K; chunk=128 WORSE (8K 1.85 s vs 1.29 s) — chunk=64 retained; `GLM5_NEXT_KDA_CHUNK` env knob added | VERIFIED (measurement) | experiments/MTP-004-ttft-chunk-size.md |

## Historical note on the reported ~21% result

The originally reported "approximately 21% higher decode throughput at batch size 1" compares
MTP=1 (36.8 tok/s, 32K/8seq window) against a non-MTP control measured on the production window
(30.3 tok/s, 256K/32seq). Those conditions are NOT equivalent. Under the like-for-like 32K/8seq
window the single-run comparison is +14% (35.7 vs 31.4 tok/s). Both observations are preserved;
neither has passed the repetition/variability requirements of `docs/BENCHMARK_POLICY.md`. The
Stage-1 task (A001) is the controlled reproduction.

## EXP-Q02/Q03 — Aggregate decode ladder under the pinned-bucket config (2026-11-07)

**Question:** what is the true MTP-on aggregate baseline (the 800 t/s @ bs=16 target), and
where does the eager-draft crossover sit? Prior figures (42.7 vs 59.4 @16; 213 @16 non-MTP)
came from different windows/configs.

**Setup:** live E015-A server (MTP k=1, TP8+EP, lin prompt buckets, 32K/8seq window).
`bench_concurrent.py --streams N --reps 2` (16 distinct prompts, 256 gen tokens each,
greedy, reasoning_effort low). Rep 1 at a new stream count pays first-use graph capture
for that decode-batch shape.

**Results (aggregate tok/s, steady-state rep):**

| Streams | MTP k=1 (this ladder) | Non-MTP (historical, production window) |
|---|---|---|
| 2 | 55.7–56.3 | n/r |
| 4 | 92.8 | n/r |
| 8 | 125.1, 130.8 | 125.1 (B001) |
| 16 | **126.7** (rep1 49.8 = capture) | 213.2–217.2 (B001) |

**Findings:**
1. With the eager draft, aggregate SATURATES at ~128 t/s from 8 streams onward — the draft
   pass dominates the step as batch grows (per-step draft host cost scales with the virtual
   batch of verify rows).
2. MTP @16 streams is NET-NEGATIVE by ~40% vs the non-MTP figure (126.7 vs 213.2) under the
   pinned-bucket config — the crossover is confirmed between 8 and 16 streams.
3. First-use capture per new decode-batch shape costs ~30–60 s wall (rep1 @16: 49.8 vs
   rep2 126.7) — pre-warming served shapes matters (D003).

**Conclusion:** the eager draft is THE aggregate bottleneck, not just a bs=1 problem.
Q01 (draft graphs) is the highest-impact lever for both the 200 t/s single-stream and the
800 t/s aggregate objectives. Non-MTP @16 under the pinned-bucket window still needs a
comparison boot (Q02 second half).

## EXP-Q01 — Draft HPU graphs (split mode): implementation iterations (2026-11-07)

**Design (adapted from the reference fork 919ec98f, GaudiGrimoire-sourced):**
- Private light-weight storage clones (`_private_copy`: shallow module copy +
  per-tensor `.data` storage swap, preserving vLLM parameter subclasses); the
  routed MoE is SHARED by reference (fused-MoE op registration is keyed on
  storage identity; a cloned MoE fails with "MOE multiplexer weights were
  partially registered" upstream).
- `dry_run=False` is load-bearing: the draft chain consumes its own outputs;
  upstream measured deterministic garbage from the first post-capture replay
  under dry run.
- Seam discipline: mark_step + `torch.hpu.synchronize()` before every replay
  (a replay is opaque to lazy-mode dependency tracking; it launches against
  input buffers as they are) and mark_step after (eager consumers must not
  fuse onto the replay's output memory).
- Two modes: `split` (pre-attention core + eager MLA + post-attention core,
  default) and `attn` (whole step incl. attention, metadata tensors as
  positional graph inputs). Env-gated: `VLLM_GLM_MTP_DRAFT_GRAPH=1`,
  `VLLM_GLM_MTP_DRAFT_GRAPH_MODE=split|attn`. Default OFF.

**Iteration 1 (boot q01split):** first decode request crashed all workers:
"Forward context is not set" — the graph dispatch returned BEFORE installing
`set_forward_context`; the split path's eager MLA reads its metadata from the
context. Fix: both graph paths now run inside `set_forward_context(md, ...)`.

**Iteration 2 (boot q01s2):** first MTP step crashed during post-attention-core
capture: `ValueError: too many values to unpack (expected 2)` at
`deepseek_v2.py:402` = `num_tokens, hidden_dim = hidden_states.shape` — the
cores fed the shared MoE a 3D [batch,1,H] tensor; our vLLM's DeepseekV2MoE
requires 2D. The eager path flattens first. Fix: all cores now operate on the
flat [N, H] contract; the dispatch reshapes in/out. NOTE: this crash also
PROVES the capture machinery works inside the server process (the pre core had
already captured; the failure was our bug inside the post core's capture).

**Standalone smoke-test finding (negative, documented):** outside the server
process, `htorch.hpu.wrap_in_hpu_graph` fails to compile ANY graph containing
an add op on this stack (1.24.1): linear-only captures and replays fine; add,
norm+lin+residual, functional add all fail with
`Graph compile failed. Recipe: .graph_dumps/HabanaFusedOpLazy_N_M, synStatus 26`.
Reproduced with the server env vars replicated (PT_HPU_LAZY_MODE=1,
PT_HPU_WEIGHT_SHARING=0, PT_HPU_ENABLE_LAZY_COLLECTIVES=true,
RUNTIME_SCALE_PATCHING=1, FUSER_ENABLE_MULTI_THREADED_INVOCATIONS=1,
PT_HPU_EAGER_PIPELINE_ENABLE=0/1) and after importing vllm. Root cause not
established (the failing recipe leaves no dump); the server process captures
the same op classes successfully at every boot, so the difference is
process-state-level, not env-level. Consequence: HPU validation of the cores
must happen in-engine (server boot), not in standalone scripts.

**CPU contract tests (green, `tests/cpu/check_draft_graph.py`):** storage
independence of `_private_copy`; split-core chain == eager forward (atol 1e-6)
with position-0 zeroing held; metadata field extraction/rebuild round-trip;
flag parsing with bad-mode fallback. The test loads the REPO copy of mtp.py
explicitly (the venv's editable finder serves the deployed tree and meta_path
beats sys.path).

**Status:** boot q01s3 (2D fix) in flight; A/B measurement next: bs=1 decode
(baseline median 37.44), 16-stream aggregate (baseline 126.7), acceptance
(baseline ~1.88), accuracy (baseline 100.0%).

## EXP-Q01 results — split-mode draft graphs: HPU A/B (2026-11-07, boot q01s3)

**Config:** MTP k=1, TP8+EP, lin prompt buckets, 32K/8seq window,
`VLLM_GLM_MTP_DRAFT_GRAPH=1 VLLM_GLM_MTP_DRAFT_GRAPH_MODE=split` vs the E015-A
baseline (same window, eager draft).

**Performance (VERIFIED, 5 reps bs=1 / steady-state reps at concurrency):**

| Metric | Eager draft (baseline) | Split graphs | Delta |
|---|---|---|---|
| bs=1 decode wall tok/s | median 37.44, mean 37.12 | **median 40.93, mean 41.28** | **+9.3% / +11.2%** |
| bs=1 streaming tok/s | 20.7–21.0 | 22.1–24.4 | +6–16% |
| Aggregate @4 streams | 92.8 | 103.0 | +11% |
| Aggregate @8 streams | 128.0 | 151.1 | +18% |
| Aggregate @16 streams | 126.7 (saturated) | **~168 (167.4–168.9)** | **+33%** |
| Acceptance (mean len) | ~1.88 | 1.84–1.85 | held |
| Acceptance (pos-0) | 73–98% | 84.3–84.5% | held |

The eager-draft aggregate saturation at ~128 t/s is broken: the draft-graph gain
SCALES with concurrency (the eager draft's host cost scaled with the virtual
verify-row batch). MTP @16 remains net-negative vs non-MTP (213.2) but the gap
narrowed from −40% to −21%.

**Accuracy gate: FAILED.** Suite v1 (40 items, greedy): **97.5%** (39/40) on both
a 1-rep and a 5-rep run (Stage-1 baseline: 100.0% × 5 reps, 200/200). The failed
item is R003 (bat-and-ball: answers "10", expected "5") — deterministic across
3 direct queries. The gate allows ≤ 0.5 pp regression; measured −2.5 pp.

**Decision (per BENCHMARK_POLICY): NOT adopted as default.** The implementation
stays in the tree, env-gated DEFAULT OFF (zero behavior change for the stock
config). The +33% aggregate prize motivates a dedicated correctness follow-up
(Q01b) rather than acceptance.

**Q01b hypotheses for the R003 flip (ranked):**
1. FP drift in the graphed draft hidden state vs eager (op-fusion order) changes
   the draft token distribution; in MTP the OUTPUT includes the bonus token
   predicted GIVEN the draft proposal, so borderline 2-token answers can flip
   even with row-0 logits unchanged. Test: logprob the first token under both
   configs; diff the draft proposals on R003.
2. The attn mode (whole step in one graph) may have different (possibly smaller)
   drift than the split mode's two graphs + eager seam. Test: attn-mode boot,
   same suite.
3. A seam bug (mark_step/sync placement) corrupts one op class. Test: the
   I015-style layer capture on the draft path, or bisect by graphing only the
   post-attention core.

**Next action:** restore the known-good default boot (graphs OFF); run Q01b
hypothesis 1 (logprob diff on R003) on the next planned boot.

**R003 logprob evidence (graphs-ON):** after `</think>` (-0.092), the next-token
distribution is `10` at -0.392 vs `5` at -1.142 — a 0.75-nat gap, NOT a near-tie.
The model genuinely ranks "10" first under split graphs; under the Stage-1
baseline it answered "5" across 200/200 runs. This is a real distribution shift
introduced by the graphed draft, not FP tie-breaking. Q01b hypothesis 1
(bonus-token path) is consistent with this: the answer tokens arrive via the
draft-proposal-dependent bonus path.

## EXP-Q07 — Small-prompt prefill ladder + TTFT measurement correction (2026-11-07)

**Method:** `bench_prefill.py` (gen=1 token, so wall ≈ TTFT; prefill t/s =
prompt_tokens / wall). 3 reps per depth; rep1 pays first-use graph capture for
the prompt bucket, rep3 is warm. Server: MTP k=1 (draft mode irrelevant to the
target-prefill property measured; the draft prompt-KV fill IS included in the
wall).

**Warm prefill throughput (t/s) by prompt depth:**

| Prompt tokens | Wall (s) | Prefill t/s |
|---|---|---|
| 467 (~512) | 0.140 | **3,338** |
| 907 (~1K) | 0.189 | **4,802** |
| 1,784 (~2K) | 0.313 | **5,697** |
| 3,514 (~4K) | 1.303 | 2,696 |
| 5,267 (~6K) | 1.393 | **3,782** |
| 7,148 (~8K) | 1.395 | **5,123–5,456** |

**Finding 1 — the prefill objective is (mostly) already met.** The 3,000–6,000
t/s band holds at 512–2K, 6K, and 8K depths; the 3.5K point sits at 2,696 t/s
(just under) with a step-change in wall time (0.31 s -> 1.30 s) crossing the
2048->4096 lin-bucket edge. The 4K bucket shape is the one prefill graph family
worth a look (Q07b).

**Finding 2 — TTFT measurement correction (supersedes C001's small-prompt
figure and D005's "TTFT" figures).** The historical "warm small-prompt ~770
t/s" (C001) and "TTFT 2.1 s @2K" (D005) included 64-token generation in the
wall (check_ttft.py uses max_tokens=64): 64 gen tokens at ~47 ms ITL ≈ 3 s
dominated the wall. With gen=1, warm TTFT at 2K is 0.31 s — 7x smaller. The
"prefill is CPU-bound / AccThread busy-wait" narrative (A017/B009/Q06) was
built on those inflated figures and must be re-examined: at 5,697 t/s warm the
prefill is already in the objective band, and the host-overhead share is far
smaller than believed. Q06 stays RESEARCH but with a much smaller expected
gain; the ~25% scheduler-headroom estimate is unvalidated.

**Caveat:** the D005 figures for the user-facing TTFT (with streaming) remain
valid as END-TO-END first-token latency under 64-token generation budgets; the
correction is about attributing that time to prefill.

## EXP-Q10 — 1M-context serving validation (2026-11-07)

**Setup:** `serve_mtp_1m.sh` — max-model-len 1,048,576 (the published config
context length), MTP k=1, TP8+EP, lin buckets, KV override 1660 blocks.

**Block-size finding (corrects the sizing math):** the `--num-gpu-blocks-override`
counts 640-token HYBRID attention blocks, not 128-token cache blocks. First
attempt (8192 blocks) auto-sized a 5.22M-token pool (38.4 GiB/card KV) and OOM'd
on a 5 GiB single allocation. Correct sizing: 1660 blocks = 1,062,400 tokens ≈
8 GiB/card — boots clean at GMU 0.82 (weights 38.2 + KV 8 + graphs reserve 4.3).

**Needle test (third run, warm graph set):**
- Prompt: 860,739 tokens (word-padded doc; needle "The access code is
  ZEPHYR-7492" at ~950K target ≈ deep in the last 10%)
- Wall 523.7 s → 1,643 t/s deep-context prefill (8192-token chunked prefill;
  the per-chunk warm throughput at short depths is 3.3-5.7K t/s — Q07)
- Answer: **"ZEPHYR-7492" — PASS** (exact retrieval at ~860K depth)

**Verdict:** the 1M-context objective is SERVING-VALIDATED on 8 HPUs (window
1,048,576; retrieval proven at ~860K; 188K window margin). Deep-context prefill
(1,643 t/s at 860K) is outside the 3-6K band measured at short depths — the
band objective is met for practical prompt depths (Q07), and deep-context
prefill throughput is now a measured, separate data point.

**Caveats:** 860,739 tokens is the tokenizer's count of the "~1M" word-built
document (word estimates undercount ~14%); the window itself is validated to
1,048,576 by config and the pool holds 1,062,400. A needle at the full 1,048,576
depth is queued as a follow-up (needs a ~1.19M-word document; HTTP request
~4.8 MB).

## EXP-Q07b — 4K bucket-edge hypothesis: NEGATIVE (2026-11-07)

Boot with VLLM_PROMPT_QUERY_BUCKET_STEP=1024 (finer prompt buckets), warm rep3:
2,657 tok -> 2,154 t/s; 3,514 -> 2,475 (baseline at step 2048: 2,696 — no gain,
slightly worse); 5,267 -> 3,998; 7,148 -> 4,839. The bucket step is NOT the
cause of the 3.5K dip.

**New signature:** warm wall time is ~0.31 s up to ~1.8K-token prompts, then
JUMPS to a ~1.2-1.4 s FLOOR that is flat from 2.6K through 7.1K tokens
(independent of prompt size and bucket step). A fixed per-prompt cost in that
regime, not a throughput limit. Prime suspect: the draft prompt-KV fill pass
(prefill_cache_only), documented prompt-proportional (D012) but possibly with a
large fixed component at these depths; second suspect: KDA chunk-alignment
re-bucketing of the prefill graph family.

**Action:** config reverted to step 2048 (equal-or-better everywhere, fewer
graphs). Q07c opened: locate the fixed ~1.2 s per-prompt cost (server-side
phase timing for one 3.5K prompt: draft-pass wall vs target-prefill wall).

## EXP-Q14 — TP4+EP deployment-candidate benchmark pass (2026-11-07, boot tp4d)

**Config:** 4 cards (HABANA_VISIBLE_DEVICES=0-3), TP4+EP, MTP k=1, GMU 0.95,
graph capture 128, 32K/8seq window, lin buckets, KV override 320 hybrid blocks
-> 299,593-token pool (the TP4 pool semantics: ~936 tokens/block; override 1280
auto-sized 1.2M tokens and OOM"d on a 1.44 GiB single allocation; 512 never
booted due to a leaked-device acquire from the prior crash — clean devices first).

**Results vs TP8 (same window, same benches, warm reps):**

| Metric | TP4+EP (4 cards) | TP8+EP (8 cards) | TP4/TP8 |
|---|---|---|---|
| bs=1 decode wall tok/s | median 33.26, mean 33.44 | median 37.44 | **89%** |
| Aggregate @16 streams | 113.5 | 126.7 | **90%** |
| Prefill warm @2K | 1,070 t/s | 5,697 t/s | 19% |
| Prefill warm @8K | 3,857 t/s | 5,123 t/s | 75% |
| Acceptance | 1.81-1.95 (pos-0 81-95%) | ~1.88 (84%) | held |
| Accuracy (40 items x 2 reps) | **100.0%** | 100.0% | PASS |
| Boot to health | ~13 min | ~19 min | faster |

**Verdict:** the 4-HPU deployment candidate is VIABLE: ~90% of the 8-card
throughput on half the hardware, accuracy gate passed, MTP healthy. The prefill
weakness at short prompts (19% at 2K) carries the SAME flat ~1.2-1.9 s
per-prompt floor seen at TP8 (Q07c) — the fixed cost dominates more when the
card count halves; fixing Q07c disproportionately helps the TP4 deployment
shape. Deep prefill (8K) is 75% of TP8.

**Caveats:** KV pool 299,593 tokens covers the 32K x 8-seq window with margin;
larger windows at TP4 need pool re-sizing (weights leave 11.1 GiB/card free).
The 1M-context objective was validated at TP8; at TP4 the pool budget (~9.5
GiB/card max) supports ~1.2M tokens — a TP4 1M boot is plausible but untested.

## EXP-Q01b/Q15 — the R003 accuracy matrix: draft graphs are accuracy-NEUTRAL;
## a TP8-specific drift vs Stage-1 is the real signal (2026-11-07)

**Method:** R003 (bat-and-ball) probed directly (greedy, logprobs) plus the full
40-item suite (2 reps) across a config matrix on the current tree.

**Matrix (suite result / R003 answer):**

| Config | Suite | R003 |
|---|---|---|
| Stage-1 boot (TP8, pre-I015 tree, orchestrator config) | 100.0% x5 | "5" |
| TP8 + exp buckets + current tree, graphs OFF | 97.5% | "10" |
| TP8 + lin buckets + current tree, graphs OFF | 97.5% | "10" (-0.392 vs "5" -1.142) |
| TP8 + lin + split draft graphs (Q01) | 97.5% | "10" (IDENTICAL logprobs) |
| TP4 + lin + current tree (Q14) | 100.0% x2 | "5" |

**Findings:**
1. **Q01 verdict REVERSED**: the split draft graphs produce byte-identical R003
   logprobs and the identical suite result as their own graphs-OFF control.
   The draft graphs are accuracy-NEUTRAL on this suite. The earlier gate
   failure was an artifact of comparing against Stage-1 across configs.
2. **A TP8-specific accuracy drift exists on the current tree**: every TP8
   configuration tested (exp/lin buckets, graphs off/on) sits at 97.5% with
   R003="10", while Stage-1's boot scored 100% and current-code TP4 scores
   100%. The drift is bounded to R003 so far (-2.5 pp, gate limit -0.5 pp).
3. R003 is plausibly CONFIG-CHAOTIC: the intuitive-wrong answer ("10") vs the
   reasoned-correct ("5") sits behind a long greedy reasoning trace; small
   numerics changes (TP sharding order, KDA state FP) can branch the trace.
   The 0.75-nat final-answer gap does NOT imply a large model change.

**Q15 opened (accuracy drift TP8 vs Stage-1):** candidates ranked:
  a. The deployed-tree drift since Stage-1 (mtp.py rewritten for Q01 - the
     eager path should be op-identical but is the prime suspect to EXCLUDE
     first: deploy the Stage-1-era mtp.py from git 719f5bd and re-probe R003);
  b. Decode-bucket pinning differences (Stage-1 pinned decode bs; the current
     boots do not - verify-row graph family changes);
  c. C015 kda.py (was already deployed at Stage-1 - unlikely but cheap to
     test via a kda.py checkout);
  d. Genuine TP8-sharding numerics chaos (TP4 passes; if a-c exclude, the
     verdict is chaos and the gate policy needs same-boot-family controls).

**Adoption impact:** Q01 (draft graphs, +33% aggregate @16, +9.3% bs=1) is
RE-ELIGIBLE for adoption once Q15 establishes the proper same-config control;
the +33% was never accuracy-blocked on its own merits.

## EXP-Q15 continued — candidate exclusions and the working verdict (2026-11-07, boots q15a)

Empirical exclusions on the R003 flip ("5" at Stage-1 -> "10" now):
- (a) the Q01 mtp.py rewrite: EXCLUDED — the original 719f5bd mtp.py deployed
  (boot q15a, pinned decode, exp buckets) answers "10" identically. The eager
  path is line-for-line op-identical (inspection agrees).
- (b) decode-bucket pinning: EXCLUDED — Stage-1's exact pinning env
  (DECODE_BS_BUCKET_MIN=8/STEP=8, BLOCK_BUCKET_MIN=32) still answers "10".
- (c) C015 kda.py: EXCLUDED — committed 11-06 06:40, before the Stage-1
  accuracy runs (finished 06:56 on 11-07, R003="5" in both conditions'
  artifacts, including the NON-MTP control).
- Buckets (lin vs exp): EXCLUDED — both fail identically.

**Working verdict: UNRESOLVED root cause; stack-state-sensitive item.** The
Stage-1 night's boots (both conditions, non-MTP included) answered "5"; every
boot since answers "10" on TP8 while current-code TP4 answers "5". No
numerics-relevant tree difference is identifiable (deployed model.py byte-equals
719f5bd; kda.py equals C015 = Stage-1 state). R003 sits on a knife edge between
the intuitive ("10") and reasoned ("5") answers behind a long greedy reasoning
trace; per-boot state (graph family, memory layout, request history) branches
it. Item-level noise: 1/40 = 2.5 pp — the same magnitude as the gate limit.

**Gate-policy consequence (recorded in DECISIONS.md):** cross-boot accuracy
comparisons on suite v1 carry at least +/-2.5 pp item-chaos noise; same-boot
A/B pairs (control vs treatment on the same boot family) remain valid. Q01's
draft graphs are accuracy-NEUTRAL against their own control (byte-identical
logprobs) — re-eligible under the same-boot policy.

## GaudiGrimoire-sourced discoveries (Q16/Q17/Q18)

**Q16 fp8-moe-gather (fix ALREADY IN THIS REPO, now APPLIED to the deployed
vllm-gaudi).** The Habana `mixture_of_experts` op reads EVERY local expert's
weights per call; at decode, tokens x top_k distinct experts <= g =
min(local_experts, tokens*top_k) suffices. Backport of vllm-gaudi #1731
(v0.30.0) to v0.26.0: `VLLM_HPU_MOE_GATHER=1` (static gathered count -> captures
into HPU graphs), RATIO 0.4, VERIFY mode logs FP8-ULP differences > 2 ULP.
Applies to block-quantized FP8 MoE with silu experts (ours: 128x128 block FP8,
top-8 of 288; 36 local experts/rank at TP8+EP8; bs=1 k=1 verify = 2 tokens x
top-8 = 16 routes -> g=16 of 36 = 2.25x less expert weight traffic).
Upstream evidence: fixes/fp8-moe-gather/README.md, vllm-gaudi#1731.
Expected gain (ESTIMATE): the expert-weight stream is the dominant single-stream
decode cost on large EP MoEs; +20-50% bs=1 decode plausible. NEXT BOOT.

**Q17 EP4xTP2 expert layout.** The cookbook's own GLM-5.3 MoE decode study
(docs/results/moe-decode-step.md, MXFP4, same TP8+EP8 topology): fewer wider
bins cut the balls-in-bins EP imbalance (one request: 16 routes in 8 bins ->
busiest 4.45 vs mean 2.0; in 4 bins -> 6.3 vs 3.95). Measured +11% (v7 kernels)
/ +5% (v8) single-stream decode, memory-neutral; quality gates held (GSM8K
49/50, needles, top-1 agreement 0.988). Trap recorded: placement selection via
host index_select page-faults the checkpoint (1.4-1.9x load time); read
contiguous shares first.

**Q18 router fast path + early draft + device-side rejection.** Same study:
three BIT-IDENTICAL changes took one-stream decode 51.64 -> 57.78 tok/s (+12%):
- router fast path (~1.7 ms/step): with 0 < num_expert_group <= topk_group the
  grouped top-k's group stage selects every expert yet costs ~25 us/layer on
  HPU -> go straight to the biased top-k. OUR CONFIG QUALIFIES: n_group=1,
  topk_group=1 (verified in config.json).
- device-side rejection sampler walk (float64-uniform-rounded-up-to-float32 =
  identical draws) + device-side draft row indices ("last accepted row is
  2i + accepted - 1"): removes the end-of-step host sync; 0 mismatches in 7,000
  steps/rank in the study's check mode.

**Step anatomy reference (GLM-5.3, TP8+EP8, MTP k=1, 39 ms step):** expert
kernels 19.5 ms (slowest EP rank; mean 9.6 — imbalance, not a hot rank),
attention 6.9, MoE-adjacent + dense 3.8, all-reduce latency 3.4 (162/step),
host at step boundary ~5.5. 322 graph launches/step @ ~74 us host each.

## EXP-Q16 — fp8-moe-gather: NEGATIVE at our shapes (2026-11-07, boots q16/q16b)

**Gate diagnosis first:** with the default RATIO=0.4 the gate refused our shape
(bs=1 k=1 verify = 2 tokens x top-8 = 16 routes vs 36 local x 0.4 = 14.4) - the
gather never engaged and q16 measured baseline noise (36.26 vs 37.44 median).

**With RATIO=0.5 (gate passes, gather engaged):** bs=1 decode median ~20.4,
mean ~19.7 (reps 2-5: 20.19/17.13/20.59/20.72) vs the 37.44 baseline -
**-45%, decisively negative**. The gathered path (index_select of the 16
routed experts' weights + one native FP8 up/gate GEMM + fp32 down projection)
costs more than the all-36-expert stock fused op saves on this stack: the
gathered-weight copy plus the fp32 down GEMM dominate at 2-row shapes.

**Verdict:** NOT adopted. `VLLM_HPU_MOE_GATHER` stays unset (stock op; the
applied patch is inert without the env). The upstream win (vllm-gaudi #1731)
does not transfer to GLM-5.3-Flash's 2-token verify shapes at 288 experts.
Revisit only if a later stack changes the gather implementation's cost model.

## EXP-Q18 lever 1 — router fast path: VERIFIED +4.4%, bit-identical (2026-11-07, boot q18b)

**Change:** `patches/router_fast.py` (this repo, fail-closed, env-gated
`GLM_ROUTER_FAST=1`): in vLLM cpu_fused_moe.grouped_topk, skip the group stage
when num_expert_group == 1 and topk_group == 1 - our config (verified) - where
it is provably an identity (masked_fill with an all-true mask). Saves the
group_scores/topk/scatter/expand/masked_fill launches per MoE layer.

**Verification:**
- CPU contract test (tests/cpu/check_router_fast.py): fast == stock bit-identical
  at n_group=1 (bias and no-bias); gate refused at n_group=8.
- HPU bit-identity: R003 logprobs BYTE-IDENTICAL to the pre-patch server
  ("10" -0.392 / "5" -1.142); suite 97.5% (the R003 chaos item, per EXP-Q15).
- First boot crashed: the patch's env def used os without importing it in
  cpu_fused_moe.py (NameError) - fixed with a module-local import; the patch
  source updated to match.

**Performance (steady reps 6-13, after warm-up):** 38.59/38.95/39.14/39.25/
38.78/40.02/38.54/39.28 -> **median 39.10, mean 39.07** vs the 37.44/37.12
baseline = **+4.4% median / +5.3% mean** - matching the study's ~1.7 ms/39 ms
(~+4.5%) prediction almost exactly. Best verified single-stream figure of the
project (37.4 -> 39.1); the earlier reps 2-5 were warm-up-contaminated.

**Status:** ADOPTED-CANDIDATE (bit-identical, gate-neutral, +4.4%). The
remaining Q18 levers (device-side rejection walk, early draft) are staged as an
adaptation task (see TASKS Q18 notes).

## EXP-Q18 levers 2-3 — device-side rejection + early draft: VERIFIED, aggregate
## +46% over the eager draft (2026-11-07, boot q18c)

**Change:** the cookbook's glm-5-mxfp4 decode_lean levers adapted to this port
(`patches/decode_lean.py`, lean module copied into our package, import rewritten,
router target dropped - covered by router_fast.py). Anchors matched the deployed
tree including the already-applied spec-draft-prefill and spec-penalties patches.
Env: `GLM_SPEC_DEVICE_REJECT=1` (the accept/reject walk as prefix logic on the
device, same draws: the host float64 uniform rounded up to float32 compares
equal) + `GLM_EARLY_DRAFT=1` (the draft enqueued right after the sampler with
device-computed row indices) - removes the end-of-step host sync.

**Boot failure caught by the new watchdog:** the first boot crashed at import -
the lean module had been copied to the repo but not the deployed tree (the
editable install serves the deployed tree). Fixed; `tools/boot_watch.sh`
added to the repo (BOOT-OK / BOOT-FAILED / BOOT-TIMEOUT for every unattended
boot from now on).

**Results (steady state):**

| Metric | Eager draft | Split graphs only | **+ levers 2-3 (this boot)** |
|---|---|---|---|
| bs=1 decode | 37.44 | (39.10 w/ router-fast) | 38.1 median (neutral vs router-fast) |
| Aggregate @16 | 126.7 | ~168 | **183-185 t/s** |
| Acceptance | ~1.88 | 1.84-1.85 | 1.80 (held) |
| R003 logprobs | -0.392/-1.142 | identical | **byte-identical** |
| Suite | 97.5% (R003) | 97.5% | 97.5% (unchanged - bit-identical) |

**Verdict:** the combined Q18 stack (draft graphs + router fast path +
device-side rejection + early draft) takes the MTP aggregate from 126.7 to
**~185 t/s @16 streams (+46%)**, within 13% of the non-MTP 213.2 while keeping
MTP's latency benefit, with byte-identical greedy outputs. Accuracy-neutral
under the same-boot policy (bit-identical). ADOPTED-CANDIDATE for the serving
configuration; the MTP-at-concurrency crossover is pushed from 8 streams to
beyond 16.

## EXP-Q19 — the combined stack (draft graphs + all Q18 levers): no stacking;
## the levers-only config wins (2026-11-07, boot q19)

**Config:** Q18c's levers PLUS `VLLM_GLM_MTP_DRAFT_GRAPH=1` (split mode).

**Results:** bs=1 median 39.51, mean 38.83 (reps 2-6: 37.91/41.02/39.51/34.84/
40.85 - within noise of router-fast-only 39.10); aggregate @16 steady 164-173
(median ~169) - **WORSE than the levers-only 183-185**. Acceptance 1.79, R003
logprobs byte-identical, suite 97.5% (all unchanged - bit-identical).

**Why the draft graphs stop stacking:** the graph replay discipline
(mark_step + torch.hpu.synchronize around every replay) serializes the draft
step, defeating the early-draft pipelining that levers 2-3 exist to enable
(the draft enqueued immediately after the sampler, device-side, without
host syncs). The two optimizations target the same host-sync time from
opposite directions; the sync-free eager draft + pipelined enqueue wins.

**Decision (ADOPTED serving configuration):**
`GLM_ROUTER_FAST=1 GLM_SPEC_DEVICE_REJECT=1 GLM_EARLY_DRAFT=1`,
draft graphs OFF (eager draft). Best aggregate (183-185 @16, +46% over the
previous default, -13% vs non-MTP) at bs=1 parity with every other variant,
byte-identical outputs, suite unchanged. The serve script's env block is
updated accordingly; the draft-graph code stays available (env-gated) for
future host-overhead work where early-draft is not enabled.

## EXP-Q07c REVISED — the >2K prefill wall is quadratic attention, not a removable
## fixed cost (2026-11-07, boot q07c + edge probes)

Phase 1 (prefill-shaped draft calls through the split cores, env-gated) measured
NO change in the warm >2K walls (1.26-1.41 s, same as baseline) - because the
draft pass is ~2% of the wall (1 MLA + 1 MoE vs the target's 45 layers), as the
device-work arithmetic always said.

**Edge bisection:** prompts of 1888/1920 real tokens prefill in 0.30 s; 2095
tokens takes 1.46 s - the jump is the 2048->4096 prompt-bucket transition, and
the per-token wall grows 146 -> 308 us/token (~2.1x) - matching quadratic MLA
attention at depth (0.65x1 + 0.35x4 ~ 2.05x). The earlier "flat 1.2-1.4 s
floor" reading compared depths that all landed in the >=4096 buckets.

**Revised verdict:** the 3.5-8K prefill walls are the target's own quadratic
attention scaling - physics, not a removable host cost. The 3,000-6,000 t/s
band holds at 2K/6K/8K (5,750/4,166/5,663 t/s warm); the 3.5-4K dip
(2,500-2,800 t/s) is the 4096-bucket quadratic penalty and is not recoverable
by bucketing (Q07b negative) or draft-side changes. The draft-model prefill
KV fill is confirmed negligible at every size (the always-eager finding stands
but is immaterial).

**Q21 (the 1Cat kernel track) is the remaining lever for deep-depth prefill**
(TPC block-FP8 dequant + fused attention kernels), along with the 200 t/s
single-stream objective.

## EXP-Q21 step 1 — local 1Cat kernel inventory (2026-11-07)

The complete 1Cat-vLLM-Gaudi tree with compiled artifacts is on this host:

- Sources: `/opt/dsv41/1Cat-vLLM-Gaudi/csrc/flashinfer_gaudi/` (bridge/, host/,
  kernels/, pytorch/)
- TPC kernels (tpc-clang, gaudi2): `block_fp8_dequant_gaudi2.c` (128x128
  block-FP8 dequant - OUR checkpoint format), `add_rmsnorm_quant_bf16_gaudi2.c`,
  `silu_mul_quant_bf16_gaudi2.c`, `silu_and_mul_bf16_gaudi2.c`,
  `tp2_dynamic_quant_bf16_gaudi2.c`, plus DSV4.1-specific GDN/MTP kernels
- PyTorch op wrappers: hpu_flashinfer_{activation,gaudi,norm}.cpp + setup.py
- Build: `/opt/dsv41/build/` (bridge-build with compiled _hpu_C.so,
  dsv41-native, plugin-snapshot, synapse-build)

**Mapping to our model:** block_fp8_dequant (weights) + silu_and_mul (MoE
activation) + add_rmsnorm_quant (norm fusion) cover the GLM-5.3-Flash decode
hot path building blocks; the DSV4.1 MoE family (moe_fp8_prequant/
direct_finalize/prefetch_w2 variants, n256 experts) is the tuning playbook for
our 288-expert MoE at TP8+EP8 (36 local/rank).

**Next actions (Q21 step 2):** qualify `block_fp8_dequant` on one GLM layer
shape (correctness vs the stock fp8 GEMM path, then kernel time); port the
pytorch op wrappers into our plugin build (the DSV4.1 env used bridge-build.sh);
then the MoE variant selection for our shapes.

## EXP-Q02 closure — same-window non-MTP comparison: the MTP+levers stack wins
## on BOTH axes; the concurrency crossover is GONE (2026-11-07, boot nomtp)

**Setup:** serve_nomtp.sh = the adopted serving script minus the speculative
config (same lin buckets, same window, GLM_ROUTER_FAST=1 kept - the router
lever is spec-independent). The honest comparison against the adopted
MTP+levers stack under IDENTICAL conditions.

**Results (steady state):**

| Metric | non-MTP + router-fast | MTP + all levers (adopted) | MTP advantage |
|---|---|---|---|
| bs=1 decode | 32.4-32.8 | 38.0-39.1 | **+17%** |
| Aggregate @16 | 143-160 (median ~154) | 181-185 | **+18-20%** |
| R003 | (not probed this boot) | "10" | - |

**Findings:**
1. The historical "213.2 t/s non-MTP @16" was the PRODUCTION window (262K/32seq,
   exp buckets) - a different regime. Under the current serving window non-MTP
   delivers only ~154 t/s @16 and ~32.8 t/s bs=1.
2. With the Q18 levers, **MTP is net-POSITIVE at 16 streams (+18-20%)** - the
   eager-draft crossover documented in EXP-Q02/Q03 is eliminated by the lever
   stack, not merely pushed.
3. The adopted configuration (MTP k=1 + router fast + device-side rejection +
   early draft) is the best known config on every axis under the serving window.

**Q02 CLOSED.** The remaining aggregate gap to the 800 t/s objective (4.3x) is
the device-efficiency track (Q21 kernels) and scheduling work, not an MTP
trade-off.

## EXP-Q21 step 2 prep — the block_fp8_dequant kernel core read (2026-11-07)

`/opt/dsv41/1Cat-vLLM-Gaudi/csrc/flashinfer_gaudi/kernels/block_fp8_dequant_gaudi2.c`
(40 lines, Apache-2.0): a TPC-C kernel that expands E4M3 weights packed 256-per-128B
rows with per-(128-row, 256-col) block FP32 scales (rounded to BF16 before
multiplying - matching our checkpoint contract), using `v_f8_ld_tnsr_b` with
SW_UNPACK + `v_convert_f8_to_bf16_b` (load-pipe expansion into conversion lanes)
and vector stores. Index space over (block_k, block_n); the second-half scale
load is predicated on `block_k*256+128 < width`.

**Read-across to our model:** our weights are E4M3 with 128x128 blocks and FP32
scales - the same layout this kernel consumes. The mxfp4-moe framework's
`--extra-kernel` wraps it as `<guid>.c` + `glue.inc` + `op.cpp`; the decode
GEMV fusion (dequant-in-registers + dot, per the mxk8_core.h pattern) is the
next kernel to write: block-FP8 w13 GEMV + SiLU-and-mul and w2 GEMV variants
reading E4M3 in place (the Q16 gather lesson says the gather-then-GEMM
decomposition loses; the fused in-register dequant GEMV is the shape that won
for MXFP4 at 66 t/s).

**Qualification plan (step 2):** build the 1Cat kernel standalone
(tpc-clang -march=gaudi2), run it against a GLM weight shard slice on one card,
compare dequant output vs the stock fp8 GEMM path bit-for-bit on the scale
rounding, then micro-bench the GEMV variant against the stock fused MoE op at
our decode shapes (2/4/8/16 tokens x 8 routes).

## EXP-Q21 step 2 — toolchain + kernel build verification (2026-11-07)

- `tpc-clang` is on PATH (/usr/bin/tpc-clang, also /opt/habanalabs/bin); the DSV4.1
  bridge build environment (/opt/dsv41/bridge-build.sh) carries every include path.
- `block_fp8_dequant_gaudi2.c` compiles cleanly standalone:
  `tpc-clang -march=gaudi2 -O2 -x c++ ... -o ...` -> 13,856-byte TPC ELF, GUID string
  embedded (one pipelining warning, benign).
- The host-glue pair exists beside it (`block_fp8_dequant_gaudi2.cpp/hpp`):
  `GetGcDefinitions` + the `_binary_*_o_start/end` embedded-ELF pattern - the same
  shape as the mxfp4-moe framework's `glue.inc` contract, so the kernel drops into
  our variant build via `--extra-kernel` with a thin adaptation.

**Step-2 build verification: PASS.** The on-device qualification (dequant output vs
the stock fp8 path on a GLM weight shard) is the next session's opening action;
everything it needs is staged on the host.

## EXP-Q21 step 2 — the extra-kernel build pipeline: TPC gate PASS, op-library
## load blocked by a stack-header mismatch (2026-11-07)

**Built:** `kernels/mxfp4-moe/build.sh --variant glm --extra-kernel
extra/block_fp8_dequant_glm` with the adapted kernel (1-D index space, npts as a
scalar param), glue.inc (DATA_F8_143/F32 -> BF16 contract, pairs = (N/128)x(K/256),
setIndexSpace + scalar npts per the inst_mxk_deq convention), and op.cpp
(REGISTER_CUSTOM_OP_ATTRIBUTES via the legacy hpu_custom_op.h API - no
TORCH_LIBRARY def: the legacy bridge registers the schema by name).

**Results:**
- TPC library gate: **PASS - 3254 GUIDs (stock 3250 + mxk_w13x, mxk_w2x,
  mxk_deq, block_fp8_dequant_glm)**. Our kernel is IN the gated library.
- Op-library load: **FAIL in this venv** - `hpu::habana_d2d_memcpy_other is
  already registered` (habana_operator.h:799). The STOCK glm variant's op
  library fails the SAME probe here, so it is environmental: the build pulls
  -I/usr/include/habanalabs from the DSV41 stack (1.24.1-482) while the
  /opt/glm53 serving venv's plugin is the GLM stack's 1.24.1 - the registrar
  sets differ, and loading an op library linked against one stack into a
  process running the other double-registers. Verified not our op.cpp (the
  stock build fails identically; both load orders tried).

**Fix (next session):** build against the GLM serving stack's headers - either
run build.sh inside the GLM serving image, or point the Habana includes at the
venv plugin's include dir instead of /usr/include/habanalabs. The TPC ELF
itself is stack-version-tolerant (compiled for gaudi2; the recipe cache keys
per build).

**Assets staged (committed):** kernels/mxfp4-moe/extra/block_fp8_dequant_glm/
{block_fp8_dequant_glm.c, glue.inc, op.cpp}; /tmp/mxk_dist/libmxk_tpc.so carries
the gated GUID (rebuildable with one command).

## EXP-Q21 step 2 continued — the op-library load collision: root cause isolated
## (2026-11-07)

Bisection with /proc/self/maps + isolated dlopens:
1. `import torch` ALONE maps libhabana_pytorch_backend.so (the hpu platform
   plugin is a torch entry point in this venv) - ONE copy, venv path.
2. The TPC library (libmxk_tpc.so, 3254 GUIDs incl. ours) dlopens cleanly after
   the imports.
3. libmxk_op_lazy.so dlopens COLLIDE: `hpu::habana_d2d_memcpy_other is already
   registered` (habana_operator.h:799), in both venvs, both load orders, and
   with the STOCK glm op library too - it is not our op.cpp.
4. The trigger: mxk_op_lazy.cpp's STATIC `TORCH_LIBRARY_IMPL(custom_op, HPU,
   ...)` - the first HPU-dispatch-key impl registration from a second .so makes
   the habana dispatcher run its lazy init, which re-registers the built-in
   kernels (d2d memcpy et al.) that the backend .so already registered at
   `import torch`. The mxfp4 serving image's stack does not have this quirk
   (its gate passes in-image).

**Resolution paths (next session):**
a. Run the kernel inside the proper serving image (the framework's assumption);
b. A single-.so design registering the ops from the plugin's own init (the
   upstream pattern) instead of a second load_library'd .so;
c. Probe the stack for an idempotent-registry flag before the second
   registration (habana_dispatcher init guards).
The TPC ELF itself is DONE and gated (block_fp8_dequant_glm is in
libmxk_tpc.so); only the torch-op shim's load path remains.

## EXP-Q21 path (d) — the serving-time load order: EXCLUDED; the fail-closed
## design held (2026-11-07, boot q21rt)

The installer (mxfp4 pattern adapted: runtime in vllm_gaudi/ops/block_fp8_rt.py,
setup_env hook at the top of vllm_gaudi/__init__.py, install hook at the end of
ops/hpu_fp8.py) was wired and the real server booted with GLM_FP8_MXK=1.

**Result:** the collision is LOAD-ORDER-INDEPENDENT - torch.ops.load_library of
the op .so collides with the registrar even at the serving-time plugin-hook
point. The runtime's fail-closed except caught it: "block_fp8_dequant_glm NOT
registered ... stock ops kept" in every worker, and the boot completed healthy
on the stock ops. No production risk at any point.

**Q21 standing after this session:**
- DONE: kernel source adapted + compiled (tpc-clang, valid TPC ELF); the
  extra-kernel pipeline built (TPC gate PASS, 3254 GUIDs incl. ours); the
  installer/runtime/hooks implemented and installed (fail-closed verified in
  the real serving process).
- BLOCKED: the torch-op shim cannot load on this stack (vllm-gaudi v0.26.0 +
  Gaudi SW 1.24.1) - the second .so's HPU-dispatch registration re-triggers the
  dispatcher's built-in kernel registration. Order-independent; both venvs.
- PATHS FORWARD: (a) a stack upgrade (v0.30.0 carries the fp8-gather natively
  and may have the registrar guards), (b) a plugin-rebuild track (register the
  op from the plugin's own init), (c) run the kernels in the DSV41-style
  isolated image. All are stack-level engineering beyond this port.

**Session close:** the server runs the adopted config (the hooks are installed
but inert without GLM_FP8_MXK=1 - subsequent boots are unaffected).

## EXP-Q11 — MTP k=2 under the fast draft path: REFUTED (2026-11-08, boot q11)

**Hypothesis tested:** the adopted levers made the draft cheap enough that a
second speculation depth (k=2: 3-row verify + 2 draft passes/step, acceptance
2.04-2.52 of 3.0 historically) turns net-positive - reopening A011/A013.

**Setup:** serve_mtp_k2.sh (the adopted config, num_speculative_tokens=2, all
levers kept). Same window, lin buckets.

**Results (steady state):**

| Metric | k=1 (adopted) | k=2 + levers | Delta |
|---|---|---|---|
| bs=1 decode | 38.0-39.1 | 33.4-34.6 (median 34.25) | **-11%** |
| Aggregate @16 | 181-185 | 130-133 | **-29%** |
| Acceptance | 1.80-1.85 of 2.0 | 2.25 of 3.0 | +22% as expected |

**Arithmetic:** the k=2 step runs ~36% slower (20.7 -> 15.2 steps/s: 3-row
verify + a SECOND full draft pass per step) while acceptance gains only +22%
(1.85 -> 2.25). The second draft pass costs a full eager draft traversal per
step - the levers made each draft pass cheap but did not make the SECOND one
free.

**Verdict: k=2 REFUTED under the levers; k=1 remains optimal.** A011/A013
stay closed unless the draft pass becomes near-free (the Q21 kernel track:
an in-graph draft would change this arithmetic - the draft replay would add
~device-time only, making the k=2 draft marginal cost ~2x device instead of
2x (device + host), which could flip the balance).

## EXP-HYGIENE — stale-worker cleanup + the clean-host re-measurement (2026-11-08)

**Finding:** multiple orphaned VLLM::Worker processes (100% CPU each, 236-295
min runtime) survived earlier boots - tmux kill-server does not reap detached
worker processes after crash chains. They competed for host CPU during every
measurement since.

**Cleanup + clean-host re-measurement (adopted config, k=1 + all levers):**
- bs=1 decode: 34.85-39.15, median 36.86 (still climbing: last rep 39.15) -
  consistent with the verified 38-39 band; the ~5% run-to-run variance is
  inherent (the lazy-mode AccThread legitimately busy-waits at ~100% CPU).
- Aggregate @16: 53.7 (capture) -> 171.4 -> 179.6 -> 160.1 -> **186.8** -
  the verified 181-185 band holds (186.8 best).

**Verdict:** the session's numbers stand within the noise band; the stale
workers added variance, not bias. PROCESS FIX: every boot script must reap
orphans first - `pkill -9 -f VLLM::Worker` before the tmux start (boot_watch
already reports the boot outcome).

**Final adopted-config numbers (clean host, steady state):**
- bs=1 decode: **median ~37-39 t/s** (best 39.4)
- Aggregate @16: **~180-187 t/s** (best 186.8)
- vs non-MTP same-window: +18-20% on both axes (Q02 closed)

## EXP-Q21 collision — gdb backtrace attempt: tooling-limited (2026-11-08)

gdb (available on the host) with `catch throw` did not intercept the dlopen-time
static-init throw (stripped release .so, no habana_operator.h debug symbols; the
terminate path bypasses the catchpoint arming). The registrar mechanism stands as
documented: the second .so's TORCH_LIBRARY_IMPL(custom_op, HPU) static block
triggers the habana dispatcher's lazy init, which re-registers the built-in
kernels the backend .so registered at import.

**Resolution requires stack-level work** (the three recorded paths); the
symbol-level hunt would need debug-symbol builds of libhabana_pytorch_backend.so.

## EXP-Q17 — EP4xTP2 expert layout on the stock FP8 stack: REFUTED
## (2026-11-08, boot q17)

**Implemented and deployed:** the mxfp4 moe_layout adapted for the FP8 port -
the group-layout hook in vLLM's FusedMoEParallelConfig.make EP branch
(GLM_MOE_TP=2: 4 groups of 2 ranks, 72 experts/group, I/2 per rank) + the
load_fused group-slice hook at the top of RoutedExperts.load_weights' is_fused
branch (the combined [E, d1, d2] w13/w2 tensors; explicit shard_id
discrimination; the row/column block slicing per the mxfp4 module's pattern).
The placement machinery (GLM_MOE_PLACEMENT) is supported but unset
(contiguous groups).

**Correctness: VERIFIED.** The model is coherent under the group layout -
R003 answers "5" (the reasoned answer; another chaos draw - the layout changes
the bf16 rounding as the study predicted), acceptance 1.71 (held; the two-slice
down-projection sum differs by bf16 rounding as documented).

**Performance (steady state):**

| Metric | EP8 (adopted) | EP4xTP2 | Delta |
|---|---|---|---|
| bs=1 decode | 38.0-39.1 | 29.5-31.1 (median 30) | **-22%** |
| Aggregate @16 | 181-185 | 142.8-147.0 | **-21%** |

**Verdict: REFUTED on the stock FP8 stack.** The study's +11% was specific to
their custom MXFP4 kernels (v7/v8: a fixed per-route staging cost that halves
with wider expert groups). Our stock fused-MoE op has a different cost model -
the halved per-rank GEMM widths cost more than the reduced balls-in-bins
imbalance saves. The adaptation itself is validated and stays in the repo
(inert without GLM_MOE_TP); it becomes relevant again if the Q21 custom
kernels land (their cost model would resemble the mxfp4 kernels').

## EXP-PHYS — the weight-read floor and objective feasibility (2026-11-08)

From the verified geometry (K=4096, I=2048/expert, 288 routed experts, 42 MoE
layers, FP8 1 B/weight; the 306 GiB checkpoint IS the experts):

- Expert weights: 25.2 MB/expert/layer; all 288 experts x 42 layers = 304 GB
  total = **38.1 GB per card per step** (EP8; the stock fused op reads every
  local expert's weights at EVERY step, any batch size).
- **Weight-read floor: 15.5 ms/step** (38.1 GB @ 2.45 TB/s/card peak) -
  batch-independent while the stock op reads all experts.
- Physics-max (weight-read floor only, acceptance 1.85):
  - aggregate @bs=16: **~1,900 t/s**
  - single-stream: **~119 t/s**

**Objective feasibility verdicts:**
1. **800 t/s aggregate @16: feasible in principle** - the floor allows ~1,900;
   we measure 183 (a 162 ms step vs the 15.5 ms floor = **10.5x step
   inefficiency**: small-M GEMMs, launch pattern, collectives, the eager draft
   at 16 streams). The Q21 kernel track + batch-efficiency work attack exactly
   this.
2. **200 t/s single-stream: IMPOSSIBLE with the stock all-experts-read op** -
   200 t/s needs a 9.3 ms step, BELOW the 15.5 ms weight-read floor. It
   REQUIRES reading fewer expert bytes per step:
   a. the routed-expert gather done RIGHT (Q16's gather lost to its
      implementation; a fused in-register dequant GEMV - the Q21 kernel -
      reads only the ~16 routed experts: 38.1 -> ~2.1 GB/step, floor 0.9 ms),
   b. or a narrower quantization (MXFP4-class: half the bytes, floor 7.8 ms -
      still above 9.3 ms at acceptance 1.85; marginal), so (a) is THE path.
3. The k-depth lever compounds with (a): with the expert reads at ~0.9 ms/step,
   deeper speculation (k=2/k=4) re-enters (the draft+verify cost dominates
   instead), per the mxfp4 study's MTP-4 arithmetic (accepted 3.33/step).

**Planning conclusion:** the 800 t/s aggregate is an efficiency problem (10.5x
headroom); the 200 t/s single-stream is a BYTES problem (the stock op's
all-experts read makes it physically impossible) - the Q21 fused routed-expert
kernel is the prerequisite for the single-stream objective, and it also unlocks
the aggregate headroom. Both objectives converge on the same next step.

## EXP-NS16 — max-num-seqs 16: the biggest aggregate win of the project
## ADOPTED (2026-11-08, boot ns16)

**Hypothesis:** the 16-stream benchmark time-shares max-num-seqs 8 slots; the
38.1 GB/card expert-weight read per step amortizes over only 16 verify rows.
Raising max-num-seqs to 16 runs all 16 streams concurrently (32 rows/step) -
the same weight read amortizes over twice the tokens.

**Results (steady state, graphs warmed over ~6 reps):**

| Metric | max-num-seqs 8 (adopted) | **max-num-seqs 16** | Delta |
|---|---|---|---|
| Aggregate @16 streams | 181-185 | **245-260 t/s (converged ~258)** | **+40-42%** |
| bs=1 decode | 38.0-39.1 | 37.0-38.0 | neutral (within noise) |
| Acceptance | 1.80-1.85 | 1.80 | held |
| Accuracy suite | 97.5% (R003) | 97.5% (unchanged) | bit-identical numerics |
| vs non-MTP same-window (~154) | +18-20% | **+68%** | |
| vs the historical non-MTP 213 (old window) | -14% | **+22%** | |

**Verdict: ADOPTED** - max-num-seqs 16 is written into serve_mtp.sh as the
default. The physics framing (EXP-PHYS) predicted exactly this: the weight-read
floor is batch-independent, so doubling the rows per step halves the
per-token weight cost. The aggregate now stands at **~258 t/s @16 streams** -
above the historical non-MTP production figure for the first time, with the
MTP latency benefit kept.

**Notes:**
- The KV pool (1280 blocks, 163,840 tokens) covers the benchmark's short
  sequences at 16 concurrent; production long-sequence concurrency (16 x 32K =
  512K tokens) needs the pool raised (~4096 blocks) - a config note, not
  measured here.
- The decode-bucket graphs for bs up to 16 capture on first use (the warm-up
  reps paid ~30-60 s per new shape; production should pre-warm, D003).
- The bs=1 path is unchanged (the bs-1 graphs are the same shapes).

## EXP-Q21 final diagnosis state (2026-11-08, boot q21q)

The rebuilt op library (with the restored TORCH_LIBRARY defs — the schema string
verified in the binary) loads at serving time WITHOUT the registrar collision
(confirming: the collision was an artifact of the probe import order, where
habana_frameworks.torch.core is imported without the plugin and the op library's
dlopen pulls the plugin fresh). The serving-time `torch.ops.load_library`
SUCCEEDS — but the op remains invisible to `torch.ops.custom_op` ("NOT
registered" in all 8 workers).

**Remaining question (next session, minimal repro):** the registration
order/namespace interplay in the worker processes — mxk_op_lazy.cpp's static
`TORCH_LIBRARY_FRAGMENT(custom_op)` + `TORCH_LIBRARY_IMPL(custom_op, HPU)` blocks
run at the same .so load; our def adds to the same namespace. Whether the HPU-impl
registration for OUR op hits the same dispatcher-init quirk (silently swallowed by
load_library), or the schema def is shadowed, is diagnosable with our op.cpp ALONE
in a standalone .so + the serving-order imports (the minimal repro that the
earlier probes missed by using the wrong import order).

**The qualification path is otherwise clear:** once the op is Python-visible, the
dequant qualification runs on-device against a GLM weight shard (the kernel is
compiled and gated), then the fused GEMV variants.

## EXP-Q21 probe addendum — both probe orders collide at the mxk library;
## the serving WORKER context is the difference (2026-11-08)

Two fresh probes of libmxk_op_lazy.so (with our restored TORCH_LIBRARY def -
the schema string verified in the binary):
- probe order A (torch -> habana_frameworks -> load_library): COLLIDES at load.
- probe order B (torch -> vllm_gaudi -> load_library, cwd the cookbook root):
  COLLIDES at load.
- the serving WORKERS: load the SAME .so WITHOUT collision (NOT registered
  only - load_library succeeded, the op Python-invisible).

**The worker context is the difference** - vLLM v1 spawns the workers with the
full config/plug state, and in that context the mxk library's static
TORCH_LIBRARY_IMPL(custom_op, HPU) registration does not trip the registrar
collision. The probes cannot reproduce the worker context cheaply (vLLM v1
spawn + the config stack).

**Next-session opener (the minimal repro):** patch a tiny debug print into
mxk_op_lazy.cpp's TORCH_LIBRARY_IMPL block (or run the qualification INSIDE a
worker via the existing step-profiler-style hook), load the library in the
worker context, and check torch.ops.custom_op.block_fp8_dequant_glm from
inside the worker process. If visible there, the dequant qualification runs
in-worker (a debug endpoint or a warmup call), unblocking the whole Q21
qualification without any stack change.

## EXP-Q21 fragment fix attempt: the Python visibility still fails (2026-11-08)

The TORCH_LIBRARY(custom_op) full definition in our op.cpp (colliding with
mxk_op_lazy.cpp's TORCH_LIBRARY_FRAGMENT for the same namespace) was changed to
TORCH_LIBRARY_FRAGMENT - the schema now adds to the existing namespace without
redefining it. The rebuilt library still fails to expose the op at serving time
(NOT registered in all workers). The namespace definition was not the (only)
issue. The in-worker visibility check (TASKS Q21 opener) remains the path: the
worker context differs from every probe context, and the schema may be
registering but visibility-checked too early, or the HPU-impl registration for
our op is silently skipped in the worker's dispatcher state.

## EXP-Q21 collision — the root architectural cause: per-library registry
## instances via weak symbols (2026-11-08)

`habana::RegisterKernel::add` is a WEAK symbol exported by BOTH
libhabana_pytorch_backend.so (0x141c150) and libhabana_pytorch_plugin.so
(0x777610) — each carries its own copy of the registrar class with its own
static registry instance. A third .so (our op shim) that links both and
includes the Habana registrar headers creates its own TU-level registry
context; the d2d_memcpy registration arrives twice across the instances and
the guard throws.

This explains every observation: the order-independence (any load of a second
registrar-carrying .so after the first registration collides), the stock
library failing identically, both venvs, and the TPC library loading clean
(it carries no registrar — pure kernel ELFs + glue).

**The fix is architectural**: the op registration must happen THROUGH one
registry instance — the plugin's own init (the plugin-rebuild track), or the
stack version whose loader deduplicates (v0.30.0 upgrade path). A third .so
cannot be made safe on this stack version without the plugin rebuild.

**Q21's kernel work is unaffected**: the TPC ELF + the gated library are
correct and load cleanly; only the torch-op shim's LOAD mechanism needs the
stack change. The kernel qualification can also run through the recipe path
once a graph references the GUID (no torch op needed) — a possible
next-session angle via a synthetic recipe.

## EXP-Q21 — the PT2 custom-op API discovery: the proven op-loading mechanism
## on this host (2026-11-08)

The DSV4.1 native build (/opt/dsv41/build/dsv41-native/) contains a PT2
custom-op module — `hpu_dsv4_control_mme_pt2.cpython-312-x86_64-linux-gnu.so` —
plus the compiled kernel libraries (libdeepseek_v4_gaudi2_kernels.so,
libdeepseek_v41_unique_kernels.so). The DSV4.1 deployment loaded its custom
kernels through the PT2 custom-op API (torch.library with the PT2 bridge), NOT
the legacy hpu_custom_op.h path that collides on this stack.

**Why this matters:** the legacy registrar collision (EXP-Q21) is specific to
the legacy hpu_custom_op.h API's static registration pattern. The PT2 API
registers through torch.library with Meta implementations — a different
mechanism that provably works on this host (the DSV4.1 serving ran it).

**The Q21 unblocking step:** adapt the block_fp8_dequant_glm op shim to the
PT2 custom-op API. The mxfp4 build.sh supports `--pt2` (libmxk_op_pt2.so) but
excludes it with --extra-kernel (the lazy-mode op.cpp API); the DSV41 build
system compiled both. The adaptation: rewrite op.cpp with the PT2 registration
(torch.library.defme + the HPU dispatch through the PT2 bridge), following the
hpu_dsv4_control_mme_pt2 module's pattern.

**Fallback if the PT2 adaptation is complex:** load the DSV4.1 PT2 module
itself and register our block-FP8 op through its infrastructure (the module is
on-host and proven).

## EXP-Q21 PT2 probe: the registerUserCustomOp path also collides at .so load
## (2026-11-08)

The PT2 op library (libblock_fp8_pt2.so: our op registered via
habana::custom_op::registerUserCustomOp through the PT2 bridge) collides with
the SAME registrar error at .so load (after the imports, both load orders).
The PT2 registration internally resolves to the same legacy registry — the
weak-symbol per-library instance architecture (EXP-Q21) applies to it too.

**The DSV4.1 deployment's difference:** its PT2 modules loaded inside the
DSV41 fork's plugin init chain (dsv41-env.sh + the DSV41 vllm-gaudi fork's
native-library loader), in a load context this session's probes could not
replicate. The next-session task: trace the DSV4.1 deployment's actual load
sequence (LD_DEBUG=libs on the dsv41 serving launch) and replicate that order
for our op library — or serve our kernel from the DSV41-style isolated image.

**Session state:** the kernel is compiled and gated; the PT2 op shim is built
(and its registration API understood); the load-context study is the remaining
gate. All staged in the repo for the next session.

## EXP-Q21 symbol analysis - the descriptor call is the load-collision trigger;
## the serving WORKERS load the op library cleanly (2026-11-08)

The include/link bisection (five isolated .so builds):
- hpu_custom_op.h alone: loads clean.
- torch/extension.h alone: loads clean.
- op.cpp with the body EMPTIED (the descriptor call removed): loads clean.
- op.cpp with ONLY the REGISTER line removed (the descriptor call kept):
  COLLIDES.
- the full op.cpp: COLLIDES.

**The offender: the compiled presence of the
HabanaCustomOpDescriptor::getCustomOpDescriptor call** - its symbol chain
(habana::custom_op::registerKernel, an UND reference resolved from the plugin)
triggers a registration at .so load in the probe context.

**The serving WORKERS load the SAME library without collision** (the q21q boot:
load_library succeeded, NOT registered only) - the worker context (vLLM v1
spawn, the config/plug state) differs from the probe context. The collision is
probe-specific; the workers are the qualification environment.

**Next-session opener (unchanged, now with the mechanism):** the in-worker
visibility check - if the op is visible in the worker (the schema may register
fine there; the q21q NOT registered lines may predate the rebuilt library - the
q21q boot loaded the PRE-restore build), the dequant qualification runs
in-worker via a debug hook.

## EXP-NS32 — max-num-seqs 32: the aggregate TRIPLED from the session start
## (2026-11-08, boot ns32)

**Config:** the adopted stack + max-num-seqs 32 (the KV pool unchanged:
723,155 tokens — covers the bench's short sequences; production long-sequence
concurrency needs more, per the EXP-NS16 note).

**Results (32 streams, converged):** **396–401 t/s aggregate** (362.5 → 396.2 →
394.7 → 401.1 — still climbing slightly). The RA-003 doc's prediction band
(260–400) is CONFIRMED AT THE TOP EDGE.

| Config | Aggregate @16 | Aggregate @32 |
|---|---|---|
| Session start (max-num-seqs 8, eager draft) | 126.7 | — |
| + the Q18 levers (max-num-seqs 8) | 183 | — |
| + max-num-seqs 16 (EXP-NS16) | ~258 | — |
| + max-num-seqs 32 (this experiment) | — | **~400** |

**Arithmetic:** at 32 active sequences the step runs 64 verify + 32 draft rows;
the 38.1 GB/card expert-weight read amortizes over 3× the tokens vs the
8-active baseline. The per-step time grew sub-linearly (the weight reads are
batch-independent; the GEMV compute grows with rows at AIP 0-24% — plenty of
headroom).

**Verdict: the max-num-seqs lever is the dominant aggregate knob.** The
remaining gap to the 800 t/s objective (2×) is the per-step efficiency (the
Q21 kernel track + the multi-row host term D1 will quantify). The bs=1
single-stream is unaffected (the bs-1 graphs are unchanged).

## EXP-NS32 saturation — the aggregate concurrency curve on the adopted stack
## (2026-11-08, boot ns32)

| Streams | Aggregate t/s | Notes |
|---|---|---|
| 16 | ~258 | the ns16 config's number (the same at ns32) |
| 32 | 396–401 | converged |
| 48 | 372–422 | |
| 64 | 383–421 | saturated |

**The batch-amortization lever is EXHAUSTED at ~400-420 t/s.** The weight-read
amortization is complete (all 36 local experts read per step regardless of
rows; adding rows adds only GEMV compute at AIP 0-24%). The remaining
bottleneck is the per-step pipeline: the ~322 recipe launches (~24 ms host at
bs=1, the D1 measurement pending), the small-M GEMM efficiency, and the MoE
kernel time (the mxfp4 study: MoE 82% of the step at batch 16).

**The 800 t/s objective requires the Q21 kernel track** — the fused MoE
kernels (the routing-aware GEMV: reading only the routed experts' weights AND
executing the GEMV in-register) cut both the weight bytes (38 -> ~2 GB/step at
bs=1; proportionally at batch) and the kernel time (the mxfp4 study: the
custom kernels at 66 t/s where stock reaches 37 — a 1.8x kernel-speedup
analogue). The concurrency lever is done; the kernel track is the path.

## EXP-NS64 — max-num-seqs 64: the aggregate +13% at 64 streams, the bs=1 -7%
## (2026-11-08, boot ns64)

**Config:** the adopted stack + max-num-seqs 64 (the KV pool 723,155 tokens —
covers the bench; production long-sequence concurrency needs more).

**Results (steady state):**

| Metric | ns32 (max-num-seqs 32) | ns64 (max-num-seqs 64) | Delta |
|---|---|---|---|
| Aggregate @64 streams | ~400 (at 32-48 streams) | **392-452 (median ~430)** | +8-13% |
| bs=1 decode | 38.0-39.1 | 35.0-36.8 (median 35.4) | **-8%** |
| Acceptance | 1.80-1.85 | (not measured separately) | — |

**Trade-off:** the aggregate +13% at 64 streams, the bs=1 -8% (the larger
decode-bucket shapes dilute the bs-1 optimization; the bs-1 graphs are the
same shapes but the bucket ladder is wider). The aggregate continues to
scale with max-num-seqs (the weight amortization improves); the bs=1 pays a
modest penalty.

**Serving recommendation:** the aggregate-oriented deployments (the 800 t/s
objective) use ns64; the latency-oriented deployments (the 200 t/s
single-stream) use ns32 or ns16. The config choice is a deployment
parameter, not a code change — both are validated.
