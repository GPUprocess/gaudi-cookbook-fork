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
