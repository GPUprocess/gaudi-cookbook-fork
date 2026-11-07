# TASKS.md — Master optimization backlog

Statuses follow `docs/SESSION_PROTOCOL.md`. Evidence pointers refer to `EXPERIMENTS.md`,
`experiments/`, and the published write-up `docs/ports/glm-5-3-flash-mtp.md`.

Status vocabulary: `NOT INVESTIGATED`, `RESEARCHING`, `READY`, `IN PROGRESS`, `WORKING`,
`VERIFIED`, `FAILED`, `BLOCKED`, `DEFERRED`, `SUPERSEDED`.

Legend for evidence pointers: **EXP-xxx** = experiment index entry; **docs** = published port
write-up; **repo** = files under `ports/glm-5-3-flash/` in this repository.

---

## A. MTP — multi-token prediction and speculative decoding

- **A001 Reproduce the existing MTP=1 improvement** — `WORKING`.
  Implementation is complete and deployed; correctness/acceptance gates passed. The reported
  ~21% batch-1 decode uplift is PROVISIONAL: the original control used the production window
  (256K context / 32 seqs) while the MTP window used 32K / 8 seqs (non-equivalent conditions).
  A like-for-like 32K/8seq single-run comparison measured +14% (35.7 vs 31.4 tok/s).
  **Next action:** Stage-1 controlled reproduction per `docs/BENCHMARK_POLICY.md`
  (≥5 repetitions per condition, median/mean/P95, alternating order). Evidence: EXP-MTP-001,
  EXP-MTP-003.
- **A002 Inspect checkpoint MTP weights and tensor names** — `VERIFIED`. Layer-45 MTP block ships
  inside the target checkpoint; names normalized on load (`model.language_model.layers.45.*` →
  `model.mtp.*`, `shared_head.norm.weight` → `shared_head_norm.weight`). Evidence: EXP-MTP-001,
  repo `mtp.py`.
- **A003 Document MTP architecture and forward dependencies** — `VERIFIED`. Draft = target's
  layer-45 re-entered with the last hidden state; position-0 embed zeroing; `eh_proj` concat;
  shared head. Evidence: docs ("The design in one paragraph"), EXP-MTP-001.
- **A004 Verify MTP weight loading and mapping** — `VERIFIED`. Load-time normalization plus a
  completeness gate that refuses startup when spec decode is on and the MTP block did not fully
  load. Evidence: EXP-MTP-001, repo `mtp.py`.
- **A005 Verify hidden-state and token-input handling** — `VERIFIED`. CPU contract test covers
  reshape, position-0 zeroing, `eh_proj` concat order, residual flow, single head application.
  Evidence: repo `tests/cpu/check_mtp_wrapper.py`.
- **A006 Measure draft generation latency** — `WORKING`. Draft runs eagerly by design; its cost
  is bounded (1 MLA + 1 MoE block) and shows up as the k=1 net gain gap. Dedicated per-phase
  draft-latency measurement pending. Evidence: EXP-MTP-001, docs.
- **A007 Measure target verification latency** — `WORKING`. Verify rows = seqs × (k+1); decode
  step time measured in aggregate. Dedicated verification-phase breakdown pending.
- **A008 Measure draft acceptance rate** — `VERIFIED`. Position-0 acceptance ≈ 73–98% on natural
  prompts; mean accepted length 1.73–1.98 of 2.0 (k=1), 2.04–2.52 of 3.0 (k=2). Evidence:
  EXP-MTP-001, EXP-MTP-002.
- **A009 Measure accepted tokens per verification step** — `VERIFIED`. Same figures as A008;
  live deployment 1.84–1.95. Evidence: EXP-MTP-001.
- **A010 Investigate MTP=2 feasibility** — `VERIFIED`. The checkpoint has a single MTP head;
  depth 2 works by chaining the same head (per-step metadata, S=4 candidate columns, 3-row
  verify). Correctness held across the full sweep. Evidence: EXP-MTP-002.
- **A011 Implement MTP=2 if supported** — `WORKING`. Functionally complete and correct; net
  decode throughput is neutral-to-worse in the eager-draft configuration (30.2–34.1 vs 36.8
  tok/s bs=1), so k=1 remains the default. Retained as a supported capability. Evidence:
  EXP-MTP-002.
- **A012 Investigate additional prediction depth** — `NOT INVESTIGATED`. k≥3 untested; blocked
  on the draft-execution lever (A014) because eager per-step cost grows with depth.
- **A013 Determine optimal speculative depth** — `WORKING`. k=1 is optimal in the eager-draft
  configuration (k=2 net-neutral; concurrent net-negative). Revisit after A014.
- **A014 Optimize draft execution** — `READY`. A working reference implementation exists:
  abotsis fork 919ec98f ships private-storage draft cores plus attention-in-graph
  (`VLLM_GLM_MTP_DRAFT_GRAPH/_ATTN_GRAPH`), measured at MTP-4: accepted 3.33 vs 2.41
  bypass, aggregate 26.0 vs 20.2 tok/s, parity 12/12. The eager draft is host-launch-bound
  (93 ms host vs 17 ms device per step). Experiment = adapt that machinery into this
  plugin, keeping our KDA candidate-slot work. Evidence: RESEARCH-001.
- **A015 Optimize target verification** — `NOT INVESTIGATED`. Vocabulary-parallel argmax in
  `get_top_tokens` is the documented draft-side lever; verification-side kernel work pending
  profiling.
- **A016 Investigate hidden-state reuse** — `NOT INVESTIGATED`.
- **A017 Investigate synchronization overhead** — `RESEARCHING`. Prefill is CPU-bound: the
  Habana AccThread busy-waits (~100% of one core) during lazy-mode dispatch; identified during
  validation. Evidence: EXP-MTP-001 (TTFT analysis).
- **A018 Investigate HPU graph compatibility for the draft** — `READY`. The blocker is
  solved upstream: private storage copies (`_private_copy`, ~0.9 GiB/rank) make target and
  draft graphs independent by construction, and the draft attention runs INSIDE the graph
  with metadata tensors as graph inputs. Reference: abotsis fork 919ec98f. Evidence:
  RESEARCH-001.
- **A019 Investigate graph capture/compilation overhead** — `PARTIAL`. Verify-row graphs
  captured at window config; every new row count adds graphs (documented constraint). Draft
  graph capture unattempted (A014).
- **A020 Validate token-position correctness** — `VERIFIED`. Prompts sized to straddle KV-block
  boundaries (639/640/641, 1279/1280/1281 tokens) produce correct output under MTP. Evidence:
  EXP-MTP-001, repo `tests/hpu/mtp/check_block_boundary.py`.
- **A021 Validate KV-cache updates and rollback** — `VERIFIED`. Per-verify-row KV block lists
  (`spec_verify_blocks`) plus boundary tests. Evidence: EXP-MTP-001, repo patches.
- **A022 Validate recurrent/KDA state updates and rollback** — `VERIFIED`. S+1 candidate-slot
  state columns; post-verify state load matches the accepted prefix; churn and chaining parity
  tests. Evidence: EXP-MTP-001, repo `tests/cpu/check_churn_slots.py`, `check_kda_spec.py`.
- **A023 Validate tensor-parallel correctness** — `VERIFIED`. TP8+EP serving passed the port
  gate suite (tools, images, streaming, structured output, logprobs); golden-parity divergences
  attributed to verify-shape FP near-ties (measured top-2 gaps), not parallelism defects.
  Evidence: EXP-MTP-001, docs.
- **A024 Investigate context-length effects** — `WORKING`. MTP validated in a 32K window
  (needle retrieval at 16K, 4K-prompt decode). The production 262K shape has not been re-run
  under MTP. Evidence: EXP-MTP-001, EXP-MTP-003.
- **A025 Investigate batch-size and concurrency effects** — `VERIFIED` for tested shapes. bs=1,
  bs=8, 16-stream concurrent at k=0/1/2: MTP net-positive single-stream, net-negative by
  16 streams (eager draft). Crossover documented. Evidence: EXP-MTP-003.
- **A026 Investigate adaptive speculation depth** — `NOT INVESTIGATED`. Depends on A013/A014.
- **A027 Measure MTP memory overhead** — `VERIFIED`. Zero marginal weight memory (bind by
  reference); one extra MLA latent cache (~0.8 GiB/rank) + KDA candidate columns (~0.3 GiB/rank
  at k=1); 64.1 GiB/card under load in the MTP window. Evidence: EXP-MTP-001, docs.
- **A028 Measure HPU utilization and power overhead** — `PARTIAL`. Pre-MTP utilization/power
  profiled (AIP 0–24% decode, 81–152 W); MTP-specific utilization delta not yet isolated.

## B. TG — token generation / decode performance

- **B001 Establish baseline decode throughput** — `VERIFIED`. Non-MTP production: 30.1/29.8
  tok/s bs=1 (two runs), 30.3 on post-rollback re-verification; 125.1 agg @8 streams; 213.2 agg
  @16 streams; re-check 217.2 agg. Evidence: EXP-OPT-001, EXP-OPT-004.
- **B002 Profile per-token latency** — `PARTIAL`. Decode is latency/sync-bound at bs=1 (AIP
  0–24%); per-op breakdown pending (Phase 10 profiling task).
- **B003 Compute- vs bandwidth-bound classification** — `PARTIAL`. bs=1 decode is not
  bandwidth-bound (low AIP, sub-linear scaling); formal roofline analysis pending.
- **B004–B008 Expert execution, MoE dispatch, placement, HBM access, weight movement** —
  `NOT INVESTIGATED`. Candidates for Stage 5 after MTP/batching stages.
- **B009 Investigate CPU synchronization** — `RESEARCHING`. Same finding as A017 (AccThread
  busy-wait during lazy dispatch; prefill CPU-bound). Scheduler-tuning headroom estimated ~25%
  in the earlier profiling phase.
- **B010–B012 Host-to-device transfers, kernel launch, graph replay** — `NOT INVESTIGATED`.
- **B013 Attention/KDA decode kernels** — `WORKING`. KDA scan/chunk knobs explored: chunk=64
  optimal, 128 worse TTFT (documented); custom kernels not attempted.
- **B014 Recurrent-state update efficiency** — `WORKING`. Candidate-slot machinery verified;
  efficiency optimization not attempted.
- **B015/B016 TP/EP communication** — `NOT INVESTIGATED`.
- **B017 Scheduling overhead** — `RESEARCHING`. ~25% headroom estimate from the earlier
  profiling phase; untested.
- **B018 Context-length effects on decode** — `PARTIAL`. Production decode flat 2K–517K
  non-MTP; MTP window 32K.
- **B019 Batch-size effects** — covered by A025.
- **B020 Adaptive scheduling** — `NOT INVESTIGATED`.
- **NEW B021 Pin decode bucket ladder** — `READY`. `VLLM_DECODE_BS_BUCKET_MIN/STEP` exist
  in our tree; abotsis measured warmup 13 → ~2 min and 12/12 serial-vs-concurrent parity
  with decode pinned to one bucket, decode within noise. Free determinism + boot-time win;
  needs a server restart (autonomous). Evidence: RESEARCH-001 addendum 2.
  Measured production boot (the MTP window, from serve_mtp.log): 31.5 min
  start-to-health, 125 decode graphs captured (the bucket config: decode bs
  [1,2,8] blocks [1,8,2048]; the MTP verify shapes roughly double the graph
  count), the graph-capture phase 13.3 min of the total. The prompt buckets:
  bs [1] (one bucket), query 13, ctx 9 = 117 prompt graphs.
- **NEW B024 Prompt-bucket taming for the production warmup** — `READY` (config-only,
  no code change). Our production warmup (~13-15 min) is dominated by the exp prompt
  bucket set: 4 bs x 13 query x 9 ctx = 468 prompt graphs. The reference recipes
  prompt-bucket taming (VLLM_PROMPT_QUERY_BUCKET_STEP=2048 VLLM_PROMPT_CTX_BUCKET_STEP=64)
  cuts it to ~128 graphs (~4 min) at the cost of coarser prompt granularity (a prompt
  just over an edge pays the next bucket). Applicable to the production launch and the
  Stage-1 boots alike; needs a restart (approval). Measured production boot (the MTP
  window, from serve_mtp.log): 31.5 min start-to-health, 125 decode graphs captured, the
  prompt buckets bs [1] query 13 ctx 9 = 117 prompt graphs, the graph-capture phase 13.3
  min of the total. Evidence: the bucket-set math in the E015 entry + serve_mtp.log.
- **NEW B023 Backport skip_layer_markstep for glm5_next** — `WORKING`. The patch is
  landed in the fail-closed contract (`patches/skip_layer_markstep.py`): gated behind
  `VLLM_GLM_SKIP_LAYER_MARKSTEP=1` (default OFF — experimental until HPU-verified in
  this tree; an explicit `VLLM_CONFIG_HIDDEN_LAYERS` always wins). Contract tests
  green (apply/idempotent/check/drift). Upstream evidence: bit-identical (greedy
  parity 8/8) and +10% single-stream decode. HPU measurement pending a window;
  deploy = apply the patch during the window boot. Evidence: RESEARCH-001 addendum 3.
  HPU MEASUREMENT (MTP k=1): the +10% claim is NOT reproduced - B023 median 36.85 vs baseline 36.91 wall tok/s (no measurable change). Our k=1 config may have different graph patterns than their MTP-4.
- **NEW B022 Port position-independent all-reduce (gather_sum)** — `READY` (upstream fix
  exists in abotsis fork; absent in our b059f9c tree). Fixes the co-batched row-order ULP
  that flips near-tie routing under concurrency. Evidence: RESEARCH-001.

## C. PP — prefill performance

- **C001 Establish baseline prefill throughput** — `VERIFIED`. 4,928 tok/s @128K prompt
  (104,867 tokens); 1,807 @32K; warm small-prompt ~770+ tok/s. Evidence: EXP-OPT-001.
- **C002 Profile prompt processing** — `PARTIAL`. Chunked prefill 8192 pipelines best at large
  chunks; 32K request paid a partial graph fill.
- **C003 Chunked prefill** — `VERIFIED` (enabled, 8192).
- **C004 Optimal prefill chunk size** — `WORKING`. KDA prefill chunk=64 optimal (128 measured
  worse: 8K TTFT 1.85 s vs 1.29 s). Evidence: EXP-MTP-004.
- **C005 Attention/KDA prefill kernels** — `READY` (upgraded). The diagonal block of
  `_gauged_scores` materializes `[sub, sub, d]` elementwise on TPCs; abotsis measured the
  anchored-MME-matmul fix at 15.6x kernel-time reduction and TTFT up to 3.23x (their commit
  c860b8ea; oracle 12/12 CPU, test_hpu_kda 14/14). Applies directly to our `_gauged_scores`
  (off-diagonal blocks already matmul). Evidence: RESEARCH-001 addendum 4. See C015.
- **NEW C015 KDA diagonal decay-dot -> anchored MME matmul** — `WORKING`. Landed in
  `_gauged_scores` (kda.py): the diagonal block now factors around its block-first-row gate
  and contracts with a matmul (row factor ≤ 1; column factor exceeds 1 only in the masked
  upper triangle, inside fp32 at ≤ e^75 with the −5/token gate bound). Oracle vs the
  pre-C015 elementwise form: 12/12 shapes, max rel err 5.7e-7 (CPU). HPU measurement
  (prefill ladder + TTFT + chunk re-check) pending a window. Evidence: RESEARCH-001
  addendum 4; repo `tests/cpu/check_c015_gauged_scores.py`.
  HPU measurement (MTP k=1, warm): no clear improvement at 2K-16K depths
  (2K: -4%, 8K: -3%, 16K: +9% - within noise). The decay-dot is a minor
  fraction of the total TTFT at these depths; the upstream 3.23x was at
  shorter prompts (507-3099 tokens) where it dominates the prefill compute.
- **C006–C012** — `NOT INVESTIGATED` (MoE prefill dispatch, prefill graph capture, allocation
  overhead, tokenization overhead, prompt batching, concurrent prefill scheduling,
  prefill/decode interference).
- **C013 Long-context prefill** — `VERIFIED` non-MTP (4,928 @128K; 1M feasibility documented in
  the earlier phase as software-blocked on prefill latency).
- **C014 Prefill/decode disaggregation** — `NOT INVESTIGATED`.

## D. TTFT — time to first token

- **D001 Establish TTFT baseline** — `VERIFIED`. Non-MTP: 128K TTFT 21.3 s (25.4 s re-check);
  warm small-prompt ~2.6 s @2K.
- **D002 Separate tokenization/queueing/prefill/first-decode** — `PARTIAL`. Phase boundaries
  identified (CPU-side metadata prep and lazy dispatch dominate warm small-prompt TTFT);
  fine-grained breakdown pending.
- **D003 Cold-start overhead** — `VERIFIED`. `VLLM_SKIP_WARMUP` defers 13–15 s graph compile per
  new shape to first use; production should pre-warm served shapes.
- **D004 Graph compilation/capture overhead** — `VERIFIED` (same finding; ~25.5 GiB ladder).
- **D005 Warm-request latency** — `VERIFIED`. MTP window: 2.1 s @2K, 3.1 s @8K, 4.2 s @16K
  (64 gen tokens, warm). Evidence: EXP-MTP-004.
- **D006–D011** — `NOT INVESTIGATED` (queue delays, prompt caching, prefix caching — prefix
  caching is architecturally OFF for this model: KDA stores no per-block states —, short-prompt
  optimization, long-prompt TTFT under MTP, concurrent-request TTFT).
- **D012 MTP effect on first-token latency** — `VERIFIED`. Draft `prefill_cache_only` pass adds
  prompt-proportional overhead; totals remain practical (D005 figures). Evidence: EXP-MTP-004.

## E. Throughput and concurrency

- **E001 Batch-size-1 throughput** — `VERIFIED`. 30.1–30.3 non-MTP; 35.7–36.8 MTP k=1.
- **E002 Batch sizes 2/4/8/16** — `PARTIAL`. 8 and 16 streams measured (production and MTP
  windows); 2 and 4 not run.
- **E003 Aggregate vs per-request** — `VERIFIED`. Documented separately in every table;
  aggregate ≠ per-request speed called out in docs.
- **E004 Per-request latency under concurrency** — `VERIFIED` for 8/16 streams (per-stream
  means recorded).
- **E005 Median and P95 inter-token latency** — `PARTIAL`. Per-stream min/max recorded;
  median/P95 ITL distribution pending.
- **E006–E012** — `NOT INVESTIGATED` (continuous batching is stock vLLM behavior; scheduling
  fairness, admission control, memory-aware batching, mixed prompt/generation lengths not
  studied).
- **E013/E014 Optimal concurrency** — `WORKING`. Latency-oriented: 1 stream (MTP net-positive);
  throughput-oriented: MTP off at 16 streams (eager draft crossover measured). Evidence:
  EXP-MTP-003.

## F. Memory, context and caching

- **F001 Per-HPU HBM baseline** — `VERIFIED`. 65.5 GiB steady / 68.2 peak (production); 64.1
  GiB (MTP window).
- **F002 Model weight allocation** — `VERIFIED`. 38.2 GiB/card at TP8+EP; 76.5 GiB/card at TP4.
- **F003 KV-cache allocation** — `VERIFIED`. ~1.8 GiB/card at the 1280-block override
  (163,840-token pool).
- **F004 Recurrent/KDA state allocation** — `VERIFIED`. Included in the MTP overhead figure
  (S+1 columns); exact per-component split pending.
- **F005 MTP memory overhead** — `VERIFIED` (A027).
- **F006–F009** — `NOT INVESTIGATED` (fragmentation, alloc/release overhead, cache reuse;
  prefix caching is architecturally disabled — KDA stores no per-block states).
- **F010 Context scaling 4K–128K** — `VERIFIED` non-MTP (prefill ladder + 256K production).
- **F011 Validate 256K context** — `VERIFIED` non-MTP (production serves 262K). MTP at 262K:
  not yet run (A024).
- **F012/F013 512K/1M context** — `PARTIAL`. 512K served non-MTP by the base port (needles at
  512K); 1M capacity feasible per the earlier feasibility study, prefill latency
  software-blocked.
- **F014 Memory growth vs context** — `PARTIAL` (KV pool sized by override; growth curve not
  swept).
- **F015–F018** — `NOT INVESTIGATED` (eviction/reclamation, long-context stability under MTP,
  state correctness after cache reuse — partially covered by A022 churn tests, BF16 state
  handling is in use and gated by correctness tests).
- **F019 HBM capacity vs movement limiting factor** — `PARTIAL`. bs=1 decode is not
  bandwidth-bound (low AIP); graphs/workspace (~25.5 GiB) is the first reclaimable block.

## G. HPU scaling and communication

- **G001 1-HPU feasibility** — `BLOCKED`. 328 GiB of weights cannot fit one 96 GiB card;
  documented hardware limit.
- **G002 2-HPU feasibility** — `BLOCKED` (same memory limit; TP2 would need ~164 GiB/card).
- **G003 4-HPU feasibility** — `WORKING`. TP4+EP measured viable: 76.5 GiB/card weights, GMU
  0.95, graph capture 128 (2048-size ladder OOMs at TP4). Full 4-HPU deployment config
  (Phase 4 of `docs/HARDWARE_ROADMAP.md`) not yet built or benchmarked. Evidence: EXP-OPT-003.
- **G004 6-HPU feasibility** — `BLOCKED`. 64 attention heads % 6 = 0; layout invalid
  (documented in the earlier phase).
- **G005 8-HPU feasibility** — `VERIFIED` (production TP8+EP).
- **G006 Utilization imbalance** — `PARTIAL` (aggregate AIP recorded; per-HPU breakdown
  pending).
- **G007–G016** — `NOT INVESTIGATED` (per-HPU HBM profiling, inter-HPU communication profiling,
  TP/EP/hybrid exploration beyond production settings, sync barriers, topology-aware placement,
  overlap; G015 "when adding HPUs helps" partially answered by E-series scaling data).
- **G017 Final 4-HPU deployment configuration** — `NOT INVESTIGATED`. Phase 4 of the hardware
  roadmap; depends on Stages 1–3.

## H. Kernels and compiler

- **H001 Identify dominant kernels via profiling** — `PARTIAL`. Earlier phase profiled at the
  phase level (host-sync-bound decode); per-kernel ranking pending.
- **H002 Custom TPC kernels** — `NOT INVESTIGATED` for this model (the mHC Sinkhorn kernel from
  the base port is in use and parity-gated).
- **H003–H012** — `NOT INVESTIGATED` (fusion, tensor conversions, layouts, data movement,
  precision/accumulation, graph compatibility beyond A018/A019, compilation overhead, launch
  overhead, compiler flags, Gaudi-specific optimizations). Rule honored: no custom kernel work
  without a measured bottleneck.

## I. Numerical accuracy and model correctness

- **I001 Baseline evaluation results** — `IN PROGRESS`. Correctness gates are established;
  the formal accuracy suite (0.5 absolute-percentage-point gate per
  `docs/BENCHMARK_POLICY.md`) is not yet built. See `ACCURACY.md`.
- **I002 Greedy decoding correctness** — `VERIFIED`. Golden-reference comparison: 2/12
  token-identical under MTP with all divergences measured as verify-shape FP near-ties;
  content-equivalent after reasoning stripping; divergence positions identical at k=1 and k=2.
  Evidence: EXP-MTP-001, EXP-MTP-002, docs.
- **I003 Speculative decoding correctness** — `VERIFIED` (I002 + feature gates + boundary
  tests + long generation zero-degeneration).
- **I004 Sampling behavior** — `VERIFIED`. top-p/top-k and temperature paths verified under
  MTP; distribution-preservation study (sampled) pending.
- **I005 Logits/probability consistency** — `VERIFIED`. top-5 logprobs verified under MTP; the
  near-tie analysis used logprob gaps directly.
- **I006 Precision-sensitive operations** — `PARTIAL`. FP near-tie class characterized at the
  decision level; op-level study pending.
- **I007 FP8 weight handling** — `VERIFIED` (block-FP8 checkpoint as published; base port
  gates).
- **I008 BF16 state handling** — `VERIFIED` (activations BF16; correctness gates).
- **I009 Long-context numerical stability** — `WORKING`. 16K needle retrieval under MTP;
  256K non-MTP production-stable. MTP at 262K pending (A024).
- **I010 Consistency across HPU configurations** — `PARTIAL`. TP8 verified; TP4 ran
  functionally in the earlier phase (pre-MTP); cross-config numerical comparison pending.
- **I011 Consistency after graph capture** — `VERIFIED` (gates run post-capture in the MTP
  window).
- **I012 Consistency after cache reuse** — `VERIFIED` (churn/slot-reuse tests).
- **I013 No silent truncation** — `VERIFIED` (needle retrieval at depth requires full context;
  block-boundary tests).
- **I014 No recurrent-state corruption** — `VERIFIED` (A022).

## J. Stability, reliability and deployment

- **J001 Long-running inference baseline** — `PARTIAL`. Production service has run for days;
  formal soak duration not standardized.
- **J002 Repeated requests** — `VERIFIED` (soak-repeat: 5 identical greedy requests, no drift).
- **J003 Concurrent requests** — `VERIFIED` (16-stream + churn tests).
- **J004 Memory reclamation** — `WORKING` (HBM stable across sweeps; formal test pending).
- **J005/J006 Recovery from cancellation/failure** — `VERIFIED` (client-disconnect mid-
  generation handled; engine survived all failure injections).
- **J007 Memory leaks** — `NOT INVESTIGATED`.
- **J008 HPU hangs/device errors** — `NOT INVESTIGATED` (none observed).
- **J009 Process cleanup** — `WORKING` (repeated k-switch restarts exercised cleanup).
- **J010 Model reload** — `NOT INVESTIGATED` (restart-from-scratch only).
- **J011 Reproducible deployment** — `VERIFIED` (serve scripts in `serving/`; boot/kill runbooks
  in the deployment doc of the earlier phase).
- **J012 Container/venv isolation** — `VERIFIED` (venv-based deployment; port Dockerfile
  provided by the base port).
- **J013 Rollback procedures** — `VERIFIED` (k-switch scripts; documented config rollback).
- **J014 Hardware/software limitations** — `VERIFIED` (documented throughout: memory limits for
  1/2 HPUs, 6-HPU invalid layout, prefix-caching constraint, eager draft, verify-row graph
  growth).

## K. Research and architecture

- **K001–K005 Upstream tracking** — `RESEARCHING`. GLM-5.3-Flash checkpoint is the reference;
  upstream vLLM/vllm-gaudi releases tracked (v0.26.0 baseline; v0.30.0 branch exists); Gaudi SW
  1.24.1.
- **K006 Compare execution architectures** — `PARTIAL`. Engine comparison from the earlier
  phase (stock vllm-gaudi vs port; the port was required — no stock path for this model).
- **K007 Portable optimizations** — `NOT INVESTIGATED`.
- **K008 Unsupported features/missing kernels** — `PARTIAL` (prefix caching unsupported by KDA;
  video not served; MTP draft graph absent).
- **K009 Reassess blocked tasks** — ongoing; G001/G002/G004 blocked on fixed architecture
  limits (not reassessable); A012/A014 reassessable after draft-graph work.
- **NEW E015 Config-delta A/B sweep from the reference recipe** — `READY` (narrowed to
  two candidates after the runtime recon): `VLLM_BUCKETING_STRATEGY=lin` (verified
  implemented: LinearBucketingStrategy selectable via the features config, default exp) and `--enable-ep-weight-filter` (consumed in
  vLLM core arg_utils/weight_utils/parallel config). `VLLM_COMPACT_GDN=0` needs no A/B:
  our runner auto-pins it when unset, so the reference value already matches our
  effective config. One variable at a time; needs a server restart
  (autonomous: the server is dedicated to development). Evidence: RESEARCH-001 addendum 7 + the runtime recon in this entry.
  Procedure (per candidate, from serve_mtp.sh): A = export VLLM_BUCKETING_STRATEGY=lin
  before boot - MUST pair it with the prompt-bucket taming the reference recipe uses
  (VLLM_PROMPT_QUERY_BUCKET_STEP=2048 VLLM_PROMPT_CTX_BUCKET_STEP=64). Quantified under
  our MTP window (8 seqs, 32K, 8192 batched, block 128): exp warms 468 prompt graphs
  (4 bs x 13 query x 9 ctx) + 60 decode graphs; lin UNTAMED would warm ~131k prompt
  graphs (8 bs x 64 query x 256 ctx) - impossible; lin TAMED warms 128 prompt graphs
  (8 bs x 4 query x 4 ctx) + 16 decode graphs (one decode bs bucket at 8, blocks
  32..2048 by 128) - SMALLER than exp, so A also speeds warmup. B = add
  --enable-ep-weight-filter to the serve args. Measure decode bs=1, 16-stream
  aggregate, TTFT ladder, and the accuracy suite (5 reps) against the Stage-1
  baseline; one candidate per boot. E015-A affects runtime (graph shapes + warmup set);
  E015-B affects load time only (each rank reads its own expert shard from disk; the
  decode path is unchanged). Empirically confirmed under our window:
  lin+tamed prompt cfgs (bs 1-8 step 1, query 128 step 2048, ctx 0 step 64) and
  decode (bs pinned 8, blocks 32..2048 step 128) -> 8x4x4 = 128 prompt graphs +
  1x16 = 16 decode graphs, matching the analytic estimate. Warmup projection (corrected): the 31.5 min boot includes ~13.3 min of graph
  compilation (593 graphs at ~1.35 s/graph avg) plus ~18 min of weight loading,
  model init, and non-graph setup (which dont change). Lin+tamed: 144 graphs at
  the same per-graph rate = ~3.2 min of compilation, total boot ~21 min
  (a ~10 min saving, not 22 min - only the graph compilation scales).
  E015-A MEASURED: decode median 37.44 (vs baseline 36.91, +1.4%),
  streaming 20.7-21.0 (vs 19.6-20.9), warmup 144 graphs (vs 593, 3.9x
  fewer). The lin bucketing gives a small decode improvement and a major
  warmup improvement.
- **NEW H013 Port the fused clamp TPC kernel from the reference fork** - `READY`.
  Our vLLM's fp8 quantization config already carries the clamped-SwiGLU contract
  (swiglu_limit / swiglu_alpha / swiglu_beta); the reference's TPC kernel and loader
  NOTE: the enable_clamped_swiglu method is an abotsis-fork addition;
  our stock vllm-gaudi MoE op lacks it. H013 requires adding the method
  to the MoE op class + porting the TPC kernel binary + the loader.
  module implement it on device. Their measurement: ~nothing on MoE decode at their
  shape; the recipe table claims ~2x on GLM-5.3 decode (contradictory - measure).
  Evidence: RESEARCH-001 addendum 11.
- **NEW K010 Evaluate the abotsis fork as the serving base** — `READY` (strategic).
  Their fork (919ec98f) is a full GLM-5.3-Flash serving stack: the draft graphs, the
  KDA spec-state, the MoE expert cache, the pad bucketing, the device sampling, the
  context buckets, the query tiling, the all-reduce fix, the mark_step skip, and the
  decay-dot fix — all measured on the same model and hardware. Recommended: adopt as
  the Phase-2 base after Stage-1 validates our current implementation (so the
  comparison has a verified baseline), porting our docs/harness/suites onto it.
  Evidence: RESEARCH-001 addenda 7-10.
- **NEW I015 Adopt boundary-level divergence diagnostics** — `READY`. The abotsis fork's
  Status: WORKING - the diagnostics core is landed in this port
  (`layer_diagnostics.py`, CPU-verified 24 checks) env-gated OFF; the runner
  integration and model hooks land with the Stage-1 window.
  layer_diagnostic hooks + diag_compare tooling prove near-tie divergences at the boundary
  level (~1 bf16 ULP at attention output, bounded 2-6e-2 amplification through mHC, no
  logic-error signature). Upgrades our near-tie evidence from logprob gaps to tensors.
  Evidence: RESEARCH-001 addendum 6.
  Porting scope confirmed GREEN: the reference modules (`layer_diagnostic.py` 427 lines
  + `kda_state_diagnostic.py` 199 lines) depend only on `runner.use_merged_prefill` and
  `runner.input_batch` - both exist in our runner - plus their own state set by
  `prepare()`; one env var (`VLLM_DIAG_LAYER_POSITIONS`); no special HPU APIs. DESIGN SIMPLIFICATION: the capture is now plugin-side SELF-ARMING - the decoder
  layer self-arms from `VLLM_GLM_LAYER_DIAG_DIR` + the operator sentinel at its first
  forward (refusing lazy mode, layer 0 only, env-gated OFF) - so NO runner changes
  are needed. Landed: `layer_diagnostics.py` (self-arming + capture + flush) and the
  decoder-layer hooks in `model.py`; CPU tests green (sentinel/admission/history/
  budget/secure-flush/nesting + the self-arming flow). The HPU capture run happens
  in a diagnostic boot (enforce_eager + the env + the sentinel) - shares the Stage-1
  window. The runner-level request mapping (their prepare()) is out of scope for v1.
  LANDED: the diag boot scripts (`serving/serve_diag.sh` non-MTP and
  `serve_diag_mtp.sh` MTP, both enforce_eager with the layer-diagnostic dir) and
  `benchmarks/diag_compare.py` (the delta-map tool: per-boundary relL2/max_abs,
  history validation; tested with synthetic payloads). I015 procedure: boot both
  diag variants -> run the golden prompts -> diag_compare the payloads.
  Evidence: RESEARCH-001 addendum 6 + the I015 porting assessment.
- **A024 update (via RESEARCH-001 addendum 6)**: MTP at the production 262K window is
  likely NET-NEGATIVE without the block-table-sharing lever (their 262K MTP-4 trial:
  decode ~6 tok/s at 219K context, each lane carrying the full block table). Plan the
  262K MTP test after A014, with the block-table-sharing lever on the list.


---

## AUTONOMOUS DISCOVERY QUEUE

Continuous-discovery backlog (directive: an empty conventional task list is NOT completion).
Entries carry: Hypothesis / Bottleneck (PP TG TTFT MTP AGG EP HBM GRAPH KERNEL SCHED KV ACC INFRA) /
Expected gain (ESTIMATE, not measured) / Evidence / Risk / Dependencies / Status / Next action.
Prioritize by: objective relevance, evidence the bottleneck exists, expected gain, feasibility,
effort, risk, independence. Re-scan after EVERY experiment: did the bottleneck move?

- **Q01 Draft HPU graphs (private-storage draft cores + attention-in-graph)** — `WORKING` (split mode implemented, performance VERIFIED +9-11% bs=1 / +33% @16 aggregate, acceptance held — but the ACCURACY GATE FAILED: 97.5% vs 100.0% baseline, R003 deterministic flip, -2.5 pp > 0.5 pp limit. NOT adopted as default (env-gated OFF). Follow-up Q01b owns the correctness investigation).
  Hypothesis: the eager draft is host-launch-bound (93 ms host vs 17 ms device per draft step);
  capturing the draft in an HPU graph with metadata tensors as inputs removes the host overhead
  and makes MTP profitable at every batch size. Bottleneck: MTP/TG/GRAPH.
  Expected gain (ESTIMATE): bs=1 decode +20-50% (37 -> 45-55 t/s); flips the 16-stream MTP
  crossover from net-negative to net-positive (aggregate 213 -> 300+); unlocks k=2/k=4 (Q11).
  Evidence: RESEARCH-001 (abotsis fork 919ec98f: private storage `_private_copy` ~0.9 GiB/rank
  makes target/draft graphs independent; attention runs INSIDE the graph, metadata as inputs;
  measured MTP-4 accepted 3.33 vs 2.41 bypass, aggregate 26.0 vs 20.2, parity 12/12);
  A014/A018; B002 (decode latency/sync-bound, AIP 0-24%).
  Risk: medium-high — graph/memory interaction with KDA candidate slots; capture-time cost per
  shape; correctness gated by existing parity/churn suites. Dependencies: none (top priority).
  Next action: port the draft-core pattern into `glm5_next_hpu` behind
  `VLLM_GLM_MTP_DRAFT_GRAPH`, CPU parity first, then HPU A/B on the live config.
- **Q02 Aggregate decode at bs=16 under the pinned-bucket config (MTP on/off)** — `WORKING` (MTP half measured: 126.7 @16, saturation ~128 from 8 streams; non-MTP comparison boot pending).
  Hypothesis: the 42.7-vs-59.4 16-stream crossover was measured under the old MTP window, the
  213 t/s aggregate under the production window — neither reflects the current pinned-bucket
  config; the true MTP-on aggregate baseline is unknown. Bottleneck: AGG/SCHED.
  Expected gain (ESTIMATE): none directly — establishes the baseline the 800 t/s target is
  measured against and locates the real MTP crossover. Evidence: A025, E013/E014, CURRENT_STATE.
  Risk: none. Dependencies: none. MEASURED (EXP-Q02/Q03): MTP @16 = 126.7 t/s aggregate (saturation ~128 from 8 streams;
  eager draft dominates). Non-MTP comparison boot pending.
- **Q03 MTP batch ladder 1/2/4/8/16** — `WORKING` (eager-draft regime measured: 2s 55.7–56.3, 4s 92.8, 8s 128.0, 16s 126.7 aggregate; per-stream 33/31/23/17.3. Re-run after Q01.)
  Hypothesis: acceptance and per-step cost scale differently with batch; the optimal MTP
  concurrency is unknown between 1 and 16. Bottleneck: MTP/AGG.
  Expected gain (ESTIMATE): identifies the concurrency band where MTP adds throughput.
  Evidence: A025 measured only 8/16 with the eager draft; E002 marks 2/4 as never run.
  Risk: none. Dependencies: parameterized concurrent bench (small addition to
  check_concurrent_16.py). Next action: add `--streams N` and run the ladder on the live server.
- **Q04 Decode-bs bucket pinning for the serving window** — `READY`. Hypothesis: pinning decode
  bs to one bucket (VLLM_DECODE_BS_BUCKET_MIN/STEP) cuts graph count, boot time, and removes
  bucket-transition replays mid-serving with decode within noise. Bottleneck: GRAPH/SCHED.
  Expected gain (ESTIMATE): boot 13.3 min graph phase -> ~2 min; decode neutral.
  Evidence: RESEARCH-001 addendum 2 (abotsis: warmup 13 -> ~2 min, serial-vs-concurrent parity
  12/12); B021 (our tree has the knobs; boot measured 31.5 min total, 125 decode graphs).
  Risk: low (config-only). Dependencies: none. Next action: fold into the next planned boot;
  measure boot time + decode regression check.
- **Q05 Whole-verify-step graph (sampler + draft handoff in-graph)** — `RESEARCH`.
  Hypothesis: between verify-graph replays, sampling/metadata prep runs on host and costs
  10-20 ms/step; capturing sampler+handoff into the graph cuts per-step time. Bottleneck:
  GRAPH/SCHED/TG. Expected gain (ESTIMATE): +10-30% bs=1 decode; must be profiled first.
  Evidence: B002 (latency/sync-bound), D002 (host phases dominate warm small-prompt TTFT).
  Risk: high (vLLM sampler is outside model graphs). Dependencies: Q01 lands first (same
  machinery); per-op profile of one verify step. Next action: after Q01, profile the step
  timeline (torch profiler / HW trace) and quantify host gaps.
- **Q06 AccThread busy-wait / scheduler headroom** — `RESEARCHING`. Hypothesis: prefill is
  CPU-bound (Habana AccThread busy-waits ~100% of one core during lazy dispatch); scheduler
  tuning (poll intervals, CPU affinity, eager-bridge flags) recovers part of an estimated ~25%
  headroom. Bottleneck: SCHED/INFRA/PP. Expected gain (ESTIMATE): REVISED DOWN — the warm prefill is already 3.3-5.7K t/s
  (EXP-Q07 correction); the ~25% headroom estimate was built on generation-inflated
  TTFT figures and is unvalidated. Evidence: A017, B009, B017, D002. Risk: low-medium (runtime
  env flags). Dependencies: host-side profile of a small-prompt prefill. Next action: profile
  one 2K prefill (pidstat + torch profiler), then sweep the runtime knobs.
- **Q07 Small-prompt prefill ladder 512/1K/2K/4K/8K** — `VERIFIED` (warm 3,338/4,802/5,697/2,696/3,782/5,123 t/s at 0.5/1/2/3.5/5/7K tokens — the 3,000-6,000 objective band holds except the 3.5K bucket-edge point at 2,696; MEASUREMENT CORRECTION: the old ~770 t/s and 2.1 s @2K figures included 64-token generation in the wall — warm TTFT at 2K is 0.31 s). Hypothesis: warm small-prompt
  prefill (~770 t/s) is far below the 3000-6000 t/s objective band and is host-bound, not
  kernel-bound; the ladder quantifies where the crossover to kernel-bound happens.
  Bottleneck: PP/TTFT. Expected gain (ESTIMATE): none directly — locates the regime where Q06/Q09
  must act. Evidence: C001, D002, EXP-OPT-001. Risk: none. Dependencies: none.
  Next action: extend check_ttft.py to report prefill t/s at 512-8K on the live server.
- **Q07b 4K-bucket prefill step** — `FAILED` (negative: query bucket step 1024 does NOT
  recover the 3.5K point - 2,475 vs 2,696 t/s at step 2048; the dip is a FIXED ~1.2-1.4 s
  per-prompt cost, flat from 2.6K to 7.1K tokens, independent of bucketing. Config reverted
  to step 2048. Superseded by Q07c.)
- **Q07c fixed per-prompt prefill cost >2K (draft prefill KV fill)** — `READY` (root cause
  found). Hypothesis CONFIRMED by source: the draft prompt-KV fill
  (`_hpu_spec_fill_draft_kv`, hpu_model_runner.py ~7476) calls the draft model with
  `bypass_hpu_graphs` - but the draft model is NEVER graph-wrapped
  (`_shares_target_modules` excludes it from the runner adapter), so the draft prefill pass
  is ALWAYS EAGER. Its ~1.2-1.4 s floor is host-side lazy dispatch of the same ~40 ops
  regardless of token count (matches the flat wall across 2.6K-7.1K prompts). The bypass
  flag is a no-op for our unwrapped draft.
  Bottleneck: PP/TTFT/MTP. Expected gain (ESTIMATE): warm TTFT at 3-8K drops ~1.2 s;
  prefill t/s at 3.5-7K roughly doubles (2,696 -> ~5,000), closing the Q07 gap; the same
  fix helps the TP4 deployment shape disproportionately (Q14).
  Evidence: EXP-Q07/Q07b walls; the bypass source; the flat-floor signature.
  Risk: prefill metadata plumbing (attn-mode machinery); capture cost per bucket shape.
  Dependencies: none (builds on the Q01 split-core machinery).
  Next action: extend the Q01 split-mode draft cores to prefill shapes ([1, N, H], N from
  the lin prompt bucket set, ~4 shapes). The abotsis warning against graphing prefill-shaped
  draft calls (fresh recipe per novel length) applies to UNBUCKETED serving; with lin
  bucketing the shape set collapses and the compile is paid once per bucket.
- **Q08 KDA anchored-MME at short prompts (C015 re-measure at 512-2K)** — `READY`. Hypothesis:
  C015 showed no TTFT change at 2K-16K, but the upstream 3.23x evidence was at 507-3099 tokens
  where the diagonal decay-dot dominates prefill compute; the win should appear BELOW 2K.
  Bottleneck: KERNEL/PP. Expected gain (ESTIMATE): TTFT -30-60% at 512-2K if the regime matches.
  Evidence: C015 HPU measurement (2K: -4%, 8K: -3%, 16K: +9%); RESEARCH-001 addendum 4
  (abotsis c860b8ea: 15.6x kernel-time reduction, TTFT up to 3.23x at 507-3099 tokens).
  Risk: none (code already landed and verified). Dependencies: Q07 ladder tooling.
  Next action: run the short-prompt ladder and compare against the pre-C015 recorded figures.
- **Q09 Host-overhead reduction in the prefill path (mark-step / dispatch batching)** —
  `RESEARCH`. Hypothesis: small-prompt prefill is dominated by per-op lazy-dispatch host work;
  strategic mark_step batching and dispatch reduction (NOT the failed decode-side B023 knob)
  cuts host time. Bottleneck: PP/INFRA/KERNEL. Expected gain (ESTIMATE): +20-50% small-prompt
  prefill. Evidence: A017/B009; B023 negative result shows the decode-side knob alone was
  inert — the new hypothesis targets the PREFILL path, a different code path. Risk: medium.
  Dependencies: Q06/Q07 profiling. Next action: profile first; do not blind-patch.
- **Q10 1M-context serving feasibility on 8 HPUs (KV math + boot test)** — `VERIFIED` (1M window boots at 1660 hybrid blocks ~8 GiB/card; needle retrieved exactly at ~860K depth; deep prefill 1,643 t/s at that depth. Full-depth needle follow-up queued.)
  Hypothesis: measured KV cost is ~11.5 KiB/token/card at TP8 (1.8 GiB per 163,840 tokens);
  a 1M-token pool needs ~11.5 GiB/card — affordable next to 38.2 GiB weights; the objective
  is capacity-blocked, not architecture-blocked, on 8 cards. Bottleneck: KV.
  Expected gain (ESTIMATE): none directly — unlocks the 1M objective for validation.
  Evidence: F003, F012/F013 (512K needles pass; 1M "prefill latency software-blocked"),
  C013. Risk: medium (pool sizing overrides, warmup graph set, first-use compile stalls).
  Dependencies: none. Next action: compute the exact block override for 1M, boot, run the
  needle at 1M minus margin, record TTFT.
- **Q11 MTP depth k=2/k=4 revival under draft graphs** — `BLOCKED` (on Q01). Hypothesis: with
  the draft in-graph, deeper speculation turns net-positive (upstream MTP-4: accepted 3.33,
  aggregate +29%); in the eager draft it was net-neutral/negative. Bottleneck: MTP/TG.
  Expected gain (ESTIMATE): bs=1 decode 50-70 t/s if acceptance ~3.3 holds at k=4.
  Evidence: EXP-MTP-002 (eager k=2 net-neutral), RESEARCH-001 (graphed MTP-4 numbers).
  Risk: verify-row graph count grows with (k+1) rows x batch (capture-time cost).
  Dependencies: Q01. Next action: after Q01 lands, sweep k=2 then k=4 with the parity suite.
- **Q12 Vocabulary-parallel argmax in the draft head (get_top_tokens)** — `RESEARCH`.
  Hypothesis: the draft's full-vocab argmax forces a logits gather/sync per draft step;
  vocabulary-parallel top-1 (local argmax + cross-rank compare) removes a host sync from the
  draft path. Bottleneck: MTP/KERNEL/GRAPH. Expected gain (ESTIMATE): part of the draft-step
  host time; material only after Q01. Evidence: A015 (documented draft-side lever).
  Risk: low-medium (numerically exact if implemented as max+index compare). Dependencies: Q01
  (draft profiling). Next action: after Q01, measure the argmax share of draft time.
- **Q13 Expert routing balance histogram** — `RESEARCH`. Hypothesis: skewed expert routing
  inflates the slowest rank's MoE time (EP straggler); quantify imbalance from live routing
  metadata and, if skewed, evaluate capacity-factor/placement levers. Bottleneck: EP/TG.
  Expected gain (ESTIMATE): unknown until measured; a 10-20% straggler tail is typical when
  imbalance exists. Evidence: B004-B008 all NOT INVESTIGATED; no routing data collected yet.
  Risk: read-only instrumentation first. Dependencies: none. Next action: hook the router
  histogram in the plugin (env-gated), run one decode batch, compute per-rank max/mean.
- **Q14 TP4+EP deployment-candidate benchmark pass** — `VERIFIED` (4 cards, ~90% of TP8
  throughput on half the hardware: decode 33.26 vs 37.44 median, agg@16 113.5 vs 126.7,
  accuracy 100.0% PASS, acceptance held; prefill@2K 19% of TP8 — the Q07c fixed cost
  dominates; KV pool 320 hybrid blocks = 299,593 tokens; override 1280 OOMs at TP4) (after 8-HPU levers land).
  Hypothesis: TP4+EP (76.5 GiB/card weights) is the final-deployment shape; it must be
  benchmarked on all objective axes (decode/aggregate/prefill/context) to know the achievable
  window. Bottleneck: INFRA/AGG/KV. Expected gain (ESTIMATE): none — measures the deployment
  target and frees 4 cards for experiments. Evidence: G003 WORKING (TP4+EP boot viable).
  Risk: KV per card doubles at TP4 (window shrinks); verify-shape graphs re-capture.
  Dependencies: prefer after Q01/Q04 (so the levers carry over). Next action: schedule a
  TP4+EP boot with the pinned buckets; run the full objective benchmark pass.

- **Q15 TP8 accuracy drift vs Stage-1 (97.5% vs 100%, R003)** — `UNRESOLVED` (all code
  hypotheses excluded empirically: original mtp.py, Stage-1 pinning, buckets, C015
  timeline. Stack-state-sensitive knife-edge item; TP4 answers "5". Gate policy updated:
  same-boot A/B pairs are the valid comparison; item chaos = 2.5 pp.) Hypothesis: a
  deployed-tree change since Stage-1 (the Q01 mtp.py rewrite is the prime suspect to EXCLUDE
  first) or decode-bucket pinning differences shifted a config-chaotic item; TP4 passes and
  every TP8 variant fails identically. Bottleneck: ACC. Expected gain: restores the 100%
  baseline and unblocks Q01 adoption. Evidence: EXP-Q01b/Q15 matrix. Risk: may be intrinsic
  TP8-sharding chaos -> the gate policy moves to same-boot-family controls. Dependencies: none.
  Next action: deploy the Stage-1-era mtp.py (git 719f5bd) at TP8 and re-probe R003; then add
  decode-bs pinning; then a kda.py checkout.

- **Q16 fp8-moe-gather decode expert gather** — `FAILED` (negative: with the ratio gate
  opened (0.5) the gather engaged and LOST 45% at bs=1 (20.4 vs 37.44 median) - the
  gathered-weight copy + fp32 down GEMM dominate at 2-row shapes; stock op stays.
  Original entry: fix APPLIED to the deployed
  vllm-gaudi; env-gated OFF). Hypothesis: at decode only min(local_experts, tokens x top_k)
  distinct experts are needed; the gather path (vllm-gaudi #1731 backport, already in
  fixes/fp8-moe-gather) cuts expert weight traffic ~2.25x at bs=1 k=1 (16 of 36 local experts),
  the dominant single-stream cost on large EP MoEs. Bottleneck: TG/HBM/KERNEL.
  Expected gain (ESTIMATE): +20-50% bs=1 decode. Evidence: fixes/fp8-moe-gather/README.md;
  upstream #1731; cookbook MoE anatomy (expert kernels 19.5 ms of a 39 ms step).
  Risk: low (verify mode checks FP8-ULP; static gather count captures into graphs).
  Dependencies: none. Next action: boot with VLLM_HPU_MOE_GATHER=1 (+VERIFY first requests),
  bs=1 decode + 16-stream aggregate + accuracy gate vs the same-family baseline.
- **Q17 EP4xTP2 expert layout** — `RESEARCH`. Hypothesis: fewer wider EP bins cut the
  balls-in-bins imbalance (16 routes: busiest 6.3/4 bins vs 4.45/8 bins). Bottleneck: EP/TG.
  Expected gain (ESTIMATE): +5-11% single-stream (measured on GLM-5.3 MXFP4, same topology,
  memory-neutral; quality gates held, NOT bit-identical). Evidence: docs/results/moe-decode-step.md.
  Risk: load-time trap (page-faulting placement selection — read contiguous shares first);
  76.5->larger per-rank weights at TP2 sharding needs the memory check. Dependencies: none.
  Next action: compute the placement offline, then a TP2-EP4 boot with the load-order fix.
- **Q18 router fast path + early draft + device-side rejection** — `VERIFIED` (all three
  levers: bs=1 38.1 median (neutral), aggregate @16 **183-185 t/s (+46% over the eager
  draft, -13% vs non-MTP)**, acceptance 1.80 held, BYTE-IDENTICAL outputs, suite
  unchanged. ADOPTED-CANDIDATE. Lever 1 was
  VERIFIED: router fast path +4.4% bs=1 (39.10 vs 37.44 median), bit-identical on HPU
  (byte-identical R003 logprobs), patches/router_fast.py env-gated GLM_ROUTER_FAST.
  Levers 2-3 APPLIED (ports/glm-5-3-flash/patches/decode_lean.py adapted from the mxfp4
  port: lean module copied into our package, import rewritten, router target dropped;
  anchors matched our deployed tree including the already-applied spec-draft-prefill and
  spec-penalties patches; measured in EXP-Q18 lever 2-3).) Hypothesis:
  three bit-identical changes (router fast path: n_group=1 qualifies on OUR config, verified;
  device-side rejection walk with float32-rounded draws; device-side draft row indices)
  remove ~3.7 ms of a ~39 ms step and the end-of-step host sync. Bottleneck: MTP/SCHED/KERNEL.
  Expected gain (ESTIMATE): +12% single-stream (bit-identical, measured on GLM-5.3 MXFP4).
  Evidence: docs/results/moe-decode-step.md. Risk: medium (sampler walk port; check mode
  Next action UPDATED (turnkey): the glm-5-mxfp4 port already implements all three
  levers as a fail-closed patch (ports/glm-5-mxfp4/patches/decode_lean.py -> vllm
  cpu_router.py + vllm_gaudi hpu_model_runner.py + hpu_rejection_sampler.py - the SAME
  files in our tree; greedy bit-identical; off by default). Try --check on our tree;
  adapt anchors; boot GLM_ROUTER_FAST=1 first, then + GLM_SPEC_DEVICE_REJECT=1 +
  GLM_EARLY_DRAFT=1. Bonus: ports/glm-5-mxfp4/patches/spec_sampling.py fixes the
  all_greedy assert (temperature > 0 crashes the engine under MTP - verify on ours).
  ADAPTATION NOTES (from inspecting both trees): the lean module is
  ports/glm-5-mxfp4/glm5_hpu/glm5_hpu/decode_lean.py (225 lines: early_draft,
  take_early_draft, make_device_reject) - copy into our glm5_next_hpu package and
  rewrite its import (from glm5_hpu -> from glm5_next_hpu). The runner anchors
  (OLD_RESET/OLD_EARLY/OLD_TAKE in patches/decode_lean.py) may have shifted under
  our mtp_runner.py patch - run --check first and re-anchor. The sampler tail wires
  make_device_reject around rejection_sampler.rejection_sample (hpu_rejection_sampler.py,
  pristine in our tree). Router fast path: DONE separately as our
  patches/router_fast.py (bit-identity contract test green; env GLM_ROUTER_FAST).

### Queue operating rule

After EVERY experiment: update the entry, ask what now limits performance, add newly discovered
opportunities with evidence, re-prioritize, start the highest-value eligible entry. A failed or
negative experiment stays in the file with its evidence (never delete history).
