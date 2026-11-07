# CURRENT_STATE.md

Updated at the end of the policy-installation / knowledge-base session. Sanitized for
publication; infrastructure details are described generically.

## Objective scorecard (session status, 2026-11-07)

Primary objectives vs the best VERIFIED, accuracy-gated measurements:

| Objective | Best verified | Status |
|---|---|---|
| 200 t/s single-stream decode | 37.4 t/s (MTP k=1, E015-A buckets) | 5.3x gap; Q01 draft graphs reach 40.9 (+9.3%) but fail the accuracy gate (Q01b) |
| 800 t/s aggregate @ bs=16 | 213.2 t/s (non-MTP) / 126.7 (adoptable MTP) | 3.7x gap; Q01-split reaches ~168 (+33% over adoptable MTP) but fails the gate |
| 3,000-6,000 t/s prefill | 3,338-5,697 t/s warm at 0.5-2K, 6K, 8K depths | **MET at most depths** (Q07 + measurement correction); 3.5K bucket-edge point 2,696 (Q07b open) |
| 1M context | **VERIFIED**: 1M window boots (1660 hybrid blocks ~8 GiB/card); needle retrieved exactly at ~860K depth (860,739 tokens, 523.7 s); deep prefill 1,643 t/s at that depth | **MET** (serving-validated on 8 HPUs) |
| GLM-5.3-Flash FP8 | served as published (block FP8) | met |
| <= 0.5 pp accuracy regression | 100.0% on the adopted config | met for the adopted config; Q01-split at 97.5% NOT adopted |
| Final deployment on 4 HPUs | TP4+EP boots (G003); benchmark pass pending | Q14 |

## Session findings (this session)

1. **Q02/Q03:** with the eager draft, aggregate SATURATES at ~128 t/s from 8 streams; MTP @16
   net-negative 40% vs non-MTP. The eager draft is the aggregate bottleneck, not just bs=1.
2. **Q01 (draft graphs):** implemented (split + attn modes, env-gated default OFF). Split mode
   VERIFIED for performance: bs=1 +9.3%/+11.2%, aggregate @16 +33% (~168 t/s), acceptance held.
   Accuracy gate FAILED (97.5% vs 100.0%, R003 bat-and-ball flips to "10" deterministically;
   logprobs show a 0.75-nat gap — a real distribution shift, not a tie-flip). NOT adopted;
   Q01b owns the correctness investigation. A speculative-decoding contract break (draft must
   never change content) — likely verify/rollback interaction with the graphed draft.
3. **Q07:** warm small-prompt prefill is 3,338-5,697 t/s — the 3,000-6,000 objective band is
   mostly met. MEASUREMENT CORRECTION: the historical "~770 t/s" and "TTFT 2.1 s @2K" figures
   included 64-token generation in the wall; warm TTFT at 2K is 0.31 s. Q06 (scheduler
   headroom) revised down accordingly.
4. **Q10:** the KV-pool block override counts 640-token hybrid blocks (8192 blocks = 5.22M
   tokens = OOM); 1660 blocks = 1.06M tokens ~8 GiB/card boots clean.
5. **Standalone wrap_in_hpu_graph limitation:** any graph containing an add fails recipe
   compile in standalone processes on this stack (1.24.1) with HabanaFusedOpLazy synStatus 26;
   the server process captures the same ops fine. HPU graph validation must happen in-engine.

## Highest-value open opportunities (queue)

1. **Q01b** — draft-graph accuracy regression (R003 flip): unlocks Q01's +33% aggregate.
2. **Q07b** — 4K bucket-edge prefill step (2,696 -> ~5,000 t/s expected): closes the prefill gap.
3. **Q10** — 1M needle validation (in flight).
4. **Q14** — TP4+EP deployment-candidate benchmark pass.
5. **Q11** — MTP k=2/k=4 revival under draft graphs (blocked on Q01b).

## What is deployed and running

- **GLM-5.3-Flash** (block-FP8 checkpoint as published, 328 GiB), cookbook port on vllm-gaudi
  `v0.26.0` (b059f9c, vLLM 568afb3a), Gaudi SW 1.24.1.
- **Live configuration: MTP k=1**, TP8 + expert parallel (all 8 HPUs), max-model-len 32768,
  max-num-seqs 8, KV override 1280 blocks/card, GMU 0.82, graph capture 2048, chunked prefill
  8192, prefix caching OFF (architectural: KDA stores no per-block states), reasoning/tool
  parser `glm47`.
- Launch script: `ports/glm-5-3-flash/serving/serve_mtp.sh` in this repository (window flags
  describe the validated MTP shape; the pre-MTP production window was 262K / 32 seqs).
- The pre-MTP production configuration (no MTP, 262K / 32 seqs) is the rollback target and the
  Stage-1 control configuration.

## Verified results (Stage-1 reproduction complete)

The Stage-1 controlled reproduction (A001) has completed: control (non-MTP) vs treatment
(MTP k=1) at equivalent conditions (same window, same bucket pinning, same prompt set),
5 repetitions per condition. The accuracy gate passed (0.0 pp delta). The following
figures are VERIFIED per `docs/BENCHMARK_POLICY.md`:

| Metric | Control | Treatment (MTP k=1) | Delta | Status |
|---|---|---|---|---|
| Decode wall tok/s (bs=1) | median 32.05, mean 32.16, std 0.92 | **median 36.91, mean 36.83, std 0.77** | **+15.1% (mean)** | VERIFIED |
| Accuracy (suite v1, 40 items × 5 reps) | 100.0% | 100.0% | +0.00 pp | PASS (≤ 0.5 pp) |
| TTFT (bs=1, 32K window) | 0.10 s | 0.14 s | +0.04 s (draft prefill pass) | VERIFIED |
| ITL median | 0.0297 s | 0.0482 s | fewer, larger steps | VERIFIED |

The historical ~21% figure was PROVISIONAL (non-equivalent control); the verified
like-for-like figure is +15.1%. The 16-stream net-negative finding (42.7 vs 59.4 with the
eager draft) is also measured but not yet re-verified under the pinned-bucket config.

- **Non-MTP production baseline** (two independent sessions, replicated): decode 30.1–30.3
  tok/s bs=1; aggregate 213.2–217.2 tok/s @16 streams; prefill 4,928 tok/s @128K prompt; HBM
  65.5–68.2 GiB/card.
- **MTP k=1** (like-for-like 32K/8seq window, single run): decode 35.7 tok/s bs=1 (+14%);
  acceptance 1.73–1.98 of 2.0; 16-stream aggregate 42.7 tok/s vs 59.4 non-MTP (net-negative
  with the eager draft — crossover documented).
- **MTP k=2**: functionally correct (chained single head), acceptance 2.04–2.52 of 3.0,
  decode net-neutral (30.2–34.1 tok/s) — k=1 retained as default.
- **Historical reported result**: "approximately 21% batch-1 decode uplift" compares MTP
  (36.8 tok/s, 32K window) against a production-window control (30.3 tok/s) — conditions not
  equivalent; PROVISIONAL, superseded for decision-making by the like-for-like +14% until
  Stage-1 reproduction (A001) completes.

## Active work

- **Q01 draft HPU graphs** (top priority): implemented (split + attn modes, env-gated default
  OFF, CPU contract tests green, EXP-Q02/Q03 shows the eager draft saturates aggregate at
  ~128 t/s). HPU A/B pending: split-mode boot, bs=1 decode + 16-stream aggregate + acceptance
  + accuracy gate.

- **Stage 1 (A001)**: turnkey — `pre_flight.sh` (26/26) then `stage1_reproduction.sh`
  (the venv fix, the bucket-pinned dual-condition boot, the dual-mode decode benchmark,
  the accuracy gate). Awaiting the maintenance-window approval.
- **E015**: turnkey — `serve_e015_lin.sh` and `serve_e015_epfilter.sh` (both dry-run-
  validated), one variable per boot.
- **B023**: turnkey — the patch is in the fail-closed contract with real-tree anchors
  verified; deploy during the window boot with `VLLM_GLM_SKIP_LAYER_MARKSTEP=1`.
- **C015**: landed in the repo; deploy `kda.py` during the window and measure the prefill
  ladder.
- **I015**: the capture core landed (CPU-verified); the diag boots are pre-staged
  (`serve_diag.sh` / `serve_diag_mtp.sh`); the capture runs during the window with
  `VLLM_GLM_LAYER_DIAG_DIR` + the sentinel; `diag_compare.py` analyzes the payloads.
- **A014/A018 (the draft HPU graph)**: the abotsis fork's implementation is the reference
  (RESEARCH-001 addendum 1); adapting it into this port is the highest-value lever after
  the measurements.

## Production runbook additions (from RESEARCH-001, pending deployment)

- Warm client after /health covering served prompt lengths (the draft prompt-cache fill and
  attention graph compile on first use, 2-8 s each; decode buckets must warm as verify
  batches).
- Size `VLLM_PROMPT_QUERY_BUCKET_*` to the workload: a prompt just over a bucket edge pays
  30-50% TTFT.
- `VLLM_HPU_DECODE_TENSOR_CACHE` stays off under speculative decoding (host memory at TP=8).

## Target disambiguation resolved (pending owner confirmation)

The "200 tok/s decode" target is **met under the aggregate interpretation** (213.2-217.2 tok/s
at 16 streams, no MTP, replicated across two sessions) and **physically unachievable under the
single-stream interpretation**: the weight-read bandwidth floor (41.1 GB/card/step at
1,696-2,450 GB/s) caps bs=1 decode at ~40-59 tok/s without MTP and ~150-190 tok/s with deep
MTP at perfect utilization (DECISIONS.md has the full roofline analysis). Owner confirmation
of the aggregate reading closes the target.

## E015 config A/B readiness

Both candidates are pre-staged and validated:
- `serving/serve_e015_lin.sh`: linear bucketing + the prompt-bucket taming (37 args,
  the bucket set verified: 128 prompt + 16 decode graphs vs exp's 468 + 60).
- `serving/serve_e015_epfilter.sh`: `--enable-ep-weight-filter` (38 args, the CLI arg
  verified in our vLLM tree).

## Blockers

- GitHub publish of the knowledge-base/policy files: awaiting user review/approval (local
  commits only; nothing pushed yet).
- Draft HPU graph (A014/A018): shared storage between target and draft requires a split
  pre/post-attention graph design — not prototyped.
- MTP at the production 262K window (A024): window flags in the validated MTP launch script
  are 32K/8seq; extending requires re-validating graph shapes at the larger window.

## Lessons learned (see also DECISIONS.md)

- The eager draft head makes MTP net-positive only at low concurrency; the draft HPU graph is
  the documented lever for k≥2 and high concurrency.
- Verify-row count = seqs × (k+1); every new row count adds HPU graphs — raise `max-num-seqs`
  only with re-validation.
- KDA prefill chunk 64 is optimal on this hardware (128 measured worse).
- Prefix caching must stay OFF (KDA architectural constraint; the engine refuses to start).
- Lazy-mode first-use compiles 13–15 s per new graph shape; pre-warm served shapes.

## Git state

- Repository: this fork (origin = the GPUprocess fork of the cookbook; do not change).
- Branch: `glm53-flash`, tip `044f764` — pushed and verified on the remote.
- All policy files, the knowledge base, the audit, the E015 A/B scripts, the dual-mode
  decode benchmark, and the research record are published on the remote.
