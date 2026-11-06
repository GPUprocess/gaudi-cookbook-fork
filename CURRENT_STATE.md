# CURRENT_STATE.md

Updated at the end of the policy-installation / knowledge-base session. Sanitized for
publication; infrastructure details are described generically.

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

## Verified results (strongest evidence to date)

All figures are measured; none has yet passed the full `docs/BENCHMARK_POLICY.md` gate
(≥5 repetitions, variability, alternating order), so `BASELINE.md` verified figures remain
unfilled and these are recorded as BASELINE CANDIDATE / PROVISIONAL in `EXPERIMENTS.md`.

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
