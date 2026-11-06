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

- **Stage 1 (next task, A001)**: controlled MTP=1 reproduction per `docs/BENCHMARK_POLICY.md`
  — non-MTP control vs MTP k=1, same window/config, ≥5 repetitions per condition, alternating
  order, median/mean/std/P95, plus the accuracy-gate setup (see `ACCURACY.md`).
- After Stage 1: MTP depth is already settled for the eager-draft configuration (k=1 optimal);
  the draft HPU graph (A014/A018) is the highest-value lever to reopen k=2+ and the concurrent
  crossover.

## Production runbook additions (from RESEARCH-001, pending deployment)

- Warm client after /health covering served prompt lengths (the draft prompt-cache fill and
  attention graph compile on first use, 2-8 s each; decode buckets must warm as verify
  batches).
- Size `VLLM_PROMPT_QUERY_BUCKET_*` to the workload: a prompt just over a bucket edge pays
  30-50% TTFT.
- `VLLM_HPU_DECODE_TENSOR_CACHE` stays off under speculative decoding (host memory at TP=8).

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
- Branch: `glm53-flash`, tip `e3c6f52` — pushed and verified on the remote.
- This session's policy/knowledge-base files are committed locally on the same branch, NOT
  pushed (awaiting user review per the project directive).
