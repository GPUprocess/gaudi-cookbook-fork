# GLM-5.3-Flash MTP speculative decoding

Native multi-token prediction (MTP) speculative decoding for the [GLM-5.3-Flash
port](glm-5-3-flash.md): the checkpoint's own MTP draft head (layer 45) drives vllm-gaudi's
speculative-decode machinery on one 8-card Gaudi 2 node, with the KDA layer keeping per-request
candidate state for the verify rows. Single-stream decode goes from 30.3 to 36.8 tok/s (+21%,
greedy, natural prompts) at a mean acceptance length of 1.8 of the k=1 maximum 2.0.

| | |
| --- | --- |
| Model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT), block FP8 as published |
| Hardware | 8x Gaudi 2 (TP 8, expert parallel), one node |
| Image | the [GLM-5.3-Flash port](glm-5-3-flash.md) image (vllm-gaudi `v0.26.0`, `b059f9c`, vLLM `568afb3a`, Gaudi SW 1.24.1) |
| Speculative config | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'` |
| Acceptance (k=1) | 1.73 – 1.98 of 2.0 (typical ≈ 1.8); live deployment 1.84 – 1.95 |
| Decode | 36.8 tok/s at bs=1 (+21% over the same weights without MTP); 37.6 tok/s after a 4K prompt |
| Concurrency | verified to 16 streams; see the crossover below |
| Scope | k=1 is the recommended setting; k=2 was swept (correct, not faster) |

## What was hard

Three things, in the order they were built.

**The draft head has no weights of its own to load.** The checkpoint ships the MTP block as part of
the target model (layer 45 plus an `eh_proj` and a shared head). The port binds the draft wrapper to
the already-loaded layer-45 modules by reference: zero marginal weight memory (no duplicate
0.9 GiB/rank), one extra MLA latent cache (~0.8 GiB/rank) and KDA candidate-state columns
(~0.3 GiB/rank at k=1). The wrapper normalises the checkpoint names (`model.language_model.layers.45.*`
to `model.mtp.*`, `shared_head.norm.weight` to `shared_head_norm.weight`) and refuses to start if
speculative decoding is on and the MTP block did not fully load — the completeness check is the
stage-1 gate.

**KDA state is per-request single-slot; the verify rows need S+1 columns.** The Kimi Delta Attention
layers keep one recurrent state per request. A verify forward presents k+1 positions per request and
may accept any prefix of them, so the decode state after verify can come from any of the k+1 rows.
The KDA layer grew candidate-slot state: S+1 state columns per request (S = k+1 at k=1), the verify
forward scattering one state per row, and the post-verify state load selecting the column that
matches the number of accepted tokens. The column algebra is shared with vllm-gaudi's compact Mamba
state layout (`base*(groups*S) + g*S + col + 1`) and is unit-tested against request churn (finish +
slot reuse, per-group disjointness, column uniqueness).

**The stock speculative path assumes GPU verify layouts.** The vllm-gaudi side needed the
[glm-5-mxfp4 port's spec patches](glm-5-mxfp4.md) (`mtp_tuple`, `mtp_residual_shape`,
`spec_sampling`, `spec_penalties`, `spec_logprobs`, `spec_structured_output`,
`spec_draft_prefill`, `spec_verify_blocks`); all eight were drift-checked against the `v0.26.0`
sources and apply cleanly. They cover the padded verify layout for sampling, penalties and logprobs,
grammar-constrained decoding, the partial-chunk draft-KV fill during chunked prefill, and
per-verify-row KV block lists.

## The design in one paragraph

`HpuEagleProposer` binds the plugin's `mtp` wrapper as the draft model. The wrapper is the target's
layer 45 re-entered with the last hidden state: embed → eh_proj(concat(enorm(emb), hnorm(h))) →
layer-45 MLA+MoE (attention input replaced by the projected draft embedding) → shared head. Position
0 of each draft embedding is zeroed so the first drafted token is a pure function of the target's
last position. The draft runs eagerly (shared storage cannot be captured in a second HPU graph);
at k=1 the remaining draft cost is bounded by the small layer (1 MLA + 1 MoE block) and the eager
run is the correctness-first choice. During prefill, a `prefill_cache_only` pass fills the draft
MLA layer's KV so the first decode step can draft immediately.

## Correctness gates

All numbers below are from an 8-card Gaudi 2 node on the port image; the harness is
`tests/hpu/mtp/mtp_validate.py` against a running server.

- **Golden parity.** 12 recorded prompts, greedy, same config. k=1 reproduces 2/12 token-identical;
  every divergence is a model-inherent near-tie flipped by the verify-shape FP shift (measured with
  top-2 logprob gaps of 0.25 vs 1.375 at the first divergence). k=0 on the same window reproduces
  6/6, which attributes the divergences to the speculative path and not to the config. A content
  comparison after reasoning stripping finds the same divergence positions at k=1 and k=2.
- **Feature gates under MTP.** Tools (single, parallel, no-call), images (4 per prompt), streaming,
  penalties (frequency, presence, repetition), sampling (top-p/top-k), JSON schema and guided-regex
  structured output, top-5 logprobs, client disconnect mid-generation, and needle retrieval at 16K
  (16/16 access codes, port gate capped to the 32K window).
- **Long generation and block boundaries.** 2034-token generations with zero degeneration; prompts
  sized to 639/640/641 and 1279/1280/1281 tokens so verify rows straddle the 640-token KV block
  boundary — all correct (`spec_verify_blocks`).
- **CPU unit tests.** The draft wrapper contract (reshape, position-0 zeroing, eh_proj order,
  residual flow, single head application, no-op load_weights, missing-MTP error), the KDA
  candidate-slot chaining parity, the churn slot algebra, and the chain-conv parity vs the
  production kernel: `tests/cpu/check_mtp_wrapper.py`, `check_kda_spec.py`, `check_churn_slots.py`,
  `check_conv_chain.py`.

## What it measured

| Metric | k=1 MTP | k=0 baseline | Notes |
| --- | --- | --- | --- |
| Decode, bs=1 greedy | **36.8 tok/s** | 30.3 tok/s | 3x 256-token completions, natural prompts |
| Decode, bs=8 | 36.5 tok/s | — | 8 concurrent ~170-token completions |
| Decode after 4K prompt | 37.6 tok/s | — | + 512 generated tokens |
| 16-stream aggregate | 42.7 tok/s | 59.4 tok/s (same window config) | eager draft costs more than acceptance buys at this batch |
| Acceptance length | 1.73 – 1.98 | 1.0 | engine metrics, natural prompts |
| HBM under load | 64.1 GiB / 96 GiB per card | ≈ 62 GiB | weights 38.6 GiB/rank + caches + candidate columns |

**k=2 sweep.** All k=2 paths work (draft chaining loop, S=4 candidate columns, 3-row verify
forward; acceptance 2.04 – 2.52 of 3.0), but decode is net-neutral (30.2 – 34.1 tok/s): the second
eager draft step costs more than the extra accepted tokens buy. The documented levers that would
flip k>1 positive are the draft HPU graph and a vocab-parallel argmax in `get_top_tokens`; both are
optimization follow-ups.

**Crossover.** MTP is net-positive for single-stream and low-concurrency serving and net-negative
by 16 streams in the eager-draft configuration. Enable it for interactive workloads; leave it off
for batch.

## Launch

`serving/serve_mtp.sh` in the port is the full verified command; the deltas from the stock port
launch are the `--speculative-config`, `--max-num-seqs 8` (verify rows = seqs x (k+1)) and the
window flags. Run the port's `install.sh` first (it applies the port patches and installs the
plugin with the MTP module), then apply the eight spec patches from
`ports/glm-5-mxfp4/patches/` to the release sources in this order — each supports `--check`:

```bash
for p in mtp_tuple mtp_residual_shape spec_sampling spec_penalties spec_logprobs \
         spec_structured_output spec_draft_prefill spec_verify_blocks; do
    python3 ports/glm-5-mxfp4/patches/$p.py --root <site-packages> || exit 1
done
```

Then run the harness: `python3 ports/glm-5-3-flash/tests/hpu/mtp/mtp_validate.py --base
http://127.0.0.1:8000`.

## What not to do

- Do not enable prefix caching: the KDA layer does not store per-block states (inherited from the
  base port; the server refuses to start).
- Do not raise `--max-num-seqs` without re-checking the verify-row graph shapes: verify rows are
  seqs x (k+1), and every new row count adds graphs.
- Do not enable MTP without the eight spec patches: the stock sampling/logprobs/structured-output
  paths fail on the padded verify layout (the engine 500s or dies on the first penalised or
  grammar-constrained request).
- Do not present the single-stream gain as a guarantee at other concurrency: the crossover to
  net-negative was measured at 16 streams in the eager-draft configuration.
