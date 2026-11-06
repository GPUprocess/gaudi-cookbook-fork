# RESEARCH-NOTE 001 — Draft-graph machinery exists upstream (abotsis fork 919ec98f)

**Date:** knowledge-base session
**Source:** GaudiGrimoire MCP search (tier D — community fork code/docs; specific and internally
consistent, not vendor-verified). Sources: `abotsis-vllm-gaudi` fork pinned `919ec98f`
(`docs/features/glm53_flash.md`, `docs/features/glm53_flash_howto.md`,
`docs/configuration/env_variables.md`, `vllm_gaudi/models/glm5_next_mtp.py`,
`vllm_gaudi/v1/worker/hpu_model_runner.py`, `tools/LAYER0_DELTA_FINDINGS.md`) and the abotsis
gist (web capture).

## What GaudiGrimoire provided (concrete, actionable)

1. **Why our eager draft is slow — quantified**: "each draft forward is hundreds of host
   launches, and on this model the host launch path is the bottleneck (**93 ms host vs 17 ms
   device per step**)". MTP-4 pays that per step. Our k=2 net-neutral result and our 16-stream
   net-negative result are consistent with this host-launch bottleneck.

2. **The shared-storage capture failure and its fix**: capturing the target's storage in a
   second graph fails at replay with *"Neither storage attached to input tensor, not its
   view"*. Their fix: `_private_copy()` — shallow-copy the module tree and replace each
   tensor's storage with a private device clone (~0.9 GiB/rank; no checkpoint re-read;
   preserves fp8 `process_weights_after_loading` layouts). Private copies make the two graphs
   independent **by construction**.

3. **Three draft execution paths, measured (their MTP-4 numbers, launcher)**:
   | Path | Accepted len | Aggregate | Single-stream | Parity |
   |---|---|---|---|---|
   | Attention bypass (their earlier default) | 2.405 | 20.16 | 17.26 | 12/12 |
   | Attention on, eager draft | 3.429 | 18.15 | 12.90 | 12/12 |
   | Split graph (pre/post cores, eager attn) | 3.429 | 22.10 | 16.16 | 11/12 (one late divergence — race suspicion, not default) |
   | **Attention-in-graph (their default since 2026-09-14)** | **3.33** | **26.0** | **20.9** | **12/12** |

4. **Draft self-attention inside the graph** (`VLLM_GLM_MTP_DRAFT_ATTN=1`,
   `VLLM_GLM_MTP_ATTN_GRAPH=1`, both default on): the draft's MLA self-attention runs INSIDE
   one HPU graph per decode-shaped draft step; attention metadata tensors (block_list,
   block_mapping, attn_bias, block_groups, slot_mapping) are graph inputs, and the forward
   context is rebuilt from them inside the captured forward so replays read fresh block
   tables. The draft's attention and MoE are its own modules (not captured by the target's
   graph — no double capture). The draft's persistent KV cache is read/written in place.

5. **Device-side rejection sampling**: temperature/top-k/top-p/softmax/recovery on device in a
   materialized segment: sampled decode 19.7 tok/s vs 21.3 greedy (was ~5 tok/s with the host
   path; `VLLM_HPU_REJECTION_HOST=1` restores). Greedy unaffected by construction.

6. **Context buckets for long prompts**: prompts > 3200 tokens were chunked with eager padding
   to model length (1.5 s per chunk); fixed with ctx buckets 0/64/128/192.

7. **Warm-up protocol**: the draft compiles one graph per decode shape + one prompt-cache fill
   per prompt bucket on first use (2–8 s each); warm with prompt lengths 100/500/1k/2k/3k/7k
   plus a burst of 8 short requests before measuring.

8. **Their measured 8× Gaudi2 numbers (greedy)**: TTFT 0.13 s (128 tok) / 0.22 s (1k) / 0.45 s
   (3k) / 1.2 s (7.7k); decode 21–24 tok/s single stream; ~75 tok/s aggregate at 4 streams;
   sampled ~19.7 tok/s.

## Relation to our implementation

Our `mtp.py` shares lineage (bind-by-reference, position-0 zeroing, `eh_proj`, shared head,
prefill cache fill — all present in their code too). Our fork does NOT contain the draft-graph
machinery: no `_private_copy`, no `_DraftGraphCore`/`_DraftAttnGraphCore`, no
`VLLM_GLM_MTP_DRAFT_GRAPH`/`_SPLIT_GRAPH`/`_ATTN_GRAPH`/`_DRAFT_ATTN` env handling (verified by
grep against our plugin and the deployed copy). Our unique contributions remain: the KDA
candidate-slot state machinery (S+1 columns, churn-safe), the verify-block KV patches, the
recorded patch set, and the validation/benchmark harness.

**Open question (theirs vs ours):** how their fork handles GLM-5.3's KDA recurrent-state
rollback across verify rows — our candidate-slot work may be complementary or duplicated.
Their docs do not mention KDA candidate slots explicitly.

## Impact on the roadmap

- **A014/A018 (draft HPU graph)**: status READY — a working reference implementation exists
  with measured numbers; the experiment is adaptation/porting, not invention.
- **Expected effect on our results**: our k=2 net-neutral and 16-stream net-negative findings
  are likely artifacts of the eager host-launch bottleneck (93 ms/step). With the draft graph,
  k=2 should flip net-positive and the concurrent crossover should move substantially higher.
- **Strategic option to evaluate**: adopt/port the abotsis draft-graph machinery into our
  plugin (keeping our KDA candidate-slot work), OR evaluate their fork directly as a base.
  Decision needed from the project owner after Stage-1.

## De-duplication statement (per protocol §5)

"Draft HPU graph" was previously NOT INVESTIGATED (documented as a lever only). This note does
not repeat an experiment — it records new evidence (a working reference implementation with
measurements) that changes feasibility, exactly the retry justification the protocol requires.


## Addendum: KDA speculative-state deduplication answered

The same fork also implements KDA candidate-slot state for speculative decoding — the problem
our KDA candidate-slot work solves — with a different mechanism:

- **Theirs** (`vllm_gaudi/models/glm5_next.py`): per-request private candidate slots APPENDED
  to the recurrent-state pool, with slot ids stable across batch condense and swap;
  `state_indices` is `[bs, 1 + num_spec + 1]` (column 0 = canonical post-prompt state,
  column 1+j = state after draft token j of the previous verify); `num_accepted` selects the
  resume slot directly (0 = fresh sentinel); the whole draft chain is scored in ONE batched
  kernel call (`chunk_kda_spec_states`) that writes every candidate state; prefix caching
  explicitly rejected under spec decode; plus layer-diagnostic tooling (kda boundary/sequence/
  pool-row hooks).
- **Ours** (`ports/glm-5-3-flash/glm5_next_hpu/`): S+1 columns addressed by the compact-GDN
  pool formula; verify-forward scatters one state per row; post-verify load selects the column
  matching the accepted prefix; churn/chain/boundary parity tests.

Same problem, two solutions; theirs is integrated with a working draft-graph stack and device
sampling and reports MTP-4 acceptance 3.33 of 4.0. A third independent GLM-5.3-Flash port
exists in the corpus (`inderjodhsingh/gaudi-llm-notes`, pinned c422c56d, KDA prefill
padding-mask fix).

**Conclusion:** the KDA spec-state problem is solved three ways in the wild; inventing a fourth
has no value. The strategic options are (a) port their draft-graph + spec-state advances into
our plugin, (b) evaluate their fork directly as the serving base, or (c) hybrid. This decision
belongs to the project owner; our recorded evidence and harness remain useful for whichever
path is chosen.


## Addendum 2: applicability check against our deployment (measured)

- **Sampled decode is NOT slow on our stack**: measured on the live MTP k=1 server, 384-token
  generations, single stream: greedy 36.2–36.3 tok/s, sampled (T=0.7/top_p 0.9) 32.8–33.7 tok/s
  (−8%). The abotsis host-rejection bottleneck (~5 tok/s at their MTP-4 shape) does not bite at
  our k=1/small-batch shape — but our sampler does move tensors to host per step
  (`hpu_rejection_sampler.py` `.cpu()` calls), so the cost grows with k and batch; their
  device-side sampling is the lever if we ever see it bite.
- **`VLLM_HPU_ALLREDUCE_MODE=gather_sum` does NOT exist in our vllm-gaudi b059f9c tree** — it
  is an abotsis-fork feature. Our concurrent-determinism findings stand as measured; porting
  the position-independent all-reduce would fix the co-batched row-order ULP issue and is a
  candidate for the A014 adaptation package.
- **`VLLM_DECODE_BS_BUCKET_*` DOES exist in our tree** (extension/features.py + bucketing/
  padding_aware.py): the abotsis bucket-pinning recipe (pin decode to one bucket) is directly
  applicable — their measurement: warmup 13 → ~2 min (one decode bucket), parity 12/12, decode
  within noise. Free win for boot time and determinism; untested here.


## Addendum 3: skip_layer_markstep backport candidate (+10% single-stream, bit-identical)

Our v0.26.0 runner installs a per-DecoderLayer forward hook that calls `mark_step()` after
every decoder layer (`modify_model_layers`, interval from `VLLM_CONFIG_HIDDEN_LAYERS`,
default 1 — our production launch does not set it). The abotsis runner (919ec98f) skips this
hook for `glm5_next`/`glm5_next_text` under HPU-graph replay:

> "The per-DecoderLayer mark_step hook predates HPU-graph replay: it flushed the lazy IR per
> layer so compilation could pipeline against the device. Under wrap_in_hpu_graph the cuts it
> makes are captured and replayed forever, and each replayed recipe boundary costs ~0.2-0.5 ms
> of host time per decode step. For glm5_next the hook fires after the MoE all-reduce and the
> trailing mHC math, splitting that math into its own recipe in every layer: 45 of 176 launches
> per decode step. Skipping it is bit-identical (greedy parity 8/8) and measured +10%
> single-stream decode."

Their skip condition: lazy mode, not enforce_eager, model_type in (glm5_next, glm5_next_text),
`VLLM_CONFIG_HIDDEN_LAYERS` unset — all match our production deployment. Reference code saved
at `research/abotsis-919ec98f/skip_layer_markstep_reference.py` (operator workspace). Proposed
as task B023: backport the skip into our runner as a fail-closed port patch, verify greedy
parity on our golden set (expected bit-identical), measure decode bs=1 and concurrency.
Related: `_decode_tensor_cache` defaults OFF when speculative decoding is on (kept tensors
cost multi-GiB host memory per worker; TP=8 warmup OOM-killed ranks on their densest host).


## Addendum 4: KDA diagonal-block decay-dot -> anchored MME matmul (major prefill candidate)

Applicability verified against our `_gauged_scores` (kda.py): our OFF-diagonal blocks already
use the anchored matmul, but the DIAGONAL block still materializes `[m, sub, sub, d]`
elementwise (`dexp = (gi.unsqueeze(2) - gi.unsqueeze(1)).clamp(max=0.0).exp()` then
`(a * b * dexp).sum(-1)`) — the exact TPC-bound pattern abotsis fixed (commit c860b8ea).

Their fix: per 16-row block, factor the exponent around the block's first-row gate —
`row_f = a * exp(g_i - g_ref)`, `col_f = b * exp(g_ref - g_j)`, both factors in (0, 1] since
`g` is non-positive decreasing and the anchor is the block's first row — then `mid = row_f @
col_f.T` on the MME. The gate lower bound (-5/token) keeps every factor inside fp32
(e^{+-80} within a block). Measured: per layer 57.6 -> 3.7 ms summed kernel time (512 tokens,
15.6x), 224 -> 14.3 ms (2048); TTFT speedups 1.11x @123 tokens to 3.23x @1995 tokens; 3099
tokens 0.915 -> 0.479 s (1.91x, 6,476 tok/s). The 2048-token cliff was the decay-dot
temporaries. Validation blueprint: oracle transition tests 12/12 (CPU), test_hpu_kda 14/14 on
HPU, serving gates (rowdep 8/8, parity 12/12, decode bench unchanged).

Interaction with our knob: our chunk=64-optimal finding (EXP-MTP-004) measured the scan
intermediates cost — with the diagonal block moved to the MME that tradeoff may shift
(re-test chunk after landing this).

Proposed task C015 (READY): replace the `mid` computation in `_gauged_scores` with the
anchored matmul; oracle-test against the current implementation; measure the prefill ladder
and TTFT; re-check the chunk-size tradeoff.


## Addendum 5: env-var applicability map (our b059f9c trees vs the reference fork)

| Reference env var | In our tree? | Note |
|---|---|---|
| `VLLM_GLM_MTP_DRAFT_GRAPH/_SPLIT_GRAPH/_ATTN_GRAPH/_DRAFT_ATTN` | no | the A014 adaptation package |
| `VLLM_HPU_ALLREDUCE_MODE=gather_sum` | no | B022 port |
| `VLLM_HPU_DECODE_TENSOR_CACHE` | no | defaults off under spec decode on the reference; host-memory note |
| `VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS` + ctx buckets | no | their long-prompt graph fix (7690-token prompt 4.1 -> 1.22 s) |
| `VLLM_HPU_GRAPH_ASYNC_REPLAY` | no | untested lever |
| `VLLM_DECODE_BS_BUCKET_MIN/STEP/MAX` | **yes** | B021 (bucket pinning) |
| `VLLM_PROMPT_QUERY_BUCKET_*` | **yes** | prompt-just-over-an-edge pays 30-50% TTFT — size to workload |
| per-DecoderLayer mark_step skip for glm5_next | no (logic absent) | B023 backport |

Conclusion: only the bucket/query-bucket knobs exist in our release; everything else arrives
with the A014 adaptation or dedicated ports. The reference implementation (919ec98f) bundles
them all and is the fastest route to most of them.


## Addendum 6: boundary diagnostics, concurrency reference, and depth sweep (their findings doc)

From their `tools/LAYER0_DELTA_FINDINGS.md` (994 lines, 27 sections):

1. **Near-tie attribution, boundary-level proof**: with bit-identical attention inputs and
   weights, `attention_raw_output` differs by ~1 bf16 ULP across batch contexts (relL2
   3-4e-3 = bf16 eps); deltas stay bounded (2-6e-2) through all 45 layers — "amplification via
   mHC mixing, no blow-up signature of a logic error". Root cause pinned to the TP/EP all-reduce
   not being row-invariant (their section 11). Our near-tie attribution (logprob-gap evidence)
   is consistent; their boundary-diagnostic tooling (`diag_window.sh`, `diag_compare.py`, the
   layer_diagnostic hooks) would upgrade our evidence from logprob-level to boundary-level.
2. **Concurrency with the draft graph** (their MTP-4, 80-lane graphs, GMU 0.35): aggregate
   15.5 / 56.0 / 106.7 / 185.9 tok/s at 1/4/8/16 streams — near-linear; the cost is
   single-stream (15.7 vs 21 at MAXSEQ=8; every step pads to 80 lanes). Quantifies the A014
   payoff for our 16-stream net-negative finding.
3. **Depth sweep with the draft graph** (their section 26): MTP-4 optimal — accepted 3.33,
   aggregate 26.3; depth 5 gains 0.1 accepted but the 48-lane verify costs more; depth 3 saves
   nothing. Greedy 5/8 vs the depth-4 reference is the verify-shape near-tie class, "not an
   error".
4. **Long prompts**: chunk-with-context steps cost ~1.4 s vs 0.45 s for a fresh 3200-token
   prompt (the draft prompt-cache fill is 5-12% of long-prompt time; the rest is the target's
   chunked prefill with context); ctx-bucket resizing helps but padding is not the main term.
   Their 262K trial (MTP-4): decode at 219K context drops to ~6 tok/s — each speculative lane
   carries the request's whole block table (10240 blocks/step); levers: share the block table
   across lanes, paged-attention kernel throughput. **Directly informs our A024: MTP at the
   production 262K window may be net-negative without those levers.**

## New task from this addendum

- **NEW I015 Adopt boundary-level divergence diagnostics** — `READY`. Their layer_diagnostic
  hooks + diag_compare tooling upgrade near-tie evidence from logprob gaps to boundary-level
  tensors (the proof class their sections 1-3 demonstrate). Strengthens ACCURACY.md and every
  future parity claim.


## Addendum 7: their serving recipe vs ours — config-delta matrix

Their stack: vLLM `3dc7a68ce4` (v0.27.2rc0 + 1041 commits) with the GLM model integrated into
their vllm-gaudi fork; GMU 0.35; MTP-4 default; block-size 128; max-num-batched-tokens 3200.
Their single-stream decode (21-24 tok/s) is SLOWER than ours (30.1 non-MTP / 35.7-36.8 MTP k=1)
— "adopt their fork" would cost per-stream speed; adapting their machinery into our faster base
is the stronger path (their value: the draft-graph stack, KDA spec-state, device sampling).

Config deltas from their recipe, checked against our trees:

| Their setting | In our trees? | Note |
|---|---|---|
| Pinned decode shapes (B021) | yes (adopted into Stage-1) | warmup + determinism |
| `VLLM_BUCKETING_STRATEGY=lin` | **yes** (vllm-gaudi) | untested here — A/B candidate |
| `VLLM_COMPACT_GDN=0` | **yes** (vllm-gaudi) | untested here — A/B candidate |
| `--enable-ep-weight-filter` | **yes** (vLLM core) | untested here — A/B candidate |
| `VLLM_KDA_SCAN=parallel` | equivalent exists (`GLM5_NEXT_KDA_SCAN=1`, set) | same feature, different name |
| `PT_HPU_LAZY_ACC_PAR_MODE=1` | no | Habana plugin env; test if our plugin accepts it |
| `PT_HPU_MEMORY_POOL=none` | no | ditto |
| `PT_HPU_GPT_MOE_WT_INTERLEAVED=0` | no | ditto |
| `VLLM_GLM_FUSED_CLAMP_MOE=1` + `VLLM_GLM_TPC_CLAMP` | no | their TPC clamp kernel — feature gap |
| `VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS` + ctx buckets | no | their long-prompt graph fix |

New task: a config-delta A/B sweep of the three "yes" candidates (one variable at a time, per
the discipline), then the Habana-env probes.


## Addendum 8: Habana runtime env audit of our live deployment

Checked against the live engine process and our v0.26.0 plugin source:

- **Auto-pinned by our plugin at import (verified in source)**: `PT_HPU_WEIGHT_SHARING=0` and
  `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` — both required for multi-rank HPU-graph serving.
  Matches the reference recipe.
- **`PT_HPU_LAZY_ACC_PAR_MODE`**: a Habana-bridge flag (vendor source: `AccThread`
  parallel-accumulation thread, default enabled with lazy mode) — the same AccThread our
  prefill profiling identified as the busy-wait core. The reference recipe pins `1` explicitly;
  our deployment does not set it and the bridge default is enabled, so behavior should match.
  Worth one explicit pin in the launch scripts for determinism of the recipe.
- **`PT_HPU_MEMORY_POOL=none`**: reference "validated-recipe value (Synapse-side pool off)".
  Not set by our plugin and not in our launch env — an untested A/B candidate for our
  deployment (needs a window).
- **`PT_HPU_GPT_MOE_WT_INTERLEAVED=0`**: the reference auto-pins it because a mismatch with
  their FUSED clamped-SwiGLU MoE kernel is SILENT corruption ("no error, just wrong expert
  math"). Our v0.26.0 plugin has no fused clamped-SwiGLU MoE kernel (that is their addition)
  and our stock MoE path passed all correctness gates — no exposure at our release. Revisit if
  we ever adopt their fused MoE kernel.
- **Their TPC clamp kernel** (`VLLM_GLM_TPC_CLAMP`/`VLLM_GLM_FUSED_CLAMP_MOE`): a feature gap
  in our release (absent from our trees). Their own measurement says arming it changed their
  decode "by ~nothing" at their shape, while the recipe table calls it "~2x on GLM-5.3 decode"
  — contradictory upstream evidence; treat as unproven until measured.

Actionable: add `PT_HPU_LAZY_ACC_PAR_MODE=1` and `PT_HPU_MEMORY_POOL=none` to both serve
scripts as explicit recipe pins (no behavior change expected for the first; untested for the
second), and re-verify at the next window.


## I015 porting assessment (reference fetched)

The reference diagnostics live as flat modules at 919ec98f:
`vllm_gaudi/v1/worker/layer_diagnostic.py` (427 lines: ticket-based capture dirs, position
filtering via `VLLM_DIAG_LAYER_POSITIONS`, boundary dumps for attention/mHC/FFN, KDA boundary
and sequence hooks, cache-written tracking, `VLLM_DEBUG=fwd` phase tracing) and
`vllm_gaudi/v1/worker/kda_state_diagnostic.py` (199 lines: KDA recurrent-state dumps). The
runner integrates them in `prepare(runner, path, context, tokens, positions, metadata,
logits_indices, mark_step, rank, baseline_graphs)` plus capture roles via
`VLLM_DIAG_SAMPLER_DIR` (must not exist at boot) and `VLLM_DEBUG=fwd`.

Porting scope to our b059f9c tree: the capture machinery is largely runner-agnostic; the
integration points (where `prepare` and the boundary hooks attach in the forward path) need
adapting to our runner — which already carries our MTP verify paths, so the highest-value
attach points are our verify forward and the sampler (exactly where our near-tie evidence
lives). Reference copies for the port are in the operator workspace at
`research/abotsis-919ec98f/ldiag/` (not committed — third-party code; the port would carry
attribution per Apache-2.0).

Sequencing: after Stage-1 and A014 — the diagnostics attach to the same forward paths those
changes touch, and the maintenance windows to exercise them already exist then.


## Addendum 9: remaining reference findings, mapped

- **Prefill is device-bound on small fp32 TPC kernels** (after their decay-dot fix);
  "attention and the mHC mixing are minor at every bucket". Their forward wall: ~80 ms floor of
  per-layer graph launches + ~0.1 ms/token. Rewriting the mHC Sinkhorn or the KDA chunk length
  at the op level did not move it (their findings doc section 20). Bounds our C-series
  expectations: after C015, prefill gains come from bucket sizing and launch-overhead work, not
  kernel rewrites.
- **Prompt-query bucket edges cost 30-50% TTFT** (a prompt just over an edge pays the next
  bucket): "size `VLLM_PROMPT_QUERY_BUCKET_*` to the workload". Runbook item for our production
  launch (our buckets are the default ladder).
- **`VLLM_KDA_SCAN=parallel`**: 7% TTFT at the 3200 bucket, neutral below, clean gates —
  equivalent to our `GLM5_NEXT_KDA_SCAN=1` (already enabled). No gap.
- **`VLLM_HPU_GRAPH_ASYNC_REPLAY=1`**: untested lever, absent from our trees.
- **`VLLM_HPU_DECODE_TENSOR_CACHE`**: on for glm5_next WITHOUT spec decode, off under it (the
  doubled graph count exceeds host memory at TP=8) — matches our MTP HBM situation; no gap.
- **Warmup under speculation runbook**: decode buckets must be warmed AS verify batches (else
  the first real request per shape compiles for tens of seconds); the draft's prompt-cache fill
  (per prompt bucket) and attention graph (per decode shape) compile on first use (2-8 s each)
  — run a warm client after /health covering the served prompt lengths. Our Stage-1
  orchestrator does one warmup rep; production should adopt the warm-client runbook.
- **FP8 all-reduce contiguity**: their fix stops round-tripping `orig_M`/`orig_N` through the
  device. Our `hpu_fp8.py` uses them only in the weight-dequant path (not the forward
  all-reduce), so the exact fix does not apply; verify our forward all-reduce tensor handling
  during the Phase-10 profiling pass (unverified check).
