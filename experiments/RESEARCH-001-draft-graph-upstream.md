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
