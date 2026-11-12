# Implementation feedback: the DFlash2 (RA-009) port state — 2026-11-11

**From:** the implementation agent (the Gaudi2 server, g2, the GLM-5.3-Flash TP8 production stack)
**To:** the research agent
**Re:** the owner's directive — "get DFlash2 working and make it predict at least 4 tokens"
**Status:** the port is STAGED at the ~40% point; the hard blockers are identified and quantified below.

## What is done and verified

1. **The checkpoint is on the server.** `incoai/GLM-5.3-Flash-DFlash2` downloaded to
   `/data/models/GLM-5.3-Flash-DFlash2` (2.2 GB; config verified: block 8, selector_top_k 16,
   selector_rank 256, mask_token_id 154856, target_layer_ids [5,14,24,33,42], 5 layers,
   sliding_window 2048, is_causal false, model_type qwen3, vocab 154880).
2. **The flashinfer_gaudi package is ported minimal.** Copied from the 1Cat checkout into
   `/opt/glm53/vllm-gaudi/flashinfer_gaudi/` with a stripped `__init__` (dflash2 + _native +
   _tactics only — the full package imports the 1Cat's compiled GDN ops that we do not have).
   All the DFlash2 imports verified: `build_attention_bias`, `prepare_device_inputs`,
   `score_and_select_path_reference`, `grouped_conv_reference`, the tactics manifest.
   `score_select native promoted: False` — the reference fallback is the active path (the
   compiled `flashinfer_gaudi_ops*.so` does not exist anywhere on this server: the 1Cat ships
   the source but never built it).
3. **The proposer and the ops are copied and GLM-swapped.** `vllm_gaudi/v1/spec_decode/
   hpu_dflash2.py` (the 421-line HpuDFlash2Proposer, an HpuEagleProposer subclass) and
   `vllm_gaudi/ops/hpu_dflash2.py` copied from the 1Cat; the two `qwen3_dflash` imports swapped
   to `laguna_dflash` (our stack's GLM DFlash model — `DFlashLagunaModel(DFlashQwen3Model)`, so
   the helper functions `dflash_has_any_non_causal` / `dflash_target_rope_is_neox_style` are
   inherited and the swap is mechanical).
4. **The drafter factory is wired.** `hpu_model_runner.py`'s speculative factory got the
   `method == "dflash"` branch ahead of the `use_eagle()` branch (upstream's `use_eagle()`
   returns true for dflash, which would have picked the wrong proposer class), setting
   `use_aux_hidden_state_outputs` from the proposer.

## The blockers, ranked by severity

### B1 — the GDN rollback machinery is 1Cat-compiled and absent here (the hard gate)
The 1Cat's DFlash2 requires `VLLM_HPU_FLASHINFER_GDN=1` + `VLLM_COMPACT_GDN=1` (its
`_validate_hpu_dflash2_config` raises without them) because DFlash2's block verification mutates
recurrent state and needs per-token rollback checkpoints. That machinery lives in the 1Cat's
compiled ops (`vllm_gaudi.ops.gdn_recurrent_compound` — imported by the flashinfer_gaudi bridge
chain, absent from our venv). **Our GLM-5.3-Flash port has its own KDA candidate-slot machinery**
(the MTP port uses it at k=1; the layout `[groups, bs, 1+num_spec+1]` is generic), so the
integration question is: does the DFlash2 proposer consume the 1Cat's GDN rollback API, or can it
ride our KDA candidate-slot path at k=7? The 1Cat proposer code imports `HpuEagleProposer` (which
we have) but the verify path needs the state-rollback call sites mapped. **This is the
first question for the next session: map `hpu_dflash2.py`'s state-rollback call sites against our
`_kda_spec_decode` candidate-slot implementation.**

### B2 — the upstream V1 rejection needs our platform gate
Upstream V1 rejects "dflash2 drafts" (the candidate selector lives in the V2 speculator; Gaudi
has no Triton). The 1Cat patches `VllmConfig._get_v1_model_runner_unsupported_features` to remove
exactly that rejection when `VLLM_HPU_FLASHINFER_DFLASH2=1` and the draft is DFlash2. Our
`platform.py` needs this monkey-patch ported (plus the `VLLM_HPU_FLASHINFER_DFLASH2` env in our
envs module). Mechanical, ~20 lines.

### B3 — the TP8 draft sharding is unmeasured
The 1Cat qualified DFlash2 at TP1, ≤16 seqs. Our production is TP8: the 1-2B draft sharded
8-way (the hidden 4096 → 512/rank) has no precedent in the 1Cat tree, and the DFlash2
convolution + selector may not shard cleanly (the conv is per-channel; the selector is
rank-256 over the full hidden — check the sharding sensitivity). If the draft does not shard,
the fallback is TP1-draft-inside-TP8-target (the draft replicated: +2 GB HBM per rank — the D13
gate at 32 seqs).

### B4 — the HBM gate (D13) quantification
The KDA per-layer state rollback at k=7: 8 candidate checkpoints per request per layer × 45
layers. The 1Cat bound was ≤16 seqs at k=7 with the GDN states; our KDA state size differs.
The quantification needs the per-layer state bytes from our runner's KV-layout (the D13
measurement protocol from the research doc).

### B5 — the license
CC BY-NC-ND 4.0: **non-commercial**. The owner needs to rule on this before any production
use. DSpine (RA-016, Apache 2.0) is the clean-license alternative but has no GLM checkpoint yet.

## What the next session should do, in order

1. **Map the state-rollback call sites** (B1): read `hpu_dflash2.py`'s verify path against our
   `_kda_spec_decode`; decide "the 1Cat GDN API" vs "our KDA candidate slots".
2. **Port the platform gate** (B2, mechanical): the monkey-patch + the env.
3. **Boot the staged arm** (`/data/glm53/boot_dflash.sh` — written, staged, expected to fail at
   B1/B2 until those land; the failures name the next gaps).
4. **Quantify the D13 HBM** at k=7 × 45 layers × 32 seqs against our KV-layout.
5. **The TP8 draft sharding decision** (B3) after the first successful boot.
6. **The suite gate** at ≥97.5% (the owner's line) with the real per-block scales.

## The instrumentation note

The F4-TS per-phase timestamp patch (`patch_f4_ts.py`, applied to multiproc_executor.py and
core.py, env `GLM_PHASE_TS=1`) is live on the current boot and will measure the DFlash2
propose-phase cost the same way it measured the MTP draft-host term. Keep it for the DFlash2
A/B; it is env-gated and costs nothing when off.
