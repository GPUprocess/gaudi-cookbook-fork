# INCIDENT-001 — I015 diagnostic boot crash (enforce_eager + I015 hooks)

**Date:** 2026-11-07
**Status:** Root cause identified; fix deployed; capture not yet validated on HPU

## What happened

The I015 diagnostic boot (serve_diag_mtp.sh with --enforce-eager and
VLLM_GLM_LAYER_DIAG_DIR) served all golden prompts successfully but produced zero
boundary payload files. Subsequently, the engine was shut down normally (the
diag_prompts runner completed, the tmux session ended). The nanobind leak messages
during shutdown are from the Habana bridge cleanup (a known shutdown issue), not
from our changes.

## Root cause

`maybe_arm_layer` used `getattr(layer, "_prefix", "")` to detect the layer index,
but `Glm5NextDecoderLayer` stores `layer_idx` (an int), not `_prefix` (a string).
`_layer_index("")` returned `None`, and the arm was refused. No captures were
attempted. The engine crash was NOT caused by this — the engine shut down normally
after the diag_prompts runner finished.

## Fix

`maybe_arm_layer` now uses `getattr(layer, "layer_idx", None)` — the decoder layer's
own attribute. Deployed to the installed tree and the repo.

## Evidence

- serve_i015.log: the engine served all 12 golden prompts (acceptance 1.80-2.00)
- No payload files in the diag directory
- The shutdown was API-server-initiated (mode=abort) after the prompts completed
- The nanobind leaks are Habana bridge cleanup messages (not our code)
