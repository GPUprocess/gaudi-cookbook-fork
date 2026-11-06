#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""skip-layer-markstep: skip the per-DecoderLayer mark_step hook for glm5_next under HPU-graph replay.

The vllm-gaudi runner installs a forward hook that calls `mark_step()` after every decoder
layer (`modify_model_layers`, interval from `VLLM_CONFIG_HIDDEN_LAYERS`, default 1). Under
HPU-graph replay the cuts the hook makes are captured and replayed forever, and each replayed
recipe boundary costs ~0.2-0.5 ms of host time per decode step. For glm5_next the hook fires
after the MoE all-reduce and the trailing mHC math, splitting that math into its own recipe in
every layer: 45 of 176 launches per decode step. The upstream reference measured skipping it
as bit-identical (greedy parity 8/8) and +10% single-stream decode.

EXPERIMENTAL in this port: gated behind `VLLM_GLM_SKIP_LAYER_MARKSTEP=1` (default 0 keeps the
hook) until verified on HPU in this tree. The upstream skip condition is honored: lazy mode,
not enforce_eager, model_type glm5_next/glm5_next_text, `VLLM_CONFIG_HIDDEN_LAYERS` unset.
Target: vllm_gaudi/v1/worker/hpu_model_runner.py.

Usage: skip_layer_markstep.py [--check] [--root DIR_HOLDING_vllm_and_vllm_gaudi]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patchlib import main, replace_once

MARKER = "[gaudi-port:glm-5-3-flash/skip-layer-markstep]"

OLD = """        try:
            hidden_layer_markstep_interval = int(os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
        except ValueError:
            logger.warning("Invalid VLLM_CONFIG_HIDDEN_LAYERS value, using default 1")
            hidden_layer_markstep_interval = 1
        model_config = getattr(self.model, "config", None)
        modify_model_layers(self.model,
                            get_target_layer_suffix_list(model_config.model_type if model_config is not None else None),
                            hidden_layer_markstep_interval)
"""

NEW = """        try:
            hidden_layer_markstep_interval = int(os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
        except ValueError:
            logger.warning("Invalid VLLM_CONFIG_HIDDEN_LAYERS value, using default 1")
            hidden_layer_markstep_interval = 1
        model_config = getattr(self.model, "config", None)
        # [gaudi-port:glm-5-3-flash/skip-layer-markstep] Under HPU-graph replay the
        # per-DecoderLayer mark_step cuts are captured and replayed forever; each replayed
        # recipe boundary costs host time per decode step (for glm5_next: 45 of 176 launches).
        # Skipping the hook is bit-identical on the reference tree (greedy parity 8/8) and
        # measured +10% single-stream decode there. Experimental here: default OFF until
        # verified on HPU in this tree; VLLM_GLM_SKIP_LAYER_MARKSTEP=1 enables the skip,
        # and an explicit VLLM_CONFIG_HIDDEN_LAYERS always wins over the skip.
        _skip_layer_markstep = (
            os.getenv("VLLM_GLM_SKIP_LAYER_MARKSTEP", "0") == "1"
            and os.getenv("VLLM_CONFIG_HIDDEN_LAYERS") is None
            and htorch.utils.internal.is_lazy()
            and not getattr(self.model_config, "enforce_eager", False)
            and getattr(model_config, "model_type", None) in ("glm5_next", "glm5_next_text")
        )
        if _skip_layer_markstep:
            logger.info(
                "[glm5_next] per-DecoderLayer mark_step hook disabled under HPU-graph replay "
                "(VLLM_GLM_SKIP_LAYER_MARKSTEP=1)"
            )
        else:
            modify_model_layers(self.model,
                                get_target_layer_suffix_list(model_config.model_type if model_config is not None else None),
                                hidden_layer_markstep_interval)
"""


def patch(src: str) -> str:
    src = replace_once(src, OLD, NEW, "mark_step interval block + modify_model_layers call")
    if MARKER not in src:
        src = src + "\n# " + MARKER + "\n"
    return src


TARGETS = (("vllm_gaudi", "v1/worker/hpu_model_runner.py", patch),)

if __name__ == "__main__":
    sys.exit(main(__doc__, MARKER, TARGETS))
