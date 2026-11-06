#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU test for the self-arming layer capture (I015): the decoder-layer hook flow.
No HPU, no server: a stub layer + a temp sentinel dir; the lazy-mode refusal is
tested by the module's own guard (habana import fails on CPU -> refuses to arm),
so the test drives the collector directly for the capture/flush flow and tests
`maybe_arm_layer`'s refusal path."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_diagnostics as ld  # noqa: E402

fails = []


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)


class StubLayer:
    """A minimal stand-in for Glm5NextDecoderLayer (prefix + layer_idx)."""

    def __init__(self, layer_idx, prefix):
        self.layer_idx = layer_idx
        self._prefix = prefix


def main():
    d = tempfile.mkdtemp(prefix="glm_diag_layer_")
    os.chmod(d, 0o700)
    sentinel = os.path.join(d, "LAYER_ENABLED")
    with open(sentinel, "w") as f:
        f.write("on")
    os.chmod(sentinel, 0o600)
    os.environ["VLLM_GLM_LAYER_DIAG_DIR"] = d
    os.environ.pop("VLLM_GLM_LAYER_DIAG_POSITIONS", None)

    layer0 = StubLayer(0, "model.layers.0")
    layer1 = StubLayer(1, "model.layers.1")

    print("=== maybe_arm_layer ===")
    ld.maybe_arm_layer(layer0)
    check("refuses to arm on CPU (habana lazy-mode guard)", ld._collector is None)

    print("=== direct collector flow (the hook path with the guard bypassed) ===")
    # Drive the collector directly: the hooks are what the decoder layer calls.
    from types import SimpleNamespace as _NS
    _runner = _NS(_glm_diag_bytes=0)
    ld._collector = dict(layer=layer0, layer_index=0, forwards=0, runner=_runner,
                         path=d, rank=0, admitted_rows=[0], payload=dict(rank=0, ordinal=0, boundaries=[]))
    t = torch.zeros(2, 8, 16)
    ld.layer_capture(layer0, "attention_raw_output", t)
    check("captures for the owning layer",
          any(b["name"].endswith("attention_raw_output")
              for b in ld._collector["payload"]["boundaries"]))
    ld.layer_capture(layer1, "attention_raw_output", t)
    check("ignores a non-owning layer",
          len(ld._collector["payload"]["boundaries"]) == 1)
    ld.layer_capture(layer0, "attention_mhc_post", t)
    check("captures the mHC post boundary",
          any(b["name"].endswith("attention_mhc_post")
              for b in ld._collector["payload"]["boundaries"]))
    ld.layer_step_done(layer0)
    check("counts the forward", ld.layer_forwards(layer0) == 1)
    saved_max = ld.MAX_FORWARDS
    ld.MAX_FORWARDS = 3
    ld.layer_step_done(layer0)
    ld.layer_step_done(layer0)
    ld.MAX_FORWARDS = saved_max
    check("auto-flushed at MAX_FORWARDS", ld._collector is None)
    files = [f for f in os.listdir(d) if f.startswith("layer-rank")]
    check("payload written", len(files) == 1)
    mode = os.stat(os.path.join(d, files[0])).st_mode
    import stat as _stat
    check("payload is 0600", _stat.S_IMODE(mode) == 0o600)

    print("=== budget ===")
    ld._collector = dict(layer=layer0, layer_index=0, forwards=0,
                         admitted_rows=[0], payload=dict(boundaries=[], truncated=False),
                         bytes_used=ld.MAX_BYTES - 1, runner=SimpleNamespace(_glm_diag_bytes=ld.MAX_BYTES - 1))
    big = torch.zeros(2, 8, 16)
    ld.layer_capture(layer0, "over_budget", big)
    check("marks truncated over budget", ld._collector["payload"]["truncated"] is True)
    ld.disarm()

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
