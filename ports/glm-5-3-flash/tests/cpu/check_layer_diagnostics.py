#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU test for the layer-diagnostics core: sentinel gate, admission, history
validation, budget, secure flush, nesting. No HPU, no server."""
import os
import stat
import sys
import tempfile

import torch
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_diagnostics as ld  # noqa: E402

fails = []


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)


def make_runner():
    return SimpleNamespace(_glm_diag_calls=0, _glm_diag_bytes=0,
                           _glm_diag_admitted=None, _glm_diag_seen=None)


def make_dir():
    d = tempfile.mkdtemp(prefix="glm_diag_test_")
    os.chmod(d, 0o700)
    sentinel = os.path.join(d, "LAYER_ENABLED")
    with open(sentinel, "w") as f:
        f.write("on")
    os.chmod(sentinel, 0o600)
    return d


def main():
    runner = make_runner()
    d = make_dir()
    runner._glm_layer_diag_dir = d
    os.environ["VLLM_GLM_LAYER_DIAG_POSITIONS"] = "0,1,2"
    prompt = list(range(100, 110))  # 10 prompt tokens

    print("=== sentinel gate ===")
    t = ld.arm(runner)
    check("arms with a valid sentinel", t is not None)
    ld.disarm()
    os.chmod(os.path.join(d, "LAYER_ENABLED"), 0o644)
    check("rejects a non-0600 sentinel", ld.arm(runner) is None)
    os.remove(os.path.join(d, "LAYER_ENABLED"))
    check("rejects a missing sentinel", ld.arm(runner) is None)
    os.chmod(d, 0o755)
    check("rejects a non-0700 dir", ld.arm(runner) is None)
    os.chmod(d, 0o700)
    with open(os.path.join(d, "LAYER_ENABLED"), "w") as f:
        f.write("on")
    os.chmod(os.path.join(d, "LAYER_ENABLED"), 0o600)
    t = ld.arm(runner)
    check("re-arms after restoration", t is not None)

    print("=== admission and history ===")
    r1 = ld.admit("req-1", 0, prompt, [200])
    check("admits req-1 at position 0", r1 is not None and r1["last_token"] == 200)
    check("records the absolute position", r1["absolute_position"] == 10)
    r2 = ld.admit("req-1", 1, prompt, [200, 201])
    check("admits req-1 at position 1", r2 is not None and r2["last_token"] == 201)
    check("dedups seen (request, position)",
          ld.admit("req-1", 1, prompt, [200, 201]) is None)
    check("admits a second request at the first allowed position (<= 3 requests)",
          ld.admit("req-2", 0, prompt, [300]) is not None)
    check("admits req-2 at position 2",
          ld.admit("req-2", 2, prompt, [300, 301, 302]) is not None)
    check("rejects a position outside the allowed list",
          ld.admit("req-3", 5, prompt, [400]) is None)
    check("rejects a NEW request off the first allowed position",
          ld.admit("req-3", 1, prompt, [400]) is None)
    check("admits a third request at the first allowed position",
          ld.admit("req-3", 0, prompt, [400]) is not None)
    check("rejects a fourth request (MAX_REQUESTS)",
          ld.admit("req-4", 0, prompt, [400]) is None)
    try:
        ld.admit("req-3", 2, prompt, [])  # empty output at position 2
        fails.append("empty output history did not raise")
    except ValueError:
        check("raises on an empty output history", True)

    print("=== budget and boundary ===")
    t0 = torch.zeros(4, 8, 16)
    ld.boundary("attn_raw_output", t0, row_indices=[0, 2])
    check("boundary captured with the admitted rows",
          any(b["name"] == "attn_raw_output" and b["rows"].shape == (2, 8, 16)
              for b in ld._collector["payload"]["boundaries"]))
    ld.boundary("out_of_range", t0, row_indices=[0, 99])
    check("rejects out-of-range rows",
          any(u["name"] == "out_of_range" for u in ld._collector["payload"].get("unavailable", [])))

    print("=== flush ===")
    ld.flush()
    files = [f for f in os.listdir(d) if f.startswith("layer-rank")]
    check("payload written", len(files) == 1)
    mode = stat.S_IMODE(os.stat(os.path.join(d, files[0])).st_mode)
    check("payload is 0600", mode == 0o600)
    import torch as _t
    payload = _t.load(os.path.join(d, files[0]), weights_only=False)
    check("payload schema 1", payload["schema"] == 1)
    check("payload has 1 boundary", len(payload["boundaries"]) == 1)
    check("bytes accounted", runner._glm_diag_bytes > 0)
    check("calls accounted", runner._glm_diag_calls >= 1)

    print("=== nesting ===")
    try:
        ld.arm(runner)
        ld.arm(runner)
        fails.append("nested arm did not raise")
    except RuntimeError:
        check("nested arm raises", True)
    ld.disarm()

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
