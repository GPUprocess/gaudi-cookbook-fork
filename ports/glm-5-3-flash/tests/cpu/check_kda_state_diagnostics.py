#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU test for the KDA state diagnostics: sentinel gate, capture, payload, cap."""
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_state_diagnostics as ksd  # noqa: E402

fails = []


def check(name, cond):
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)


class Glm5NextHPUKDAAttention:
    def __init__(self, pool_rows, pool_dim, device="cpu"):
        self.kv_cache = [torch.zeros(pool_rows, pool_dim, device=device),
                         torch.zeros(pool_rows, pool_dim, device=device)]
        self.cache_group_idx = torch.tensor([0], device=device)


class StubLayer:
    def __init__(self):
        self._glm_state_calls = 0
        self._glm_state_bytes = 0
        self.kda_attn = Glm5NextHPUKDAAttention(64, 128)

    def named_modules(self):
        yield "self_attn", self.kda_attn


def make_metadata(is_prompt=True):
    fields = {f: None for f in ksd.FIELDS}
    fields["load_indices_tensor"] = torch.arange(8)
    fields["store_indices_tensor"] = torch.arange(8)
    fields["query_start_loc"] = torch.tensor([0, 8])
    fields["num_accepted_tokens"] = torch.tensor([1])
    fields["seq_lens_tensor"] = torch.tensor([8])
    fields["is_prompt"] = is_prompt
    return SimpleNamespace(**fields)


def main():
    layer = StubLayer()
    d = tempfile.mkdtemp(prefix="glm_kda_state_test_")
    os.chmod(d, 0o700)
    sentinel = os.path.join(d, "STATE_ENABLED")
    with open(sentinel, "w") as f:
        f.write("on")
    os.chmod(sentinel, 0o600)
    os.environ["VLLM_GLM_LAYER_DIAG_DIR"] = d

    tokens = torch.tensor([[42]])
    positions = torch.tensor([[10]])
    metadata = make_metadata(is_prompt=False)

    print("=== sentinel gate ===")
    check("state dir armed with a valid sentinel", ksd.state_directory(layer) == d)
    os.chmod(sentinel, 0o644)
    check("rejects a non-0600 sentinel", ksd.state_directory(layer) is None)
    os.chmod(sentinel, 0o600)

    print("=== capture ===")
    ksd.maybe_capture(layer, metadata, tokens, positions)
    files = [f for f in os.listdir(d) if f.startswith("state-rank")]
    check("payload written", len(files) == 1)
    mode = stat.S_IMODE(os.stat(os.path.join(d, files[0])).st_mode)
    check("payload is 0600", mode == 0o600)
    payload = torch.load(os.path.join(d, files[0]), weights_only=False)
    check("payload schema 1", payload["schema"] == 1)
    check("payload raw_state", payload["raw_state"] is True)
    check("payload phase decode", payload["phase"] == "decode")
    check("metadata maps present", "load_indices_tensor" in payload["metadata"])
    check("one KDA layer found", len(payload["layers"]) == 1)
    pools = payload["layers"][0]["pools"]
    check("conv + recurrent pools", [p["kind"] for p in pools] == ["conv", "recurrent"])
    check("pool shapes recorded", all(p["shape"] == (64, 128) for p in pools))
    check("raw pool rows captured", all(len(p["raw"]) == ksd.MAX_CAPTURES for p in pools))
    check("calls accounted", layer._glm_state_calls == 1)
    check("bytes accounted", layer._glm_state_bytes > 0)

    print("=== capture cap ===")
    for _ in range(ksd.MAX_CAPTURES + 2):
        ksd.maybe_capture(layer, metadata, tokens, positions)
    check("stops at MAX_CAPTURES", layer._glm_state_calls == ksd.MAX_CAPTURES)
    total_files = len([f for f in os.listdir(d) if f.startswith("state-rank")])
    check("payload count == MAX_CAPTURES", total_files == ksd.MAX_CAPTURES)

    print("=== skipped without the sentinel ===")
    os.remove(sentinel)
    layer._glm_state_calls = 0
    ksd.maybe_capture(layer, metadata, tokens, positions)
    check("no capture without the sentinel", layer._glm_state_calls == 0)

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
