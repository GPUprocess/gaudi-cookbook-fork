#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe-layout for the GLM-5.3-Flash FP8 port (Q17): the EP4xTP2 group layout.

    python3 moe_layout_fp8_patch.py [--check] [--root SITE_PACKAGES_PARENT]
Exit: 0 applied/already, 1 not applied (anchors match), 2 drifted.

Two targets (the mxfp4 moe_layout patch adapted; the MXFP4 loader/router parts
are N/A for the FP8 path):
  1. vllm/model_executor/layers/fused_moe/config.py - the group layout in
     FusedMoEParallelConfig.make's EP branch (GLM_MOE_TP=t forms rank groups).
  2. vllm/model_executor/layers/fused_moe/routed_experts.py - the load_fused
     group-slice hook at the top of load_weights' is_fused branch (our combined
     [E, d1, d2] w13/w2 checkpoint tensors).
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[glm-5-3-flash:moe-layout]"

CFG_OLD = """\
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        ep_size = tp_size
"""
CFG_NEW = """\
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        from glm5_next_hpu import moe_layout as _glm_moe  # [glm-5-3-flash:moe-layout] GLM_MOE_TP

        _glm = _glm_moe.group_layout(tp_size, tp_rank, dp_size, pcp_size)
        if _glm is not None:
            return FusedMoEParallelConfig(
                tp_size=_glm[0],
                tp_rank=_glm[1],
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                ep_size=_glm[2],
                ep_rank=_glm[3],
                sp_size=sp_size_,
                use_ep=True,
                all2all_backend=vllm_parallel_config.all2all_backend,
                enable_eplb=vllm_parallel_config.enable_eplb,
            )
        ep_size = tp_size
"""

RT_OLD = """\
                if is_fused:
                    # w1 and w3 share one fused tensor; use a local copy so the
"""
RT_NEW = """\
                if is_fused:
                    from glm5_next_hpu import moe_layout as _glm_moe  # [glm-5-3-flash:moe-layout]

                    if _glm_moe.load_fused(self, param, loaded_weight, shard_id):
                        matched = True
                        continue
                    # w1 and w3 share one fused tensor; use a local copy so the
"""


def locate(pkg, rel, root=None):
    if root:
        p = Path(root) / pkg / rel
        return p if p.is_file() else None
    spec = importlib.util.find_spec(pkg)
    for loc in (spec.submodule_search_locations or []) if spec else []:
        p = Path(loc) / rel
        if p.is_file():
            return p
    return None


def targets(root=None):
    return [
        ("vllm", "model_executor/layers/fused_moe/config.py", CFG_OLD, CFG_NEW, "config"),
        ("vllm", "model_executor/layers/fused_moe/routed_experts.py", RT_OLD, RT_NEW, "loader"),
    ]


def run(check, root):
    ok = True
    for pkg, rel, old, new, label in targets(root):
        path = locate(pkg, rel, root)
        if path is None:
            print(f"{MARKER} {pkg}/{rel} not found", file=sys.stderr)
            return 2
        src = path.read_text()
        if MARKER in src:
            print(f"{MARKER} {label}: already applied")
            continue
        if src.count(old) != 1:
            print(f"{MARKER} {label}: anchor found {src.count(old)} times (drift)", file=sys.stderr)
            ok = False
            continue
        if check:
            print(f"{MARKER} {label}: not applied (anchor matches)")
            continue
        path.write_text(src.replace(old, new, 1))
        cache = path.parent / "__pycache__"
        if cache.is_dir():
            for pyc in cache.glob("*.pyc"):
                pyc.unlink(missing_ok=True)
        print(f"{MARKER} {label}: patched {path}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=None, help="parent dir holding vllm/")
    a = ap.parse_args()
    return run(a.check, a.root)


if __name__ == "__main__":
    sys.exit(main())
