#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""router-fast: skip the grouped top-k group stage when it is an identity.

GLM-5.3-Flash routes with n_group=1, topk_group=1 (verified in config.json).
In vLLM's `grouped_topk` (cpu_fused_moe.py), the group stage
(group_scores -> group_idx -> group_mask -> score_mask -> masked_fill) then
selects EVERY expert: with one group, `score_mask` is all-true and
`tmp_scores = scores.masked_fill(~all_true, -inf)` is exactly `scores`.
The stage costs several op launches per MoE layer (~25 us/layer on HPU per the
cookbook's GLM-5.3 MoE decode study) for zero effect.

GLM_ROUTER_FAST=1 short-circuits `tmp_scores = scores` in that case.
Bit-identical by construction (masked_fill with an all-true mask is an
identity op); env-gated default OFF per the project's HPU-verify-first policy.
Reference: ports/glm-5-mxfp4/patches/decode_lean.py (GLM_ROUTER_FAST);
docs/results/moe-decode-step.md ("Router fast path (about 1.7 ms)").

Usage: router_fast.py [--check] [--root SITE_PACKAGES]
Exit codes: 0 applied or already applied, 1 (--check) not applied, 2 drifted.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patchlib import main, replace_once  # noqa: E402

MARKER = "[gaudi-port:glm-5-3-flash/router-fast]"

OLD = """    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
"""

NEW = """    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
    # [gaudi-port:glm-5-3-flash/router-fast] GLM_ROUTER_FAST=1: with one expert
    # group and topk_group == 1 the group stage below selects every expert and
    # its masked_fill is an identity; skip it (bit-identical, saves the
    # group_scores/topk/scatter/expand/masked_fill launches per MoE layer).
    if (
        _GLM_ROUTER_FAST
        and num_expert_group == 1
        and topk_group == 1
    ):
        tmp_scores = scores  # identity: the single group is always selected
    else:
        if e_score_correction_bias is not None:
            group_scores = (
                scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )
        else:
            group_scores = (
                scores.view(num_token, num_expert_group, -1).max(dim=-1).values
            )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
"""

ENV_DEF = '''# [gaudi-port:glm-5-3-flash/router-fast] skip the identity group stage
_GLM_ROUTER_FAST = os.environ.get("GLM_ROUTER_FAST", "0") == "1"
'''


def patch(src: str) -> str:
    src = replace_once(src, "def grouped_topk(", ENV_DEF + "\ndef grouped_topk(", "env-def")
    src = replace_once(src, OLD, NEW, "group-stage")
    if MARKER not in src:
        src = src.rstrip("\n") + "\n\n# " + MARKER + "\n"
    return src


def _load_patchlib():
    spec = importlib.util.spec_from_file_location(
        "patchlib", Path(__file__).resolve().parent / "patchlib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def locate_root(args_root: str | None) -> Path:
    if args_root:
        return Path(args_root)
    for cand in ("/opt/glm53/vllm",):
        if (Path(cand) / "vllm" / "model_executor" / "layers" / "fused_moe" / "cpu_fused_moe.py").exists():
            return Path(cand)
    import vllm  # noqa: F401
    return Path(vllm.__file__).resolve().parent.parent


def do_check(root: Path) -> int:
    target = root / "vllm" / "model_executor" / "layers" / "fused_moe" / "cpu_fused_moe.py"
    src = target.read_text()
    if MARKER in src:
        print("[gaudi-port:glm-5-3-flash/router-fast] already applied")
        return 0
    try:
        patch(src)
    except SystemExit:
        print("[gaudi-port:glm-5-3-flash/router-fast] anchors do not match (drifted)")
        return 2
    print("[gaudi-port:glm-5-3-flash/router-fast] not applied (anchors match)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = locate_root(a.root)
    if a.check:
        return do_check(root)
    target = root / "vllm" / "model_executor" / "layers" / "fused_moe" / "cpu_fused_moe.py"
    src = target.read_text()
    if MARKER in src:
        print("[gaudi-port:glm-5-3-flash/router-fast] already applied")
        return 0
    patched = patch(src)
    target.write_text(patched)
    print(f"[gaudi-port:glm-5-3-flash/router-fast] patched {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
