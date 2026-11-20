#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q21-moe wiring for the GLM-5.3-Flash port (Q21 step 3): the fused routed-expert
two-pass TPC path at decode shapes.

    python3 q21_moe_wiring.py [--check] [--root SITE_PACKAGES_PARENT of vllm_gaudi]
Exit: 0 applied/already, 1 not applied (anchors match), 2 drifted.

Target: vllm_gaudi/ops/hpu_fused_moe.py
  1. The helper functions (_q21_moe_active / _q21_two_pass_moe) before
     _normalize_moe_activation.
  2. The elif branch at the standard-activation moe_op call site.

Runtime gates: GLM_Q21_MOE=1, x.shape[0] <= GLM_Q21_MOE_MAX_TOKENS (default 32),
the FP8 weights retained on the layer, both custom ops registered. Anything else
stays on the stock path (fail closed). The Q21 kernels: pass-1 = the block-FP8
dequant GEMV raw[T,K,2I] (the -1 sentinel = the non-local EP pairs, contributing
0 through the existing TP all-reduce); the between math = SiLU(gate)*up*weight;
pass-2 = the accumulate over the K routed experts. Gate: the 12-stage HPU
correctness gate PASS (pass_1 1.2e-4, pass_2 3.6e-8 vs the dequant reference).
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[glm-5-3-flash:q21-moe]"

HELPERS_OLD = """\
def _normalize_moe_activation(activation):"""

HELPERS_NEW = """\
def _q21_moe_active(layer, tokens):
    \"\"\"[glm-5-3-flash:q21-moe] The fail-closed gate for the Q21 two-pass path.\"\"\"
    if os.environ.get("GLM_Q21_MOE", "0") != "1":
        return False
    if tokens > int(os.environ.get("GLM_Q21_MOE_MAX_TOKENS", "32")):
        return False
    w13 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    if w13 is None or w2 is None:
        return False
    if w13.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return False
    if w2.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return False
    ops = getattr(torch.ops, "custom_op", None)
    return (ops is not None and hasattr(ops, "block_fp8_moe_w13_glm")
            and hasattr(ops, "block_fp8_moe_w2_glm"))


def _q21_two_pass_moe(layer, x, topk_ids, topk_weights):
    \"\"\"[glm-5-3-flash:q21-moe] The Q21 fused routed-expert two-pass (decode shapes).

    Pass 1 (TPC): raw[t,k,2I] = W13[e] . x with the in-kernel block-FP8 dequant;
    the -1 sentinel pairs (the non-local EP experts) contribute 0 and combine via
    the existing TP all-reduce. Between math: SiLU(gate) * up * routing weight.
    Pass 2 (TPC): y[t] = sum_k W2[e] . act.
    \"\"\"
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    e_local = layer.w13_weight.shape[0]
    local_ids = topk_ids - experts_min
    local_ids = torch.where((local_ids >= 0) & (local_ids < e_local), local_ids,
                            torch.full_like(local_ids, -1)).to(torch.int32)
    raw = torch.ops.custom_op.block_fp8_moe_w13_glm(
        layer.w13_weight, layer.w13_weight_scale, x, local_ids)
    i = raw.shape[-1] // 2
    act = (torch.nn.functional.silu(raw[..., :i].float()) * raw[..., i:].float()
           * topk_weights[..., None].float()).to(torch.bfloat16)
    return torch.ops.custom_op.block_fp8_moe_w2_glm(
        layer.w2_weight, layer.w2_weight_scale, act, local_ids)


def _normalize_moe_activation(activation):"""

CALL_OLD = """\
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:"""

CALL_NEW = """\
        elif _q21_moe_active(layer, x.shape[0]):                     # [glm-5-3-flash:q21-moe]
            output = _q21_two_pass_moe(layer, x, topk_ids, topk_weights)
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:"""


def find_root() -> Path:
    for cand in [Path("/opt/glm53/vllm-gaudi"),
                 Path.home() / "publish_work/fork"]:
        if (cand / "vllm_gaudi" / "ops" / "hpu_fused_moe.py").is_file():
            return cand
    raise SystemExit("vllm_gaudi tree not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else find_root()
    target = root / "vllm_gaudi" / "ops" / "hpu_fused_moe.py"
    src = target.read_text()
    if MARKER in src:
        print(f"{MARKER} already applied at {target}")
        return 0
    if HELPERS_OLD not in src:
        print(f"{MARKER} DRIFT: the helper anchor not found in {target}")
        return 2
    if CALL_OLD not in src:
        print(f"{MARKER} DRIFT: the call-site anchor not found in {target}")
        return 2
    if args.check:
        print(f"{MARKER} anchors match, not applied")
        return 1
    src = src.replace(HELPERS_OLD, HELPERS_NEW, 1)
    src = src.replace(CALL_OLD, CALL_NEW, 1)
    target.write_text(src)
    print(f"{MARKER} applied at {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
