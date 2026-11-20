#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q21-hpu-fp8 wiring (Q21 step 3c — THE CORRECT SITE): the fused routed-expert
two-pass TPC path inside HPUFp8MoEMethod.apply_monolithic (vllm_gaudi/ops/hpu_fp8.py).

    python3 q21_hpu_fp8_wiring.py [--check] [--root /opt/glm53/vllm-gaudi]
Exit: 0 applied/already, 1 not applied (anchors match), 2 drifted.

The runtime-path proof (the layer-level DIAG, boot10): RoutedExperts.forward_monolithic →
qmethod=HPUFp8MoEMethod, w13 = [36, 4096, 4096] torch.float8_e4m3fn (36 local experts,
FP8 retained — the loader's load_fused copies the checkpoint bytes directly, no dequant).
The earlier patch sites (vllm fp8.py Fp8MoEMethod, the vllm_gaudi unquantized method) are
dead code for this model.

The swap point: inside apply_monolithic AFTER the routing computes topk_ids/topk_weights
(the same select_experts_from_routed the stock path uses) and before the stock
_gaudi_fix_moe_gather_apply. The EP mapping: layer.expert_map[global_id] -> the local id
or -1 (the Q21 kernels' -1 sentinel; the non-local pairs contribute 0 and the existing TP
all-reduce combines the partials). The gates (fail closed): GLM_Q21_MOE=1,
tokens <= GLM_Q21_MOE_MAX_TOKENS (32), the FP8 weights, the scale attrs
(w13/w2_weight_scale_inv), both custom ops registered.
"""
import argparse
import sys
from pathlib import Path

MARKER = "[glm-5-3-fp8-moe]"

CALL_OLD = """        # [gaudi-fix:fp8-moe-gather] gathered-expert combine at low token counts, else the stock op.
        output = _gaudi_fix_moe_gather_apply(self, layer, x, topk_ids, topk_weights,
                                             _normalize_moe_activation(layer.activation))
        return output.view(*(output.size(0), *input_shape[1:]))"""

CALL_NEW = """        # [glm-5-3-fp8-moe] THE Q21 TWO-PASS (decode shapes, fail-closed): the fused
        # routed-expert TPC kernels with the in-kernel block-FP8 dequant. The routing
        # is already computed above (the same select_experts the stock path uses);
        # the EP mapping = layer.expert_map (the global->local ids, -1 = the
        # non-local pairs -> the kernels contribute 0 and the existing TP all-reduce
        # combines). Everything else falls through to the stock path.
        if (_q21_moe_gate(layer, x.shape[0])):
            out = _q21_fp8_two_pass(self, layer, x, topk_ids, topk_weights)
            return out.view(*(out.size(0), *input_shape[1:]))

        # [gaudi-fix:fp8-moe-gather] gathered-expert combine at low token counts, else the stock op.
        output = _gaudi_fix_moe_gather_apply(self, layer, x, topk_ids, topk_weights,
                                             _normalize_moe_activation(layer.activation))
        return output.view(*(output.size(0), *input_shape[1:]))"""

HELPERS = """

# [glm-5-3-fp8-moe] THE Q21 TWO-PASS GATE + THE CALL (fail-closed).
_Q21_ACTIVE_LOGGED = False


def _q21_moe_gate(layer, tokens):
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
    if _q21_s13(layer) is None or _q21_s2(layer) is None:
        return False
    ops = getattr(torch.ops, "custom_op", None)
    return (ops is not None and hasattr(ops, "block_fp8_moe_w13_glm")
            and hasattr(ops, "block_fp8_moe_w2_glm"))


def _q21_s13(layer):
    for name in ("w13_weight_scale_inv", "w13_weight_scale"):
        s = getattr(layer, name, None)
        if s is not None:
            return s
    return None


def _q21_s2(layer):
    for name in ("w2_weight_scale_inv", "w2_weight_scale"):
        s = getattr(layer, name, None)
        if s is not None:
            return s
    return None


def _q21_fp8_two_pass(method, layer, x, topk_ids, topk_weights):
    global _Q21_ACTIVE_LOGGED
    if not _Q21_ACTIVE_LOGGED:
        _Q21_ACTIVE_LOGGED = True
        print(f"[glm-5-3-fp8-moe] ACTIVE: tokens={x.shape[0]} "
              f"e_local={layer.w13_weight.shape[0]} w13={layer.w13_weight.dtype} "
              f"s13={_q21_s13(layer).dtype} s2={_q21_s2(layer).dtype} "
              f"emap={getattr(layer, 'expert_map', None) is not None}", flush=True)
    emap = getattr(layer, "expert_map", None)
    if emap is not None:
        local_ids = emap[topk_ids.to(torch.long)].to(torch.int32)
    else:
        local_ids = topk_ids.to(torch.int32)
    s13 = _q21_s13(layer)
    s2 = _q21_s2(layer)
    if s13.dtype != torch.float32:
        s13 = s13.float()
    if s2.dtype != torch.float32:
        s2 = s2.float()
    raw = torch.ops.custom_op.block_fp8_moe_w13_glm(layer.w13_weight, s13, x, local_ids)
    i2 = raw.shape[-1] // 2
    act = (torch.nn.functional.silu(raw[..., :i2].float()) * raw[..., i2:].float()
           * topk_weights[..., None].float()).to(torch.bfloat16)
    return torch.ops.custom_op.block_fp8_moe_w2_glm(layer.w2_weight, s2, act, local_ids)

"""

HELPER_ANCHOR = "\n# [glm-fp8-mxk] decode-sized block-FP8 calls -> custom TPC kernels (inert unless GLM_FP8_MXK=1)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default="/opt/glm53/vllm-gaudi")
    args = ap.parse_args()
    target = Path(args.root) / "vllm_gaudi" / "ops" / "hpu_fp8.py"
    src = target.read_text()
    if MARKER in src:
        print(f"{MARKER} already applied at {target}")
        return 0
    if CALL_OLD not in src:
        print(f"{MARKER} DRIFT: the call-site anchor not found")
        return 2
    if HELPER_ANCHOR not in src:
        print(f"{MARKER} DRIFT: the helper anchor not found")
        return 2
    if args.check:
        print(f"{MARKER} anchors match, not applied")
        return 1
    src = src.replace(CALL_OLD, CALL_NEW, 1)
    src = src.replace(HELPER_ANCHOR, HELPERS + HELPER_ANCHOR, 1)
    target.write_text(src)
    import ast
    ast.parse(src)
    print(f"{MARKER} applied at {target} (parses OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
