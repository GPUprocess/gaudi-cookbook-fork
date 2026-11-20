#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q21-fp8-moe wiring (Q21 step 3b): the fused routed-expert two-pass TPC path for the
FP8 (block-quant) MoE — the actual GLM-5.3-Flash runtime path.

    python3 q21_fp8_moe_wiring.py [--check] [--root /opt/glm53/vllm]
Exit: 0 applied/already, 1 not applied (anchors match), 2 drifted.

Target: vllm/model_executor/layers/quantization/fp8.py — Fp8MoEMethod.apply_monolithic
(the oracle modular-kernel path the block-FP8 checkpoint actually runs; the earlier
vllm_gaudi HPUUnquantizedFusedMoEMethod patch only covers the BF16 unquantized method).

Runtime gates (fail closed): GLM_Q21_MOE=1, tokens <= GLM_Q21_MOE_MAX_TOKENS (32),
the FP8 weights retained on the layer, the weight scales present, both custom ops
registered. The routing: the port's own select_experts_from_routed (the laguna-fix
helper = the upstream select_experts with the layer's grouped-topk parameters). The
EP mapping: layer.expert_map[global_id] -> the local id or -1 — exactly the Q21
kernels' -1 sentinel (the non-local pairs contribute 0 and combine via the existing
TP all-reduce). Gate evidence: the 12-stage HPU correctness gate PASS.
"""
import argparse
import os
import sys
from pathlib import Path

MARKER = "[glm-5-3-flash:q21-fp8-moe]"

HELPERS_OLD = """    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None"""

HELPERS_NEW = """    def _q21_fp8_active(self, layer, tokens: int) -> bool:
        \"\"\"[glm-5-3-fp8-moe] The fail-closed gate for the Q21 two-pass decode path.\"\"\"
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
        s13 = self._q21_s13(layer)
        s2 = self._q21_s2(layer)
        if s13 is None or s2 is None:
            return False
        ops = getattr(torch.ops, "custom_op", None)
        return (ops is not None and hasattr(ops, "block_fp8_moe_w13_glm")
                and hasattr(ops, "block_fp8_moe_w2_glm"))

    @staticmethod
    def _q21_s13(layer):
        for name in ("w13_weight_scale_inv", "w13_weight_scale"):
            s = getattr(layer, name, None)
            if s is not None:
                return s
        return None

    @staticmethod
    def _q21_s2(layer):
        for name in ("w2_weight_scale_inv", "w2_weight_scale"):
            s = getattr(layer, name, None)
            if s is not None:
                return s
        return None

    _q21_logged = False

    def _q21_fp8_two_pass(self, layer, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        \"\"\"[glm-5-3-fp8-moe] The Q21 fused routed-expert two-pass (decode shapes).

        Pass 1 (TPC): raw[t,k,2I] = W13[e] . x with the in-kernel block-FP8 dequant;
        the expert_map's -1 entries (the non-local EP experts) contribute 0 and the
        per-rank partial y combines via the existing TP all-reduce. Between math:
        SiLU(gate) * up * routing weight. Pass 2 (TPC): y[t] = sum_k W2[e] . act.
        \"\"\"
        if not self._q21_logged:
            type(self)._q21_logged = True
            print(f"[glm-5-3-fp8-moe] ACTIVE: tokens={x.shape[0]} "
                  f"e_local={layer.w13_weight.shape[0]} w13={layer.w13_weight.dtype} "
                  f"w2={layer.w2_weight.dtype} s13={self._q21_s13(layer).dtype} "
                  f"s2={self._q21_s2(layer).dtype} "
                  f"emap={'expert_map' in dir(layer)}", flush=True)
        from vllm_gaudi.vllm_gaudi.ops.hpu_fused_moe import select_experts_from_routed
        topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        emap = getattr(layer, "expert_map", None)
        if emap is not None:
            local_ids = emap[topk_ids.to(torch.long)].to(torch.int32)
        else:
            local_ids = topk_ids.to(torch.int32)
        s13 = self._q21_s13(layer)
        s2 = self._q21_s2(layer)
        if s13.dtype != torch.float32:
            s13 = s13.float()
        if s2.dtype != torch.float32:
            s2 = s2.float()
        raw = torch.ops.custom_op.block_fp8_moe_w13_glm(layer.w13_weight, s13, x, local_ids)
        i = raw.shape[-1] // 2
        act = (torch.nn.functional.silu(raw[..., :i].float()) * raw[..., i:].float()
               * topk_weights[..., None].float()).to(torch.bfloat16)
        return torch.ops.custom_op.block_fp8_moe_w2_glm(layer.w2_weight, s2, act, local_ids)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._q21_fp8_active(layer, x.shape[0]):                  # [glm-5-3-fp8-moe]
            return self._q21_fp8_two_pass(layer, x, router_logits)
        assert self.is_monolithic
        assert self.moe_kernel is not None"""


def find_root() -> Path:
    for cand in [Path("/opt/glm53/vllm")]:
        if (cand / "vllm" / "model_executor" / "layers" / "quantization" / "fp8.py").is_file():
            return cand
    raise SystemExit("vllm tree not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else find_root()
    target = root / "vllm" / "model_executor" / "layers" / "quantization" / "fp8.py"
    src = target.read_text()
    if MARKER in src:
        print(f"{MARKER} already applied at {target}")
        return 0
    if HELPERS_OLD not in src:
        print(f"{MARKER} DRIFT: the anchor not found in {target}")
        return 2
    if args.check:
        print(f"{MARKER} anchors match, not applied")
        return 1
    src = src.replace(HELPERS_OLD, HELPERS_NEW, 1)
    if "\nimport os" not in src and "import os\n" not in src:
        src = src.replace("\nimport torch", "\nimport os\nimport torch", 1)
    target.write_text(src)
    print(f"{MARKER} applied at {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
