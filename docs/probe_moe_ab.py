#!/usr/bin/env python3
"""Multi-expert Q21 orchestration A/B (run on g2).

Measures the realistic bs=1 MoE layer execution for the 8 routed experts:
  arm A: stock path — per expert: block_fp8_dequant_glm + bf16 mm (the w13),
         SiLU gate, the w2 mm, weighted sum
  arm B: Q21 path — per expert: block_fp8_gemv_glm (the fused dequant+GEMV),
         SiLU gate, the w2 GEMV, weighted sum
Both use the same real checkpoint weights and the same routing.
"""
import torch, glob, time
import habana_frameworks.torch.hpu as hpu
import vllm_gaudi  # noqa: F401
hpu.init()
from safetensors import safe_open

f = sorted(glob.glob("/data/models/GLM-5.3-Flash/model-*.safetensors"))[0]
w13_e = []
w2_e = None
with safe_open(f, framework="pt") as fh:
    for k in fh.keys():
        if "mlp.experts.0." not in k or "weight_scale" in k or "weight_shape" in k:
            continue
        if "down_proj.weight" in k and w2_e is None:
            t = fh.get_tensor(k)
            w2_e = (t.float() * 0.5).to(torch.float8_e4m3fn).to("hpu")
        elif "gate_proj.weight" in k or "up_proj.weight" in k:
            t = fh.get_tensor(k)
            w13_e.append((t.float() * 0.5).to(torch.float8_e4m3fn).to("hpu"))
        if len(w13_e) >= 2 and w2_e is not None:
            break

print("w13 pieces:", len(w13_e) if w13_e else 0, "| w2 shape:", tuple(w2_e.shape) if w2_e is not None else None)
# the w13: the gate_proj [2048, 4096] + the up_proj [2048, 4096]: the fused: the cat → the [4096, 4096]
if len(w13_e) >= 2:
    w13 = torch.cat([w13_e[0], w13_e[1]], dim=0)  # the [4096, 4096]
else:
    w13 = w13_e[0]
w2 = w2_e  # the [4096, 2048]
print("w13:", tuple(w13.shape), "w2:", tuple(w2.shape))

# the scales: the shapes: the w13: [2, 32]?? — the real scales: the [N/128, K/128]: the per-piece!
# for the timing A/B the scales: the ones (the numerics: not the target here)
s13 = torch.ones(w13.shape[0] // 128, w13.shape[1] // 128, dtype=torch.float32).to("hpu")
s2 = torch.ones(w2.shape[0] // 128, w2.shape[1] // 128, dtype=torch.float32).to("hpu")

TOPK = 8
ids = torch.randperm(288)[:TOPK]  # the routed experts (the timing-only: the same weight reused)
weights = torch.softmax(torch.randn(TOPK), dim=0).to("hpu")
x = torch.randn(4096, dtype=torch.bfloat16).to("hpu")

torch.ops.load_library("/tmp/mxk_gemv_dist/libmxk_op_lazy.so")
import habana_frameworks.torch.core as htcore

def arm_a():
    outs = []
    for _ in range(TOPK):
        wd = torch.ops.custom_op.block_fp8_dequant_glm(w13, s13)
        y13 = wd @ x
        act = torch.nn.functional.silu(y13[:2048]) * y13[2048:]
        y2 = torch.ops.custom_op.block_fp8_dequant_glm(w2, s2) @ act
        outs.append(y2)
    return (torch.stack(outs) * weights[:, None]).sum(0)

def arm_b():
    outs = []
    for _ in range(TOPK):
        y13 = torch.ops.custom_op.block_fp8_gemv_glm(w13, s13, x)
        act = torch.nn.functional.silu(y13[:2048]) * y13[2048:]
        y2 = torch.ops.custom_op.block_fp8_gemv_glm(w2, s2, act)
        outs.append(y2)
    return (torch.stack(outs) * weights[:, None]).sum(0)

# the correctness cross-check
ya = arm_a(); yb = arm_b(); htcore.mark_step()
print("A vs B max abs:", (ya.float() - yb.float()).abs().max().item())

for name, fn in (("A-stock", arm_a), ("B-q21", arm_b)):
    for _ in range(3): fn()
    htcore.mark_step()
    t0 = time.perf_counter(); R = 20
    for _ in range(R): fn()
    htcore.mark_step()
    t1 = time.perf_counter()
    print(f"arm {name}: {(t1-t0)/R*1e3:.2f} ms per MoE layer")
