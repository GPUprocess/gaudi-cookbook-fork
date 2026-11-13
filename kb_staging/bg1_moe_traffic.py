#!/usr/bin/env python3
"""BG-1: the dequant-vs-GEMM traffic measurement for GLM's FP8 MoE op (standalone, eager).

Builds VllmMixtureOfExpertsOpFP8 with GLM-5.3-Flash's per-rank shapes
(36 local experts, hidden 4096, moe_intermediate 2048, block-FP8 128x128),
then times, with torch.hpu events:
  - the dequant phase: get_dequant_weight() for all 36 experts (the stock cast+mul
    chain, as production runs with GLM_FP8_MXK=0)
  - the GEMM phase: torch.ops.hpu.mixture_of_experts on the dequantized BF16
at bs=1 (1 token) and @16-like (32 rows = 16 requests x (1 draft + 1)).

Answers: how much of the step the dequant traffic costs eagerly, and whether the
GEMM on the dequantized weights is consistent with the fused-traffic model.
"""
import os
import sys
import time

import torch

os.environ.setdefault("PT_HPU_LAZY_MODE", "1")

from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpFP8, MoeFP8Matmul  # noqa: E402

H = 4096
I = 2048
E_LOCAL = 12  # the 1/3 scale: fits HBM beside serving; the timing is per-expert linear
BLOCK = (128, 128)
LAYERS = 42

dev = "hpu"


def build_op():
    op = VllmMixtureOfExpertsOpFP8(
        global_num_experts=288, num_experts=E_LOCAL, experts_min=0, experts_max=E_LOCAL - 1
    )
    for j in range(E_LOCAL):
        w13 = torch.randn(I * 2, H, device=dev).clamp(-3, 3).to(torch.float8_e4m3fn)
        w2 = torch.randn(H, I, device=dev).clamp(-3, 3).to(torch.float8_e4m3fn)
        s13 = torch.ones(I * 2 // BLOCK[0], H // BLOCK[1], device=dev, dtype=torch.float32)
        s2 = torch.ones(H // BLOCK[0], I // BLOCK[1], device=dev, dtype=torch.float32)
        op.w13_list[j].set_weight(w13)
        op.w13_list[j].set_scale_inv_fp8(s13)
        op.w13_list[j].scale_inv_fp8 = s13
        op.w13_list[j].set_weight_block_size(BLOCK)
        op.w2_list[j].set_weight(w2)
        op.w2_list[j].scale_inv_fp8 = s2
        op.w2_list[j].set_weight_block_size(BLOCK)
    return op


def time_phase(fn, reps=5):
    """Returns the list of per-rep device times in ms via the hpu events."""
    times = []
    for _ in range(reps):
        s = torch.hpu.Event(enable_timing=True)
        e = torch.hpu.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.hpu.synchronize()
        times.append(s.elapsed_time(e))
    return times, out


def main():
    print("building the op (the FP8 weights on HPU)...")
    t0 = time.time()
    op = build_op()
    torch.hpu.synchronize()
    print(f"built in {time.time()-t0:.1f}s; the weights on HBM")

    w_bytes_fp8 = E_LOCAL * 3 * I * H  # the per-layer per-rank expert bytes, FP8
    print(f"the per-layer expert weights (12 experts): {w_bytes_fp8/1e9:.2f} GB FP8 "
          f"(the full 36: {w_bytes_fp8*3/1e9:.2f} GB; x42 layers = {w_bytes_fp8*3*LAYERS/1e9:.1f} GB/rank)")

    for tag, tokens in (("bs=1", 1), ("@16-verify", 32)):
        x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16)
        topk_ids = torch.randint(0, 288, (tokens, 8), device=dev, dtype=torch.int32) % E_LOCAL
        topk_w = torch.rand(tokens, 8, device=dev, dtype=torch.float32)
        # normalize the routing rows
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)

        # warmup
        op.forward(x, topk_ids, topk_w)
        torch.hpu.synchronize()

        # the dequant phase alone (the stock naive path, as production)
        d_times, _ = time_phase(lambda: [op.w13_list[j].get_dequant_weight() for j in range(E_LOCAL)])
        # the GEMM phase on the PRE-dequantized weights: monkeypatch get_dequant_weight
        # to return the cached tensor, isolating the mixture_of_experts call
        cached = [op.w13_list[j].get_dequant_weight() for j in range(E_LOCAL)]
        cached2 = [op.w2_list[j].get_dequant_weight() for j in range(E_LOCAL)]
        orig = [(op.w13_list[j], op.w2_list[j]) for j in range(E_LOCAL)]

        def gemm_only():
            w13l = cached
            w2l = cached2
            return torch.ops.hpu.mixture_of_experts(
                hidden_states=x, expert_routing_table=topk_ids, router_weights=topk_w,
                w12=w13l, w3=w2l, permuted_weights=True, activation="silu",
                experts_min=0, experts_max=E_LOCAL - 1,
            )

        g_times, out = time_phase(gemm_only)

        d_med = sorted(d_times)[len(d_times) // 2]
        g_med = sorted(g_times)[len(g_times) // 2]
        print(f"\n[{tag}] tokens={tokens}")
        print(f"  the dequant phase (the 36 experts, the stock naive): "
              f"median {d_med:.2f} ms  (the reads {w_bytes_fp8/1e9:.2f} GB FP8 + the writes "
              f"{2*w_bytes_fp8/1e9:.2f} GB BF16)")
        print(f"  the GEMM phase (the mixture_of_experts on the BF16): median {g_med:.2f} ms")
        print(f"  the eager total: {d_med + g_med:.2f} ms")
        if d_med > 0:
            implied = 3 * w_bytes_fp8 / (d_med / 1e3) / 1e12
            print(f"  the implied dequant BW: {implied:.2f} TB/s")


if __name__ == "__main__":
    main()
