#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# test_block_fp8_moe_glm.py — the Q21/RA-002 step-2 correctness gate (GLM-5.3-Flash MoE).
#
# Compares the two-pass Q21 TPC kernels against the PyTorch dequant reference.
# Every stage is individually marked and error-isolated so one run in the
# load-window sidecar bisects any bridge failure:
#   s1 the device f32 randn → s2 the device fp8 cast → s3 the f32 scales →
#   s4 the bf16 x → s5 the i32 ids → s6 the f32 weights → s7 the reference
#   pulls → s8 the pass-1 op → s9 the between-pass math → s10 the pass-2 op →
#   s11 the sync + compare.
import argparse
import json
import os
import sys

import torch  # noqa: F401 — must precede the habana import

E, T, H, I2, K, SEED = 8, 4, 512, 512, 8, 20261114
RTOL, ATOL = 3e-2, 1e-3


def _stage(name, fn, acc):
    try:
        acc[name] = fn()
        print(f"[Q21-GATE] s:{name} OK", flush=True)
        return True
    except Exception as exc:
        print(f"[Q21-GATE] s:{name} FAIL {repr(exc)[:160]}", flush=True)
        return False


def _dequant_bf16(w_rq, s_f32):
    """The block-FP8 dequant on the requantized f32 masters:
    f32(w8) * f32(bf16(s)) cast to bf16 (the kernel casts the scales to bf16)."""
    er, r, c = w_rq.shape
    s_exp = s_f32.to(torch.bfloat16).float().repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)[:, :r, :c]
    return (w_rq * s_exp).to(torch.bfloat16)


def run_gate():
    import habana_frameworks.torch  # noqa: F401 — the HPU backend init

    acc = {}
    dev = torch.device("hpu")
    torch.manual_seed(SEED)
    ok = _stage("s0_load_op_lib", lambda: torch.ops.load_library(
        os.environ.get("Q21_OP_LIB", "/data/glm53/q21_dist/libmxk_op_lazy.so")), acc)
    ok = ok and _stage("s1_w_f32_dev", lambda: torch.randn(E, I2, H, device=dev) * 0.05, acc)
    if ok:
        ok = _stage("s2_w_fp8_cast", lambda: acc["s1_w_f32_dev"].to(torch.float8_e4m3fn), acc)
    ok = ok and _stage("s3_scales_f32_dev",
                       lambda: torch.rand(E, I2 // 128, H // 128, device=dev) * 0.02 + 0.005, acc)
    ok = ok and _stage("s4_x_bf16_dev",
                       lambda: (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16), acc)
    ok = ok and _stage("s5_ids_i32_dev",
                       lambda: torch.randint(0, E, (T, K), device=dev).to(torch.int32), acc)
    if ok:
        def _ids_perm():
            g = torch.Generator().manual_seed(SEED + 1)
            return torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32).to(dev)
        ok = _stage("s5b_ids_perm_dev", _ids_perm, acc)
    ok = ok and _stage("s6_wts_f32_dev", lambda: torch.rand(T, K, device=dev) / T * 8, acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "input staging failed", "stages": sorted(acc)}
    w13, s13, x, ids, wts = (acc["s2_w_fp8_cast"], acc["s3_scales_f32_dev"],
                             acc["s4_x_bf16_dev"], acc["s5b_ids_perm_dev"], acc["s6_wts_f32_dev"])
    ok = _stage("s6b_w2_cast", lambda: (torch.randn(E, H, I2 // 2, device=dev) * 0.05).to(torch.float8_e4m3fn), acc)
    ok = ok and _stage("s6c_s2_rand", lambda: torch.rand(E, H // 128, (I2 // 2) // 128, device=dev) * 0.02 + 0.005, acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "w2 staging failed", "stages": sorted(acc)}
    w2, s2 = acc["s6b_w2_cast"], acc["s6c_s2_rand"]

    ok = _stage("s7_ref_pulls", lambda: (
        w13.float().cpu(), s13.float().cpu(), w2.float().cpu(), s2.float().cpu(),
        x.float().cpu(), ids.cpu(), wts.cpu()), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "reference pulls failed", "stages": sorted(acc)}
    w13_rq, s13_c, w2_rq, s2_c, x_c, ids_c, wts_c = acc["s7_ref_pulls"]

    ok = _stage("s8_pass1_op", lambda: torch.ops.custom_op.block_fp8_moe_w13_glm(w13, s13, x, ids), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "pass-1 op failed", "stages": sorted(acc)}
    raw_q = acc["s8_pass1_op"]
    i = I2 // 2
    ok = _stage("s9_between_math",
                lambda: (torch.nn.functional.silu(raw_q[..., :i].float()) * raw_q[..., i:].float()
                         * wts[..., None].float()).to(torch.bfloat16), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "between-pass math failed", "stages": sorted(acc)}
    act_pre = acc["s9_between_math"]
    ok = _stage("s10_pass2_op", lambda: torch.ops.custom_op.block_fp8_moe_w2_glm(w2, s2, act_pre, ids), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "pass-2 op failed", "stages": sorted(acc)}
    y_q = acc["s10_pass2_op"]

    def _refs():
        w13_dq = _dequant_bf16(w13_rq, s13_c).float()
        w2_dq = _dequant_bf16(w2_rq, s2_c).float()
        xf = x_c.float()
        raw = torch.empty(T, K, I2, dtype=torch.float32)
        for t in range(T):
            for k in range(K):
                raw[t, k] = w13_dq[ids_c[t, k].long()] @ xf[t]
        act = (torch.nn.functional.silu(raw[..., :i]) * raw[..., i:] * wts_c[..., None]).to(torch.bfloat16).float()
        y = torch.zeros(T, H, dtype=torch.float32)
        for t in range(T):
            for k in range(K):
                y[t] += w2_dq[ids_c[t, k].long()] @ act[t, k]
        return raw, y

    ok = _stage("s11_refs", _refs, acc)
    ok = ok and _stage("s12_sync", lambda: (raw_q.float().cpu(), y_q.float().cpu()), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "sync/refs failed", "stages": sorted(acc)}
    raw_ref, y_ref = acc["s11_refs"]
    raw_q_c, y_q_c = acc["s12_sync"]

    def cmp(q, r, name):
        err = (q - r).abs()
        rel = (err / r.abs().clamp_min(1e-6)).max().item()
        okc = bool((err <= ATOL + RTOL * r.abs()).all().item())
        return {"tensor": name, "max_abs": err.max().item(), "max_rel": rel, "pass": okc}

    res_raw, res_y = cmp(raw_q_c, raw_ref, "raw"), cmp(y_q_c, y_ref, "y")
    return {"gate": "q21_block_fp8_moe_glm",
            "shapes": {"E": E, "T": T, "H": H, "I2": I2, "K": K},
            "pass_1": res_raw, "pass_2": res_y,
            "pass": bool(res_raw["pass"] and res_y["pass"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/glm53/q21_gate_result.json")
    args = ap.parse_args()
    try:
        result = run_gate()
    except Exception as exc:  # the gate must report, not crash the host process
        import traceback
        traceback.print_exc()
        result = {"gate": "q21_block_fp8_moe_glm", "pass": False, "error": repr(exc)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
