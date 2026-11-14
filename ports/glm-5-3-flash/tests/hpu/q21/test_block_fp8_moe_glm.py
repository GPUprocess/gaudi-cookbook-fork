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


def _interleave_pairs(w):
    """The interleaved-chunk-pair layout along the last dim (the v3 kernel
    contract): the span [p*256+2j] = the chunk 2p's element j, the span
    [p*256+2j+1] = the chunk (2p+1)'s element j. The converted fp8 pairs then
    split cell-wise (SW_UNPACK_EVEN/ODD_LANES) into the vectors aligned with
    the two 128-element activation chunks per pair."""
    e, r, c = w.shape
    return w.view(e, r, c // 256, 2, 128).permute(0, 1, 2, 4, 3).reshape(e, r, c).contiguous()


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
                       lambda: torch.rand(E, I2 // 128, H // 128, device=dev) * 0.02 + 0.005
                       if os.environ.get("GLM_GATE_SCALES_ONES", "0") != "1"
                       else torch.ones(E, I2 // 128, H // 128, device=dev), acc)
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
    ok = ok and _stage("s6c_s2_rand", lambda: torch.rand(E, H // 128, (I2 // 2) // 128, device=dev) * 0.02 + 0.005
                       if os.environ.get("GLM_GATE_SCALES_ONES", "0") != "1"
                       else torch.ones(E, H // 128, (I2 // 2) // 128, device=dev), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "w2 staging failed", "stages": sorted(acc)}
    w2, s2 = acc["s6b_w2_cast"], acc["s6c_s2_rand"]
    w13_q, w2_q = w13, w2  # the v5: the STANDARD order (the no interleave)

    ok = _stage("s7_ref_pulls", lambda: (
        w13.float().cpu(), s13.float().cpu(), w2.float().cpu(), s2.float().cpu(),
        x.float().cpu(), ids.cpu(), wts.cpu()), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "reference pulls failed", "stages": sorted(acc)}
    w13_rq, s13_c, w2_rq, s2_c, x_c, ids_c, wts_c = acc["s7_ref_pulls"]

    ok = _stage("s8_pass1_op", lambda: torch.ops.custom_op.block_fp8_moe_w13_glm(w13_q, s13, x, ids), acc)
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
    ok = _stage("s10_pass2_op", lambda: torch.ops.custom_op.block_fp8_moe_w2_glm(w2_q, s2, act_pre, ids), acc)
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

    # The fp8 unpack RUNTIME probe FIRST (its own tiny graph + sync, isolated
    # from the custom-op graph): the [256] f8 in → the [0..128) = the EVEN
    # halves (the expected in[2i]), the [128..256) = the ODD (the in[2i+1])
    def _probe():
        # The i64→fp8 device cast trips the bridge; use the proven f32→fp8 path.
        # The 2-D [1,256]: the 1-D custom-op outputs trip the bridge descriptor.
        inp = torch.arange(256, device=dev).float().to(torch.float8_e4m3fn).view(1, 256)
        out_p = torch.ops.custom_op.f8_unpk_probe(inp)
        return out_p.cpu().flatten()

    probe_verdict = None
    if _stage("s12_probe_op", _probe, acc):
        po = acc["s12_probe_op"].float()
        inp = torch.arange(256).float().to(torch.float8_e4m3fn).float()
        ev_ok = bool(torch.all(po[:128] == inp[0::2]).item())
        od_ok = bool(torch.all(po[128:] == inp[1::2]).item())
        probe_verdict = {"even_first_128_matches_in2i": ev_ok,
                         "odd_second_128_matches_in2i1": od_ok}
        print(f"[Q21-GATE] probe: {probe_verdict}", flush=True)

    ok = ok and _stage("s13a_raw_cpu", lambda: raw_q.cpu(), acc)
    ok = ok and _stage("s13b_y_cpu", lambda: y_q.cpu(), acc)
    ok = ok and _stage("s13c_sync", lambda: (raw_q.cpu(), y_q.cpu()), acc)
    if not ok:
        return {"gate": "q21_block_fp8_moe_glm", "pass": False,
                "error": "sync/refs failed", "stages": sorted(acc),
                "probe": probe_verdict}
    raw_ref, y_ref = acc["s11_refs"]
    # The device-side .float() of the custom-op outputs trips the bridge
    # ("Empty tensor optional") — cast on the CPU after the bf16 pull
    raw_q_c, y_q_c = acc["s13c_sync"][0].float(), acc["s13c_sync"][1].float()

    # The ONE-HOT x probes: the convert's per-position mapping, determined
    # exactly. With x[0] = the unit vector at h*, raw[0,k,col] must equal
    # w13_dq[ids0k, col, h*] * s13[ids0k, col//128, h*//128]. The five
    # positions bracket the 128-value boundary of the convert's output pair.
    onehot = {}
    for h_star in (0, 1, 127, 128, 255):
        def _p(hs=h_star):
            xh = x.clone()
            xh[0] = 0
            xh[0, hs] = 1.0
            r = torch.ops.custom_op.block_fp8_moe_w13_glm(w13_q, s13, xh, ids)
            return r.cpu()
        if _stage(f"s14_hot{h_star}", _p, acc):
            r = acc[f"s14_hot{h_star}"].float()[0]
            ids0 = ids_c[0].long()
            colr = torch.arange(I2)
            exp = w13_rq[ids0][:, colr, h_star] * s13_c[ids0][:, colr // 128, h_star // 128]
            onehot[h_star] = (r - exp).abs().max().item()
            if h_star == 0:
                # The per-expert breakdown: the r[0,:,0] vs the exp[:,0] — the
                # cross-expert contamination, if any, is visible in the 8 pairs
                for k in range(K):
                    print(f"[Q21-GATE] hot0 k={k} id={int(ids0[k])} "
                          f"r={r[k, 0].item():.6f} exp={exp[k, 0].item():.6f}", flush=True)
        else:
            onehot[h_star] = None
    print(f"[Q21-GATE] one-hot errs: {onehot}", flush=True)

    def cmp(q, r, name):
        err = (q - r).abs()
        scale = max(r.abs().max().item(), 1e-6)
        # The scale-based allclose: the near-zero refs make the per-element
        # rel meaningless; judge against the tensor's dynamic range
        okc = bool((err <= ATOL + RTOL * scale).all().item())
        return {"tensor": name, "max_abs": err.max().item(),
                "rel_to_scale": (err.max().item() / scale), "ref_max": scale, "pass": okc}

    res_raw, res_y = cmp(raw_q_c, raw_ref, "raw"), cmp(y_q_c, y_ref, "y")
    return {"gate": "q21_block_fp8_moe_glm",
            "shapes": {"E": E, "T": T, "H": H, "I2": I2, "K": K},
            "pass_1": res_raw, "pass_2": res_y,
            "probe": probe_verdict,
            "onehot_errs": onehot,
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
