#!/usr/bin/env python3
"""CPU math-parity test for the T3 closed-form KDA conv-chain rewrite.

Reference: the per-token loop in kda_layer.py (spec_len>1 branch):
  full[j]  = cat([cur_win_j, x_cat[:, j]], dim=1)     # [bs, W, C]
  o[j]     = (full[j] * w_t).sum(dim=1) + bias        # [bs, C]
  outs[j]  = silu(o[j]);  cur_win_{j+1} = full[j][:, 1:]

Closed form: with stream = cat([win, x_cat], dim=1)  # [bs, W-1+L, C]
  full[j] = stream[:, j:j+W, :]
  o = sum_t stream[:, t:t+L, :] * w_t[t]              # W slice-MACs, no loop over L
  new_win[j] = stream[:, j+1:j+W, :]

Verifies exact equality in fp32 and measures the bf16 ULP delta.
"""
import torch

torch.manual_seed(0)

def reference_loop(win, x_cat, w_t, bias):
    bs = win.shape[0]
    width = w_t.shape[0]
    cur_win = win.contiguous()
    outs, new_wins = [], []
    for j in range(x_cat.shape[1]):
        full = torch.cat([cur_win, x_cat[:, j].unsqueeze(1)], dim=1).contiguous()
        w_exp = w_t.reshape(1, width, w_t.shape[1]).expand(bs, width, w_t.shape[1])
        o = (full * w_exp).sum(dim=1) + bias.view(1, -1)
        outs.append(torch.nn.functional.silu(o))
        cur_win = full[:, 1:].contiguous()
        new_wins.append(cur_win)
    return torch.stack(outs, dim=1), torch.stack(new_wins, dim=1)

def closed_form(win, x_cat, w_t, bias):
    L = x_cat.shape[1]
    W = w_t.shape[0]
    stream = torch.cat([win, x_cat], dim=1)  # [bs, W-1+L, C]
    o = torch.zeros_like(x_cat)
    for t in range(W):  # W=4 vectorized slice-MACs, no loop over L
        o = o + stream[:, t:t+L, :] * w_t[t].view(1, 1, -1)
    o = o + bias.view(1, -1)
    outs = torch.nn.functional.silu(o)
    idx = (torch.arange(L).unsqueeze(1) + 1 + torch.arange(W - 1).unsqueeze(0)).reshape(-1)
    new_wins = stream.index_select(1, idx).reshape(x_cat.shape[0], L, W - 1, x_cat.shape[-1])
    return outs, new_wins

def run(dtype, bs, L, W, C):
    win = torch.randn(bs, W - 1, C, dtype=dtype)
    x_cat = torch.randn(bs, L, C, dtype=dtype)
    w_t = torch.randn(W, C, dtype=dtype)
    bias = torch.randn(C, dtype=dtype)
    o_ref, nw_ref = reference_loop(win, x_cat, w_t, bias)
    o_new, nw_new = closed_form(win, x_cat, w_t, bias)
    if dtype == torch.float32:
        assert torch.equal(o_ref, o_new), f"fp32 output mismatch (max {(o_ref-o_new).abs().max()})"
        assert torch.equal(nw_ref, nw_new), "fp32 window mismatch"
        return "exact"
    delta = (o_ref.float() - o_new.float()).abs().max().item()
    scale = o_ref.float().abs().max().item()
    ulps = delta / (torch.finfo(dtype).eps * max(scale, 1e-9))
    return f"max |delta|={delta:.2e} ({ulps:.1f} bf16 ULPs of peak)"

print("fp32:", run(torch.float32, 4, 5, 4, 96))
print("bf16:", run(torch.bfloat16, 4, 5, 4, 96))
print("bf16 large:", run(torch.bfloat16, 80, 5, 4, 24576))
print("T3-CLOSED-FORM-PARITY-PASS")
