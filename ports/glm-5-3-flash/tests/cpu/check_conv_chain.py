# SPDX-License-Identifier: Apache-2.0
"""CPU parity: the MTP chain-conv formulation vs hpu_causal_conv1d_update (production kernel),
mirroring the KDA layer's exact decode call."""

import sys

import torch

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "glm5_next_hpu"))


def main():
    from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update

    torch.manual_seed(11)
    C, W, slots, bs, L = 1024, 4, 16, 2, 2

    conv_w = torch.randn(C, 1, W) * 0.3
    conv_b = torch.randn(C) * 0.1
    win0 = torch.randn(bs, W - 1, C) * 0.5
    x_seq = torch.randn(bs, L, C)
    idx = torch.arange(bs, dtype=torch.int32)
    qsl = torch.arange(0, bs * L + 1, L, dtype=torch.int32)  # spans of L tokens

    # --- reference: production kernel, one token at a time, chaining the in-place window
    pool = torch.zeros(slots, W - 1, C)
    pool[:bs] = win0
    outs, wins = [], []
    for j in range(L):
        st = pool[: slots].clone()
        o = hpu_causal_conv1d_update(
            x=x_seq[:, j].contiguous(),
            conv_state=st,
            weight=conv_w.view(C, W),
            bias=conv_b,
            activation="silu",
            conv_state_indices=idx,
            query_start_loc=torch.arange(0, bs + 1, dtype=torch.int32),
            validate_data=False,
        )
        outs.append(o[:bs])
        wins.append(st[:bs].clone())
        pool[:bs] = st[:bs]
    ref_out = torch.stack(outs, dim=1)
    ref_win = torch.stack(wins, dim=1)

    # --- mine: closed-form chain (the kda_layer spec branch formulation)
    inp = torch.cat([win0, x_seq], dim=1).contiguous()  # [bs, W-1+L, C]
    w_t = conv_w.view(C, W).t().contiguous()  # [W, C]
    cur_win = win0.contiguous()
    outs2, wins2 = [], []
    for j in range(L):
        full = torch.cat([cur_win, x_seq[:, j].unsqueeze(1)], dim=1).contiguous()  # [bs, W, C]
        w_exp = w_t.reshape(1, W, C).expand(bs, W, C)
        o = torch.nn.functional.silu((full * w_exp).sum(dim=1) + conv_b.view(1, -1))
        outs2.append(o)
        cur_win = full[:, 1:].contiguous()
        wins2.append(cur_win)
    mine_out = torch.stack(outs2, dim=1)
    mine_win = torch.stack(wins2, dim=1)

    d_out = float((mine_out - ref_out).abs().max())
    d_win = float((mine_win - ref_win).abs().max())
    print(f"max|out diff| = {d_out:.3e}   max|win diff| = {d_win:.3e}")
    assert d_out < 1e-5, "conv output mismatch"
    assert d_win < 1e-5, "window mismatch"
    print("chain conv parity vs production kernel: OK")


if __name__ == "__main__":
    sys.exit(main())
