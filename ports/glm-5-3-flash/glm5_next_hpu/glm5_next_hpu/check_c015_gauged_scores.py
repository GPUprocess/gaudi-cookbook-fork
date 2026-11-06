#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Oracle test for C015: anchored-MME-matmul diagonal block vs the current elementwise form.

Oracle for the landed C015 change: compares the shipped anchored-matmul `_gauged_scores_v2`
(from kda.py) against the pre-C015 elementwise diagonal block. Runs on CPU; no HPU, no
server. Keep in sync if `_gauged_scores` changes shape handling.

The transformation: within a sub-block, for i >= j (both relative to the block start),
  exp(g_i - g_j) = exp(g_i - g_ref) * exp(g_ref - g_j),
both factors in (0, 1] because g is non-positive and decreasing within the chunk and g_ref is
the block's first row. So mid = (a * exp(g_i - g_ref)) @ (b * exp(g_ref - g_j)).T — an MME
matmul instead of a [sub, sub, D] elementwise product reduced on the TPCs.
"""
import sys
import torch

sys.path.insert(0, ".")
from kda import _gauged_scores as _gauged_scores_v2  # the landed C015 implementation


def _gauged_scores_ref(a, b, g, sub, strict, keep):
    """Pre-C015 reference: elementwise diagonal block (for the oracle)."""
    *lead, c, d = a.shape
    if c % sub:
        raise ValueError(f"chunk {c} not divisible by sub-block {sub}")
    nb = c // sub
    m = 1
    for x in lead:
        m *= x
    a = a.reshape(m, c, d)
    b = b.reshape(m, c, d)
    g = g.reshape(m, c, d)

    rows = []
    for i_blk in range(nb):
        i0, i1 = i_blk * sub, (i_blk + 1) * sub
        g_ref = g[:, i0 : i0 + 1, :]
        gi = g[:, i0:i1, :]

        # Diagonal block: anchored factorization. The ROW factor is <= 1
        # (g decreasing, anchor = first row). The COL factor exceeds 1 for
        # j > anchor (up to e^75 with the -5/token gate bound) - inside fp32;
        # those entries sit in the masked upper triangle and are discarded by
        # `keep`, while the kept lower triangle is bounded by 1.
        row_f = a[:, i0:i1, :] * (gi - g_ref).clamp(max=0.0).exp()
        col_f = b[:, i0:i1, :] * (g_ref - gi).exp()
        mid = (row_f @ col_f.transpose(-1, -2)) * keep

        pieces = []
        if i0 > 0:
            row = a[:, i0:i1, :] * (gi - g_ref).clamp(max=0.0).exp()
            col = b[:, :i0, :] * (g_ref - g[:, :i0, :]).clamp(max=0.0).exp()
            pieces.append(row @ col.transpose(-1, -2))
        pieces.append(mid)
        if i1 < c:
            pieces.append(a.new_zeros(m, sub, c - i1))
        rows.append(torch.cat(pieces, dim=-1))

    return torch.cat(rows, dim=-2).reshape(*lead, c, c)


def make_inputs(lead, c, d, sub, seed):
    torch.manual_seed(seed)
    a = torch.randn(lead, c, d).abs() * 0.5 + 0.5
    b = torch.randn(lead, c, d).abs() * 0.5 + 0.5
    # g: within-chunk cumulative decay — non-positive, decreasing per position.
    per_tok = -torch.rand(lead, c, d) * 5.0  # gate lower bound -5/token
    g = torch.cumsum(per_tok, dim=1)
    return a.float(), b.float(), g.float()


def keep_mask(sub, strict):
    i = torch.arange(sub).view(-1, 1)
    j = torch.arange(sub).view(1, -1)
    return ((j < i) if strict else (j <= i)).float()


def main():
    fails = 0
    total = 0
    max_err = 0.0
    for strict in (True, False):
        keep = keep_mask(16, strict)
        for lead, c, d, sub, seed in [
            (1, 64, 128, 16, 0),
            (2, 64, 128, 16, 1),
            (4, 64, 128, 16, 2),
            (1, 64, 64, 16, 3),
            (1, 32, 128, 16, 4),
            (3, 64, 96, 16, 5),
        ]:
            a, b, g = make_inputs(lead, c, d, sub, seed)
            old = _gauged_scores_ref(a, b, g, sub, strict, keep)
            new = _gauged_scores_v2(a, b, g, sub, strict, keep)
            err = (old - new).abs().max().item()
            scale = old.abs().max().item()
            rel = err / max(scale, 1e-9)
            total += 1
            max_err = max(max_err, rel)
            # fp32 matmul vs elementwise-sum: relative error should be tiny (~1e-6 class)
            if rel > 1e-4:
                fails += 1
                print(f"FAIL lead={lead} c={c} d={d} strict={strict}: rel_err={rel:.3e}")
            else:
                print(f"ok  lead={lead} c={c} d={d} strict={strict}: rel_err={rel:.2e}")
    print(f"\noracle: {total - fails}/{total} shapes match; max relative error {max_err:.2e}")
    if fails:
        sys.exit(1)
    print("C015 oracle PASS — the anchored matmul is numerically equivalent on CPU")


if __name__ == "__main__":
    main()
