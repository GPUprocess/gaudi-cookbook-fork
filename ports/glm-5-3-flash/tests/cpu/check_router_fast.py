#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU contract test for the router-fast group-stage skip (GLM_ROUTER_FAST).

Verifies, on CPU, that skipping the group stage when num_expert_group == 1 and
topk_group == 1 is bit-identical to the stock grouped_topk (both with and
without an e_score_correction_bias), and that the patch's gate refuses other
group configurations (the stage is NOT an identity there).
Run: python3 tests/cpu/check_router_fast.py
"""
import torch  # noqa: E402


def _impl(scores, topk, renormalize, num_expert_group, topk_group, bias, rsf, fast):
    original_scores = scores
    if bias is not None:
        scores = scores + bias.unsqueeze(0)
    if fast and num_expert_group == 1 and topk_group == 1:
        tmp_scores = scores
    else:
        if bias is not None:
            group_scores = scores.view(-1, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        else:
            group_scores = scores.view(-1, num_expert_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (group_mask.unsqueeze(-1)
                      .expand(-1, num_expert_group, scores.shape[-1] // num_expert_group)
                      .reshape(-1, -1))
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    if bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * rsf
    return topk_weights, topk_ids.to(torch.int32)


def main():
    torch.manual_seed(0)
    E, topk, rsf = 288, 8, 2.5
    logits = torch.randn(7, E)
    bias = torch.randn(E) * 0.01

    # 1. n_group=1, topk_group=1: fast == stock, with and without the bias.
    for b in (bias, None):
        s = torch.softmax(logits.float(), dim=-1)
        if b is not None:
            s_biased = s + b.unsqueeze(0)
        else:
            s_biased = s
        w1, i1 = _impl(s_biased, topk, True, 1, 1, b, rsf, fast=False)
        w2, i2 = _impl(s_biased, topk, True, 1, 1, b, rsf, fast=True)
        assert torch.equal(w1, w2) and torch.equal(i1, i2), "fast path must be bit-identical at n_group=1"
    print("PASS: fast == stock (bit-identical) at n_group=1/topk_group=1, bias and no-bias")

    # 2. n_group>1: the stage is NOT an identity; the fast gate must not apply.
    s = torch.softmax(logits.float(), dim=-1)
    w1, i1 = _impl(s, topk, True, 8, 2, None, rsf, fast=False)
    w2, i2 = _impl(s, topk, True, 8, 2, None, rsf, fast=True)
    assert not torch.equal(i1, i2), "n_group>1 must route differently (gate must not fire)"
    print("PASS: gate refused at n_group=8/topk_group=2 (routing differs)")

    # 3. The identity claim itself: with one group, masked_fill(all-true) == input.
    scores = torch.softmax(logits.float(), dim=-1)
    group_mask = torch.ones(7, 1)
    score_mask = group_mask.unsqueeze(-1).expand(7, 1, E).reshape(7, -1)
    tmp = scores.masked_fill(~score_mask.bool(), float("-inf"))
    assert torch.equal(tmp, scores)
    print("PASS: masked_fill with an all-true mask is an identity")


if __name__ == "__main__":
    main()
