# SPDX-License-Identifier: Apache-2.0
"""CPU-only smoke test for glm5_next_hpu.mtp (no HPU, no vLLM engine).

Validates the draft wrapper contract with stub modules: flat/lead reshape,
position-0 embed zeroing, eh_proj concat order, shape-presidual adds,
single post-norm output, no-op load_weights.
Run: python3 test_mtp_wrapper_cpu.py
"""

import sys
import types

import torch
from torch import nn


class StubAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, positions, hidden_states):
        self.calls.append((positions.clone(), hidden_states.clone()))
        # distinguishable transform: x + 100
        return hidden_states + 100.0


class StubLogitsProcessor:
    def __call__(self, lm_head, hidden_states, embedding_bias=None):
        return lm_head(hidden_states)


def build():
    from glm5_next_hpu.mtp import Glm5NextMTPDecoderLayer, Glm5NextMTPModel

    torch.manual_seed(7)
    H = 16
    mtp = Glm5NextMTPDecoderLayer.__new__(Glm5NextMTPDecoderLayer)  # skip heavy init
    nn.Module.__init__(mtp)
    mtp.enorm = nn.Identity()
    mtp.hnorm = nn.Identity()
    mtp.eh_proj = nn.Linear(2 * H, H, bias=False)
    mtp.input_layernorm = nn.Identity()
    mtp.post_attention_layernorm = nn.Identity()
    mtp.shared_head_norm = nn.Identity()
    mtp.self_attn = StubAttn()
    mtp.mlp = nn.Identity()

    target = types.SimpleNamespace()
    target.mtp = mtp
    inner = types.SimpleNamespace()
    inner.embed_tokens = nn.Embedding(32, H)
    target.model = inner
    target.lm_head = nn.Linear(H, 32, bias=False)
    target.logits_processor = StubLogitsProcessor()

    wrapper = Glm5NextMTPModel(target)
    return wrapper, mtp, H


def main():
    wrapper, mtp, H = build()

    # load_weights is a no-op
    assert wrapper.load_weights([]) == set(), "load_weights must return an empty set"
    assert wrapper._shares_target_modules is True

    # decode-shaped call: [batch, 1, H] hidden, ids/pos [batch]
    B = 3
    ids = torch.tensor([5, 6, 7])
    pos = torch.tensor([10, 11, 12])
    hidden = torch.randn(B, 1, H)
    out = wrapper(input_ids=ids, positions=pos, hidden_states=hidden, attn_metadata=None)
    assert out.shape == (B, 1, H), f"lead shape must be preserved, got {out.shape}"

    # flow check: eh_proj consumed cat(enorm(emb), hnorm(hidden)); emb zeroed at pos 0
    attn_pos, attn_in = mtp.self_attn.calls[-1]
    assert attn_pos.shape == (B,) and attn_in.shape == (B, H)
    emb = wrapper.embed_tokens(ids)
    expect_in = mtp.eh_proj(torch.cat([emb, hidden.reshape(B, H)], dim=-1))
    assert torch.allclose(attn_in, expect_in, atol=1e-5), "attention input must be eh_proj(cat(enorm(emb), hnorm(h)))"

    # position 0 zeroing
    pos0 = torch.tensor([0, 1, 2])
    out0 = wrapper(input_ids=ids, positions=pos0, hidden_states=hidden, attn_metadata=None)
    emb0 = wrapper.embed_tokens(ids).clone()
    emb0[0] = 0.0
    expect0 = mtp.eh_proj(torch.cat([emb0, hidden.reshape(B, H)], dim=-1))
    attn_pos, attn_in = mtp.self_attn.calls[-1]
    assert torch.allclose(attn_in, expect0, atol=1e-5), "position-0 embedding must be zeroed"

    # output == shared_head_norm(final residual): recompute the tail
    x = expect0
    x = x + (x + 100.0)  # residual + attn
    x = x + x  # residual + mlp (Identity post-norm)
    assert torch.allclose(out0.reshape(B, H), x, atol=1e-5), "tail flow mismatch"

    # compute_logits: single norm already applied -> logits = lm_head(hidden)
    logits = wrapper.compute_logits(out.reshape(B, H))
    assert logits.shape == (B, 32)
    assert torch.allclose(logits, wrapper.lm_head(out.reshape(B, H)), atol=1e-6)

    # flat-shaped call (prefill-like): [1, L, H]
    L = 5
    out_p = wrapper(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        positions=torch.tensor([[0, 1, 2, 3, 4]]),
        hidden_states=torch.randn(1, L, H),
        attn_metadata=None,
    )
    assert out_p.shape == (1, L, H)

    # missing mtp module raises
    try:
        from glm5_next_hpu.mtp import Glm5NextMTPModel as M

        M(types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=None)))
    except ValueError as e:
        assert "no .mtp module" in str(e)
    else:
        raise AssertionError("expected ValueError for missing mtp module")

    print("MTP wrapper CPU smoke: ALL OK")


if __name__ == "__main__":
    sys.exit(main())
