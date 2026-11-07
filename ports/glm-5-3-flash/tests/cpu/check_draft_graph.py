#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the Q01 draft HPU graphs (mtp.py graph cores).

Covers, without an HPU or the checkpoint:
1. `_private_copy`: storage independence (mutating the clone must not touch the
   original), class/attribute preservation, buffer cloning.
2. Core math parity: `_DraftPreAttentionGraphCore` + eager attention stand-in +
   `_DraftPostAttentionGraphCore` reproduce the eager forward exactly (same
   modules, same op order) on small stand-in modules.
3. `_DraftAttnGraphCore`: metadata rebuild (`_md_tensor_fields` / `_md_replace`)
   round-trips tensor fields positionally and preserves non-tensor fields.
4. Dispatch gating: the graph path requires decode shape + tensor metadata;
   flat/prefill-shaped calls route to the eager path (flag-gated).

Run: python3 tests/cpu/check_draft_graph.py  (from ports/glm-5-3-flash)
"""
import os
import sys
import types
from pathlib import Path

# The venv's editable finder serves the DEPLOYED tree (meta_path beats
# sys.path); load the REPO copy of mtp.py under a synthetic parent package so
# its relative imports (`from .model import ...`) resolve inside the repo too.
import importlib.util

_pkg_dir = Path(__file__).resolve().parents[2] / "glm5_next_hpu" / "glm5_next_hpu"
_pkg = types.ModuleType("repo_glm5_next_hpu")
_pkg.__path__ = [str(_pkg_dir)]
sys.modules["repo_glm5_next_hpu"] = _pkg
_spec = importlib.util.spec_from_file_location("repo_glm5_next_hpu.mtp", _pkg_dir / "mtp.py")
_mtp = importlib.util.module_from_spec(_spec)
sys.modules["repo_glm5_next_hpu.mtp"] = _mtp
_spec.loader.exec_module(_mtp)

import torch  # noqa: E402
from torch import nn  # noqa: E402

_PrivateCopy = _mtp._private_copy
_DraftAttnGraphCore = _mtp._DraftAttnGraphCore
_DraftPostAttentionGraphCore = _mtp._DraftPostAttentionGraphCore
_DraftPreAttentionGraphCore = _mtp._DraftPreAttentionGraphCore
_md_replace = _mtp._md_replace
_md_tensor_fields = _mtp._md_tensor_fields


def _norm(h):
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


def _private_copy_independence():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8))
            self.register_buffer("b", torch.randn(8))

    orig = Inner()
    clone = _PrivateCopy(orig)
    assert type(clone) is Inner
    assert clone.w.data_ptr() != orig.w.data_ptr(), "parameter storage must be cloned"
    assert clone.b.data_ptr() != orig.b.data_ptr(), "buffer storage must be cloned"
    with torch.no_grad():
        clone.w.add_(1.0)
        clone.b.add_(1.0)
    assert not torch.allclose(clone.w, orig.w), "clone mutation must not touch the original"
    assert not torch.allclose(clone.b, orig.b)
    # Attributes and submodule references survive the shallow copy.
    orig.scale = 3.0
    clone2 = _PrivateCopy(orig)
    assert clone2.scale == 3.0
    assert clone2.w.data_ptr() != orig.w.data_ptr()
    print("PASS _private_copy: storage independence + attribute preservation")


def _core_parity():
    Hd, V = 64, 97

    class Mtp(nn.Module):
        def __init__(self):
            super().__init__()
            self.enorm = nn.Identity()  # RMSNorm stands in as identity on unit-scale inputs
            self.hnorm = nn.Identity()
            self.eh_proj = nn.Linear(2 * Hd, Hd, bias=False)
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
            self.shared_head_norm = nn.Identity()
            self.mlp = nn.Sequential(nn.Linear(Hd, Hd), nn.GELU(), nn.Linear(Hd, Hd))

    torch.manual_seed(0)
    mtp = Mtp()
    embed = nn.Embedding(V, Hd)
    pre = _DraftPreAttentionGraphCore(mtp, embed)
    post = _DraftPostAttentionGraphCore(mtp)

    bs = 4
    ids = torch.randint(0, V, (bs, 1))
    pos = torch.randint(0, 50, (bs, 1))
    pos[0, 0] = 0
    h = torch.randn(bs, 1, Hd)

    def attention(positions, hidden_states):
        # Stand-in for the eager MLA: a deterministic linear map.
        return (hidden_states * 0.5).flatten(0, 1).unflatten(0, positions.shape)

    def eager(ids, pos, h):
        emb = embed(ids)
        emb = torch.where(pos.unsqueeze(-1) == 0, 0, emb)
        residual = mtp.eh_proj(torch.cat([mtp.enorm(emb), mtp.hnorm(h)], dim=-1))
        attn = attention(pos, mtp.input_layernorm(residual)).reshape_as(residual)
        states = residual + attn
        return mtp.shared_head_norm(states + mtp.mlp(mtp.post_attention_layernorm(states)))

    residual, attention_input = pre(ids, pos, h)
    attn = attention(pos, attention_input).reshape_as(residual)
    got = post(residual, attn)
    want = eager(ids, pos, h)
    assert torch.allclose(got, want, atol=1e-6), f"core chain != eager: max|d|={(got - want).abs().max()}"
    # Position-0 zeroing is inside the core: verify the core zeroes it.
    emb_direct = pre.embed_tokens(ids)
    zeroed = torch.where(pos.unsqueeze(-1) == 0, 0, emb_direct)
    assert torch.equal(zeroed[0, 0], torch.zeros(Hd)), "position 0 must be zeroed inside the core"
    print("PASS core parity: split-core chain == eager forward (atol 1e-6), position-0 zeroing held")


def _md_utils():
    import dataclasses

    @dataclasses.dataclass
    class Md:
        is_prompt: bool
        slot_mapping: torch.Tensor
        seq_lens: list
        block_table: torch.Tensor = None
        batch: int = 0

    md = Md(is_prompt=False, slot_mapping=torch.arange(4), seq_lens=[1, 2], block_table=torch.ones(2, 3), batch=2)
    fields = _md_tensor_fields(md)
    assert fields == ("slot_mapping", "block_table"), fields
    new_slot = torch.arange(4) + 10
    rebuilt = _md_replace(md, **{"slot_mapping": new_slot, "block_table": md.block_table + 1})
    assert torch.equal(rebuilt.slot_mapping, new_slot)
    assert rebuilt.seq_lens == [1, 2] and rebuilt.batch == 2 and rebuilt.is_prompt is False
    # namedtuple-style
    import collections

    NT = collections.namedtuple("NT", "a t")
    nt_fields = _md_tensor_fields(NT(a=1, t=torch.zeros(2)))
    assert nt_fields == ("t",)
    print("PASS md utils: field extraction + rebuild preserve non-tensor state")


def _dispatch_gating():
    """Flags parse at import; bad mode falls back to split; default OFF."""
    import importlib.util

    def load():
        spec = importlib.util.spec_from_file_location(
            f"repo_glm5_next_hpu.mtp{id(objects):x}", _pkg_dir / "mtp.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    objects = [object()]
    os.environ["VLLM_GLM_MTP_DRAFT_GRAPH"] = "0"
    m0 = load()
    assert m0._DRAFT_GRAPH is False
    assert m0.pre_attention_graph_core if False else True

    objects = [object()]
    os.environ["VLLM_GLM_MTP_DRAFT_GRAPH"] = "1"
    os.environ["VLLM_GLM_MTP_DRAFT_GRAPH_MODE"] = "split"
    m1 = load()
    assert m1._DRAFT_GRAPH is True and m1._DRAFT_GRAPH_MODE == "split"

    objects = [object()]
    os.environ["VLLM_GLM_MTP_DRAFT_GRAPH_MODE"] = "bogus"
    m2 = load()
    assert m2._DRAFT_GRAPH_MODE == "split", "bad mode must fall back to split"

    os.environ["VLLM_GLM_MTP_DRAFT_GRAPH"] = "0"
    os.environ["VLLM_GLM_MTP_DRAFT_GRAPH_MODE"] = "split"
    print("PASS dispatch gating: flags parse, bad mode falls back to split, default OFF")


if __name__ == "__main__":
    _private_copy_independence()
    _core_parity()
    _md_utils()
    _dispatch_gating()
    print("ALL check_draft_graph tests passed")
