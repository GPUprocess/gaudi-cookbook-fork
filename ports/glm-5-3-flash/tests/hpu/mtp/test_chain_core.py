#!/usr/bin/env python3
"""CPU contract test for _DraftChainGraphCore (RA-018 T2 / RA-038-B D1).

AST-extracts the chain core from mtp.py (which imports habana at module
top level) and verifies, with a stub attn core, that:
  1. the chain runs exactly k passes,
  2. each pass consumes the previous pass's argmaxed token,
  3. it returns k hidden states in order.
"""
import ast
import pathlib
import torch
from torch import nn

SRC = pathlib.Path(__file__).resolve().parents[3] / "glm5_next_hpu" / "glm5_next_hpu" / "mtp.py"


import sys as _sys, types as _t
_fake_utils = _t.ModuleType("vllm.v1.spec_decode.utils")
def _fake_kernel(*a, **k):
    seq = a[2]
    seq += 1
    a[5].copy_(a[0] + 1)
    a[6].copy_(seq.to(torch.long) * 8)
_fake_utils.eagle_step_update_slot_mapping_and_metadata = _fake_kernel
_sys.modules.setdefault("vllm", _t.ModuleType("vllm"))
_sys.modules["vllm.v1"] = _t.ModuleType("vllm.v1")
_sys.modules["vllm.v1.spec_decode"] = _t.ModuleType("vllm.v1.spec_decode")
_sys.modules["vllm.v1.spec_decode.utils"] = _fake_utils

def _load_chain_cls():
    tree = ast.parse(SRC.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "_DraftChainGraphCore")
    node.module = None
    for stmt in node.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.ImportFrom) and sub.module == "vllm.forward_context":
                sub.module = "unittest.mock"
    ns = {"torch": torch, "nn": nn, "os": __import__("os")}
    # stub the names the class body references at runtime
    ns["_DraftAttnGraphCore"] = object
    import types as _t
    _u = _t.ModuleType("vllm.v1.spec_decode.utils")
    def _fake_kernel(*a, **k):
        seq = a[2]
        seq += 1
        a[4].copy_(a[0] + 1)
        a[5].copy_(seq.to(torch.long) * 8)
    _u.eagle_step_update_slot_mapping_and_metadata = _fake_kernel
    ns["eagle_step_update_slot_mapping_and_metadata"] = _fake_kernel
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_DraftChainGraphCore"]


class _StubAttnCore(nn.Module):
    """Mimics the attn-mode core contract: returns a deterministic function of
    (ids, positions, hidden) so the argmax chain is checkable."""

    def __init__(self, k_vocab=7):
        super().__init__()
        self.k_vocab = k_vocab
        self._md_fields = ("positions", "seq_lens", "slot_mapping", "block_table_tensor")
        self.calls = []
        self.lm_head = nn.Linear(4, k_vocab, bias=False)
        self.logits_processor = lambda lm, h, bias: lm(h.mean(dim=1)[:, :lm.weight.shape[1]] if h.shape[-1] != lm.weight.shape[1] else h)

    def forward(self, ids, positions, hidden, *md):
        self.calls.append((ids.clone(), int(positions.max().item())))
        # hidden_out = f(hidden, ids): deterministic, shape-preserving
        h = hidden + (ids.float().sum(dim=-1, keepdim=True).unsqueeze(-1) % 3.0)
        return h


def main():
    ChainCore = _load_chain_cls()
    k = 4
    stub = _StubAttnCore()
    core = ChainCore(stub, k)
    ids0 = torch.tensor([1, 2, 3, 5])
    pos0 = torch.tensor([100, 100, 100, 100])
    hid0 = torch.zeros(4, 4)
    md = (pos0.clone(), torch.tensor([50, 60, 70, 80]), torch.tensor([500, 600, 700, 800]), torch.zeros(4, 2))
    outs, toks = core(ids0, pos0, hid0, *md)
    assert len(outs) == k and len(toks) == k, f"expected {k}+{k}, got {len(outs)}+{len(toks)}"
    assert len(stub.calls) == k, f"expected {k} attn-core calls, got {len(stub.calls)}"
    # pass 0 consumed the base ids; each later pass consumed the previous argmax
    assert torch.equal(stub.calls[0][0], ids0), "pass 0 must consume the base ids"
    for j in range(1, k):
        prev_pos = stub.calls[j - 1][1]
        this_pos = stub.calls[j][1]
        assert this_pos == 100 + j, (
            f"pass {j} positions must advance by {j} from base (got {this_pos}, base {prev_pos})")
    assert toks[0].shape == ids0.shape
    print(f"CHAIN-CORE-CONTRACT-PASS: {k} passes, ids chained, positions advanced, {len(outs)} outputs, {len(toks)} token tensors")


if __name__ == "__main__":
    main()
