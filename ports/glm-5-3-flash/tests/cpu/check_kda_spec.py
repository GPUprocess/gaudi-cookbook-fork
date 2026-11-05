# SPDX-License-Identifier: Apache-2.0
"""CPU parity tests for the GLM-5.3 KDA speculative-decode state machinery.

1. hpu_recurrent_kda(return_all_states=True) over [bs, L] == L sequential
   single-token calls (chaining), and state after token j == all_states[:, j].
2. _kda_load_slots column selection (canonical / candidate / clamp / 1-D).
3. _scatter_candidate_states identity guard for -1 (padding) targets.
Run: python3 test_kda_spec_cpu.py  (no HPU, no vLLM imports)
"""

import sys

import torch

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "glm5_next_hpu"))


def main():
    from glm5_next_hpu.kda import hpu_recurrent_kda
    from glm5_next_hpu.kda_layer import _kda_load_slots, _scatter_candidate_states

    torch.manual_seed(3)
    B, L, H, D, V = 3, 2, 4, 8, 8
    q = torch.randn(B, L, H, D)
    k = torch.randn(B, L, H, D)
    v = torch.randn(B, L, H, V)
    g = -torch.rand(B, L, H, D) * 0.5 - 0.1  # negative decay (log-domain)
    beta = torch.rand(B, L, H)
    s0 = torch.randn(B, H, D, V)

    # chained multi-token call with per-position states
    out_chain, _, states = hpu_recurrent_kda(q, k, v, g, beta, initial_state=s0, return_all_states=True)
    assert states.shape == (B, L, H, D, V), states.shape

    # sequential reference: L single-token calls, chaining states
    state = s0.clone()
    outs_seq, states_seq = [], []
    for j in range(L):
        o, s = hpu_recurrent_kda(
            q[:, j : j + 1], k[:, j : j + 1], v[:, j : j + 1], g[:, j : j + 1], beta[:, j : j + 1],
            initial_state=state, output_final_state=True,
        )
        outs_seq.append(o[:, 0])
        states_seq.append(s)
        state = s
    out_seq = torch.stack(outs_seq, dim=1)
    states_seq = torch.stack(states_seq, dim=1)

    assert torch.allclose(out_chain, out_seq, atol=1e-5), "chained output != sequential"
    assert torch.allclose(states, states_seq, atol=1e-4), "per-position states != sequential"
    print("recurrence chaining parity: OK")

    # state after token j must equal all_states[:, j] and the final chain state == states[:, -1]
    _, final = hpu_recurrent_kda(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    assert torch.allclose(final, states[:, -1], atol=1e-6), "final != last per-position state"
    print("final-state consistency: OK")

    # ---- slot selection ----
    idx = torch.arange(3 * 6, dtype=torch.int32).view(3, 6)  # [bs, S=6] canonical + 5 candidates
    acc = torch.tensor([0, 3, 5], dtype=torch.int32)
    got = _kda_load_slots(idx, acc, 3)
    assert got.tolist() == [0, 9, 17], got.tolist()  # acc=0 -> col0; acc=3 -> col3; acc=5 -> col5
    assert _kda_load_slots(idx, torch.tensor([99, 0], dtype=torch.int32), 2).tolist() == [5, 6]  # clamp to last col
    assert _kda_load_slots(idx, None, 2).tolist() == [0, 6]  # no info -> canonical
    flat = torch.tensor([9, 8, 7], dtype=torch.int32)
    assert _kda_load_slots(flat, None, 2).tolist() == [9, 8]  # 1-D pass-through
    print("load-slot selection: OK")

    # ---- scatter guard ----
    pool = torch.zeros(8, 2, 2)
    pool[3] = torch.ones(2, 2) * 7
    tgt = torch.tensor([1, -1, 3])
    new = torch.arange(12, dtype=torch.float32).view(3, 2, 2) + 100
    _scatter_candidate_states(pool, tgt, new)
    assert torch.allclose(pool[1], new[0]), "live target must be written"
    assert torch.allclose(pool[3], new[2]), "second live target must be written"
    assert torch.allclose(pool[7], torch.zeros(2, 2)), "-1 target must be an identity write (wraps to slot 7)"
    print("scatter pad guard: OK")

    print("KDA spec CPU parity: ALL OK")


if __name__ == "__main__":
    sys.exit(main())
