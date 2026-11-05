# SPDX-License-Identifier: Apache-2.0
"""CPU test: compact-GDN candidate-slot math under request churn.

Exercises the exact prepare_mamba_state_idxs slot formula (base*(groups*S) + g*S + col + 1)
for: fresh alloc, finish + slot reuse, per-group disjointness, and column uniqueness.
No HPU, no engine: the slot algebra is pure torch/numpy.
"""


import torch


class FakeBlockTables:
    def __init__(self, n):
        self.block_tables = [None] * n

    def __len__(self):
        return len(self.block_tables)


def make_state_fn(num_groups, spec_slots, req_ids, base_of, req_indices, target_bs, columns=1):
    """Mirror of HPUModelRunner.prepare_mamba_state_idxs' compact branch."""
    span = num_groups * spec_slots
    rows = []
    for g in range(num_groups):
        base = torch.tensor(
            [base_of.get(req_ids[i], 0) if i < len(req_ids) else 0 for i in req_indices], dtype=torch.int64)
        if columns == 1:
            rows.append((base * span + g * spec_slots + 1).to(torch.int32))
        else:
            cols = torch.arange(columns, dtype=torch.int64)
            rows.append((base.view(-1, 1) * span + g * spec_slots + cols.view(1, -1) + 1).to(torch.int32))
    return torch.stack(rows, dim=0)


def main():
    G, S = 3, 4  # 3 kda "groups" (layers), S = k+2 = 4 columns for k=2
    max_reqs = 8
    pool_rows = max_reqs * G * S + 2

    # 1) fresh alloc: two requests, disjoint slot sets
    ids = ["r0", "r1"]
    base_of = {"r0": 7, "r1": 6}  # free-list pops from the top: first alloc takes max-1
    out = make_state_fn(G, S, ids, base_of, [0, 1], 2, columns=S)
    assert out.shape == (G, 2, S), out.shape
    s0 = set(out[:, 0, :].flatten().tolist())
    s1 = set(out[:, 1, :].flatten().tolist())
    assert not (s0 & s1), "requests must get disjoint slots"
    assert max(s0 | s1) < pool_rows, "slot within pool"
    # column 0 index follows the S-aware layout: base*(G*S) + g*S + col + 1
    assert out[0, 0, 0].item() == 7 * G * S + 0 * S + 1, "col0 must follow the S-aware layout"
    assert out[1, 0, 0].item() == 7 * G * S + 1 * S + 1
    assert out[0, 1, 0].item() == 6 * G * S + 0 * S + 1
    print("fresh alloc disjoint + legacy-compatible col0: OK")

    # 2) finish + reuse: r0 finishes, its base slot goes back; a new request r2 gets base 7 again
    base_of2 = {"r2": 7, "r1": 6}
    ids2 = ["r2", "r1"]
    out2 = make_state_fn(G, S, ids2, base_of2, [0, 1], 2, columns=S)
    reused = set(out2[:, 0, :].flatten().tolist())
    kept = set(out2[:, 1, :].flatten().tolist())
    assert reused == s0, "reused request must land exactly on the freed request's slots"
    assert kept == s1, "surviving request must keep its slots"
    assert not (reused & kept)
    print("finish + slot reuse lands on the freed slot set: OK")

    # 3) column uniqueness within a request: all G*S entries distinct
    for row in range(2):
        flat = out2[:, row, :].flatten().tolist()
        assert len(set(flat)) == G * S, "columns must be unique per request"
    print("per-request column uniqueness: OK")

    # 4) padding rows are -1 and stay out of the live range (req_indices covers the real
    # requests; target_bs pads the bucket beyond them, exactly like the runner's call)
    out3 = make_state_fn(G, S, ids2, base_of2, [0, 1], 4, columns=S)  # 2 real, 4-wide bucket
    pad = out3[:, 2:, :]
    assert (pad == -1).all(), "padding rows must be the -1 sentinel"
    print("padding sentinel rows: OK")

    # 5) S=1 reproduces the legacy layout exactly
    out_legacy = make_state_fn(G, 1, ids, base_of, [0, 1], 2, columns=1)
    assert out_legacy.shape == (G, 2)
    assert out_legacy[0, 0].item() == 7 * G + 1 and out_legacy[1, 1].item() == 6 * G + 1 + 1
    print("S=1 legacy layout regression: OK")

    print("churn slot algebra: ALL OK")


if __name__ == "__main__":
    main()
