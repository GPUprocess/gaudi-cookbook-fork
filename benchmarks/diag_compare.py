#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two I015 layer-boundary payloads (the delta map).

Usage: diag_compare.py capture_a.pt capture_b.pt [--max-boundaries N]

For each boundary name present in both payloads: relL2 and max_abs of the
difference, plus the shape/dtype. The reference interpretation (abotsis
findings doc sections 1-3): ~1 bf16 ULP at the attention output (relL2
3-4e-3), bounded 2-6e-2 amplification through mHC across layers, and no
O(1) blow-up (which would indicate a logic error rather than a numeric
shape shift).
"""
import argparse
import json

import torch


def rel_stats(a, b):
    diff = (a.float() - b.float()).abs()
    denom = a.float().abs().max().clamp_min(1e-12)
    rel_l2 = (diff.pow(2).sum().sqrt() / a.float().pow(2).sum().sqrt().clamp_min(1e-12)).item()
    return rel_l2, diff.max().item(), denom.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture_a")
    ap.add_argument("capture_b")
    ap.add_argument("--max-boundaries", type=int, default=50)
    args = ap.parse_args()

    a = torch.load(args.capture_a, weights_only=False)
    b = torch.load(args.capture_b, weights_only=False)
    pa = {x["name"]: x for x in a["payload"]["boundaries"]}
    pb = {x["name"]: x for x in b["payload"]["boundaries"]}
    common = sorted(set(pa) & set(pb))
    only_a = sorted(set(pa) - set(pb))
    only_b = sorted(set(pb) - set(pa))

    print(f"capture A: {args.capture_a} ({len(pa)} boundaries, "
          f"{a['payload'].get('bytes', 0)} bytes)")
    print(f"capture B: {args.capture_b} ({len(pb)} boundaries, "
          f"{b['payload'].get('bytes', 0)} bytes)")
    if only_a:
        print(f"only in A: {only_a}")
    if only_b:
        print(f"only in B: {only_b}")

    print(f"\n{'boundary':<44} {'relL2':>10} {'max_abs':>10}  shape")
    for name in common[: args.max_boundaries]:
        ta, tb = pa[name]["rows"], pb[name]["rows"]
        if ta.shape != tb.shape:
            print(f"{name:<44} {'SHAPE MISMATCH':>10} {ta.shape} vs {tb.shape}")
            continue
        rel, mx, _ = rel_stats(ta, tb)
        print(f"{name:<44} {rel:>10.3e} {mx:>10.3e}  {tuple(ta.shape)} {pa[name]['dtype']}")

    rec_a = {r["request_id"]: r for r in a["payload"]["records"]}
    rec_b = {r["request_id"]: r for r in b["payload"]["records"]}
    shared = sorted(set(rec_a) & set(rec_b))
    if shared:
        print("\nrecord history (admitted requests):")
        for rid in shared:
            ra, rb = rec_a[rid], rec_b[rid]
            same = (ra["prompt_token_ids"] == rb["prompt_token_ids"]
                    and ra["output_token_ids"] == rb["output_token_ids"])
            print(f"  {rid}: pos {ra['output_position']}, "
                  f"last token {ra['last_token']}, history match: {same}")


if __name__ == "__main__":
    main()
