# SPDX-License-Identifier: Apache-2.0
"""Routed-expert group layout for GLM-5.3-Flash at TP 8 (Q17, ``GLM_MOE_TP``).

Adapted from the glm-5-mxfp4 port's moe_layout.py for the block-FP8 checkpoint
(unpacked E4M3 weights + FP32 128x128 block scales; the mxfp4 packed-u8 handling
is dropped, the row/column block slicing is kept).

With ``--enable-expert-parallel`` vLLM puts 36 whole experts on each of the 8
ranks (EP 8, contiguous). Every decode MoE layer waits at its all-reduce for the
rank with the most local routes: 16 routes at one stream (verify token + MTP
draft, top-8 each) into 8 ranks give the busiest rank ~4.5 routes against a mean
of 2 (balls in bins). The fix is a wider layout:

``GLM_MOE_TP=2`` (unset or 1: EP 8)
    The 8 ranks form 4 expert groups of 2 ranks ("EP 4 x TP 2"). Group g holds
    experts [72g, 72g+72); inside a group, rank r holds rows [r I/2, (r+1) I/2)
    of each expert's gate+up (w13) and columns [r I/2, (r+1) I/2) of the down
    (w2). Every rank still holds 36 experts' worth of weights. Outputs differ
    from EP 8 by bf16 rounding (each route's down projection sums over two
    partial slices in the all-reduce).

``GLM_MOE_PLACEMENT=<file.json>`` (unset: contiguous groups)
    Per-layer expert placement, as in the mxfp4 module (the router ids map to
    placement slots through a per-layer table).
"""

import ctypes
import json
import os
import re
import time

import torch

TP = int(os.environ.get("GLM_MOE_TP", "1") or 1)
PLACEMENT = os.environ.get("GLM_MOE_PLACEMENT", "")
_cache = {}


def group_layout(tp_size, tp_rank, dp_size, pcp_size):
    """(moe tp_size, moe tp_rank, ep_size, ep_rank) for GLM_MOE_TP, or None for vLLM's own EP layout."""
    t = TP
    if t > 1 and dp_size == 1 and pcp_size == 1 and tp_size % t == 0 and tp_size > t:
        return t, tp_rank % t, tp_size // t, tp_rank // t
    return None


def placement():
    if "layers" not in _cache:
        layers = {}
        if PLACEMENT:
            with open(PLACEMENT) as f:
                d = json.load(f)
            ne = int(d["n_experts"])
            for k, v in d["layers"].items():
                v = [int(x) for x in v]
                if sorted(v) != list(range(ne)):
                    raise ValueError(f"{PLACEMENT}: layer {k} is not a permutation of 0..{ne - 1}")
                layers[int(k)] = v
            print(f"[glm-moe] placement {PLACEMENT}: {len(layers)} layers, {ne} experts, {d['n_ranks']} groups")
        _cache["layers"] = layers
    return _cache["layers"]


def experts_by_slot(layer_name):
    if not PLACEMENT:
        return None
    m = re.search(r"layers\.(\d+)\.", layer_name or "")
    return placement().get(int(m.group(1))) if m else None


def load_fused(layer, param, loaded_weight, shard_id):
    """Group-layout loader hook for the block-FP8 MoE params (w13 / w2 and their
    128x128 block scales). `loaded_weight` is the combined [E, d1, d2] checkpoint
    tensor; `shard_id` discriminates w13 vs w2 (the scale tensors share the row /
    column structure of their weight at the block granularity). Returns True when
    this rank's group slice was copied, False to let the stock loader run."""
    pc = layer.moe_config.moe_parallel_config
    if not (pc.tp_size > 1 and pc.use_ep):
        return False
    t, r = pc.tp_size, pc.tp_rank
    n_local = param.shape[0]
    num_e = loaded_weight.shape[0]
    if num_e <= n_local:
        return False
    ep = pc.ep_rank
    ebs = experts_by_slot(getattr(layer, "layer_name", ""))
    ids = ebs[ep * n_local : (ep + 1) * n_local] if ebs is not None else list(range(ep * n_local, (ep + 1) * n_local))
    d1, d2 = loaded_weight.shape[1], loaded_weight.shape[2]
    p1, p2 = param.shape[1], param.shape[2]
    if shard_id in ("w1", "w3", "w13"):
        if d1 != t * p1 or d2 != p2:
            return False
        _load_w13_rows(param.data, loaded_weight, ids, r, t)
        return True
    if shard_id == "w2":
        if d2 != t * p2 or d1 != p1:
            return False
        _load_w2_cols(param.data, loaded_weight, ids, r, t)
        return True
    return False


def _load_w13_rows(p, lw, ids, r, t):
    """w13 [E, 2I, K] (or its scale [E, 2I/b, K/b]): rank r copies gate rows
    [r*I/t, (r+1)*I/t) and up rows [I + r*I/t, ...) of each expert in `ids`."""
    i2 = lw.shape[1]  # 2I (weights) or 2I/b (scales)
    h = i2 // 2 // t
    half = i2 // 2
    for i, e in enumerate(ids):
        p[i, :h].copy_(lw[e, r * h : r * h + h])
        p[i, h : 2 * h].copy_(lw[e, half + r * h : half + r * h + h])


def _load_w2_cols(p, lw, ids, r, t):
    """w2 [E, K, I] (or its scale [E, K/b, I/b]): rank r copies the column block
    [r*I/t, (r+1)*I/t) of each expert in `ids`. Single-threaded gather (the
    mxfp4 module's finding: multi-threaded host gathers thrash the mmap)."""
    w = lw.shape[2] // t
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        p[:, :, :w].copy_(
            lw.narrow(2, r * w, w).index_select(0, torch.tensor(ids, dtype=torch.long))
        )
    finally:
        torch.set_num_threads(nt)


def map_ids(layer, topk_ids):
    """Router hook: global expert ids -> slot ids (no-op without a placement for this layer)."""
    tab = getattr(layer, "_glm_slot_of", None)
    if tab is None:
        return topk_ids
    return torch.index_select(tab, 0, topk_ids.reshape(-1)).view(topk_ids.shape)
