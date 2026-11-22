# SPDX-License-Identifier: Apache-2.0
"""KDA attention layer for GLM-5.3-Flash on vllm-gaudi.

A subclass of vLLM's `KimiGatedDeltaNetAttention`: the parameters (projections, `dt_bias`, `A_log`, the three short
convolutions, `o_norm`, `o_proj`) and the state shape and dtype are inherited, so checkpoint names match. Only
`_forward` is replaced. The upstream one reads a per-layer dict of `GDNAttentionMetadata` and calls Triton kernels;
vllm-gaudi publishes one metadata object with attributes (`is_prompt`, `load_indices_tensor`, `query_start_loc_p`,
`has_initial_states_p`, `padding_mask_flat`), like its own Qwen3.5 GDN layer.

Differences from vllm-gaudi's GDN layer that matter here: KDA has three separate q/k/v short convolutions over one
conv state split three ways, the decay is per channel, and the gate is GLM's bounded sigmoid.
"""

from __future__ import annotations

import os

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn, hpu_causal_conv1d_update

from .kda import KdaConsts, hpu_chunk_kda, hpu_chunk_kda_scan, hpu_kda_gate, hpu_recurrent_kda, make_kda_consts

# Prefill chunk geometry. The layer builds the kernel's constant buffers for exactly this geometry.
_KDA_CHUNK = int(os.environ.get("GLM5_NEXT_KDA_CHUNK", "64"))
_KDA_SUB = 16

# GLM5_NEXT_KDA_SCAN=1: prefill carries the state between chunks with the log-depth scan; 0 (default) runs the
# reference loop. Read once per process so every rank takes the same path.
KDA_SCAN = os.environ.get("GLM5_NEXT_KDA_SCAN", "0") == "1"

logger = init_logger(__name__)


@torch._dynamo.disable
def _save_ssm_state(core_attn_out, ssm_state, final_state, state_indices):
    """Write the final recurrent states into the cache and pass `core_attn_out` through.

    Same shape as vllm-gaudi's GDN layer: outside dynamo (torch.compile drops an in-place index_copy_ into an aliased
    state), and the caller must use the return value (a call whose result is unused can be eliminated, which also
    broke graph replay with "Neither storage attached to input tensor").
    """
    safe = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
    return core_attn_out


@torch._dynamo.disable
def _scatter_candidate_states(state_tensor, tgt, new_rows):
    """index_copy_ candidate rows with an identity guard for padding (-1) targets.

    `tgt`: [R] slot ids (may contain -1 for padded lanes); `new_rows`: [R, ...]. A -1 target would wrap onto the
    pool's last live slot via remainder(), so those rows write back their current value instead.
    """
    pool = state_tensor.shape[0]
    flat_idx = torch.remainder(tgt, pool).long()
    cur = state_tensor.index_select(0, flat_idx)
    new_rows = torch.where((tgt < 0).view(-1, *([1] * (new_rows.dim() - 1))), cur, new_rows.to(cur.dtype))
    state_tensor.index_copy_(0, flat_idx, new_rows)
    return state_tensor


def _kda_load_slots(state_indices, num_accepted_tokens, n):
    """Slot holding the INCOMING recurrent state for each decode sequence.

    Under speculative decode the per-group indices are [bs, 1 + num_spec + 1]: column 0 is the CANONICAL slot (where
    prefill stored the post-prompt state) and column 1+j is private candidate slot j, holding the state after chain
    token j of the previous verify step. ``num_accepted`` selects directly: 0 is the fresh sentinel (resume from
    canonical), j >= 1 resumes from candidate j-1. Without speculation the indices are 1-D/2-D col-0 and there is
    exactly one slot. (Slot layout mirrors the validated abotsis GLM-5.3 MTP implementation.)
    """
    if state_indices is None:
        return None
    if state_indices.dim() < 2 or state_indices.shape[-1] == 1:
        return state_indices.reshape(-1)[:n]
    slots = state_indices.shape[-1]
    if num_accepted_tokens is None:
        return state_indices[:n, 0].contiguous()
    # [glm-mtp] num_accepted_tokens is built from the real (unpadded) requests; the padded batch is wider.
    # Pad with the fresh sentinel (0 = canonical) so gather keeps bs_rows rows — padding lanes are -1-guarded
    # at the store anyway.
    if num_accepted_tokens.numel() < n:
        pad = torch.zeros(n - num_accepted_tokens.numel(), dtype=num_accepted_tokens.dtype,
                          device=num_accepted_tokens.device)
        num_accepted_tokens = torch.cat([num_accepted_tokens, pad])
    col = torch.clamp(num_accepted_tokens[:n].long(), 0, slots - 1)
    return state_indices[:n].gather(1, col.view(-1, 1)).squeeze(1).contiguous()


def _a_log_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Shard `A_log` over heads. The checkpoint stores it flat, [num_heads]; vLLM's Kimi layer declares
    [1, 1, local_heads, 1] and narrows axis 2, which fails on a 1-D tensor (only visible at TP > 1)."""
    if loaded_weight.dim() == 1:
        loaded_weight = loaded_weight.view(1, 1, -1, 1)
    shard = param.data.shape[2]
    start = get_tensor_model_parallel_rank() * shard
    return default_weight_loader(param, loaded_weight.narrow(2, start, shard))


class Glm5NextHPUKDAAttention(KimiGatedDeltaNetAttention):
    def __init__(self, *args, gate_lower_bound: float, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the inherited A_log loader through set_weight_attrs, which is where vllm-gaudi wraps loaders with the
        # torch.hpu.synchronize() every HPU weight load needs. It refuses to overwrite, so drop the old one first.
        if hasattr(self.A_log, "weight_loader"):
            del self.A_log.weight_loader
        set_weight_attrs(self.A_log, {"weight_loader": _a_log_weight_loader})

        # Kernel constants as non-persistent module buffers: owned and placed like weights, so they are stable across
        # every captured graph, and absent from the state dict.
        consts = make_kda_consts(_KDA_CHUNK, _KDA_SUB, torch.float32, self.A_log.device)
        self.register_buffer("kda_eye", consts.eye, persistent=False)
        self.register_buffer("kda_keep_strict", consts.keep_strict, persistent=False)
        self.register_buffer("kda_keep_diag", consts.keep_diag, persistent=False)
        if KDA_SCAN:
            self.register_buffer(
                "kda_eye_k", torch.eye(self.head_dim, dtype=torch.float32, device=self.A_log.device), persistent=False
            )

        # This layer implements the state contract without prefix caching: one index for load and store, one final
        # state per sequence. With prefix caching vllm-gaudi splits the load and store indices and stores one state per
        # chunk, so a reused prefix would start from another request's state. Refuse rather than corrupt silently.
        if getattr(getattr(self, "cache_config", None), "enable_prefix_caching", False):
            raise ValueError(
                "GLM-5.3-Flash KDA does not support prefix caching on vllm-gaudi; serve with --no-enable-prefix-caching"
            )

        self.gate_lower_bound = gate_lower_bound
        # Set by the model runner (cache-group-idx patch): which row of the 2-D state index tensor is this layer's.
        self.cache_group_idx = None
        self._cg_int = None

    @property
    def kda_consts(self) -> KdaConsts:
        return KdaConsts(self.kda_eye, self.kda_keep_strict, self.kda_keep_diag)

    def _resolve_state_indices(self, attn_metadata, field: str = "load_indices_tensor"):
        """This layer's row of the per-cache-group state index tensor, as a static slice (resolved to an int once).

        Under MTP speculative decode the tensor is 3-D ([groups, bs, 1 + num_spec + 1]); the squeeze leaves a [bs, S]
        view whose base is not a graph input, so materialise it with index_select (replay-safe).
        """
        idx = getattr(attn_metadata, field, None)
        if idx is not None and idx.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None, "cache_group_idx not set for a multi-D state index"
            if self._cg_int is None:
                self._cg_int = int(cg.item()) if torch.is_tensor(cg) else int(cg)
            # int indexing returns a strided view; .contiguous() materialises it (a strided view of the group
            # tensor replays as "Neither storage attached to input tensor" under HPU graphs).
            idx = idx[self._cg_int].contiguous()
        return idx

    def _forward(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out) -> None:
        md = get_forward_context().attn_metadata
        if md is None:
            return  # profile run
        if isinstance(md, dict):
            md = md.get(self.prefix, md)

        for apc in ("mamba_chunks_to_block_mapping", "blocks_caching_range", "seqlens_offsets_for_blocks"):
            if getattr(md, apc, None) is not None:
                raise RuntimeError(
                    f"GLM-5.3-Flash KDA saw `{apc}`: prefix caching is on, which this layer does not support. "
                    "Serve with --no-enable-prefix-caching."
                )

        is_prompt = bool(getattr(md, "is_prompt", False))
        state_indices = self._resolve_state_indices(md)
        store_indices = self._resolve_state_indices(md, "store_indices_tensor")
        if store_indices is None:
            store_indices = state_indices
        num_accepted_tokens = getattr(md, "num_accepted_tokens", None)
        query_start_loc = getattr(md, "query_start_loc_p", None)
        if query_start_loc is None:
            query_start_loc = getattr(md, "query_start_loc", None)
        has_initial_state = getattr(md, "has_initial_states_p", None)
        # vllm-gaudi pads a prompt batch to its bucket. Upstream's _forward trims the padding itself; this override
        # must zero it, or the padded tail runs through the recurrence into the state decode starts from.
        padding_mask_flat = getattr(md, "padding_mask_flat", None)

        if state_indices is None and query_start_loc is None:
            return  # warmup / profile metadata carries no state map

        conv_state, ssm_state = self.kv_cache[0], self.kv_cache[1]
        n = q_proj_states.size(0)

        # Mask before the conv as well as after: the padded tail can hold NaN, and the conv writes its state from the
        # tail of its input. nan_to_num because 0 * NaN is NaN.
        if padding_mask_flat is not None:
            m = padding_mask_flat.reshape(-1, 1)
            q_proj_states = torch.nan_to_num(q_proj_states * m.to(q_proj_states.dtype))
            k_proj_states = torch.nan_to_num(k_proj_states * m.to(k_proj_states.dtype))
            v_proj_states = torch.nan_to_num(v_proj_states * m.to(v_proj_states.dtype))

        # The conv state is (kernel - 1, 3 * dim) with q, k, v side by side on the last axis, the layout vllm-gaudi's
        # conv kernels read. The kernels update their state in place, and an in-place write through a strided view of
        # a graph input fails at replay, so they work on contiguous copies that are written back as one tensor.
        dim3 = conv_state.shape[-1] // 3
        cs_q = conv_state[..., 0:dim3].contiguous()
        cs_k = conv_state[..., dim3 : 2 * dim3].contiguous()
        cs_v = conv_state[..., 2 * dim3 :].contiguous()

        def cw(conv):
            return conv.weight.view(conv.weight.size(0), conv.weight.size(2))

        # Tokens per decode sequence this step: 1 normally, num_spec+1 when verifying an MTP draft chain. Under
        # speculative decode the per-group state indices are 3-D before the group-row squeeze: [bs, S] with column 0
        # = canonical slot and columns 1+j = candidate slot j (state/window after chain token j).
        spec_len = 1
        spec_cols = None
        if not is_prompt and state_indices is not None:
            bs_rows = state_indices.shape[0]
            if bs_rows > 0 and n % bs_rows == 0:
                spec_len = n // bs_rows
            if state_indices.dim() >= 2 and state_indices.shape[-1] > 1:
                spec_cols = state_indices
                store_cols = store_indices if store_indices.dim() >= 2 else None

        if is_prompt:
            conv_args = dict(
                query_start_loc=query_start_loc,
                cache_indices=state_indices,
                has_initial_state=has_initial_state,
                activation=self.activation,
                block_size_to_align=self.cache_config.mamba_block_size,
                is_prompt=True,
            )
            q, k, v = (
                hpu_causal_conv1d_fn(
                    x=x.transpose(0, 1), weight=cw(conv), bias=conv.bias, conv_states=cs, **conv_args
                ).transpose(0, 1)
                for x, conv, cs in (
                    (q_proj_states, self.q_conv1d, cs_q),
                    (k_proj_states, self.k_conv1d, cs_k),
                    (v_proj_states, self.v_conv1d, cs_v),
                )
            )
        elif spec_cols is not None and spec_len > 1:
            # --- MTP verify: chained depthwise conv over each sequence's draft chain.
            # Runs the whole chain from the canonical window (column-selected by num_accepted), writes the window
            # after token j to candidate column 1+j, and re-writes the loaded window to column 0 so the canonical
            # slot always holds the freshest committed state (padded lanes are guarded to identity writes).
            conv_state_shape = conv_state.shape
            pool = conv_state_shape[0]
            bs_rows = spec_cols.shape[0]
            if not getattr(self, "_spec_shapes_logged", False):
                self._spec_shapes_logged = True
                logger.warning(
                    "GLM KDA spec shapes: conv_state=%s dim3=%s q=%s k=%s v=%s spec_cols=%s store=%s "
                    "qconv_w=%s qconv_b=%s",
                    tuple(conv_state.shape), dim3, tuple(q_proj_states.shape), tuple(k_proj_states.shape),
                    tuple(v_proj_states.shape), tuple(spec_cols.shape), tuple(store_cols.shape),
                    tuple(self.q_conv1d.weight.shape),
                    tuple(self.q_conv1d.bias.shape) if self.q_conv1d.bias is not None else None,
                )
            load_slots = _kda_load_slots(spec_cols, num_accepted_tokens, bs_rows)
            win = conv_state.index_select(0, torch.remainder(load_slots, pool).long())  # [bs, W-1, 3*dim3]
            x_cat = torch.cat((q_proj_states, k_proj_states, v_proj_states), dim=-1).view(bs_rows, spec_len, -1)
            w_cat = torch.cat([cw(c) for c in (self.q_conv1d, self.k_conv1d, self.v_conv1d)], dim=0)
            b_cat = torch.cat(
                [c.bias if c.bias is not None else torch.zeros(cw(c).shape[0], dtype=w_cat.dtype, device=w_cat.device)
                 for c in (self.q_conv1d, self.k_conv1d, self.v_conv1d)], dim=0)
            width = self.q_conv1d.weight.shape[-1]
            C = x_cat.shape[-1]
            w_t = w_cat.t().contiguous()  # [W, C]
            bias_c = b_cat.to(x_cat.dtype) if b_cat is not None else None
            # out[b, j, c] = silu(sum_t win_or_x[b, j+t, c] * w[c, t] + b[c]) — an explicit per-token chain:
            # Tensor.unfold hits a CPU fallback under HPU graph capture and Synapse rejects broadcast forms
            # here, so every op is same-shape elementwise over [bs, W, C].
            cur_win = win.contiguous()  # [bs, W-1, C]
            outs, new_wins = [], []
            try:
                for j in range(spec_len):
                    full = torch.cat([cur_win, x_cat[:, j].unsqueeze(1)], dim=1).contiguous()  # [bs, W, C]
                    w_exp = w_t.reshape(1, width, C).expand(bs_rows, width, C)
                    o = (full * w_exp).sum(dim=1) + (bias_c.view(1, -1) if bias_c is not None else 0)
                    outs.append(torch.nn.functional.silu(o))
                    cur_win = full[:, 1:].contiguous()  # window after token j
                    new_wins.append(cur_win)
            except RuntimeError:
                logger.warning(
                    "GLM KDA conv-chain failure: win=%s x_cat=%s w_t=%s b_cat=%s bs=%s L=%s C=%s",
                    tuple(win.shape), tuple(x_cat.shape), tuple(w_t.shape),
                    tuple(b_cat.shape) if b_cat is not None else None, bs_rows, spec_len, C,
                )
                raise
            out_chain = torch.stack(outs, dim=1)  # [bs, L, C]
            new_win = torch.stack(new_wins, dim=1)  # [bs, L, W-1, C]
            tgt = torch.cat([store_cols[:bs_rows, 0:1], store_cols[:bs_rows, 1 : 1 + spec_len]], dim=1).reshape(-1)
            rows = torch.cat([win.unsqueeze(1), new_win], dim=1).reshape(
                bs_rows * (spec_len + 1), win.shape[1], win.shape[2])
            _scatter_candidate_states(conv_state, tgt, rows)
            q, k, v = (t.reshape(bs_rows * spec_len, -1) for t in out_chain.split(dim3, dim=-1))
        else:
            if spec_cols is not None:
                # Column scheme: resume reads the num_accepted-selected slot (canonical or candidate) but the new
                # window must land in the CANONICAL slot — seed the canonical slots with the resumed windows first,
                # then update in place there. Identity for fresh rows (num_accepted == 0 loads the canonical slot).
                bs_c = spec_cols.shape[0]
                load_c = _kda_load_slots(spec_cols, num_accepted_tokens, bs_c)
                col0 = spec_cols[:, 0]
                pool_c = conv_state.shape[0]
                seed = conv_state.index_select(0, torch.remainder(load_c, pool_c).long())
                safe0 = torch.remainder(col0, pool_c).long()
                cs_q.index_copy_(0, safe0, seed[..., 0:dim3])
                cs_k.index_copy_(0, safe0, seed[..., dim3 : 2 * dim3])
                cs_v.index_copy_(0, safe0, seed[..., 2 * dim3 :])
                conv_slots = col0
            else:
                conv_slots = state_indices
            upd = dict(
                activation=self.activation,
                conv_state_indices=conv_slots,
                query_start_loc=query_start_loc,
                validate_data=False,
            )
            q, k, v = (
                hpu_causal_conv1d_update(x=x, conv_state=cs, weight=cw(conv), bias=conv.bias, **upd)
                for x, conv, cs in (
                    (q_proj_states, self.q_conv1d, cs_q),
                    (k_proj_states, self.k_conv1d, cs_k),
                    (v_proj_states, self.v_conv1d, cs_v),
                )
            )

        if spec_cols is None or spec_len <= 1 or is_prompt:
            # The kernels above updated the contiguous cs_* copies in place; write them back as one tensor. The MTP
            # verify branch scatters its windows into conv_state directly and must not be clobbered here.
            conv_state.copy_(torch.cat((cs_q, cs_k, cs_v), dim=-1))

        d = self.head_dim
        if padding_mask_flat is not None:
            m_flat = padding_mask_flat.reshape(-1, 1).to(q.dtype)
            q = torch.nan_to_num(q * m_flat)
            k = torch.nan_to_num(k * m_flat)
            v = torch.nan_to_num(v * m_flat)

        q = q.reshape(1, n, -1, d).contiguous()
        k = k.reshape(1, n, -1, d).contiguous()
        v = v.reshape(1, n, -1, d).contiguous()

        g = hpu_kda_gate(g1.reshape(1, n, -1), self.A_log, d, g_bias=self.dt_bias, lower_bound=self.gate_lower_bound)
        if padding_mask_flat is not None:
            # the gate is never 0 on padding (lower_bound * sigmoid(...)), so it must be masked too
            g = torch.nan_to_num(g * padding_mask_flat.reshape(1, -1, 1, 1).to(g.dtype))
            beta = torch.nan_to_num(beta * padding_mask_flat.reshape(1, -1, 1).to(beta.dtype))

        if is_prompt:
            init = None
            n_seqs = 1
            if state_indices is not None:
                init = ssm_state.index_select(0, state_indices.long()).contiguous()
                if has_initial_state is not None:
                    # a multiply, not a boolean index (which lowers to scatter_nd)
                    init = init * has_initial_state.bool().view(-1, 1, 1, 1).to(init.dtype)
                n_seqs = int(state_indices.numel())

            # One recurrence per sequence: fold the flat prompt block into [n_seqs, seq_len, H, D].
            seq_len = n // n_seqs if n_seqs > 0 else n
            if n_seqs > 1 and seq_len * n_seqs == n:
                qkv = [t.reshape(n_seqs, seq_len, -1, d).contiguous() for t in (q, k, v)]
                gg = g.reshape(n_seqs, seq_len, -1, d).contiguous()
                bb = beta.reshape(n_seqs, seq_len, -1).contiguous()
            else:
                qkv, gg, bb = [q, k, v], g, beta

            kw = dict(
                chunk_size=_KDA_CHUNK,
                sub_block=_KDA_SUB,
                initial_state=init,
                output_final_state=True,
                consts=self.kda_consts,
            )
            if KDA_SCAN:
                out, final = hpu_chunk_kda_scan(qkv[0], qkv[1], qkv[2], gg, bb, eye_k=self.kda_eye_k, **kw)
            else:
                out, final = hpu_chunk_kda(qkv[0], qkv[1], qkv[2], gg, bb, **kw)
            out = out.reshape(1, n, -1, d).contiguous()
        else:
            pool = ssm_state.shape[0]
            bs_rows = state_indices.shape[0] if state_indices is not None else n
            load_slots = _kda_load_slots(state_indices, num_accepted_tokens, bs_rows)
            state = (
                ssm_state.index_select(0, torch.remainder(load_slots, pool).long()).contiguous()
                if load_slots is not None
                else None
            )

            if spec_len > 1 and spec_cols is not None:
                # --- MTP verify: one chained recurrence per sequence over its draft chain, keeping the state after
                # EVERY token: acceptance is only known after sampling, and unlike the conv cache a contracted
                # recurrent state cannot be rewound. Candidate column 1+j (store column 1+j) holds the state after
                # chain token j; the next step resumes from column num_accepted (0 = fresh sentinel = canonical).
                qr, kr, vr = (t.reshape(bs_rows, spec_len, -1, d).contiguous() for t in (q, k, v))
                gr = g.reshape(bs_rows, spec_len, -1, d).contiguous()
                br = beta.reshape(bs_rows, spec_len, -1).contiguous()
                out_r, _, states_r = hpu_recurrent_kda(
                    qr, kr, vr, gr, br, initial_state=state, output_final_state=False, return_all_states=True
                )
                out = out_r.reshape(1, n, -1, d).contiguous()
                if load_slots is not None and store_cols is not None:
                    tgt = torch.cat(
                        [store_cols[:bs_rows, 0:1], store_cols[:bs_rows, 1 : 1 + spec_len]], dim=1
                    ).reshape(-1)
                    rows = torch.cat([state.unsqueeze(1), states_r.to(state.dtype)], dim=1)
                    _scatter_candidate_states(
                        ssm_state, tgt, rows.reshape(bs_rows * (spec_len + 1), *state.shape[1:])
                    )
                    final = None  # per-position states already published to their slots
            else:

                def to_seq(t):
                    # [1, n, H, D] -> [n, 1, H, D], contiguous: strided inputs to a captured graph fail at replay
                    return t.reshape(-1, *t.shape[2:]).unsqueeze(1).contiguous()

                out, final = hpu_recurrent_kda(
                    to_seq(q),
                    to_seq(k),
                    to_seq(v),
                    to_seq(g),
                    beta.reshape(-1, beta.shape[-1]).unsqueeze(1).contiguous(),
                    initial_state=state,
                    output_final_state=True,
                )
                out = out.reshape(1, -1, *out.shape[2:]).contiguous()
        if state_indices is not None and not (spec_len > 1 and spec_cols is not None):
            # Non-speculative store: the canonical (column-0) slot. num_gdn_decodes (when the metadata carries it)
            # excludes padded lanes, so trim the final states to the rows actually stored.
            store_slots = (store_indices[:, 0] if store_indices.dim() >= 2 else store_indices).reshape(-1)
            if final is not None and final.shape[0] != store_slots.shape[0]:
                final = final[: store_slots.shape[0]]
            out = _save_ssm_state(out, ssm_state, final, store_slots)

        out = torch.nan_to_num(out).to(core_attn_out.dtype)
        if out.shape == core_attn_out.shape:
            core_attn_out.copy_(out)
        else:
            core_attn_out[0, :n] = out[0, :n]
