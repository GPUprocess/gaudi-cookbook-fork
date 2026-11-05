# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash MTP (Multi-Token Prediction) draft head for vLLM on Gaudi 2.

The MTP layer ships inside the checkpoint as `model.language_model.layers.45.*`
(`num_nextn_predict_layers=1`). The target model loads those tensors into
`self.mtp` (see model.py) only when speculative decoding is on. This module
provides the two pieces upstream vLLM gives DeepSeek-style models:

* `Glm5NextMTPDecoderLayer` -- the layer-45 body. The checkpoint has no `hc_*`
  tensors for the MTP layer, so unlike the target layers it runs without mHC:
  NoPE MLA (dense, no indexer) + routed-MoE block with the standard residual
  flow.
* `Glm5NextMTPModel` -- the draft head `HpuEagleProposer` binds to the target's
  already-loaded modules (no second checkpoint pass; GLM-5.3's MTP block has no
  own `shared_head.head`, so a standalone draft could not own its head anyway).

One draft step (mirrors core vLLM's DeepSeekMTP / Glm4MoeMTP and the validated
abotsis HPU wrapper):

    emb = embed(input_ids); emb[positions == 0] = 0      # no predecessor at 0
    x = eh_proj(cat([enorm(emb), hnorm(hidden_states)], -1))
    residual = x
    x = input_layernorm(x)
    x = self_attn(positions, x)        # draft MLA over the draft's own KV
    x = residual + x
    x = post_attention_layernorm(x)
    x = mlp(x)
    x = residual + x                   # shape-preserving: [N,1,H]+[N,H] must not broadcast
    return shared_head_norm(x)         # post-final-norm hidden, single tensor

`compute_logits` applies the target's `lm_head` directly: the forward already
applied the final norm, so logits and the recycled hidden each get exactly one
final norm (the same contract as DeepSeekMTP's tuple, collapsed to the single
tensor the HPU proposer's `mtp` branch expects).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

import torch
from torch import nn

from .model import Glm5NextMLAAttention, _apply_swiglu_clamp


class Glm5NextMTPDecoderLayer(nn.Module):
    """Layer-45 body: MLA + MoE, standard (non-mHC) residual flow."""

    def __init__(self, config, quant_config, cache_config, parallel_config, prefix: str = ""):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

        # MTP glue: enorm/hnorm/eh_proj combine the embedding of the next token with the target's last hidden
        # state; shared_head_norm is the draft's own final norm before the (shared) lm_head.
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Glm5NextMLAAttention(config, quant_config, cache_config, prefix=f"{prefix}.self_attn")
        # The checkpoint's layer 45 is a sparse MoE (288 routed + shared experts); text_config.mlp_layer_types
        # covers only the 45 main layers, so the type is not read for the MTP layer.
        self.mlp = DeepseekV2MoE(
            config=config, parallel_config=parallel_config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )
        _apply_swiglu_clamp(self.mlp, float(getattr(config, "swiglu_limit", 10.0)), "MTP layer shared expert")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(residual)
        x = self.self_attn(positions, x)
        x = residual + x  # same-numel add: never let [N,1,H] + [N,H] broadcast to [N,N,H]
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


def _current_vllm_config():
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:
        return None


class Glm5NextMTPModel(nn.Module):
    """Draft head over the target's already-loaded MTP (layer 45) modules.

    Constructed by the patched `HpuEagleProposer.load_model` with the target
    model. Every parameter below is the TARGET's (held by reference; nothing is
    copied or re-read), so `load_weights` is a no-op and the runner must not
    wrap this module in a second HPU graph over the same storage.
    """

    _shares_target_modules = True

    def __init__(self, target: nn.Module, vllm_config=None):
        super().__init__()
        mtp = getattr(target, "mtp", None)
        if mtp is None:
            raise ValueError(
                "target model has no .mtp module -- GLM-5.3 MTP speculative decode needs "
                "num_nextn_predict_layers > 0 and the layer-45 weights present"
            )
        self.mtp = mtp
        self._target = target
        self._vllm_config = vllm_config
        self.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        self.logits_processor = target.logits_processor

    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings=None) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step. Returns post-final-norm hidden states (single tensor).

        The wrapper is deliberately NOT wrapped in the runner's HPU graph/adapter
        (it shares the target's already-captured weights), so it installs its own
        forward context: the proposer hands `attn_metadata` straight to us and
        the MLA layer reads it from the context.
        """
        from vllm.forward_context import set_forward_context

        # Work flat like every layer in this port: the proposer hands
        # [batch, seq, ...] (or [tokens, ...]); the output keeps the token count.
        lead = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        n_tokens = flat.shape[0]
        ids = input_ids.reshape(-1)[:n_tokens]
        pos = positions.reshape(-1)[:n_tokens]

        inputs_embeds = self.embed_tokens(ids) if inputs_embeds is None else inputs_embeds.reshape(n_tokens, -1)
        # Position 0 has no predecessor to predict from; zero it rather than
        # feed the raw embedding (matches glm4_moe_mtp.py).
        inputs_embeds = torch.where(pos.unsqueeze(-1) == 0, 0, inputs_embeds)

        x = self.mtp.eh_proj(torch.cat([self.mtp.enorm(inputs_embeds), self.mtp.hnorm(flat)], dim=-1))
        vllm_config = self._vllm_config or _current_vllm_config()
        with contextlib.ExitStack() as stack:
            if attn_metadata is not None:
                stack.enter_context(set_forward_context(attn_metadata, vllm_config))
            residual = x
            x = self.mtp.input_layernorm(x)
            x = self.mtp.self_attn(pos, x)
            x = residual + x
            residual = x
            x = self.mtp.post_attention_layernorm(x)
            x = self.mtp.mlp(x)
        x = residual + x
        return self.mtp.shared_head_norm(x).reshape(*lead, -1)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Nothing to load: every parameter is the target's, already populated.
        return set()
