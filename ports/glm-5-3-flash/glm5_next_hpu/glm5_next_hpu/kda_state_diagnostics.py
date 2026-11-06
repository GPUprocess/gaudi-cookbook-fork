# SPDX-License-Identifier: Apache-2.0
"""Bounded, opt-in incoming RAW KDA state snapshots for the GLM-5.3-Flash port.

Adapted from abotsis/vllm-gaudi 919ec98f `vllm_gaudi/v1/worker/kda_state_diagnostic.py`
(Apache-2.0) for this port's plugin architecture: the KDA layer self-arms from
`VLLM_GLM_LAYER_DIAG_DIR` + a `STATE_ENABLED` sentinel (root-owned 0600) in an
enforce_eager diagnostic boot, and captures the RAW incoming recurrent-state rows
(never the effective masked state) for the first decode steps.

No serving-stack imports: this module is exercised by CPU-only tests.

Simplification vs the reference: the runner-side request-history validation is
dropped (the layer has no request context in this port's architecture); the
payload carries the metadata maps and the raw pool rows for the admitted decode
steps, which is the evidence class needed for candidate-slot verification.
"""
import os
import stat

import torch

FIELDS = (
    "load_indices_tensor",
    "store_indices_tensor",
    "query_start_loc_p",
    "query_start_loc",
    "has_initial_states_p",
    "padding_mask_flat",
    "num_accepted_tokens",
    "seq_lens_tensor",
    "context_lens_tensor",
    "prep_initial_states",
    "last_chunk_indices_p",
    "blocks_caching_range",
    "mamba_chunks_to_block_mapping",
    "seqlens_offsets_for_blocks",
)
MAX_BYTES = 256 * 1024 * 1024
MAX_CAPTURES = 9

_collector = None


def state_directory(layer) -> str | None:
    """The diag dir iff the operator-created STATE_ENABLED sentinel allows capture."""
    path = os.environ.get("VLLM_GLM_LAYER_DIAG_DIR", "")
    if not path:
        return None
    try:
        info = os.lstat(os.path.join(path, "STATE_ENABLED"))
        if (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600):
            return path
    except OSError:
        pass
    return None


def maybe_capture(layer, metadata, tokens, positions, mark_step=None) -> None:
    """Self-arming KDA state capture: called from the KDA layer's forward.

    Arms from the sentinel-gated directory, captures the RAW state maps and the
    first pool rows for this forward, writes the payload, and disarms. No-op
    unless the sentinel allows it; bounded by MAX_CAPTURES and MAX_BYTES.
    """
    global _collector
    if _collector is not None:
        return
    directory = state_directory(layer)
    if directory is None:
        return
    ordinal = getattr(layer, "_glm_state_calls", 0)
    if ordinal >= MAX_CAPTURES:
        return
    layer._glm_state_calls = ordinal + 1
    _collector = dict(layer=layer, path=directory, rank=0, ordinal=ordinal,
                      payload=dict(schema=1, rank=0, ordinal=ordinal,
                                   raw_state=True,
                                   phase="prefill" if metadata.is_prompt else "decode",
                                   layer=type(layer).__name__,
                                   records=[], layers=[], skipped=None,
                                   truncated=False))
    try:
        _capture_inner(layer, metadata, tokens, positions, mark_step)
    finally:
        payload = _collector["payload"]
        runner_bytes = getattr(layer, "_glm_state_bytes", 0)
        payload["bytes"] = runner_bytes
        _collector = None
        _write_payload(directory, payload, ordinal)


def _capture_inner(layer, metadata, tokens, positions, mark_step):
    if not metadata.is_prompt and tokens.shape[1] != 1:
        _collector["payload"]["skipped"] = "State diagnostic requires NSPEC0 single-token decode"
        return
    modules = [(name, module) for name, module in layer.named_modules()
               if type(module).__name__ == "Glm5NextHPUKDAAttention"]
    if not modules:
        _collector["payload"]["skipped"] = "No actual KDA modules found"
        return
    raw_metadata = {field: getattr(metadata, field, None) for field in FIELDS}
    tensors = [value for value in raw_metadata.values() if isinstance(value, torch.Tensor)]
    size = sum(t.numel() * t.element_size() for t in tensors)
    size += tokens.numel() * tokens.element_size() + positions.numel() * positions.element_size()
    for _, module in modules:
        if len(module.kv_cache) != 2:
            _collector["payload"]["skipped"] = "Unexpected KDA pool layout"
            return
        size += module.cache_group_idx.numel() * module.cache_group_idx.element_size()
        for pool in module.kv_cache[:2]:
            size += pool.numel() * pool.element_size()
    runner_bytes = getattr(layer, "_glm_state_bytes", 0)
    if runner_bytes + size > MAX_BYTES:
        _collector["payload"]["truncated"] = True
        return
    layer._glm_state_bytes = runner_bytes + size

    def cpu(tensor):
        return tensor.detach().to(device="cpu", non_blocking=False, copy=True)

    if mark_step is not None:
        mark_step()
    maps = {key: cpu(value) if isinstance(value, torch.Tensor) else value
            for key, value in raw_metadata.items()}
    layers = []
    for name, module in modules:
        group = cpu(module.cache_group_idx)
        layer_entry = dict(name=name, group=group, pools=[])
        for pool_index, pool in enumerate(module.kv_cache[:2]):
            kind = "conv" if pool_index == 0 else "recurrent"
            # RAW pool rows only: the first rows (the diagnostic evidence class;
            # full padded load/store maps remain in `maps`).
            selected = torch.arange(min(MAX_CAPTURES, pool.shape[0]), device=pool.device)
            if metadata.is_prompt and ((selected < -pool.shape[0]) | (selected >= pool.shape[0])).any():
                raise ValueError("Prefill index outside native indexing range")
            read_rows = torch.remainder(selected, pool.shape[0]).long()
            owned = pool.index_select(0, read_rows.to(pool.device)).detach().clone()
            if mark_step is not None:
                mark_step()
            raw = cpu(owned)
            layer_entry["pools"].append(
                dict(kind=kind,
                     shape=tuple(pool.shape),
                     dtype=str(pool.dtype),
                     raw=raw,
                     read_rows=read_rows.tolist()))
        layers.append(layer_entry)
    _collector["payload"]["metadata"] = maps
    _collector["payload"]["layers"] = layers


def _write_payload(directory, payload, ordinal):
    dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(dfd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("State diagnostic directory changed")
        fd = os.open(f"state-rank{payload['rank']}-{ordinal:02d}.pt",
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
    finally:
        os.close(dfd)
