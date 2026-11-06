# SPDX-License-Identifier: Apache-2.0
"""Bounded layer-boundary capture for GLM-5.3-Flash near-tie forensics on vllm-gaudi.

Adapted from abotsis/vllm-gaudi 919ec98f `vllm_gaudi/v1/worker/layer_diagnostic.py`
(Apache-2.0), reworked for this port's plugin architecture and enforce_eager
diagnostic boots: the capture requires no graph bypass here — a diagnostic boot
runs with `--enforce-eager` and an operator-created sentinel file.

Activation (all must hold):
  * `VLLM_GLM_LAYER_DIAG_DIR` points at a directory created for this purpose.
  * That directory contains `LAYER_ENABLED`, a regular file owned by the running
    uid with mode 0600 (the operator creates it after boot: `install -m 600
    /dev/null $DIR/LAYER_ENABLED`).
  * `VLLM_GLM_LAYER_DIAG_POSITIONS` (default "0,1,2") lists the output positions
    admitted; earlier positions stay under normal execution so the captured
    forward sees state exactly as production wrote it.

Safety properties (inherited from the reference design):
  * Admission is bounded: <= 3 requests, <= 3 positions each, <= 9 captures per
    boot, <= 64 MiB of boundary payload per boot.
  * Every dump validates the exact request history (prompt ids, output ids,
    absolute position) before any tensor is copied.
  * Writes use O_NOFOLLOW on a directory fd re-verified as 0700 and owned by
    the running uid; payloads are 0600.
  * The capture copies only the admitted rows of each boundary tensor — never
    the KV pool, slot tables, or block mappings.
"""
from __future__ import annotations

import os
import stat

import torch

MAX_BYTES = 64 * 1024 * 1024
MAX_CAPTURES = 9
MAX_REQUESTS = 3
MAX_POSITIONS = 3

_collector: dict | None = None


def directory(runner) -> str | None:
    """The diag dir iff the operator-created sentinel allows capture this boot."""
    path = getattr(runner, "_glm_layer_diag_dir", None)
    if not path:
        return None
    try:
        info = os.lstat(os.path.join(path, "LAYER_ENABLED"))
        if (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600):
            return path
    except OSError:
        pass
    return None


def cpu(tensor):
    return tensor.detach().to(device="cpu", non_blocking=False, copy=True)


def allowed_positions() -> list[int]:
    raw = os.environ.get("VLLM_GLM_LAYER_DIAG_POSITIONS", "0,1,2")
    try:
        values = sorted({int(x) for x in raw.split(",") if x.strip() != ""})
    except ValueError:
        values = list(range(MAX_POSITIONS))
    return values[:MAX_POSITIONS] or [0]


def arm(runner, *, rank: int = 0) -> dict | None:
    """Begin a capture ticket for this forward. Returns None unless armed.

    The caller supplies, in `commit`, the per-row request context this module
    validates before any boundary is captured.
    """
    global _collector
    if _collector is not None:
        raise RuntimeError("Nested layer-diagnostic forward")
    path = directory(runner)
    if path is None:
        return None
    ordinal = getattr(runner, "_glm_diag_calls", 0)
    used = getattr(runner, "_glm_diag_bytes", 0)
    if ordinal >= MAX_CAPTURES or used >= MAX_BYTES:
        return None
    runner._glm_diag_calls = ordinal + 1
    if getattr(runner, "_glm_diag_admitted", None) is None:
        runner._glm_diag_admitted = []
    if getattr(runner, "_glm_diag_seen", None) is None:
        runner._glm_diag_seen = set()
    
    _collector = dict(path=path,
                      rank=rank,
                      ordinal=ordinal,
                      bytes_used=used,
                      positions=allowed_positions(),
                      runner=runner,
                      admitted=runner._glm_diag_admitted,
                      seen_set=runner._glm_diag_seen,
                      payload=dict(schema=1, rank=rank, ordinal=ordinal,
                                   records=[], boundaries=[], truncated=False))
    return _collector


def admit(request_id, position: int, prompt_token_ids: list[int],
          output_token_ids: list[int]) -> dict | None:
    """Admit one (request, output position) record with exact history validation."""
    if _collector is None:
        return None
    positions = _collector["positions"]
    admitted: list = _collector["admitted"]
    seen: set = _collector["seen_set"]
    if position not in positions or (request_id, position) in seen:
        return None
    if request_id not in admitted:
        if position != positions[0] or len(admitted) >= MAX_REQUESTS:
            return None
        admitted.append(request_id)
    if len(seen) >= MAX_CAPTURES:
        return None
    seen.add((request_id, position))
    if not output_token_ids or not prompt_token_ids:
        raise ValueError("Layer diagnostic: empty token history")
    # Exact per-decode-step contract: this forward verifies generated token
    # `position`; the caller passes the request's generated tokens INCLUDING it.
    if len(output_token_ids) != position + 1:
        raise ValueError("Layer diagnostic: decode-step mismatch")
    record = dict(request_id=request_id,
                  output_position=position,
                  absolute_position=len(prompt_token_ids) + position,
                  prompt_token_ids=list(prompt_token_ids),
                  output_token_ids=list(output_token_ids),
                  last_token=output_token_ids[position])
    _collector["payload"]["records"].append(record)
    return record


def boundary(name: str, tensor, *, row_indices: list[int] | None = None) -> None:
    """Copy the admitted rows of one boundary tensor into the ticket."""
    if _collector is None:
        return
    runner = _collector["runner"]
    selected = list(row_indices) if row_indices is not None else _collector["admitted_rows"]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] == 0:
        _collector["payload"].setdefault("unavailable", []).append(dict(name=name, reason="empty"))
        return
    if not selected or min(selected) < 0 or max(selected) >= tensor.shape[0]:
        _collector["payload"].setdefault("unavailable", []).append(
            dict(name=name, reason="row index outside leading dimension"))
        return
    size = len(selected) * (tensor.numel() // tensor.shape[0]) * tensor.element_size()
    used = getattr(runner, "_glm_diag_bytes", 0)
    if used + size > MAX_BYTES:
        _collector["payload"]["truncated"] = True
        return
    runner._glm_diag_bytes = used + size
    index = torch.tensor(selected, dtype=torch.long, device=tensor.device)
    owned = tensor.index_select(0, index).detach().clone()
    _collector["payload"]["boundaries"].append(
        dict(name=name, shape=tuple(tensor.shape), dtype=str(tensor.dtype),
             rows=cpu(owned), row_indices=list(selected)))


def flush(mark_step=None) -> None:
    """Persist the ticket: sync, re-verify the directory, write 0600 O_NOFOLLOW."""
    global _collector
    if _collector is None:
        return
    ticket, _collector = _collector, None
    if mark_step is not None:
        mark_step()
    payload = ticket["payload"]
    runner = ticket["runner"]
    payload["bytes"] = getattr(runner, "_glm_diag_bytes", 0)
    path = ticket["path"]
    dfd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(dfd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Layer diagnostic directory changed")
        fd = os.open(f"layer-rank{payload['rank']}-{payload['ordinal']:02d}.pt",
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
    finally:
        os.close(dfd)


def disarm() -> None:
    global _collector
    _collector = None
