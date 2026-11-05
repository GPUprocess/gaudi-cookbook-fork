#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mtp-attn: the MTP draft-path changes for the GLM-5.3-Flash port: speculative-decode plumbing, verify-row layouts and state handling recorded from the verified deployment

Generated from the verified deployment of GLM-5.3-Flash MTP speculative decoding
(docs/ports/glm-5-3-flash-mtp.md). Applies on top of the port image state: the
release fix profile (v0.26.0) and the port's own patches (install.sh) must already
be applied. Every replacement anchors exactly once; a mismatch writes nothing and
exits 2. Target: vllm_gaudi/v1/attention/backends/hpu_attn.py.

Usage: mtp_attn.py [--check] [--root DIR_HOLDING_vllm_and_vllm_gaudi]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patchlib import main, replace_once

MARKER = "[gaudi-port:glm-5-3-flash/mtp-attn]"

REPLACEMENTS = (
    ('    mamba_chunks_to_block_mapping: Optional[torch.Tensor] = None\n    seqlens_offsets_for_blocks: Optional[torch.Tensor] = None\n    window_block_list: Optional[torch.Tensor] = None\n\n    def seq_len(self):\n        return self.slot_mapping.size(-1)\n', '    mamba_chunks_to_block_mapping: Optional[torch.Tensor] = None\n    seqlens_offsets_for_blocks: Optional[torch.Tensor] = None\n    window_block_list: Optional[torch.Tensor] = None\n    # [glm-mtp] Speculative decode: per-sequence accepted-token count from the PREVIOUS verify step. The\n    # linear-attention layer resumes its recurrent state from candidate slot num_accepted (0 = fresh sentinel =\n    # canonical slot). num_gdn_decodes = real (unpadded) decode-sequence count for the step.\n    num_accepted_tokens: Optional[torch.Tensor] = None\n    num_gdn_decodes: Optional[int] = None\n\n    def seq_len(self):\n        return self.slot_mapping.size(-1)\n', 'region@orig-line68'),
    ('                             load_indices_tensor=None,\n                             store_indices_tensor=None,\n                             query_start_loc=None,\n                             seq_lens_tensor=None):\n        return cls(is_prompt=False,\n                   block_mapping=None,\n                   alibi_blocks=None,\n', '                             load_indices_tensor=None,\n                             store_indices_tensor=None,\n                             query_start_loc=None,\n                             seq_lens_tensor=None,\n                             num_accepted_tokens=None,\n                             num_gdn_decodes=None):\n        return cls(is_prompt=False,\n                   block_mapping=None,\n                   alibi_blocks=None,\n', 'region@orig-line138'),
    ('                   load_indices_tensor=load_indices_tensor,\n                   store_indices_tensor=store_indices_tensor,\n                   query_start_loc=query_start_loc,\n                   query_start_loc_p=query_start_loc)\n', '                   load_indices_tensor=load_indices_tensor,\n                   store_indices_tensor=store_indices_tensor,\n                   query_start_loc=query_start_loc,\n                   query_start_loc_p=query_start_loc,\n                   num_accepted_tokens=num_accepted_tokens,\n                   num_gdn_decodes=num_gdn_decodes)\n', 'region@orig-line161'),
)


def patch(src: str) -> str:
    for old, new, label in REPLACEMENTS:
        src = replace_once(src, old, new, label)
    if MARKER not in src:
        src = src.rstrip("\n") + "\n\n# " + MARKER + "\n"
    return src


TARGETS = (("vllm_gaudi", "v1/attention/backends/hpu_attn.py", patch),)

if __name__ == "__main__":
    sys.exit(main(__doc__, MARKER, TARGETS))
