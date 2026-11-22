# RA-041 Implementation Plan — KDA Prefix Cache Lattice

Status: DESIGN COMPLETE, IMPLEMENTATION STAGED (the owner directive: bf16 KV mandatory)

## Problem
The KDA (Kimi Delta Attention) layer stores per-block states that vLLM's
block-granular prefix cache cannot index, so `--enable-prefix-caching` is
refused at boot (the model refuses to start). Long shared prefixes re-prefill
on every request.

## Design (the lattice)
1. **State lattice**: index KDA per-block states at the KDA block granularity
   (not the vLLM KV block granularity). Each KDA block carries
   `(layer_id, block_seq_start, state_hash)`. The cache key lattice is the
   tree of `(parent_state_hash, new_block_tokens_hash)` chains, so a hit at
   depth d implies all ancestors hold.
2. **Hit path**: on a chunked-prefill boundary, look up the deepest matched
   state for the incoming prefix; copy the matched states into the request's
   state slots (a bounded copy: 11 DSA/MLA layers x block states), skip the
   covered tokens, run the remaining prefill.
3. **Invalidation**: KDA states decay; the lattice stores the state as
   produced, no recompute. Eviction is LRU at the block-state level.
4. **bf16 invariant**: states stored and copied in bf16 per the owner
   directive; no conversion in the hit path.

## Measured KV geometry (the lattice's constants)
- 819,200-token KV pool (the measured config), KDA block = 128 tokens
- 11 hybrid attention layers carry the state; the remaining layers are the
  standard block-indexed KV cache (already cacheable by vLLM).

## Acceptance
- Hit at depth d must reproduce the prefill logits parity of the reference
  (the T1 gates run on a warm prefix-cache request).
- Measured: prefix re-prefill tokens per request drop from N to N mod 128;
  TTFT improvement recorded at 4K/8K/16K shared prefixes.

## Risk
The state copy is on the critical path; the hit path must be cheaper than
the covered prefill (~1.4 s per 4K at the measured ladder). The copy is
bounded (11 layers x block states), so the win is at the 4K+ prefixes.
