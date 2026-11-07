#!/bin/bash
# GLM-5.3-Flash + native MTP speculative decoding (k=1) launch script.
#
# Differences from the stock launch (see ../../README.md "Serve" and the MTP docs page):
#   * --speculative-config with the port mtp draft head (k=1)
#   * --max-num-seqs 8 (verify rows = 8*(k+1) = 16; raise together with graph capture)
#   * reduced max-model-len and --num-gpu-blocks-override for the validation window
# Requires: the spec patches from ../../patches applied (install.sh chain) and the
# plugin mtp module (part of this port).
set -euo pipefail
: "${COOKBOOK:?set COOKBOOK to the repository checkout}"
: "${GLM53_MODEL_PATH:?set GLM53_MODEL_PATH to the checkpoint directory}"

export PT_HPU_LAZY_MODE=1
# Habana runtime recipe pins (see experiments/RESEARCH-001 addendum 8):
# ACC_PAR_MODE=1 matches the bridge default but is pinned explicitly for a
# deterministic recipe; MEMORY_POOL=none is the reference validated value.
export PT_HPU_LAZY_ACC_PAR_MODE=1
export PT_HPU_MEMORY_POOL=none
export VLLM_SKIP_WARMUP=true
export VLLM_ENGINE_ITERATION_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export GLM5_NEXT_VISION=1
export GLM5_NEXT_IMAGE_MAX_TOKENS=2048
export GLM5_NEXT_KDA_SCAN=1
export GLM5_NEXT_DECODE_LAYER_GROUP=45
export GLM5_NEXT_MHC_SINKHORN=kernel
export OMPI_MCA_btl_vader_single_copy_mechanism=none
# [glm-5-3-flash:decode-lean] adopted serving levers (EXP-Q18/Q19): bit-identical
# greedy outputs; aggregate @16 126.7 -> ~185 t/s; bs=1 neutral. The draft HPU
# graphs (VLLM_GLM_MTP_DRAFT_GRAPH) stay OFF: their replay syncs defeat the
# early-draft pipelining (EXP-Q19).
export GLM_ROUTER_FAST=1
export GLM_SPEC_DEVICE_REJECT=1
export GLM_EARLY_DRAFT=1


python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 glm47-streaming-text-markers || exit 1

exec vllm serve "$GLM53_MODEL_PATH" \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --num-gpu-blocks-override 1280 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 2048 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.82 \
  --no-enable-prefix-caching --limit-mm-per-prompt "{\"image\": 4}" \
  --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": 1}" \
  --host 0.0.0.0 --port 8000
