#!/bin/bash
# Control launch for Stage-1 MTP reproduction: identical window to serve_mtp.sh
# but WITHOUT speculative decoding. Same model, TP8+EP, 32K window, 8 seqs,
# KV override, GMU, capture size, chunked prefill, parsers.
# This is the non-MTP control for task A001 (docs/BENCHMARK_POLICY.md equivalent conditions).
set -euo pipefail
: "${COOKBOOK:?set COOKBOOK to the repository checkout}"
: "${GLM53_MODEL_PATH:?set GLM53_MODEL_PATH to the checkpoint directory}"

export PT_HPU_LAZY_MODE=1
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
  --host 0.0.0.0 --port 8000
