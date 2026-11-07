#!/usr/bin/env bash
# Q10: 1M-context feasibility boot. Same as serve_mtp.sh except:
#   --max-model-len 1048576 (the published config context length)
#   --num-gpu-blocks-override 8192 (8192*128 = 1,048,576-token KV pool;
#     measured KV cost 11.5 KiB/token/card -> ~11.8 GiB/card, affordable
#     next to 38.2 GiB weights at GMU 0.82 on 96 GiB cards)
# Graphs capture lazily (VLLM_SKIP_WARMUP=true): the first 1M-shaped prompt
# pays its capture.
set -e
export COOKBOOK="${COOKBOOK:-/home/tmpai2/publish_work/fork}"
export GLM53_MODEL_PATH="${GLM53_MODEL_PATH:-/data/models/GLM-5.3-Flash}"
export PT_HPU_LAZY_MODE=1
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
export VLLM_BUCKETING_STRATEGY=lin
export VLLM_PROMPT_QUERY_BUCKET_STEP=2048
export VLLM_PROMPT_CTX_BUCKET_STEP=64

python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 glm47-streaming-text-markers || exit 1

exec vllm serve "$GLM53_MODEL_PATH" \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 1048576 \
  --max-num-seqs 8 \
  --num-gpu-blocks-override 8192 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 2048 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.82 \
  --no-enable-prefix-caching --limit-mm-per-prompt "{\"image\": 4}" \
  --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice \
  --trust-remote-code \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": 1}" \
  --host 0.0.0.0 --port 8000
