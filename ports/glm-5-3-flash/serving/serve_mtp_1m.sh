#!/bin/bash
# serve_mtp_1m.sh: the 1M-context variant of serve_mtp.sh (RA-047 / D17 window).
export COOKBOOK=/home/tmpai2/publish_work/fork
export GLM53_MODEL_PATH=/data/models/GLM-5.3-Flash
export GLM_FP8_MXK=0
export GLM_SPEC_DEVICE_REJECT=1
export GLM_EARLY_DRAFT=1

python3 "$COOKBOOK/fixes/apply_all.py" --profile v0.26.0 glm47-streaming-text-markers || exit 1

exec vllm serve "$GLM53_MODEL_PATH" \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
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
