#!/bin/bash
# Stage-1 controlled MTP=1 reproduction (task A001) — orchestrator.
#
# REQUIRES APPROVAL before running: it restarts the serving engine (production
# interruption) twice, once per condition. Each boot is ~15-20 min; the whole
# run is ~1.5-2 h.
#
# Design (per docs/BENCHMARK_POLICY.md):
#   - Control: non-MTP, window identical to the treatment (serve_control_window.sh).
#   - Treatment: MTP k=1 (serve_mtp.sh), same window/config otherwise.
#   - 6 decode runs per condition (first rep is warm-up, discarded; 5 measured).
#     Per-rep alternation is impractical with ~20-min boots; control block runs
#     first, treatment second — order bias documented as a confounder.
#   - Accuracy suite: 5 paired reps per condition (evaluate.py), gate applied.
#   - MTP-specific engine metrics (acceptance) captured from the server log.
#
# Usage: bash stage1_reproduction.sh [BASE_DIR]
#   BASE_DIR defaults to the repository checkout (required files relative to it).
set -uo pipefail
REPO="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVE="$REPO/ports/glm-5-3-flash/serving"
TESTS="$REPO/ports/glm-5-3-flash/tests/hpu/mtp"
ACC="$REPO/benchmarks/accuracy"
OUT="$REPO/benchmarks/stage1_results"
LOGD="$OUT/logs"
mkdir -p "$OUT" "$LOGD"

export COOKBOOK="$REPO"
export GLM53_MODEL_PATH="${GLM53_MODEL_PATH:?set GLM53_MODEL_PATH to the checkpoint directory}"

# Bucket pinning (applied to BOTH conditions equally — it is not the Stage-1
# variable; MTP on/off is). Removes decode-ladder noise from the comparison and
# cuts warmup ~13 -> ~2 min. Reference recipe at max-num-seqs 8; see
# experiments/RESEARCH-001-draft-graph-upstream.md addendum 2.
export VLLM_DECODE_BS_BUCKET_MIN=8
export VLLM_DECODE_BS_BUCKET_STEP=8
export VLLM_DECODE_BLOCK_BUCKET_MIN=32

REPS_MEASURED=5
REPS_TOTAL=6   # first rep is warm-up, discarded
TOKENS=256

wait_healthy() {
  for i in $(seq 1 160); do
    if curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "healthy after ~$((i*12))s"; return 0
    fi
    sleep 12
  done
  echo "server failed to become healthy"; return 1
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  for p in $(pgrep -f "VLLM::Worker" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 3
}

# The serve scripts and the fix step require the deployment venv python
# (the system python fails the fix-step target resolution).
source /opt/glm53/venv/bin/activate

run_condition() {
  local label=$1 serve_script=$2 log=$3
  echo "=== booting $label ==="
  stop_server
  rm -f "$log"
  cd "$(dirname "$serve_script")"
  setsid nohup bash "$serve_script" > "$log" 2>&1 < /dev/null &
  wait_healthy || return 1
  echo "=== warmup run (discarded) ==="
  python3 "$TESTS/bench_decode.py" --tokens "$TOKENS" --label "$label-warmup" > /dev/null
  echo "=== $REPS_MEASURED measured decode runs ==="
  for r in $(seq 1 "$REPS_MEASURED"); do
    python3 "$TESTS/bench_decode.py" --tokens "$TOKENS" --label "$label-rep$r" \
      | tee "$OUT/decode_${label}_rep${r}.json"
  done
  echo "=== accuracy suite ($REPS_MEASURED reps) ==="
  python3 "$ACC/evaluate.py" run --base http://127.0.0.1:8000 \
    --label "$label" --reps "$REPS_MEASURED" --suite "$ACC/suite_v1.jsonl" \
    --outdir "$OUT" | tee "$OUT/accuracy_${label}.txt"
  if grep -q "speculative" "$log"; then
    echo "=== MTP engine metrics (from server log) ==="
    grep -iE "accepted|acceptance" "$log" | tail -5 > "$OUT/mtp_metrics_${label}.txt" || true
  fi
}

run_condition control "$SERVE/serve_control_window.sh" "$LOGD/control.log" || exit 1
run_condition mtp1   "$SERVE/serve_mtp.sh"         "$LOGD/mtp1.log"    || exit 1

echo "=== summarize decode runs ==="
STAGE1_OUT="$OUT" python3 - << "PYEOF"
import json, os, statistics, glob
OUT = os.environ["STAGE1_OUT"]
for label in ("control", "mtp1"):
    vals, ttfts, med, p95 = [], [], [], []
    for f in sorted(glob.glob(f"{OUT}/decode_{label}_rep*.json")):
        d = json.load(open(f))
        vals.append(d["decode_tok_s"]); ttfts.append(d["ttft_s"])
        med.append(d["itl_median_s"]); p95.append(d["itl_p95_s"])
    if vals:
        print(f"{label}: tok/s median {statistics.median(vals)}, mean {statistics.mean(vals):.2f}, "
              f"std {statistics.stdev(vals) if len(vals) > 1 else 0:.2f}, all {vals}")
        print(f"         ttft mean {sum(ttfts)/len(ttfts):.2f}s, itl median {statistics.median(med)}, p95 {max(p95)}")
PYEOF

echo "=== accuracy gate ==="
if [ -f "$OUT/control.json" ] && [ -f "$OUT/mtp1.json" ]; then
  python3 "$ACC/evaluate.py" gate --control "$OUT/control.json" --treatment "$OUT/mtp1.json"
else
  echo "SKIP: accuracy gate needs both condition results (a condition failed - see logs above)"
fi

echo "=== DONE — raw runs in $OUT; update EXPERIMENTS.md / BASELINE.md per the policy ==="
