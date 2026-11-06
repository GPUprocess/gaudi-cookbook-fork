#!/bin/bash
# pre_flight.sh — read-only validation of every Stage-1 prerequisite (task A001).
# Run BEFORE requesting the maintenance window. Changes nothing.
set -u
PASS=0; FAIL=0
ok()   { echo "  OK   $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL $1"; FAIL=$((FAIL+1)); }
note() { echo "  note $1"; }

echo "=== Stage-1 pre-flight ==="

# 1. Repository layout
REPO="${1:-$(cd "$(dirname "$0")" && pwd)/fork}"
for f in "$REPO/benchmarks/stage1_reproduction.sh" \
         "$REPO/benchmarks/accuracy/evaluate.py" \
         "$REPO/benchmarks/accuracy/suite_v1.jsonl" \
         "$REPO/ports/glm-5-3-flash/serving/serve_mtp.sh" \
         "$REPO/ports/glm-5-3-flash/serving/serve_control_window.sh" \
         "$REPO/ports/glm-5-3-flash/tests/hpu/mtp/bench_decode.py" \
         "$REPO/TASKS.md" "$REPO/EXPERIMENTS.md" "$REPO/BASELINE.md" \
         "$REPO/docs/BENCHMARK_POLICY.md"; do
  [ -f "$f" ] && ok "present: $(basename "$f")" || bad "missing: $f"
done

# 2. Scripts parse
for s in "$REPO/benchmarks/stage1_reproduction.sh" \
         "$REPO/ports/glm-5-3-flash/serving/serve_mtp.sh" \
         "$REPO/ports/glm-5-3-flash/serving/serve_control_window.sh"; do
  bash -n "$s" 2>/dev/null && ok "syntax: $(basename "$s")" || bad "syntax: $s"
done
python3 -m py_compile "$REPO/benchmarks/accuracy/evaluate.py" \
  "$REPO/ports/glm-5-3-flash/tests/hpu/mtp/bench_decode.py" 2>/dev/null \
  && ok "py_compile: evaluator + decode bench" || bad "py_compile failed"

# 3. Model checkpoint
MP="${GLM53_MODEL_PATH:-/data/models/GLM-5.3-Flash}"
if [ -d "$MP" ] && [ -f "$MP/config.json" ]; then
  ok "checkpoint: $MP (config.json present)"
  SZ=$(du -s --block-size=1G "$MP" 2>/dev/null | cut -f1)
  note "checkpoint size: ${SZ} GiB"
else
  bad "checkpoint missing or no config.json: $MP"
fi

# 4. Deployed plugin state (the live server's code)
PLUG=/opt/glm53/gaudi-cookbook/ports/glm-5-3-flash/glm5_next_hpu/glm5_next_hpu
if [ -f "$PLUG/mtp.py" ]; then
  ok "deployed plugin has mtp.py"
  DEPLOYED_SERVE=/data/glm53/serve_glm_mtp.sh
  if [ -f "$DEPLOYED_SERVE" ] && grep -q "num_speculative_tokens" "$DEPLOYED_SERVE"; then
    ok "deployed serve script carries the speculative config"
  else
    bad "deployed serve script missing or lost the spec config: $DEPLOYED_SERVE"
  fi
else
  bad "deployed plugin missing mtp.py: $PLUG"
fi

# 5. The four vllm-gaudi MTP patches applied in the deployed tree (marker spot-checks)
VG=/opt/glm53/vllm-gaudi/vllm_gaudi
for marker_file in "v1/attention/backends/hpu_attn.py" "v1/spec_decode/hpu_eagle.py" \
                   "v1/worker/hpu_model_runner.py" "v1/sample/hpu_rejection_sampler.py"; do
  if grep -qE "gaudi-port:glm-5-3-flash/mtp|glm-mtp|spec-draft-prefill|glm-5-mxfp4:spec-" "$VG/$marker_file" 2>/dev/null; then
    ok "patches present: $marker_file"
  else
    bad "patches MISSING: $marker_file"
  fi
done

# 6. C015 state: the live plugin must run the PRE-C015 kernel for the Stage-1
#    control/treatment comparison; the repo carries the landed C015 for the window.
if grep -q "row_f = a\[:, i0:i1, :\]" "$PLUG/kda.py" 2>/dev/null; then
  note "deployed kda.py ALREADY has C015 - the window must measure it (not pre-C015)"
else
  ok "deployed kda.py is pre-C015 (C015 deploys during the window)"
fi
grep -q "row_f = a\[:, i0:i1, :\]" "$REPO/ports/glm-5-3-flash/glm5_next_hpu/glm5_next_hpu/kda.py" \
  && ok "repo kda.py carries landed C015" || bad "repo kda.py missing C015"

# 7. Git sync
cd "$REPO" || exit 1
BRANCH=$(git branch --show-current)
DIRTY=$(git status --short | wc -l)
[ "$DIRTY" -eq 0 ] && ok "git tree clean on $BRANCH" || bad "git tree dirty ($DIRTY entries)"
git fetch origin -q 2>/dev/null
LOCAL=$(git rev-parse --short HEAD); REMOTE=$(git rev-parse --short "origin/$BRANCH" 2>/dev/null)
[ "$LOCAL" = "$REMOTE" ] && ok "synced with origin ($LOCAL)" || bad "local $LOCAL != origin $REMOTE"

# 8. Engine reachability right now (informational: the window will stop it)
if curl -s -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  note "a server is LIVE on 8000 - the window will stop and restart it twice"
else
  note "no server currently on 8000"
fi

# 9. Disk space for logs/results (need > 1 GiB free on the repo filesystem)
FREE=$(df --output=avail -BG "$REPO" | tail -1 | tr -dc "0-9")
[ "$FREE" -gt 1 ] && ok "disk free: ${FREE} GiB" || bad "disk free low: ${FREE} GiB"

echo "=== pre-flight: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && echo "READY for the maintenance window." || echo "NOT READY - fix the failures above."
