#!/usr/bin/env bash
# boot_watch: definitive vLLM boot outcome for unattended sessions.
# Waits for the serve log to reach "Application startup complete" (BOOT-OK, exit 0)
# or a fatal failure (BOOT-FAILED: <error>, exit 1) or timeout (BOOT-TIMEOUT, exit 1).
# Benign log noise (vllm._C import warnings, Triton, nanobind, resource tracker) is
# filtered; a traceback only counts as fatal when the log stops making progress.
# Usage: tools/boot_watch.sh <logfile> [timeout_s=1800]
# Boot watchdog: waits for a vLLM serve log to reach startup-complete or fail.
# Usage: boot_watch.sh <logfile> [timeout_s=1800]
# Prints BOOT-OK or BOOT-FAILED: <first matching error line> and exits 0/1.
LOG="$1"; TO="${2:-1800}"
SEEN_ERR=0
for i in $(seq 1 $((TO / 10))); do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "BOOT-OK after ~$((i * 10))s"; exit 0
  fi
  if grep -qE "Engine core initialization failed|Traceback \(most recent call last\)|EngineDeadError|Address already in use" "$LOG" 2>/dev/null; then
    # a traceback may be a transient warning; confirm it is fatal by absence of progress
    LAST=$(grep -E "Error|error" "$LOG" | grep -viE "ModuleNotFoundError.*vllm._C|Triton|nanobind|resource_tracker" | tail -1 | cut -c1-200)
    if [ -n "$LAST" ] && ! tail -20 "$LOG" | grep -q "Loading safetensors\|Capturing\|init engine"; then
      echo "BOOT-FAILED: $LAST"; exit 1
    fi
  fi
  sleep 10
done
echo "BOOT-TIMEOUT after ${TO}s: $(tail -1 "$LOG" | cut -c1-160)"; exit 1
