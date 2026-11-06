#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve the golden prompts against an I015 diagnostic boot.

The layer-boundary capture is passive (the model self-arms from
VLLM_GLM_LAYER_DIAG_DIR + the sentinel); this runner just drives the
golden prompts serially so the payloads accumulate, then lists the diag
directory. Run against a diagnostic boot (serve_diag.sh or
serve_diag_mtp.sh) AFTER creating the sentinel.

Usage: diag_prompts.py [--base URL] [--golden PATH] [--reps N]
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--golden", default=None,
                    help="golden_reference.json (default: the port's tests/hpu copy)")
    ap.add_argument("--reps", type=int, default=1,
                    help="how many times to run the full prompt set")
    args = ap.parse_args()

    golden = args.golden
    if golden is None:
        cand = Path(__file__).resolve().parents[3] / "tests/hpu/golden_reference.json"
        golden = str(cand) if cand.exists() else "golden_reference.json"
    ref = json.loads(Path(golden).read_text())
    cases = ref["cases"] if isinstance(ref, dict) else ref
    print(f"golden prompts: {len(cases)} x {args.reps} reps")

    for rep in range(args.reps):
        for i, case in enumerate(cases):
            prompt = case["prompt"] if isinstance(case, dict) else case
            body = {"model": "zai-org/GLM-5.3-Flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64, "temperature": 0,
                    "chat_template_kwargs": {"reasoning_effort": "low"}}
            req = urllib.request.Request(args.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=280).read())
                n = r["usage"]["completion_tokens"]
                print(f"  rep {rep} prompt {i}: {n} tokens")
            except Exception as e:  # noqa: BLE001
                print(f"  rep {rep} prompt {i}: FAILED {e}")

    print("\ndiag dir contents (the layer-rank*.pt payloads):")
    diag_dir = os.environ.get("VLLM_GLM_LAYER_DIAG_DIR", "")
    if diag_dir and os.path.isdir(diag_dir):
        for f in sorted(os.listdir(diag_dir)):
            p = os.path.join(diag_dir, f)
            print(f"  {f} ({os.path.getsize(p)} bytes)")
    else:
        print("  (VLLM_GLM_LAYER_DIAG_DIR not set or unreadable from this process)")


if __name__ == "__main__":
    main()
