#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-run batch-1 decode benchmark for Stage-1 reproduction (task A001).

Streams one completion of a fixed prompt and reports:
  - decode tok/s (generated tokens / wall time from first token to last)
  - median and P95 inter-token latency (ITL)
  - first-token latency (TTFT for this request)

One invocation = one measured run. Repetitions and ordering are handled by
benchmarks/stage1_reproduction.sh. Fixed prompt and generation length keep conditions
equivalent across control and treatment runs.
"""
import argparse
import json
import time
import urllib.request

PROMPT = ("Write a detailed paragraph about the history of maritime navigation, "
          "covering compasses, celestial navigation, and modern GPS.")
MAX_TOKENS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    itls = []
    tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                now = time.time()
                if first is None:
                    first = now
                else:
                    itls.append(now - last)
                last = now
                tokens += 1
    total = time.time() - t0
    decode_wall = (last - first) if (first and tokens > 1) else 0
    itls.sort()
    out = {
        "label": args.label,
        "prompt_tokens_fixed": True,
        "generated_tokens": tokens,
        "ttft_s": round(first - t0, 3) if first else None,
        "decode_tok_s": round(tokens / decode_wall, 2) if decode_wall else None,
        "itl_median_s": round(itls[len(itls) // 2], 4) if itls else None,
        "itl_p95_s": round(itls[int(len(itls) * 0.95)], 4) if itls else None,
        "total_s": round(total, 2),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
