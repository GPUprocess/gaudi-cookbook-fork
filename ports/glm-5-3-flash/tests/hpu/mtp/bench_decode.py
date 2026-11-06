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
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Non-streaming pass first: the primary Stage-1 metric (wall-clock tok/s,
    # directly comparable with the historical 30.1/35.7 figures).
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = time.time() - t0
    tokens = r["usage"]["completion_tokens"]
    wall_tok_s = round(tokens / wall, 2)

    # Streaming pass: TTFT and the inter-token latency distribution.
    body_s = dict(body, stream=True)
    req_s = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body_s).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0s = time.time()
    first = None
    itls = []
    stokens = 0
    with urllib.request.urlopen(req_s, timeout=600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta") or {}
            if delta.get("content"):
                now = time.time()
                if first is None:
                    first = now
                else:
                    itls.append(now - last)
                last = now
                stokens += 1
    decode_wall = (last - first) if (first and stokens > 1) else 0
    itls.sort()
    out = {
        "label": args.label,
        "prompt_tokens_fixed": True,
        "generated_tokens": tokens,
        "wall_tok_s": wall_tok_s,
        "wall_total_s": round(wall, 2),
        "ttft_s": round(first - t0s, 3) if first else None,
        "stream_tok_s": round(stokens / decode_wall, 2) if decode_wall else None,
        "itl_median_s": round(itls[len(itls) // 2], 4) if itls else None,
        "itl_p95_s": round(itls[int(len(itls) * 0.95)], 4) if itls else None,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
