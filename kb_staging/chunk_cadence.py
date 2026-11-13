#!/usr/bin/env python3
"""The @16 iteration-cadence arbiter: one streaming request among 16-stream load.

The chunks arrive per engine iteration; the inter-chunk gap = the iteration time and
the tokens/chunk = the per-request tokens per iteration. Reads the client-side
timestamps.
"""
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
FILLERS = [
    "Write a short paragraph about lighthouses.",
    "Summarize how a bicycle drivetrain works.",
    "Explain the difference between weather and climate.",
    "Describe the taste of sourdough bread.",
    "What does a catalyst do?",
    "Name three winter constellations.",
    "Why do leaves change color in autumn?",
    "Explain what a semaphore flag signal is.",
    "Write a paragraph about coral reefs.",
    "Summarize how a four-stroke engine works.",
    "Explain why ice floats.",
    "Describe the lifecycle of a butterfly.",
    "What causes the seasons?",
    "Explain how a compass works.",
    "Write a paragraph about the water cycle.",
]


def post(payload, timeout=280):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def filler(i):
    try:
        post({"model": "zai-org/GLM-5.3-Flash",
              "messages": [{"role": "user", "content": FILLERS[i % len(FILLERS)]}],
              "max_tokens": 256, "temperature": 0,
              "chat_template_kwargs": {"reasoning_effort": "low"}}).read()
    except Exception:
        pass


def main_stream():
    body = {"model": "zai-org/GLM-5.3-Flash",
            "messages": [{"role": "user", "content": "Count from 1 to 500 slowly, one number per line."}],
            "max_tokens": 256, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    resp = post(body)
    gaps = []
    toks_per_chunk = []
    t_prev = None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except Exception:
            continue
        t_now = time.perf_counter()
        choices = d.get("choices") or []
        n_toks = 0
        if choices:
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            r = delta.get("reasoning_content")
            if c:
                n_toks = max(1, len(c) // 4)
            elif r:
                n_toks = max(1, len(r) // 4)
            elif delta:
                n_toks = 1
        if t_prev is not None:
            gaps.append((t_now - t_prev) * 1e3)
            toks_per_chunk.append(n_toks)
        t_prev = t_now
    return gaps, toks_per_chunk


# 1. the bs=1 baseline (no load)
g0, k0 = main_stream()
import statistics
print("bs=1 alone: chunks=%d gap_mean=%.1f ms gap_p50=%.1f" %
      (len(g0), statistics.mean(g0), statistics.median(g0)))

# 2. under the 16-stream load
threads = [threading.Thread(target=filler, args=(i,)) for i in range(15)]
for t in threads:
    t.start()
time.sleep(1.0)
g1, k1 = main_stream()
for t in threads:
    t.join()
print("under load: chunks=%d gap_mean=%.1f ms gap_p50=%.1f gap_p10=%.1f gap_p90=%.1f" %
      (len(g1), statistics.mean(g1), statistics.median(g1),
       sorted(g1)[len(g1)//10], sorted(g1)[len(g1)*9//10]))
print("tokens/chunk under load: mean=%.2f p50=%.1f" %
      (statistics.mean(k1), statistics.median(k1)))
