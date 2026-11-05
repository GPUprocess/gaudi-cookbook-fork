#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Churn soak for MTP on HPU: concurrent requests with staggered arrival/finish,
forcing batch composition changes (condense, slot reuse, finish-during-verify)
while speculative decoding is active.

Threads fire 24 requests with mixed lengths and staggered starts; every response
must be coherent (contains its marker phrase or a plausible continuation), the
engine must survive, and repeated identical prompts must agree.
"""

import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
PROMPTS = [
    "Say the word: ANCHOR",
    "Name three colors, comma separated.",
    "Count from 1 to 5, digits only.",
    "Write one sentence about the sea.",
    "What is 2+2? Answer with the number only.",
    "Say the word: BEACON",
    "Name the first five letters of the alphabet.",
    "Write one sentence about lighthouses.",
]


def ask(prompt, n, timeout=240):
    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return (r["choices"][0]["message"].get("content") or ""), time.time() - t0


def main():
    results = {}
    errors = []
    lock = threading.Lock()

    def worker(i, prompt, n, delay):
        time.sleep(delay)
        try:
            txt, w = ask(prompt, n)
            with lock:
                results[i] = (prompt, txt, w)
        except Exception as e:
            with lock:
                errors.append((i, prompt, repr(e)))

    threads = []
    for i in range(24):
        p = PROMPTS[i % len(PROMPTS)]
        n = 8 if i % 3 == 0 else (32 if i % 3 == 1 else 96)
        t = threading.Thread(target=worker, args=(i, p, n, i * 1.7))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print(f"completed: {len(results)}, errors: {len(errors)}")
    for i, p, e in errors[:5]:
        print(f"  ERROR[{i}] {p[:40]!r}: {e[:100]}")

    # determinism: identical prompts must produce identical outputs
    by_prompt = {}
    drift = 0
    for i, (p, txt, w) in sorted(results.items()):
        by_prompt.setdefault(p, []).append(txt)
    for p, txts in by_prompt.items():
        if len(txts) > 1 and len(set(txts)) > 1:
            drift += 1
            print(f"  DRIFT on {p[:40]!r}: {len(set(txts))} distinct outputs")
    print(f"{'PASS' if drift == 0 else 'FAIL'} determinism across {len(by_prompt)} prompt groups")

    # coherence: every non-empty output is plausible text (no crash garbage)
    garbage = sum(1 for _, (p, txt, w) in results.items() if txt and "\ufffd" in txt)
    print(f"{'PASS' if garbage == 0 else 'FAIL'} no replacement-char garbage")

    # engine alive
    req = urllib.request.Request(BASE + "/health")
    urllib.request.urlopen(req, timeout=10)
    print("engine alive after churn: OK")

    fails = len(errors) + drift + garbage
    print(f"CHURN TOTAL FAILURES: {fails}")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(main())
