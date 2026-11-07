import argparse
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
PROMPTS = [
    "Write a short paragraph about why lighthouses use rotating beams.",
    "Summarize how a bicycle drivetrain works in four sentences.",
    "Explain the difference between weather and climate.",
    "Describe the taste and texture of sourdough bread.",
    "What does a catalyst do in a chemical reaction? Two sentences.",
    "Name three constellations visible in winter and one fact about each.",
    "Why do leaves change color in autumn? Three sentences.",
    "Explain what a semaphore flag signal is in plain words.",
    "Write a short paragraph about coral reef ecosystems.",
    "Summarize how a four-stroke engine works in plain words.",
    "Explain why ice floats on water. Three sentences.",
    "Describe the lifecycle of a butterfly in four stages.",
    "What causes the seasons on Earth? Two sentences.",
    "Explain how a compass works in plain words.",
    "Write a short paragraph about the water cycle.",
    "Describe what happens during photosynthesis. Three sentences.",
]


def worker(idx, prompt, results, errors):
    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=580).read())
        w = time.time() - t0
        results[idx] = (r["usage"]["completion_tokens"], w)
    except Exception as e:
        errors[idx] = repr(e)


def run_batch(n):
    results, errors = {}, {}
    t0 = time.time()
    threads = []
    for i in range(n):
        p = PROMPTS[i % len(PROMPTS)]
        t = threading.Thread(target=worker, args=(i, p, results, errors))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0
    total = sum(v[0] for v in results.values())
    per_stream = [v[0] / v[1] if v[1] > 0 else 0 for v in results.values()]
    return {
        "streams": n,
        "aggregate_tok_s": total / wall if wall > 0 else 0,
        "total_tokens": total,
        "wall_s": wall,
        "per_stream_mean": sum(per_stream) / len(per_stream) if per_stream else 0,
        "per_stream_min": min(per_stream) if per_stream else 0,
        "per_stream_max": max(per_stream) if per_stream else 0,
        "errors": list(errors.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for rep in range(a.reps):
        r = run_batch(a.streams)
        r["rep"] = rep + 1
        r["label"] = a.label
        rows.append(r)
        print(json.dumps(r))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
