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


# warm up
for p in PROMPTS[:2]:
    worker(0, p, {}, {})

# concurrent test: 16 streams, 256 tokens each
results, errors = {}, {}
t0 = time.time()
threads = []
for i, p in enumerate(PROMPTS):
    t = threading.Thread(target=worker, args=(i, p, results, errors))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
wall = time.time() - t0

total_tokens = sum(v[0] for v in results.values())
aggregate = total_tokens / wall if wall > 0 else 0
per_stream = [v[0] / v[1] if v[1] > 0 else 0 for v in results.values()]
print(f"concurrent 16-stream: {aggregate:.1f} tok/s aggregate over {total_tokens} tokens in {wall:.1f}s")
print(f"  per-stream mean: {sum(per_stream)/len(per_stream):.1f} tok/s  min: {min(per_stream):.1f}  max: {max(per_stream):.1f}")
print(f"  errors: {len(errors)}")
if aggregate > 150:
    print(f"  vs production k=0 baseline 217.2 tok/s: {aggregate/217.2*100:.0f}%")
