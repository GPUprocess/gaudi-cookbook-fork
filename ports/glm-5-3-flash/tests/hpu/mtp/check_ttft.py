import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def ttft(prompt_tokens, seed):
    import random
    words = [
    "harbor", "lantern", "engine", "meadow", "signal", "granite", "orbit", "willow", "cipher",
    "tide", "ember", "quarry", "beacon", "thicket", "prism", "delta", "cascade", "marlin",
    "zenith", "cobalt", "fathom", "bramble", "kestrel",
]
    rng = random.Random(seed)
    parts, n = [], 0
    while n < prompt_tokens:
        s = " ".join(rng.choice(words) for _ in range(rng.randint(12, 20)))
        parts.append(s)
        n += len(s) // 4
    doc = "\n".join(parts)
    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": doc + "\n\nSummarize the above in one sentence."}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=580).read())
    w = time.time() - t0
    n_gen = r["usage"]["completion_tokens"]
    n_prompt = r["usage"]["prompt_tokens"]
    return n_prompt, n_gen, w


# warm up
ttft(128, 0)

for depth in (2048, 8192, 16384):
    # 2 runs: first may include graph compile, second is warm
    for rep in range(2):
        n_prompt, n_gen, w = ttft(depth, seed=depth * 10 + rep)
        tag = "warm" if rep else "first"
        print(f"TTFT {depth//1000}K ({tag}): prompt={n_prompt} gen={n_gen} total={w:.1f}s")
    print()
