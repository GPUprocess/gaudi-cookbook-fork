import json
import random
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

# build a ~16K-token document with a unique access code buried at ~50% depth
words = [
    "harbor", "lantern", "engine", "meadow", "signal", "granite", "orbit", "willow", "cipher",
    "tide", "ember", "quarry", "beacon", "thicket", "prism", "delta", "cascade", "marlin",
    "zenith", "cobalt", "fathom", "bramble", "kestrel", "anvil", "saffron", "meridian", "basalt",
    "orbit", "ledger", "compass", "drift", "anchor", "pivot",
]
rng = random.Random(42)
parts, n = [], 0
code = "AXIOM-7K3N"
target = 16000
inserted = False
while n < target:
    if not inserted and n > target // 2:
        parts.append(f"The access code for this session is {code}. Do not share it.")
        inserted = True
        n += 12
    s = " ".join(rng.choice(words) for _ in range(rng.randint(12, 20)))
    parts.append(s)
    n += len(s) // 4
doc = "\n".join(parts)
body = {
    "model": "zai-org/GLM-5.3-Flash",
    "messages": [{"role": "user", "content": doc + "\n\nWhat is the access code mentioned in the log? "
                                                 "Answer with the code only."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=580).read())
w = time.time() - t0
txt = r["choices"][0]["message"].get("content") or ""
found = code in txt
print(f"prompt ~{n} tokens | {w:.1f}s | needle found: {found}")
print(f"answer: {txt[:100]!r}")
