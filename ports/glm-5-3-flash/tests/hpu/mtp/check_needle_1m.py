import argparse
import json
import random
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
WORDS = [
    "harbor", "lantern", "engine", "meadow", "signal", "granite", "orbit", "willow", "cipher",
    "tide", "ember", "quarry", "beacon", "thicket", "prism", "delta", "cascade", "marlin",
    "zenith", "cobalt", "fathom", "bramble", "kestrel",
]


def build(target_tokens, needle_at, needle):
    rng = random.Random(1234)
    parts, n = [], 0
    needle_placed = False
    while n < target_tokens:
        if not needle_placed and n >= needle_at:
            parts.append(f"The access code is {needle}. Memorize it.")
            n += 12
            needle_placed = True
            continue
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 20)))
        parts.append(s)
        n += len(s) // 4
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1000000)
    ap.add_argument("--needle-at", type=int, default=950000)
    ap.add_argument("--code", default="ZEPHYR-7492")
    a = ap.parse_args()
    doc = build(a.tokens, a.needle_at, a.code)
    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": doc + "\n\nWhat is the access code? Reply with only the code."}],
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    print(f"request built: ~{len(doc)/1e6:.1f} MB of text, needle at ~{a.needle_at}")
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=14400).read())
    w = time.time() - t0
    u = r["usage"]
    content = r["choices"][0]["message"]["content"]
    print(f"wall {w:.1f}s  prompt_tokens {u['prompt_tokens']}  gen {u['completion_tokens']}")
    print(f"prefill t/s: {u['prompt_tokens']/w:.0f}")
    print("answer:", content[:120])
    print("PASS" if a.code in content else "MISS")


if __name__ == "__main__":
    main()
