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


def build_prompt_tokens(target):
    rng = random.Random(target * 7 + 1)
    parts, n = [], 0
    while n < target:
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 20)))
        parts.append(s)
        n += len(s) // 4
    return "\n".join(parts)


def prefill(target, seed, gen=1):
    doc = build_prompt_tokens(target)
    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": doc + "\n\nReply with OK."}],
        "max_tokens": gen,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=580).read())
    w = time.time() - t0
    n_prompt = r["usage"]["prompt_tokens"]
    # TTFT approximated by total wall minus generation time share; with gen=1
    # token the wall is essentially TTFT + one decode step.
    return n_prompt, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="512,1024,2048,4096,8192")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="prefill")
    a = ap.parse_args()
    # warm
    prefill(256, 0, gen=8)
    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        for rep in range(a.reps):
            n_prompt, w = prefill(depth, seed=depth * 100 + rep, gen=1)
            tps = n_prompt / w
            rows.append({"depth_target": depth, "prompt_tokens": n_prompt, "wall_s": w,
                         "prefill_tok_s": tps, "rep": rep + 1, "label": a.label,
                         "tag": "warm" if rep else "first"})
            print(json.dumps(rows[-1]))
    if len(sys.argv) > 1 and "--out" in sys.argv:
        pass


if __name__ == "__main__":
    main()
