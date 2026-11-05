import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def gen(prompt, n):
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
    r = json.loads(urllib.request.urlopen(req, timeout=590).read())
    return r, time.time() - t0


# long generation: 2048 tokens, coherence checked at the span
r, w = gen("Write a detailed essay about the history and future of space telescopes.", 2048)
n = r["usage"]["completion_tokens"]
txt = r["choices"][0]["message"].get("content") or ""
print(f"2048-max: {n} tokens in {w:.1f}s = {n/w:.1f} tok/s")
# degeneration check: repetition of long loops
words = txt.split()
rep = 0
for size in (8, 16, 32):
    for i in range(0, len(words) - 2 * size, size):
        if words[i:i + size] == words[i + size:i + 2 * size]:
            rep += 1
print(f"degeneration loops: {rep} | tail: {txt[-120:]!r}")

# second long generation on a different topic (state carry-over from the first)
r2, w2 = gen("Now write a short poem about the ocean.", 256)
n2 = r2["usage"]["completion_tokens"]
txt2 = r2["choices"][0]["message"].get("content") or ""
print(f"follow-up: {n2} tokens in {w2:.1f}s = {n2/w2:.1f} tok/s | tail: {txt2[-100:]!r}")

# third: long generation again (state re-stress after the short one)
r3, w3 = gen("Explain in detail how a nuclear reactor works, section by section.", 2048)
n3 = r3["usage"]["completion_tokens"]
txt3 = r3["choices"][0]["message"].get("content") or ""
print(f"second long: {n3} tokens in {w3:.1f}s = {n3/w3:.1f} tok/s | tail: {txt3[-120:]!r}")
