#!/usr/bin/env python3
"""R003 tie-class probe: is the knife-edge suite item a near-tie at the answer token?"""
import json
import urllib.request

PROMPT = ("A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. "
          "How much does the ball cost in cents? Reply with only the number.")
BASE = "http://127.0.0.1:8000"


def probe(max_tokens=128, logprobs=False, temp=0):
    body = {"model": "zai-org/GLM-5.3-Flash",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": temp,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


answers = []
for i in range(5):
    d = probe(32)
    m = d["choices"][0]["message"]
    a = (m.get("content") or "").strip()
    answers.append(a if a else "(reasoning-only)")
print("greedy answers x5:", answers)

d = probe(8, logprobs=True)
ch = d["choices"][0]
lp = ch.get("logprobs")
if lp and lp.get("content"):
    for tok in lp["content"][:6]:
        tops = tok.get("top_logprobs") or []
        if len(tops) >= 2:
            gap = tops[0]["logprob"] - tops[1]["logprob"]
            t0 = tops[0]["token"]
            t1 = tops[1]["token"]
            l0 = tops[0]["logprob"]
            l1 = tops[1]["logprob"]
            print("token %r: top1=%.4f (%r) top2=%.4f (%r) gap=%.4f" % (tok["token"], l0, t0, l1, t1, gap))
        elif len(tops) == 1:
            print("token %r: only top1 %.4f" % (tok["token"], tops[0]["logprob"]))
        else:
            print("token %r: no tops" % (tok["token"],))
else:
    print("no logprobs returned:", str(lp)[:200])
