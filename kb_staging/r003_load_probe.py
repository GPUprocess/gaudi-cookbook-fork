#!/usr/bin/env python3
"""R003 under the concurrent load: does the batch-shape numerics flip the near-tie answer?

Fires 16 concurrent copies of DIFFERENT prompts with R003 as one of them, so R003's
verify rows land in a 32-row batched decode step (the same shape the accuracy suite's
concurrent arm produces). Compares the R003 answer alone vs under load, 5x each.
"""
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
R003 = ("A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. "
        "How much does the ball cost in cents? Reply with only the number.")
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


def chat(prompt, max_tokens=128):
    body = {"model": "zai-org/GLM-5.3-Flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=280).read())
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip() or "(reasoning-only)"


def r003_alone(n=5):
    return [chat(R003) for _ in range(n)]


def r003_under_load(n=5):
    answers = []

    def burst(i):
        if i == 0:
            answers.append(chat(R003))
        else:
            chat(FILLERS[(i - 1) % len(FILLERS)])

    for rep in range(n):
        threads = [threading.Thread(target=burst, args=(k,)) for k in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return answers


a0 = r003_alone()
print("alone:", a0)
a1 = r003_under_load()
print("under 16-stream load:", a1)
