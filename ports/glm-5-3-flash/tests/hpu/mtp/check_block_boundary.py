import json
import urllib.request

BASE = "http://127.0.0.1:8000"

# The attention block size is 640 tokens. The spec_verify_blocks patch fixes
# per-row block lists when verify rows cross a block boundary. Testing prompts
# of 639/640/641 tokens puts the first generated token at positions 639/640/641,
# straddling the boundary.
#
# We build prompts to exact token counts using the /tokenize endpoint.

for n_tok in (639, 640, 641):
    # build a prompt, adjusting word count to hit the target token count
    body_tok = {"model": "zai-org/GLM-5.3-Flash",
                "messages": [{"role": "user", "content": "x"}],
                "add_generation_prompt": True}
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body_tok).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    base_count = r["count"]

    # pad with filler words to reach the target
    filler_words = n_tok - base_count
    filler = " ".join(["data"] * max(filler_words, 0))
    prompt = f"{filler}\n\nWhat is 2+3? Answer with the number only."

    # verify actual token count
    body_tok2 = {"model": "zai-org/GLM-5.3-Flash",
                 "messages": [{"role": "user", "content": prompt}],
                 "add_generation_prompt": True}
    req2 = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body_tok2).encode(),
                                  headers={"Content-Type": "application/json"})
    r2 = json.loads(urllib.request.urlopen(req2, timeout=30).read())
    actual = r2["count"]

    # generate
    body_gen = {"model": "zai-org/GLM-5.3-Flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32, "temperature": 0,
                "chat_template_kwargs": {"reasoning_effort": "low"}}
    req3 = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body_gen).encode(),
                                  headers={"Content-Type": "application/json"})
    r3 = json.loads(urllib.request.urlopen(req3, timeout=280).read())
    txt = (r3["choices"][0]["message"].get("content") or "").strip()
    print(f"prompt_tokens={actual} (target {n_tok}): {txt[:80]!r}")
