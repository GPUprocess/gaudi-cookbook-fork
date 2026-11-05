import json
import urllib.request

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(__import__("os").environ.get("GLM53_MODEL_PATH", "zai-org/GLM-5.3-Flash"), trust_remote_code=True)
g = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[0] / "golden_reference.json"))
BASE = "http://127.0.0.1:8000"
THINK = "</think>"


def strip_reason(text):
    """Strip everything up to and including the </think> tag."""
    if THINK in text:
        return text[text.index(THINK) + len(THINK):]
    return text


identical = 0
for ci in range(len(g["cases"])):
    case = g["cases"][ci]
    ref_ids = case["token_ids"]
    ref_full = tok.decode(ref_ids)
    ref_content = strip_reason(ref_full)

    body = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=280).read())
    except Exception as e:
        print(f"golden[{ci}]: MTP FAIL {e}")
        continue
    mtp_content = r["choices"][0]["message"].get("content") or ""

    if mtp_content == ref_content:
        print(f"golden[{ci}]: CONTENT IDENTICAL ({len(mtp_content)} chars)")
        identical += 1
    else:
        min_len = min(len(mtp_content), len(ref_content))
        div = next((j for j in range(min_len) if mtp_content[j] != ref_content[j]), min_len)
        if mtp_content[:div] == ref_content[:div] and div == min_len:
            # one is a prefix of the other — same content, different length
            longer = "MTP" if len(mtp_content) > len(ref_content) else "REF"
            print(f"golden[{ci}]: CONTENT IDENTICAL (prefix match, {longer} has {abs(len(mtp_content)-len(ref_content))} extra chars)")
            identical += 1
        else:
            print(f"golden[{ci}]: CONTENT diverges at char {div}")
            print(f"  MTP: ...{mtp_content[max(0, div-20):div+60]!r}")
            print(f"  REF: ...{ref_content[max(0, div-20):div+60]!r}")

print(f"\n{identical}/12 cases have identical content after reasoning stripping")
