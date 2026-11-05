import json
import urllib.request
import base64
import io
from PIL import Image

BASE = "http://127.0.0.1:8000"

# 1) frequency penalty under spec decode (spec_penalties patch)
body = {
    "model": "zai-org/GLM-5.3-Flash",
    "messages": [{"role": "user", "content": "Write five sentences about the ocean."}],
    "max_tokens": 256,
    "temperature": 0.7,
    "frequency_penalty": 1.5,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=280).read())
    txt = r["choices"][0]["message"].get("content") or ""
    print(f"penalty: {r['usage']['completion_tokens']} tok, content head: {txt[:80]!r}")
except urllib.error.HTTPError as e:
    print(f"penalty HTTP {e.code}: {e.read()[:200]}")

# 2) image limit: 5 images should be rejected
img = Image.new("RGB", (64, 64), (200, 200, 200))
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}] * 5
body2 = {
    "model": "zai-org/GLM-5.3-Flash",
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 8,
    "temperature": 0,
}
req2 = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body2).encode(),
                              headers={"Content-Type": "application/json"})
try:
    r2 = json.loads(urllib.request.urlopen(req2, timeout=280).read())
    print(f"5-images: ACCEPTED (should be rejected!) content={str(r2['choices'][0]['message'])[:80]}")
except urllib.error.HTTPError as e:
    print(f"5-images: HTTP {e.code} (rejected ✓)")

# 3) structured output (JSON mode) under spec decode
body3 = {
    "model": "zai-org/GLM-5.3-Flash",
    "messages": [{"role": "user", "content": "Return a JSON object with key 'color' set to 'blue'."}],
    "max_tokens": 64,
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
req3 = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body3).encode(),
                              headers={"Content-Type": "application/json"})
try:
    r3 = json.loads(urllib.request.urlopen(req3, timeout=280).read())
    txt3 = r3["choices"][0]["message"].get("content") or ""
    parsed = json.loads(txt3) if txt3.strip().startswith("{") else None
    print(f"json-mode: {txt3[:80]!r}, parsed={parsed}")
except urllib.error.HTTPError as e:
    print(f"json-mode HTTP {e.code}: {e.read()[:200]}")
except json.JSONDecodeError:
    print(f"json-mode: content not valid JSON: {txt3[:80]!r}")
