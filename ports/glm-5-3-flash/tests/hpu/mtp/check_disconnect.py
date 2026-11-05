import json
import socket
import urllib.request
BASE = "http://127.0.0.1:8000"


# 1) start a long generation with a raw socket, then abruptly close
body = {
    "model": "zai-org/GLM-5.3-Flash",
    "messages": [{"role": "user", "content": "Write a very long detailed essay about quantum computing."}],
    "max_tokens": 2048,
    "temperature": 0,
    "chat_template_kwargs": {"reasoning_effort": "low"},
}
data = json.dumps(body).encode()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 8000))
sock.sendall(
    f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
    f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data
)
# read a few bytes then close abruptly
sock.recv(256)
sock.close()
print("client disconnected mid-generation")

# 2) immediately send normal requests — the engine should handle the orphaned request
import time
time.sleep(2)
for i in range(3):
    body2 = {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [{"role": "user", "content": f"Say the number {i + 1}."}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body2).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        txt = (r["choices"][0]["message"].get("content") or "").strip()
        print(f"post-disconnect [{i}]: {txt[:50]!r}")
    except Exception as e:
        print(f"post-disconnect [{i}]: FAIL {e}")
        break

# 3) verify engine health
req = urllib.request.Request(BASE + "/health")
urllib.request.urlopen(req, timeout=10)
print("engine: alive after disconnect")
