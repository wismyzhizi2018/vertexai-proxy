#!/usr/bin/env python3
"""
诊断工具：捕获 gemini-3.1-flash-lite-preview 真实 SSE 响应，
找出 thought_signature 的真实 JSON 位置。

用法：
  VERTEX_AI_PROJECT=你的项目ID python3 sig_probe.py
"""
import os, json, subprocess
import httpx

PROJECT = os.getenv("VERTEX_AI_PROJECT", "YOUR_PROJECT_ID")
REGION  = os.getenv("VERTEX_AI_REGION", "us-west1")
MODEL   = "google/gemini-3.1-flash-lite-preview"

# gemini-3.x 用 global endpoint
URL = (
    f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}"
    f"/locations/global/endpoints/openapi/chat/completions"
)

def get_token():
    r = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True, text=True, check=True
    )
    return r.stdout.strip()

payload = {
    "model": MODEL,
    "stream": True,
    "tools": [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    }],
    "messages": [
        {"role": "user", "content": "Please read the file /etc/hostname"}
    ]
}

print(f"=== Model : {MODEL}")
print(f"=== URL   : {URL}")
print(f"=== Project: {PROJECT}\n")

token = get_token()
line_no = 0

with httpx.Client(timeout=60) as client:
    with client.stream(
        "POST", URL, json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    ) as resp:
        print(f"HTTP Status: {resp.status_code}\n{'='*60}")
        if resp.status_code >= 400:
            print(resp.read().decode())
            exit(1)

        for line in resp.iter_lines():
            line_no += 1
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                print(f"[line {line_no}] {data_str or '(empty)'}")
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                print(f"[line {line_no}] PARSE ERROR: {line[:200]}")
                continue

            raw = json.dumps(data, ensure_ascii=False)

            # 打印所有 chunk（完整结构）
            print(f"\n[chunk {line_no}]")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # 特别标注含签名相关字段的 chunk
            if any(k in raw for k in ["thought_signature", "thought", "signature"]):
                print(f"  ^^^ CONTAINS SIGNATURE-RELATED FIELD ^^^")

print(f"\n{'='*60}")
print(f"Total lines processed: {line_no}")
