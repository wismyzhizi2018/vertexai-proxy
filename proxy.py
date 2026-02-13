#!/usr/bin/env python3
"""
Vertex AI Reasoning Proxy – 终极稳定版 (Fixed Streaming)
- 修复流式请求生命周期问题 (Fix: httpx client lifecycle)
- 动态端点选择
- 增强错误诊断
"""

import os
import json
import subprocess
from typing import Dict, Any, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, Response
import httpx
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vertex AI Reasoning Proxy")

# ========== 配置 ==========
VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT", "")
VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION", "us-west1")

PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))

REASONING_LEVELS = {
    "none": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high"
}

# ========== 辅助函数 ==========
def get_vertex_token() -> str:
    try:
        # 优化：优先使用环境变量中的 token (如果存在)，方便调试
        if os.getenv("VERTEX_ACCESS_TOKEN"):
            return os.getenv("VERTEX_ACCESS_TOKEN")
            
        env = os.environ.copy()
        env.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[AUTH ERROR] {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Failed to get access token: {e}")

def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    for suffix, effort in REASONING_LEVELS.items():
        if model_id.endswith(f"-{suffix}"):
            base_model = model_id.rsplit(f"-{suffix}", 1)[0]
            return base_model, effort
    return model_id, None

def determine_reasoning_effort(
    base_model: str,
    client_effort: Optional[str],
    suffix_effort: Optional[str]
) -> Optional[str]:
    if suffix_effort is not None:
        effective = suffix_effort
    elif client_effort is not None:
        effective = client_effort
    else:
        effective = "high" if "gemini-3-" in base_model else "medium"
    return "minimal" if effective == "none" else effective

def get_endpoint_url(base_model: str) -> str:
    if "gemini-3-" in base_model:
        return f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}/locations/global/endpoints/openapi"
    else:
        return f"https://{VERTEX_AI_REGION}-aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}/locations/{VERTEX_AI_REGION}/endpoints/openapi"

# ========== API 端点 ==========
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    client = None
    try:
        body = await request.json()
        model_id = body.get("model", "")
        base_model, suffix_effort = parse_model_id(model_id)
        client_effort = body.get("reasoning_effort")
        effective_effort = determine_reasoning_effort(base_model, client_effort, suffix_effort)

        vertex_body = body.copy()
        vertex_body["model"] = base_model
        if effective_effort is not None:
            vertex_body["reasoning_effort"] = effective_effort
        else:
            vertex_body.pop("reasoning_effort", None)

        # 角色转换逻辑
        if "messages" in vertex_body:
            for msg in vertex_body["messages"]:
                role = msg.get("role")
                if role == "developer":
                    msg["role"] = "system"
                elif role not in ["system", "user", "assistant", "tool"]:
                    msg["role"] = "system"

        endpoint_url = get_endpoint_url(base_model)
        print(f"[PROXY] {model_id} -> {base_model} | Effort: {effective_effort} | Stream: {vertex_body.get('stream', False)}")

        token = get_vertex_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # 创建 Client (注意：这里不使用 async with，手动管理生命周期)
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, read=600.0, connect=60.0),
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=60.0),
            http2=False
        )

        req = client.build_request(
            "POST",
            endpoint_url + "/chat/completions",
            json=vertex_body,
            headers=headers
        )

        # 发送请求头，保持连接开启
        response = await client.send(req, stream=True)

        # 1. 如果上游报错 (非 200)
        if response.status_code >= 400:
            try:
                error_body = await response.aread()
                print(f"[ERROR] Upstream {response.status_code}: {error_body.decode('utf-8', errors='ignore')}")
                return Response(
                    content=error_body,
                    status_code=response.status_code,
                    media_type="application/json"
                )
            finally:
                await response.aclose()
                await client.aclose()

        # 2. 如果是流式请求 (Stream = True)
        if vertex_body.get("stream", False):
            async def stream_generator():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                except Exception as e:
                    print(f"[STREAM ERROR] {e}")
                    # 在流中断时，无法更改状态码，只能断开连接
                    raise e
                finally:
                    # 关键修复：数据传输完成后，关闭上游连接
                    await response.aclose()
                    await client.aclose()
                    print("[PROXY] Stream finished, connection closed.")

            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                media_type="text/event-stream"
            )

        # 3. 如果是普通请求 (Stream = False)
        else:
            try:
                content = await response.aread()
                # 尝试解析 JSON 以确保完整性
                try:
                    json_content = json.loads(content)
                    return JSONResponse(content=json_content, status_code=response.status_code)
                except json.JSONDecodeError:
                    return Response(content=content, status_code=response.status_code)
            finally:
                await response.aclose()
                await client.aclose()

    except Exception as e:
        # 发生未预期的异常时，确保 client 被关闭
        if client:
            await client.aclose()
        import traceback
        print(f"[EXCEPTION] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "fixed-lifecycle"}

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
