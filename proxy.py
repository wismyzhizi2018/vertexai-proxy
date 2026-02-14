#!/usr/bin/env python3
"""
Vertex AI Proxy – v22.0 (Active Reporter / 主动汇报版)
- 继承: v21.0 的所有底层优化 (HTTP池, 流监控, 防400报错)。
- 修复: 解决模型"干完活不说话"的问题。
- 策略: 调整 System Prompt，强制模型在工具执行后必须向用户发送"完成确认"。
"""

import os
import json
import subprocess
import re
import time
from typing import Dict, Any, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# ========== 全局资源管理 ==========
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=60.0),
        limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=120.0),
        http2=False,
        verify=True
    )
    print("[SYSTEM] Global HTTP Client Initialized.")
    yield
    if http_client:
        await http_client.aclose()
        print("[SYSTEM] Global HTTP Client Closed.")

app = FastAPI(title="Vertex AI Proxy v22.0", lifespan=lifespan)

# ========== 配置 ==========
VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT") # Required env var
VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION", "us-west1")
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
MAX_LOG_LENGTH = 30000 

REASONING_LEVELS = {
    "none": "minimal", "low": "low", "medium": "medium", "high": "high"
}

# ========== 核心：角色反转与清洗 ==========
def role_flip_sanitize(body: Dict[str, Any]) -> Dict[str, Any]:
    if "messages" not in body: return body

    cleaned_messages = []
    
    # === 关键修改：更人性化的汇报指令 ===
    system_instruction = {
        "role": "system",
        "content": """[SYSTEM INSTRUCTION:
1. You are an autonomous Agent.
2. The '[System Log]' in history represents the OUTPUT of tools you just ran.
3. **DO NOT repeat the raw logs.** The user cannot read them.
4. **MANDATORY:** After reading the logs, you **MUST** reply to the user immediately.
   - If success: Say "Done" or briefly describe what changed (e.g., "Code committed and pushed.").
   - If error: Briefly explain the error.
5. **NEVER stay silent** after a tool execution.]"""
    }
    cleaned_messages.append(system_instruction)

    for msg in body["messages"]:
        new_msg = msg.copy()
        original_role = new_msg.get("role")
        
        # 1. Content 拍扁
        raw_content = new_msg.get("content")
        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item: text_parts.append(str(item["text"]))
                elif isinstance(item, str): text_parts.append(item)
            new_msg["content"] = "\n".join(text_parts)
        
        if not isinstance(new_msg.get("content"), str):
            new_msg["content"] = str(new_msg.get("content") or "")

        content_str = new_msg["content"]

        # 2. 角色兼容
        if original_role == "developer": new_msg["role"] = "system"

        # 3. Assistant 消息反转
        if original_role in ["assistant", "model"]:
            should_flip = False
            log_suffix = ""

            if "tool_calls" in new_msg:
                for tc in new_msg["tool_calls"]:
                    fname = tc.get("function", {}).get("name", "tool")
                    log_suffix += f"\n[Log: Tool '{fname}' called]"
                del new_msg["tool_calls"]
                should_flip = True

            if "function_call" in new_msg:
                fc = new_msg["function_call"]
                log_suffix += f"\n[Log: Function '{fc.get('name')}' called]"
                del new_msg["function_call"]
                should_flip = True

            if re.search(r"\[Past Action|\[System Log|\[Log:|\[Internal Context", content_str):
                should_flip = True
            
            if should_flip:
                new_msg["role"] = "user"
                new_msg["content"] = f"[Internal Context: Assistant's previous action]\n{content_str}\n{log_suffix}"
            
        # 4. Tool 消息处理
        if original_role == "tool" or original_role == "function":
            new_msg["role"] = "user"
            
            if len(content_str) > MAX_LOG_LENGTH:
                cut_len = len(content_str) - MAX_LOG_LENGTH
                head = content_str[:MAX_LOG_LENGTH // 2]
                tail = content_str[-MAX_LOG_LENGTH // 2:]
                content_str = f"{head}\n\n[...System: Output Truncated ({cut_len} chars removed)...]\n\n{tail}"
            
            # === 关键修改：在日志末尾追加“催促”提示 ===
            # 这会像有人在背后推了模型一把：“喂，结果出来了，快去告诉用户！”
            new_msg["content"] = f"[Internal Execution Result - READ ONLY]\n{content_str}\n\n[SYSTEM: Action finished. Please report status to user now.]"
            new_msg.pop("tool_call_id", None)
            new_msg.pop("name", None)

        if not new_msg.get("content"): new_msg["content"] = "."
        cleaned_messages.append(new_msg)

    body["messages"] = cleaned_messages
    return body

# ========== 辅助函数 ==========
def get_vertex_token() -> str:
    try:
        if os.getenv("VERTEX_ACCESS_TOKEN"): return os.getenv("VERTEX_ACCESS_TOKEN")
        env = os.environ.copy()
        env.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
        result = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"], capture_output=True, text=True, check=True, env=env)
        return result.stdout.strip()
    except Exception: return ""

def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    for suffix, effort in REASONING_LEVELS.items():
        if model_id.endswith(f"-{suffix}"):
            return model_id.rsplit(f"-{suffix}", 1)[0], effort
    return model_id, None

def get_endpoint_url(base_model: str) -> str:
    if "gemini-3-" in base_model:
        return f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}/locations/global/endpoints/openapi"
    return f"https://{VERTEX_AI_REGION}-aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}/locations/{VERTEX_AI_REGION}/endpoints/openapi"

# ========== API 端点 ==========
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = str(int(time.time() * 1000))[-6:]
    print(f"[{request_id}] New Request Received.")
    
    try:
        raw_body = await request.json()
        
        # 1. 清洗
        body = role_flip_sanitize(raw_body)
        
        if "tools" in body and "tool_choice" not in body:
            body["tool_choice"] = "auto"
            
        model_id = body.get("model", "")
        base_model, suffix_effort = parse_model_id(model_id)
        
        vertex_body = body.copy()
        vertex_body["model"] = base_model
        vertex_body.pop("reasoning_effort", None)

        endpoint_url = get_endpoint_url(base_model)
        token = get_vertex_token()
        headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }

        # 2. 发起请求
        print(f"[{request_id}] Sending to Vertex AI ({base_model})...")
        req = http_client.build_request("POST", endpoint_url + "/chat/completions", json=vertex_body, headers=headers)
        
        start_time = time.time()
        response = await http_client.send(req, stream=True)

        if response.status_code >= 400:
            err = await response.aread()
            print(f"[{request_id}] [ERROR] Vertex responded {response.status_code}")
            return Response(content=err, status_code=response.status_code)

        # 3. 流式监控
        async def monitored_stream_generator():
            chunk_count = 0
            total_bytes = 0
            first_byte_time = None
            
            try:
                async for chunk in response.aiter_bytes():
                    if chunk_count == 0:
                        first_byte_time = time.time()
                        latency_ms = int((first_byte_time - start_time) * 1000)
                        print(f"[{request_id}] [STREAM START] TTFT: {latency_ms}ms")
                    
                    yield chunk
                    
                    chunk_len = len(chunk)
                    total_bytes += chunk_len
                    chunk_count += 1
                    
                    if chunk_count % 50 == 0:
                        print(f"[{request_id}] [STREAMING] > {chunk_count} chunks...")
                        
            except Exception as e:
                print(f"[{request_id}] [STREAM ERROR] {e}")
                raise e
            finally:
                await response.aclose()
                duration = time.time() - start_time
                print(f"[{request_id}] [STREAM DONE] Total: {total_bytes/1024:.1f} KB in {duration:.2f}s")

        return StreamingResponse(monitored_stream_generator(), status_code=response.status_code, media_type="text/event-stream")

    except Exception as e:
        print(f"[{request_id}] [EXCEPTION] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
