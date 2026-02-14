#!/usr/bin/env python3
"""
Vertex AI Proxy – v20.0 (Production Ready / 生产环境优化版)
- 核心继承: v19.2 的 Role-Flip + Flattener 逻辑。
- 优化 1: 全局 HTTP Client 连接池 (提升并发性能)。
- 优化 2: 日志截断 (防止 git diff 等长输出撑爆 Token)。
- 优化 3: 更好的错误日志格式。
"""

import os
import json
import subprocess
import re
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
    # 启动时初始化连接池
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=60.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
        http2=False, # Vertex 偶尔对 HTTP/2 不友好，保持 v1.1
        verify=True
    )
    print("[SYSTEM] Global HTTP Client Initialized.")
    yield
    # 关闭时清理
    if http_client:
        await http_client.aclose()
        print("[SYSTEM] Global HTTP Client Closed.")

app = FastAPI(title="Vertex AI Proxy v20.0", lifespan=lifespan)

# ========== 配置 ==========
VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT", "gen-lang-client-0041139433")
VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION", "us-west1")
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
# 最大保留的日志长度 (字符数)，约等于 5000-8000 tokens
MAX_LOG_LENGTH = 30000 

REASONING_LEVELS = {
    "none": "minimal", "low": "low", "medium": "medium", "high": "high"
}

# ========== 核心：角色反转与清洗 ==========
def role_flip_sanitize(body: Dict[str, Any]) -> Dict[str, Any]:
    if "messages" not in body: return body

    cleaned_messages = []
    
    # 系统指令
    system_instruction = {
        "role": "system",
        "content": """[SYSTEM CRITICAL INSTRUCTION:
1. You are an autonomous Agent.
2. The '[System Log]' entries in history are INTERNAL execution data for your reference only.
3. **DO NOT repeat or quote the raw logs to the user.**
4. **Summarize the result in concise natural language.**
5. To take action, MUST emit a standard Tool Call object directly.]"""
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
                    # 仅保留工具名，参数如果太长可以省略，节省 history token
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
            
        # 4. Tool 消息处理 (带截断功能)
        if original_role == "tool" or original_role == "function":
            new_msg["role"] = "user"
            
            # --- 优化：日志截断 ---
            if len(content_str) > MAX_LOG_LENGTH:
                cut_len = len(content_str) - MAX_LOG_LENGTH
                head = content_str[:MAX_LOG_LENGTH // 2]
                tail = content_str[-MAX_LOG_LENGTH // 2:]
                content_str = f"{head}\n\n[...System: Output Truncated ({cut_len} chars removed) to save memory...]\n\n{tail}"
            
            new_msg["content"] = f"[Internal Execution Result - READ ONLY]\n{content_str}"
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
    # 修复了之前版本的语法错误
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

        # 2. 使用全局 Client 发送请求
        req = http_client.build_request("POST", endpoint_url + "/chat/completions", json=vertex_body, headers=headers)
        response = await http_client.send(req, stream=True)

        if response.status_code >= 400:
            err = await response.aread()
            print(f"[ERROR] {response.status_code}: {err.decode('utf-8', errors='ignore')}")
            return Response(content=err, status_code=response.status_code)

        async def stream_generator():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except Exception: pass
            finally: await response.aclose()

        return StreamingResponse(stream_generator(), status_code=response.status_code, media_type="text/event-stream")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
