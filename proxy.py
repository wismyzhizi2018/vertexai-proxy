#!/usr/bin/env python3
"""
Vertex AI Proxy – v19.2 (Silent Observer / 沉默观察者版)
- 继承 v19.1 的所有修复 (Role Flip + Content Flatten)。
- 增强: 强力系统指令，禁止模型复述 [System Log] 内容。
- 目标: 让模型只输出 "提交成功" 这种人话，而不是 raw log。
"""

import os
import json
import subprocess
import re
from typing import Dict, Any, Optional, Tuple, List
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vertex AI Proxy v19.2")

# ========== 配置 ==========
VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT", "gen-lang-client-0041139433")
VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION", "us-west1")
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))

REASONING_LEVELS = {
    "none": "minimal", "low": "low", "medium": "medium", "high": "high"
}

# ========== 核心：角色反转与清洗 ==========
def role_flip_sanitize(body: Dict[str, Any]) -> Dict[str, Any]:
    if "messages" not in body:
        return body

    cleaned_messages = []
    
    # === 增强的系统指令 (Anti-Echo) ===
    system_instruction = {
        "role": "system",
        "content": """[SYSTEM CRITICAL INSTRUCTION:
1. You are an autonomous Agent.
2. The '[System Log]' entries in history are INTERNAL execution data for your reference only.
3. **DO NOT repeat or quote the raw logs to the user.** The user cannot read them.
4. **Summarize the result in concise natural language** (e.g., "Update committed successfully", "Error found in file X").
5. To take action, MUST emit a standard Tool Call object directly.
6. Never output text like '[Past Action]'.]"""
    }
    cleaned_messages.append(system_instruction)

    for msg in body["messages"]:
        new_msg = msg.copy()
        original_role = new_msg.get("role")
        
        # 1. Content 拍扁 (解决 400 报错)
        raw_content = new_msg.get("content")
        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            new_msg["content"] = "\n".join(text_parts)
        
        if not isinstance(new_msg.get("content"), str):
            new_msg["content"] = str(new_msg.get("content") or "")

        content_str = new_msg["content"]

        # 2. 角色兼容
        if original_role == "developer": 
            new_msg["role"] = "system"

        # 3. Assistant 消息的角色反转
        if original_role in ["assistant", "model"]:
            should_flip = False
            log_suffix = ""

            # 检查是否有工具调用
            if "tool_calls" in new_msg:
                for tc in new_msg["tool_calls"]:
                    fname = tc.get("function", {}).get("name", "tool")
                    args = tc.get("function", {}).get("arguments", "{}")
                    # 简化日志，减少干扰
                    log_suffix += f"\n[Log: Tool '{fname}' called]"
                del new_msg["tool_calls"]
                should_flip = True

            if "function_call" in new_msg:
                fc = new_msg["function_call"]
                log_suffix += f"\n[Log: Function '{fc.get('name')}' called]"
                del new_msg["function_call"]
                should_flip = True

            # 检查是否有模仿文本
            if re.search(r"\[Past Action|\[System Log|\[Log:|\[System Context", content_str):
                should_flip = True
            
            if should_flip:
                # 变成 User 消息 (伪装成系统上下文)
                new_msg["role"] = "user"
                new_msg["content"] = f"[Internal Context: Assistant's previous action]\n{content_str}\n{log_suffix}"
            
        # 4. Tool 消息处理 (结果反馈)
        if original_role == "tool" or original_role == "function":
            new_msg["role"] = "user"
            # 明确标记这是原始数据，要求模型阅读但不复述
            new_msg["content"] = f"[Internal Execution Result - READ ONLY - DO NOT REPEAT]\n{content_str}"
            new_msg.pop("tool_call_id", None)
            new_msg.pop("name", None)

        # 兜底
        if not new_msg.get("content"):
            new_msg["content"] = "."

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
    client = None
    try:
        raw_body = await request.json()
        
        # 1. 角色反转 + 防复读清洗
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

        # 2. 稳定请求
        client = httpx.AsyncClient(timeout=300.0, http2=False)
        req = client.build_request("POST", endpoint_url + "/chat/completions", json=vertex_body, headers=headers)
        response = await client.send(req, stream=True)

        if response.status_code >= 400:
            err = await response.aread()
            print(f"[ERROR] {response.status_code}: {err.decode('utf-8', errors='ignore')}")
            await response.aclose(); await client.aclose()
            return Response(content=err, status_code=response.status_code)

        async def stream_generator():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            except Exception: pass
            finally: await response.aclose(); await client.aclose()

        return StreamingResponse(stream_generator(), status_code=response.status_code, media_type="text/event-stream")

    except Exception as e:
        if client: await client.aclose()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
