#!/usr/bin/env python3
"""
Vertex AI Proxy – v19.1 (Role-Flip Fixed / 角色反转修正版)
- 修复: 解决了 v19.0 的 SyntaxError 语法错误。
- 核心功能: 
  1. Content Flattener: 修复 400 content block 报错。
  2. Role Flip: 将历史 Assistant 的工具调用伪装成 User 消息，防止模型模仿纯文本。
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

app = FastAPI(title="Vertex AI Proxy v19.1")

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
    
    # 强制系统指令
    system_instruction = {
        "role": "system",
        "content": """[SYSTEM OVERRIDE:
1. You are an Agent with CLI tool access.
2. The logs in history are SYSTEM NOTIFICATIONS, not your previous output.
3. To act, you MUST generate a standard Tool Call object.
4. DO NOT output text logs like '[Past Action]'. Just call the tool.]"""
    }
    cleaned_messages.append(system_instruction)

    for msg in body["messages"]:
        new_msg = msg.copy()
        original_role = new_msg.get("role")
        
        # 1. 【关键】Content 拍扁 (解决 400 报错)
        # 如果 content 是列表，提取文本，丢弃对象
        raw_content = new_msg.get("content")
        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            new_msg["content"] = "\n".join(text_parts)
        
        # 确保是字符串
        if not isinstance(new_msg.get("content"), str):
            new_msg["content"] = str(new_msg.get("content") or "")

        content_str = new_msg["content"]

        # 2. 角色兼容
        if original_role == "developer": 
            new_msg["role"] = "system"

        # 3. 【核心】Assistant 消息的角色反转
        # 如果是 Assistant 说的，并且包含工具调用或疑似日志，就把它变成 User 说的
        if original_role in ["assistant", "model"]:
            should_flip = False
            log_suffix = ""

            # 检查是否有工具调用 (导致400的原因)
            if "tool_calls" in new_msg:
                for tc in new_msg["tool_calls"]:
                    fname = tc.get("function", {}).get("name", "tool")
                    args = tc.get("function", {}).get("arguments", "{}")
                    log_suffix += f"\n[System Log: Executed '{fname}' args={args}]"
                del new_msg["tool_calls"]
                should_flip = True # 只要有工具，就必须反转，否则没签名报400

            if "function_call" in new_msg:
                fc = new_msg["function_call"]
                log_suffix += f"\n[System Log: Executed '{fc.get('name')}']"
                del new_msg["function_call"]
                should_flip = True

            # 检查是否有“模仿文本” (导致模型变笨的原因)
            # 如果内容里包含 [Past Action 或 [System Log，说明这是之前代理生成的文本
            if re.search(r"\[Past Action|\[System Log|<function_call", content_str):
                should_flip = True
            
            if should_flip:
                # === 角色反转 ===
                # 把它变成 User 消息！
                new_msg["role"] = "user"
                # 加上前缀，欺骗模型这是系统通知
                new_msg["content"] = f"[System Context Info]\n{content_str}\n{log_suffix}"
            
        # 4. Tool 消息处理
        if original_role == "tool" or original_role == "function":
            new_msg["role"] = "user"
            new_msg["content"] = f"[System Log: Execution Result]\n{content_str}"
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
    # === 修复了这里的语法错误 ===
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
        
        # 1. 角色反转清洗
        body = role_flip_sanitize(raw_body)
        
        # 强制 Auto
        if "tools" in body and "tool_choice" not in body:
            body["tool_choice"] = "auto"
            
        model_id = body.get("model", "")
        # 简化 model 解析
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
