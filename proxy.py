#!/usr/bin/env python3
"""
Vertex AI Proxy – v26.0 (生产就绪版)

修复历程总结：
  v22  原始版本，工具调用链断裂 + 幻觉问题
  v23  移除 tool 消息内联指令注入（防幻觉），system prompt 幂等注入
  v24  引入 thought_signature 缓存，但 key 设计错误导致永远 Miss
  v25  改用 tool_call_id 做 key，修复签名写回位置（extra_content.google），
       增加签名广播（Vertex 只给第1个 tool_call 签名时补全其余）
  v26  生产收尾：
       - 移除 DEBUG 原始消息打印
       - token 缓存（避免每请求 fork gcloud 子进程）
       - extract_signatures + extract_all_ids 合并为一次遍历
       - 缓存大小硬上限（防内存泄漏）
       - 日志分级：INFO / WARN / ERROR，减少正常流水日志噪音
       - 健康检查返回更多诊断信息
"""

import os, json, subprocess, time
from typing import Dict, Any, Optional, Tuple, List
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import httpx, uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  全局资源
# ═══════════════════════════════════════════════════════════════
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=60.0),
        limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=120.0),
        http2=False,
        verify=True,
    )
    print("[SYSTEM] HTTP Client Initialized.")
    yield
    if http_client:
        await http_client.aclose()
        print("[SYSTEM] HTTP Client Closed.")

app = FastAPI(title="Vertex AI Proxy v26.0", lifespan=lifespan)

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════
VERTEX_AI_PROJECT  = os.getenv("VERTEX_AI_PROJECT")
VERTEX_AI_REGION   = os.getenv("VERTEX_AI_REGION", "us-west1")
PROXY_HOST         = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT         = int(os.getenv("PROXY_PORT", "8000"))
MAX_TOOL_CONTENT   = 30_000       # tool 结果最大字符数
CACHE_TTL_SECONDS  = 3600         # thought_signature 缓存 1 小时
CACHE_MAX_ENTRIES  = 2000         # 缓存硬上限，防内存泄漏
TOKEN_REFRESH_SECS = 1800         # gcloud token 缓存刷新间隔（30 分钟）

REASONING_LEVELS = {"none": "minimal", "low": "low", "medium": "medium", "high": "high"}

AGENT_SYSTEM_INSTRUCTION = (
    "[AGENT INSTRUCTIONS]\n"
    "1. You are an autonomous Agent with tool-use capability.\n"
    "2. After receiving a tool result, you MUST reply to the user immediately.\n"
    "3. On success: briefly state what was accomplished.\n"
    "4. On error: briefly explain the error and suggest next steps.\n"
    "5. Never stay silent after a tool execution."
)

# ═══════════════════════════════════════════════════════════════
#  gcloud token 缓存（避免每请求 fork 子进程）
# ═══════════════════════════════════════════════════════════════
_token_cache: Dict[str, Any] = {"token": "", "ts": 0.0}

def get_vertex_token() -> str:
    now = time.time()
    if os.getenv("VERTEX_ACCESS_TOKEN"):
        return os.getenv("VERTEX_ACCESS_TOKEN")
    # 缓存未过期直接返回
    if _token_cache["token"] and now - _token_cache["ts"] < TOKEN_REFRESH_SECS:
        return _token_cache["token"]
    try:
        env = os.environ.copy()
        env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        r = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True, text=True, check=True, env=env,
        )
        token = r.stdout.strip()
        _token_cache["token"] = token
        _token_cache["ts"]    = now
        return token
    except Exception as e:
        print(f"[ERROR] Token refresh failed: {e}")
        return _token_cache["token"]   # 降级：返回旧 token

# ═══════════════════════════════════════════════════════════════
#  thought_signature 缓存
#  key: tool_call_id  →  {"sig": str, "ts": float}
#
#  为什么用 tool_call_id 而不是会话 hash？
#  tool_call_id 由 Vertex 生成且全局唯一，客户端原样传回，
#  是跨轮连接同一个 tool_call 的唯一稳定标识。
# ═══════════════════════════════════════════════════════════════
_sig_cache: Dict[str, Dict] = {}

def _sig_cache_evict() -> None:
    """过期淘汰 + 超上限时 LRU 淘汰"""
    now = time.time()
    # 1. 清过期
    expired = [k for k, v in _sig_cache.items() if now - v["ts"] > CACHE_TTL_SECONDS]
    for k in expired:
        del _sig_cache[k]
    # 2. 超上限：按时间戳排序，删最旧的
    if len(_sig_cache) > CACHE_MAX_ENTRIES:
        oldest = sorted(_sig_cache, key=lambda k: _sig_cache[k]["ts"])
        for k in oldest[:len(_sig_cache) - CACHE_MAX_ENTRIES]:
            del _sig_cache[k]

def sig_cache_put(tool_call_id: str, signature: str) -> None:
    _sig_cache[tool_call_id] = {"sig": signature, "ts": time.time()}
    if len(_sig_cache) % 100 == 0:   # 每 100 次写入触发一次淘汰
        _sig_cache_evict()

def sig_cache_get(tool_call_id: str) -> Optional[str]:
    e = _sig_cache.get(tool_call_id)
    return e["sig"] if e and time.time() - e["ts"] <= CACHE_TTL_SECONDS else None

# ═══════════════════════════════════════════════════════════════
#  thought_signature 读写辅助
#  Vertex 要求位置: tool_calls[n].extra_content.google.thought_signature
# ═══════════════════════════════════════════════════════════════
def _get_sig(tc: Dict) -> Optional[str]:
    extra  = tc.get("extra_content") or {}
    google = extra.get("google") or {}
    return google.get("thought_signature") or tc.get("thought_signature")

def _set_sig(tc: Dict, sig: str) -> Dict:
    tc = {k: v for k, v in tc.items() if k != "thought_signature"}  # 清除顶层错误字段
    existing_extra  = tc.get("extra_content") or {}
    existing_google = existing_extra.get("google") or {}
    tc["extra_content"] = {
        **existing_extra,
        "google": {**existing_google, "thought_signature": sig}
    }
    return tc

# ═══════════════════════════════════════════════════════════════
#  SSE 流解析（单次遍历，同时提取签名 + 所有 tool_call_id）
# ═══════════════════════════════════════════════════════════════
def parse_stream_tool_calls(sse_chunks: List[bytes]) -> Tuple[Dict[str, str], List[str]]:
    """
    一次遍历完成两件事：
      1. 提取 {tool_call_id: thought_signature}（仅有签名的）
      2. 提取所有 tool_call_id（含无签名的）

    返回: (sigs_dict, all_ids_list)
    """
    tc_map: Dict[int, Dict] = {}   # index → {id, thought_signature}

    for chunk in sse_chunks:
        for line in chunk.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = ((data.get("choices") or [{}])[0]).get("delta") or {}
            for tc_delta in delta.get("tool_calls") or []:
                idx = tc_delta.get("index", 0)
                if idx not in tc_map:
                    tc_map[idx] = {"id": "", "thought_signature": ""}
                tc = tc_map[idx]

                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]

                fn     = tc_delta.get("function") or {}
                extra  = tc_delta.get("extra_content") or {}
                google = extra.get("google") or {}
                sig = (
                    google.get("thought_signature")      # ← 真实位置（已确认）
                    or tc_delta.get("thought_signature") # 备用1
                    or fn.get("thought_signature")       # 备用2
                    or ""
                )
                if sig:
                    tc["thought_signature"] += sig   # 拼接分片

    sigs: Dict[str, str] = {}
    all_ids: List[str]   = []
    for tc in tc_map.values():
        tc_id = tc.get("id")
        if not tc_id:
            continue
        all_ids.append(tc_id)
        if tc.get("thought_signature"):
            sigs[tc_id] = tc["thought_signature"]

    return sigs, all_ids

# ═══════════════════════════════════════════════════════════════
#  消息清洗 + thought_signature 补全
# ═══════════════════════════════════════════════════════════════
def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(i.get("text", "")) if isinstance(i, dict) else str(i)
            for i in content
        )
    return str(content or "")

def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_CONTENT:
        return text
    half = MAX_TOOL_CONTENT // 2
    cut  = len(text) - MAX_TOOL_CONTENT
    return text[:half] + f"\n\n[...Truncated {cut} chars...]\n\n" + text[-half:]

def _restore_tool_calls(tool_calls: List[Dict], req_id: str) -> List[Dict]:
    """
    对一批 tool_calls：
      1. 按 id 从缓存补回签名
      2. 若仍有缺签名的，从同批已有签名广播
    """
    # 第1步：缓存补全
    restored = []
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not _get_sig(tc):
            sig = sig_cache_get(tc_id)
            if sig:
                tc = _set_sig(tc, sig)
                print(f"[{req_id}] [SIG] ✓ Restored id={tc_id}")
            else:
                print(f"[{req_id}] [SIG] ✗ Miss id={tc_id}")
        restored.append(tc)

    # 第2步：广播（Vertex 只给第1个签名时补全其余）
    any_sig = next((_get_sig(tc) for tc in restored if _get_sig(tc)), None)
    if any_sig:
        final = []
        for tc in restored:
            if not _get_sig(tc):
                tc = _set_sig(tc, any_sig)
                print(f"[{req_id}] [SIG] ↗ Broadcast id={tc.get('id','?')}")
            final.append(tc)
        return final

    return restored

def sanitize_and_restore(body: Dict[str, Any], req_id: str) -> Dict[str, Any]:
    if "messages" not in body:
        return body

    raw: List[Dict]  = body["messages"]
    cleaned: List[Dict] = []
    found_system = agent_injected = False

    for msg in raw:
        role    = msg.get("role", "")
        content = _flatten(msg.get("content"))

        # ── system / developer ───────────────────────────────────
        if role in ("system", "developer"):
            found_system = True
            if not agent_injected:
                content = content.rstrip() + "\n\n" + AGENT_SYSTEM_INSTRUCTION
                agent_injected = True
            cleaned.append({"role": "system", "content": content})

        # ── assistant → model ────────────────────────────────────
        elif role == "assistant":
            new_msg: Dict[str, Any] = {"role": "model", "content": content}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                new_msg["tool_calls"] = _restore_tool_calls(tool_calls, req_id)
            cleaned.append(new_msg)

        # ── tool ─────────────────────────────────────────────────
        elif role == "tool":
            new_msg = {"role": "tool", "content": _truncate(content)}
            if "tool_call_id" in msg: new_msg["tool_call_id"] = msg["tool_call_id"]
            if "name"         in msg: new_msg["name"]         = msg["name"]
            cleaned.append(new_msg)

        # ── user ─────────────────────────────────────────────────
        elif role == "user":
            cleaned.append({"role": "user", "content": content})

        # ── 其他（原样保留）──────────────────────────────────────
        else:
            cleaned.append({**msg, "content": content})

    if not found_system:
        cleaned.insert(0, {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION})

    body["messages"] = cleaned
    return body

# ═══════════════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════════════
def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    for s in REASONING_LEVELS:
        if model_id.endswith(f"-{s}"):
            return model_id.rsplit(f"-{s}", 1)[0], s
    return model_id, None

def get_endpoint_url(base_model: str) -> str:
    if "gemini-3" in base_model:
        return (f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}"
                f"/locations/global/endpoints/openapi")
    return (f"https://{VERTEX_AI_REGION}-aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}"
            f"/locations/{VERTEX_AI_REGION}/endpoints/openapi")

def _chain_summary(messages: List[Dict]) -> List[str]:
    summary = []
    for m in messages:
        role = m.get("role", "?")
        tcs  = m.get("tool_calls") or []
        if tcs:
            missing = sum(1 for tc in tcs if not _get_sig(tc))
            tag = "sig=OK" if missing == 0 else f"sig=MISSING{missing}/{len(tcs)}"
            summary.append(f"{role}[{tag}]")
        else:
            summary.append(role)
    return summary

# ═══════════════════════════════════════════════════════════════
#  流式生成器：透传 + 收集 + 缓存签名
# ═══════════════════════════════════════════════════════════════
async def stream_and_cache(response: httpx.Response, req_id: str, start: float):
    count = total = 0
    collected: List[bytes] = []
    first = False

    try:
        async for chunk in response.aiter_bytes():
            if not first:
                print(f"[{req_id}] [TTFT] {int((time.time()-start)*1000)}ms")
                first = True
            yield chunk
            collected.append(chunk)
            total += len(chunk)
            count += 1
            if count % 100 == 0:
                print(f"[{req_id}] [STREAM] {count} chunks / {total/1024:.1f} KB")

    except Exception as e:
        print(f"[{req_id}] [ERROR] Stream error: {e}")
        raise
    finally:
        await response.aclose()
        duration = time.time() - start
        print(f"[{req_id}] [DONE] {total/1024:.1f} KB in {duration:.2f}s")

        if not collected:
            return

        sigs, all_ids = parse_stream_tool_calls(collected)

        # 广播：同批内有签名的补给无签名的
        if sigs and all_ids:
            any_sig = next(iter(sigs.values()))
            for tc_id in all_ids:
                if tc_id not in sigs:
                    sigs[tc_id] = any_sig
                    print(f"[{req_id}] [SIG] ↗ Broadcast to id={tc_id}")

        if sigs:
            for tc_id, sig in sigs.items():
                sig_cache_put(tc_id, sig)
            print(f"[{req_id}] [SIG] Cached {len(sigs)} signature(s)")

# ═══════════════════════════════════════════════════════════════
#  API 端点
# ═══════════════════════════════════════════════════════════════
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_id = str(int(time.time() * 1000))[-6:]
    print(f"[{req_id}] ← Request")

    try:
        raw_body = await request.json()
        body = sanitize_and_restore(raw_body, req_id)

        if "tools" in body and "tool_choice" not in body:
            body["tool_choice"] = "auto"

        base_model, _ = parse_model_id(body.get("model", ""))
        vertex_body   = {**body, "model": base_model}
        vertex_body.pop("reasoning_effort", None)

        chain = _chain_summary(vertex_body.get("messages", []))
        print(f"[{req_id}] Chain({len(chain)}): {chain}")

        token = get_vertex_token()
        if not token:
            raise HTTPException(500, "Failed to obtain Vertex AI access token.")

        endpoint_url = get_endpoint_url(base_model)
        req_obj = http_client.build_request(
            "POST",
            endpoint_url + "/chat/completions",
            json=vertex_body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        print(f"[{req_id}] → Vertex ({base_model})")
        start = time.time()
        resp  = await http_client.send(req_obj, stream=True)

        if resp.status_code >= 400:
            err = await resp.aread()
            print(f"[{req_id}] [ERROR] Vertex {resp.status_code}: {err[:500]}")
            return Response(content=err, status_code=resp.status_code,
                            media_type="application/json")

        return StreamingResponse(
            stream_and_cache(resp, req_id, start),
            status_code=resp.status_code,
            media_type="text/event-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

# ═══════════════════════════════════════════════════════════════
#  健康检查
# ═══════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    token_age = int(time.time() - _token_cache["ts"]) if _token_cache["ts"] else -1
    return {
        "status":            "ok",
        "version":           "v26.0",
        "cached_signatures": len(_sig_cache),
        "token_age_seconds": token_age,
        "token_valid":       token_age < TOKEN_REFRESH_SECS if token_age >= 0 else False,
    }

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
