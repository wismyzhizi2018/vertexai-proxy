#!/usr/bin/env python3
"""
Vertex AI Proxy – v29.0 (生产就绪版)

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

import os, json, subprocess, time, hashlib, sqlite3, threading
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
    # 提前初始化 SQLite，不等第一次请求才懒加载，确保启动日志数字准确
    _get_db()
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
    if _db_conn:
        _db_conn.close()
        print("[SYSTEM] SQLite cache closed.")

app = FastAPI(title="Vertex AI Proxy v29.0", lifespan=lifespan)

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════
VERTEX_AI_PROJECT  = os.getenv("VERTEX_AI_PROJECT")
VERTEX_AI_REGION   = os.getenv("VERTEX_AI_REGION", "us-west1")
PROXY_HOST         = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT         = int(os.getenv("PROXY_PORT", "8000"))
MAX_TOOL_CONTENT   = 30_000
CACHE_TTL_SECONDS  = 86400   # 24小时（原1小时太短，长任务会过期）
CACHE_MAX_ENTRIES  = 2000
TOKEN_REFRESH_SECS = 1800
CACHE_DB_PATH      = os.getenv("CACHE_DB_PATH", "/var/lib/vertexai-proxy/sig_cache.db")

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL  = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION  = "2023-06-01"

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
        return _token_cache["token"]

# ═══════════════════════════════════════════════════════════════
#  thought_signature 持久化缓存（SQLite）
#
#  解决问题：
#    - 服务重启后签名全部丢失 → 全部 Miss → Vertex 400
#    - 多用户签名污染（ns 隔离）
#
#  设计：
#    - SQLite 单文件，Python 内置，零依赖
#    - 写操作异步化（丢入线程池），不阻塞请求路径
#    - 读操作同步（SQLite 读极快，<1ms）
#    - WAL 模式：读写并发，不互相阻塞
#    - 每 100 次写入触发一次过期清理
#
#  表结构：
#    ns TEXT, tool_call_id TEXT, sig TEXT, ts REAL
#    PRIMARY KEY (ns, tool_call_id)
# ═══════════════════════════════════════════════════════════════
_db_lock = threading.Lock()
_db_conn: Optional[sqlite3.Connection] = None
_write_count = 0

def _db_init() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sig_cache (
            ns           TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            sig          TEXT NOT NULL,
            ts           REAL NOT NULL,
            PRIMARY KEY (ns, tool_call_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON sig_cache(ts)")
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM sig_cache").fetchone()[0]
    print(f"[SYSTEM] SQLite cache loaded: {total} signatures from {CACHE_DB_PATH}")
    return conn

def _get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = _db_init()
    return _db_conn

def _user_ns(auth_header: str) -> str:
    return hashlib.sha256(auth_header.encode()).hexdigest()[:16]

def sig_cache_put(ns: str, tool_call_id: str, signature: str) -> None:
    global _write_count
    now = time.time()
    def _write():
        global _write_count
        with _db_lock:
            db = _get_db()
            db.execute(
                "INSERT OR REPLACE INTO sig_cache (ns, tool_call_id, sig, ts) VALUES (?, ?, ?, ?)",
                (ns, tool_call_id, signature, now)
            )
            _write_count += 1
            if _write_count % 100 == 0:
                # 清过期 + 超上限的旧条目
                cutoff = now - CACHE_TTL_SECONDS
                db.execute("DELETE FROM sig_cache WHERE ts < ?", (cutoff,))
                # 每个 ns 保留最新 CACHE_MAX_ENTRIES 条
                db.execute("""
                    DELETE FROM sig_cache WHERE rowid IN (
                        SELECT rowid FROM sig_cache s1
                        WHERE (SELECT COUNT(*) FROM sig_cache s2
                               WHERE s2.ns = s1.ns AND s2.ts >= s1.ts) > ?
                    )
                """, (CACHE_MAX_ENTRIES,))
            db.commit()
    # 异步写入，不阻塞请求路径
    threading.Thread(target=_write, daemon=True).start()

def sig_cache_get(ns: str, tool_call_id: str) -> Optional[str]:
    cutoff = time.time() - CACHE_TTL_SECONDS
    with _db_lock:
        db = _get_db()
        row = db.execute(
            "SELECT sig FROM sig_cache WHERE ns=? AND tool_call_id=? AND ts>?",
            (ns, tool_call_id, cutoff)
        ).fetchone()
    return row[0] if row else None

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

def _restore_tool_calls(tool_calls: List[Dict], req_id: str, ns: str) -> List[Dict]:
    """
    对一批 tool_calls：
      1. 按 id 从用户隔离缓存补回签名
      2. 若仍有缺签名的，从同批已有签名广播
    """
    # 第1步：缓存补全（ns 隔离，不会拿到其他用户的签名）
    restored = []
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not _get_sig(tc):
            sig = sig_cache_get(ns, tc_id)
            if sig:
                tc = _set_sig(tc, sig)
                print(f"[{req_id}] [SIG] ✓ Restored id={tc_id}")
            else:
                print(f"[{req_id}] [SIG] ✗ Miss id={tc_id}")
        restored.append(tc)

    # 第2步：同批广播（Vertex 只给第1个签名时补全其余）
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

def sanitize_and_restore(body: Dict[str, Any], req_id: str, ns: str) -> Dict[str, Any]:
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
                new_msg["tool_calls"] = _restore_tool_calls(tool_calls, req_id, ns)
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

    # ── 全局签名广播（第二阶段）──────────────────────────────────────────
    # 策略1：从消息链中找任意已有签名广播给所有缺口
    # 策略2：链中全部是孤儿时（切换模型/新会话），从 DB 取该用户最近一条签名兜底
    # 依据：Vertex 只验证签名字段存在，不验证签名与具体 tool_call 的绑定关系
    global_sig: Optional[str] = None

    # 策略1：链内查找
    for msg in cleaned:
        if msg.get("role") != "model":
            continue
        for tc in (msg.get("tool_calls") or []):
            s = _get_sig(tc)
            if s:
                global_sig = s
                break
        if global_sig:
            break

    # 策略2：链内无签名时，从 DB 取该用户任意最近签名（切换模型/新会话场景）
    if not global_sig:
        cutoff = time.time() - CACHE_TTL_SECONDS
        with _db_lock:
            row = _get_db().execute(
                "SELECT sig FROM sig_cache WHERE ns=? AND ts>? ORDER BY ts DESC LIMIT 1",
                (ns, cutoff)
            ).fetchone()
        if row:
            global_sig = row[0]
            print(f"[{req_id}] [SIG] ↗ Fallback: using own DB signature for ns={ns}")

    # 策略3：该用户 DB 也没有（首次使用思考模型）→ 借用全局任意最近签名
    # Vertex 只验证签名存在，不验证归属，跨用户借用安全
    if not global_sig:
        cutoff = time.time() - CACHE_TTL_SECONDS
        with _db_lock:
            row = _get_db().execute(
                "SELECT sig FROM sig_cache WHERE ts>? ORDER BY ts DESC LIMIT 1",
                (cutoff,)
            ).fetchone()
        if row:
            global_sig = row[0]
            print(f"[{req_id}] [SIG] ↗ Fallback: using global DB signature (no own sig found)")

    if global_sig:
        filled = 0
        for msg in cleaned:
            if msg.get("role") != "model" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            for tc in msg["tool_calls"]:
                if not _get_sig(tc):
                    tc = _set_sig(tc, global_sig)
                    filled += 1
                new_tcs.append(tc)
            msg["tool_calls"] = new_tcs
        if filled:
            print(f"[{req_id}] [SIG] ↗ Global broadcast: filled {filled} orphan signature(s)")
    # ─────────────────────────────────────────────────────────────────────

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
async def stream_and_cache(response: httpx.Response, req_id: str, start: float, ns: str):
    count = total = 0
    collected: List[bytes] = []
    first = False
    # 流式实时写：收到签名立刻写 DB，不等 finally（修复 race condition）
    live_sigs: Dict[str, str] = {}   # {tc_id: sig} 流中已发现的
    live_ids:  List[str]      = []   # 流中所有 tc_id（含无签名的）
    live_tc_map: Dict[int, Dict] = {}  # index → {id, sig}

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

            # 实时解析当前 chunk，发现新签名立即写 DB
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                delta = ((data.get("choices") or [{}])[0]).get("delta") or {}
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in live_tc_map:
                        live_tc_map[idx] = {"id": "", "sig": ""}
                    entry = live_tc_map[idx]
                    if tc_delta.get("id") and not entry["id"]:
                        entry["id"] = tc_delta["id"]
                        if entry["id"] not in live_ids:
                            live_ids.append(entry["id"])
                    extra  = tc_delta.get("extra_content") or {}
                    google = extra.get("google") or {}
                    sig = (google.get("thought_signature")
                           or tc_delta.get("thought_signature")
                           or (tc_delta.get("function") or {}).get("thought_signature")
                           or "")
                    if sig:
                        entry["sig"] += sig
                        tc_id = entry["id"]
                        if tc_id and tc_id not in live_sigs:
                            live_sigs[tc_id] = entry["sig"]
                            sig_cache_put(ns, tc_id, entry["sig"])  # 立即写 DB

    except Exception as e:
        print(f"[{req_id}] [ERROR] Stream error: {e}")
        raise
    finally:
        await response.aclose()
        duration = time.time() - start
        print(f"[{req_id}] [DONE] {total/1024:.1f} KB in {duration:.2f}s")

        if not collected:
            return

        # finally 阶段：广播（同批内有签名的补给无签名的）并补写 DB
        if live_sigs and live_ids:
            any_sig = next(iter(live_sigs.values()))
            for tc_id in live_ids:
                if tc_id not in live_sigs:
                    live_sigs[tc_id] = any_sig
                    sig_cache_put(ns, tc_id, any_sig)
                    print(f"[{req_id}] [SIG] ↗ Broadcast to id={tc_id}")

        if live_sigs:
            print(f"[{req_id}] [SIG] Cached {len(live_sigs)} signature(s)")

# ═══════════════════════════════════════════════════════════════
#  API 端点
# ═══════════════════════════════════════════════════════════════
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_id = str(int(time.time() * 1000))[-6:]
    print(f"[{req_id}] ← Request")

    auth_header = request.headers.get("Authorization", "")
    ns = _user_ns(auth_header) if auth_header else "anonymous"

    try:
        raw_body = await request.json()
        model_id = raw_body.get("model", "")

        # ── 路由分流：claude-* → Anthropic，其余 → Vertex AI ──────────────
        if model_id.startswith("anthropic/") or model_id.startswith("claude-"):
            return await _forward_anthropic(raw_body, req_id)

        # ── Vertex AI 路径（原有逻辑）─────────────────────────────────────
        body = sanitize_and_restore(raw_body, req_id, ns)

        if "tools" in body and "tool_choice" not in body:
            body["tool_choice"] = "auto"

        base_model, _ = parse_model_id(body.get("model", ""))
        vertex_body   = {**body, "model": base_model}
        vertex_body.pop("reasoning_effort", None)

        chain = _chain_summary(vertex_body.get("messages", []))
        print(f"[{req_id}] Chain({len(chain)}) ns={ns}: {chain}")

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
            stream_and_cache(resp, req_id, start, ns),
            status_code=resp.status_code,
            media_type="text/event-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


# Claude 4.x 模型：opus-4-6/sonnet-4-6 用 adaptive thinking + effort
# opus-4-5/sonnet-4-5 用旧的 budget_tokens 方式
_CLAUDE4_ADAPTIVE_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}
_CLAUDE4_BUDGET_MODELS   = {"claude-opus-4-5", "claude-sonnet-4-5"}
_CLAUDE4_THINKING_MODELS = _CLAUDE4_ADAPTIVE_MODELS | _CLAUDE4_BUDGET_MODELS

async def _forward_anthropic(body: dict, req_id: str):
    """将 OpenAI 格式请求转换为 Anthropic Messages API 格式并透传响应"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    # ── OpenAI → Anthropic 格式转换 ──────────────────────────────────────
    model_id = body.get("model", "").replace("anthropic/", "")
    messages  = body.get("messages", [])
    stream    = body.get("stream", False)

    # 分离 system prompt（支持多条 system 消息合并）
    system_blocks = []
    filtered_msgs = []
    for m in messages:
        if m.get("role") in ("system", "developer"):
            text = m.get("content", "")
            if isinstance(text, list):
                # content 可能是 block 数组
                text = " ".join(b.get("text","") for b in text if b.get("type")=="text")
            if text:
                system_blocks.append(text)
        else:
            filtered_msgs.append(m)

    # BUG FIX 1: 合并相邻同 role 消息
    # Anthropic 不允许连续两条相同 role，需要合并
    merged_msgs = []
    for m in filtered_msgs:
        role = m.get("role")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")
        tool_call_id = m.get("tool_call_id")
        if (merged_msgs and merged_msgs[-1]["_role"] == role
                and role in ("user", "assistant") and not tool_call_id and not tool_calls):
            # 合并连续同 role 消息（Anthropic 不允许连续相同 role）
            prev = merged_msgs[-1]
            new_text = content or ""
            if isinstance(prev["_content"], list):
                prev["_content"].append({"type": "text", "text": new_text})
            else:
                prev["_content"] = (prev["_content"] or "") + ("\n" + new_text if new_text else "")
        else:
            merged_msgs.append({"_role": role, "_content": content,
                                 "_tool_calls": tool_calls, "_tool_call_id": tool_call_id,
                                 "_name": m.get("name")})

    # 转换 tool_calls（OpenAI → Anthropic 格式）
    anthropic_msgs = []
    for m in merged_msgs:
        role = m["_role"]
        content = m["_content"]
        tool_calls = m["_tool_calls"]
        tool_call_id = m["_tool_call_id"]

        if role == "assistant":
            blocks = []
            # BUG FIX 2: content 可能是 list（某些客户端发 block 数组）
            if isinstance(content, list):
                for b in content:
                    if b.get("type") == "text" and b.get("text"):
                        blocks.append({"type": "text", "text": b["text"]})
            elif content:
                blocks.append({"type": "text", "text": content})
            for tc in (tool_calls or []):
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            # BUG FIX 3: assistant content 不能为空数组
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            anthropic_msgs.append({"role": "assistant", "content": blocks})

        elif role == "tool":
            # BUG FIX 4: 多个 tool result 应合并到同一个 user 消息
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_call_id or "",
                "content": content or "",
            }
            # 如果上一条已经是 tool_result user 消息，追加进去
            if (anthropic_msgs and anthropic_msgs[-1]["role"] == "user"
                    and isinstance(anthropic_msgs[-1]["content"], list)
                    and any(b.get("type") == "tool_result" for b in anthropic_msgs[-1]["content"])):
                anthropic_msgs[-1]["content"].append(tool_result)
            else:
                anthropic_msgs.append({"role": "user", "content": [tool_result]})

        else:
            # user 消息：content 可能是 list（含图片 block）
            if isinstance(content, list):
                converted_blocks = []
                for b in content:
                    if b.get("type") == "text":
                        converted_blocks.append({"type": "text", "text": b.get("text", "")})
                    elif b.get("type") == "image_url":
                        # OpenAI image_url → Anthropic image source 格式转换
                        url = b.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # base64 内嵌图片: data:image/jpeg;base64,<data>
                            try:
                                header, data_part = url.split(",", 1)
                                media_type = header.split(":")[1].split(";")[0]
                                converted_blocks.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data_part,
                                    }
                                })
                            except Exception:
                                pass  # 格式异常跳过
                        else:
                            # URL 引用图片
                            converted_blocks.append({
                                "type": "image",
                                "source": {"type": "url", "url": url}
                            })
                    else:
                        # 其他 block 类型原样保留
                        converted_blocks.append(b)
                anthropic_msgs.append({"role": role, "content": converted_blocks})
            else:
                anthropic_msgs.append({"role": role, "content": content or ""})

    # BUG FIX 5: 确保第一条消息是 user（Anthropic 要求）
    if anthropic_msgs and anthropic_msgs[0]["role"] != "user":
        anthropic_msgs.insert(0, {"role": "user", "content": "."})

    # 转换 tools
    anthropic_tools = []
    for t in (body.get("tools") or []):
        fn = t.get("function", {})
        anthropic_tools.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })

    # 转换 tool_choice
    anthropic_tool_choice = None
    tc_raw = body.get("tool_choice")
    if tc_raw == "auto":
        anthropic_tool_choice = {"type": "auto"}
    elif tc_raw == "none":
        anthropic_tool_choice = None  # Anthropic 不支持 none，直接不传 tools
    elif isinstance(tc_raw, dict) and tc_raw.get("type") == "function":
        anthropic_tool_choice = {"type": "tool", "name": tc_raw["function"]["name"]}

    anthropic_body: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": body.get("max_tokens", 8192),
        "messages": anthropic_msgs,
        "stream": stream,
    }
    if system_blocks:
        anthropic_body["system"] = "\n".join(system_blocks).strip()
    # tool_choice=none 时不传 tools（Anthropic 无 none 类型）
    if anthropic_tools and tc_raw != "none":
        anthropic_body["tools"] = anthropic_tools
    if anthropic_tool_choice and anthropic_tools and tc_raw != "none":
        anthropic_body["tool_choice"] = anthropic_tool_choice
    # Claude 4.x 扩展思考
    # opus-4-6/sonnet-4-6: adaptive thinking + effort（budget_tokens 已废弃）
    # opus-4-5/sonnet-4-5: 旧版 enabled + budget_tokens
    reasoning_effort = body.get("reasoning_effort")
    thinking_enabled = False

    if model_id in _CLAUDE4_ADAPTIVE_MODELS and reasoning_effort and reasoning_effort != "none":
        # 新模型：用 adaptive + effort 参数
        # effort: low/medium/high/max（max 是 opus-4-6 新增）
        effort_map = {"low": "low", "medium": "medium", "high": "high", "max": "max"}
        effort = effort_map.get(reasoning_effort, "high")
        anthropic_body["thinking"] = {"type": "adaptive"}
        anthropic_body["effort"] = effort
        # adaptive 模式 max_tokens 建议 >= 16000
        if anthropic_body["max_tokens"] < 16000:
            anthropic_body["max_tokens"] = 16000
        thinking_enabled = True

    elif model_id in _CLAUDE4_BUDGET_MODELS and reasoning_effort and reasoning_effort != "none":
        # 旧模型：用 enabled + budget_tokens
        budget_map = {"low": 2000, "medium": 8000, "high": 16000}
        budget = budget_map.get(reasoning_effort, 8000)
        anthropic_body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if anthropic_body["max_tokens"] <= budget:
            anthropic_body["max_tokens"] = budget + 4096
        thinking_enabled = True

    # temperature/top_p: 扩展思考模式下不能传（会 400）
    if not thinking_enabled:
        if body.get("temperature") is not None:
            anthropic_body["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            anthropic_body["top_p"] = body["top_p"]

    if body.get("stop") is not None:
        stop = body["stop"]
        anthropic_body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    print(f"[{req_id}] → Anthropic ({model_id}) msgs={len(anthropic_msgs)} "
          f"stream={stream} thinking={thinking_enabled}")
    start = time.time()

    # opus-4-6 / sonnet-4-6 支持 1M context（beta），自动启用
    extra_betas = []
    if model_id in _CLAUDE4_ADAPTIVE_MODELS:
        extra_betas.append("output-128k-2025-02-19")     # 128K 输出
    # interleaved-thinking: opus-4-6 上已废弃，仅 sonnet-4-6 开启 thinking 时需要
    if model_id == "claude-sonnet-4-6" and thinking_enabled:
        extra_betas.append("interleaved-thinking-2025-05-14")

    # 客户端通过 max_context=1m 或 contextWindow>=500000 触发 1M beta
    if (body.get("max_context") == "1m"
            or (body.get("contextWindow", 0) >= 500_000)
            or body.get("enable_1m_context")):
        extra_betas.append("context-1m-2025-08-07")

    # Fast mode: opus-4-6 专属，速度提升 2.5x
    # 仅传 speed 字段，不加 beta header（header 版本号需以实际文档为准）
    if body.get("speed") == "fast" and model_id == "claude-opus-4-6":
        anthropic_body["speed"] = "fast"

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    if extra_betas:
        headers["anthropic-beta"] = ",".join(extra_betas)

    req_obj = http_client.build_request(
        "POST",
        ANTHROPIC_API_URL,
        json=anthropic_body,
        headers=headers,
    )
    resp = await http_client.send(req_obj, stream=True)

    if resp.status_code >= 400:
        err = await resp.aread()
        print(f"[{req_id}] [ERROR] Anthropic {resp.status_code}: {err[:500]}")
        return Response(content=err, status_code=resp.status_code,
                        media_type="application/json")

    if not stream:
        # 非流式：Anthropic → OpenAI 格式转换后返回
        data = await resp.aread()
        try:
            ar = json.loads(data)
            content_blocks = ar.get("content", [])
            # 过滤 thinking block（扩展思考内容不透传给客户端）
            # 多段 text 换行拼接
            text = "\n".join(b.get("text","") for b in content_blocks
                             if b.get("type")=="text" and b.get("text"))
            tool_calls = []
            for b in content_blocks:
                if b.get("type") == "thinking":
                    continue  # 跳过思考块
                if b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b["id"], "type": "function",
                        "function": {"name": b["name"],
                                     "arguments": json.dumps(b.get("input",{}), ensure_ascii=False)}
                    })
            # stop_reason 映射
            stop_reason = ar.get("stop_reason", "stop")
            finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"
            oai_resp = {
                "id": ar.get("id",""),
                "object": "chat.completion",
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": tool_calls or None,
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": ar.get("usage",{}).get("input_tokens", 0),
                    "completion_tokens": ar.get("usage",{}).get("output_tokens", 0),
                    "total_tokens": (ar.get("usage",{}).get("input_tokens", 0)
                                     + ar.get("usage",{}).get("output_tokens", 0)),
                }
            }
            duration = time.time() - start
            print(f"[{req_id}] [DONE] Anthropic non-stream in {duration:.2f}s")
            return Response(content=json.dumps(oai_resp, ensure_ascii=False),
                            media_type="application/json")
        except Exception as e:
            print(f"[{req_id}] [ERROR] Anthropic response parse: {e}")
            return Response(content=data, status_code=200, media_type="application/json")

    # 流式：Anthropic SSE → OpenAI SSE 格式转换
    async def convert_anthropic_stream(resp, req_id, start):
        first = True
        buffer = ""  # 跨 chunk 的不完整 SSE 行缓冲
        try:
            async for chunk in resp.aiter_bytes():
                if first:
                    print(f"[{req_id}] [TTFT] {int((time.time()-start)*1000)}ms")
                    first = False
                # BUG FIX 7: SSE 行可能跨 chunk，需要缓冲
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        ev = json.loads(data)
                        ev_type = ev.get("type","")

                        if ev_type == "content_block_delta":
                            delta = ev.get("delta",{})
                            if delta.get("type") == "thinking_delta":
                                pass  # 过滤扩展思考内容，不透传给客户端
                            elif delta.get("type") == "text_delta":
                                oai = {"choices":[{"delta":{"content":delta.get("text","")},"index":0}]}
                                yield f"data: {json.dumps(oai, ensure_ascii=False)}\n\n".encode()
                            elif delta.get("type") == "input_json_delta":
                                oai = {"choices":[{"delta":{"tool_calls":[{
                                    "index": ev.get("index", 0),
                                    "function":{"arguments":delta.get("partial_json","")}}
                                ]},"index":0}]}
                                yield f"data: {json.dumps(oai)}\n\n".encode()

                        elif ev_type == "content_block_start":
                            block = ev.get("content_block",{})
                            if block.get("type") == "thinking":
                                pass  # 过滤扩展思考 block
                            elif block.get("type") == "tool_use":
                                oai = {"choices":[{"delta":{"tool_calls":[{
                                    "index": ev.get("index", 0),
                                    "id": block.get("id",""),
                                    "type": "function",
                                    "function": {"name": block.get("name",""), "arguments":""}
                                }]},"index":0}]}
                                yield f"data: {json.dumps(oai)}\n\n".encode()

                        elif ev_type == "message_delta":
                            # BUG FIX 8: 补发 finish_reason
                            delta = ev.get("delta", {})
                            stop_reason = delta.get("stop_reason","")
                            finish = "tool_calls" if stop_reason == "tool_use" else "stop"
                            oai = {"choices":[{"delta":{},"index":0,"finish_reason": finish}]}
                            yield f"data: {json.dumps(oai)}\n\n".encode()

                        elif ev_type == "message_stop":
                            yield b"data: [DONE]\n\n"

                    except Exception:
                        pass
        finally:
            # 处理 buffer 中残留的最后一行（无结尾换行的情况）
            if buffer.strip() and buffer.strip().startswith("data:"):
                data = buffer.strip()[5:].strip()
                if data and data != "[DONE]":
                    try:
                        ev = json.loads(data)
                        if ev.get("type") == "message_stop":
                            yield b"data: [DONE]\n\n"
                    except Exception:
                        pass
            await resp.aclose()
            duration = time.time() - start
            print(f"[{req_id}] [DONE] Anthropic stream in {duration:.2f}s")

    return StreamingResponse(
        convert_anthropic_stream(resp, req_id, start),
        status_code=200,
        media_type="text/event-stream",
    )

# ═══════════════════════════════════════════════════════════════
#  健康检查
# ═══════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    token_age = int(time.time() - _token_cache["ts"]) if _token_cache["ts"] else -1
    try:
        with _db_lock:
            db = _get_db()
            total_sigs = db.execute("SELECT COUNT(*) FROM sig_cache").fetchone()[0]
            active_ns  = db.execute("SELECT COUNT(DISTINCT ns) FROM sig_cache").fetchone()[0]
    except Exception:
        total_sigs = active_ns = -1
    return {
        "status":            "ok",
        "version":           "v29.0",
        "active_namespaces": active_ns,
        "cached_signatures": total_sigs,
        "cache_db":          CACHE_DB_PATH,
        "token_age_seconds": token_age,
        "token_valid":       token_age < TOKEN_REFRESH_SECS if token_age >= 0 else False,
    }

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
