#!/usr/bin/env python3
"""Vertex AI Proxy – v29.1"""

import os, json, subprocess, time, hashlib, sqlite3, threading
from typing import Dict, Any, Optional, Tuple, List
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import httpx, uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════
VERTEX_AI_PROJECT  = os.getenv("VERTEX_AI_PROJECT")
VERTEX_AI_REGION   = os.getenv("VERTEX_AI_REGION", "us-west1")
PROXY_HOST         = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT         = int(os.getenv("PROXY_PORT", "8000"))
MAX_TOOL_CONTENT   = 30_000
CACHE_TTL_SECONDS  = 86400
CACHE_MAX_ENTRIES  = 2000
TOKEN_REFRESH_SECS = 1800
CACHE_DB_PATH      = os.getenv("CACHE_DB_PATH", "/var/lib/vertexai-proxy/sig_cache.db")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL  = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION  = "2023-06-01"
REASONING_LEVELS   = {"none": "minimal", "low": "low", "medium": "medium", "high": "high"}
AGENT_SYSTEM_INSTRUCTION = (
    "[AGENT INSTRUCTIONS]\n"
    "1. You are an autonomous Agent with tool-use capability.\n"
    "2. After receiving a tool result, you MUST reply to the user immediately.\n"
    "3. On success: briefly state what was accomplished.\n"
    "4. On error: briefly explain the error and suggest next steps.\n"
    "5. Never stay silent after a tool execution."
)
_CLAUDE4_ADAPTIVE = {"claude-opus-4-6", "claude-sonnet-4-6"}
_CLAUDE4_BUDGET   = {"claude-opus-4-5", "claude-sonnet-4-5"}

# ═══════════════════════════════════════════════════════════════
#  HTTP 客户端 + 生命周期
# ═══════════════════════════════════════════════════════════════
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    _get_db()
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=60.0),
        limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=120.0),
        http2=False, verify=True,
    )
    print("[SYSTEM] HTTP Client Initialized.")
    yield
    if http_client: await http_client.aclose()
    if _db_conn:    _db_conn.close()

app = FastAPI(title="Vertex AI Proxy v29.1", lifespan=lifespan)

# ═══════════════════════════════════════════════════════════════
#  gcloud token 缓存
# ═══════════════════════════════════════════════════════════════
_token_cache: Dict[str, Any] = {"token": "", "ts": 0.0}

def get_vertex_token() -> str:
    if os.getenv("VERTEX_ACCESS_TOKEN"):
        return os.getenv("VERTEX_ACCESS_TOKEN")
    now = time.time()
    if _token_cache["token"] and now - _token_cache["ts"] < TOKEN_REFRESH_SECS:
        return _token_cache["token"]
    try:
        env = {**os.environ}; env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        r = subprocess.run(["gcloud","auth","application-default","print-access-token"],
                           capture_output=True, text=True, check=True, env=env)
        _token_cache.update(token=r.stdout.strip(), ts=now)
        return _token_cache["token"]
    except Exception as e:
        print(f"[ERROR] Token refresh failed: {e}")
        return _token_cache["token"]

# ═══════════════════════════════════════════════════════════════
#  SQLite 签名缓存
# ═══════════════════════════════════════════════════════════════
_db_lock  = threading.Lock()
_db_conn: Optional[sqlite3.Connection] = None
_write_count = 0

def _get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn: return _db_conn
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    c = sqlite3.connect(CACHE_DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=NORMAL")
    c.execute("""CREATE TABLE IF NOT EXISTS sig_cache(
        ns TEXT NOT NULL, tool_call_id TEXT NOT NULL, sig TEXT NOT NULL, ts REAL NOT NULL,
        PRIMARY KEY(ns, tool_call_id))""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON sig_cache(ts)")
    c.commit()
    total = c.execute("SELECT COUNT(*) FROM sig_cache").fetchone()[0]
    print(f"[SYSTEM] SQLite loaded: {total} sigs from {CACHE_DB_PATH}")
    _db_conn = c; return c

def _user_ns(auth: str) -> str:
    return hashlib.sha256(auth.encode()).hexdigest()[:16]

def sig_cache_put(ns: str, tc_id: str, sig: str) -> None:
    global _write_count
    now = time.time()
    def _write():
        global _write_count
        with _db_lock:
            db = _get_db()
            db.execute("INSERT OR REPLACE INTO sig_cache VALUES(?,?,?,?)", (ns, tc_id, sig, now))
            _write_count += 1
            if _write_count % 100 == 0:
                db.execute("DELETE FROM sig_cache WHERE ts<?", (now - CACHE_TTL_SECONDS,))
                db.execute("""DELETE FROM sig_cache WHERE rowid IN(
                    SELECT rowid FROM sig_cache s1 WHERE
                    (SELECT COUNT(*) FROM sig_cache s2 WHERE s2.ns=s1.ns AND s2.ts>=s1.ts)>?)""",
                    (CACHE_MAX_ENTRIES,))
            db.commit()
    threading.Thread(target=_write, daemon=True).start()

def sig_cache_get(ns: str, tc_id: str) -> Optional[str]:
    with _db_lock:
        row = _get_db().execute(
            "SELECT sig FROM sig_cache WHERE ns=? AND tool_call_id=? AND ts>?",
            (ns, tc_id, time.time() - CACHE_TTL_SECONDS)).fetchone()
    return row[0] if row else None

def sig_cache_latest(ns: Optional[str] = None) -> Optional[str]:
    """取最近一条签名，ns=None 时全局查找"""
    cutoff = time.time() - CACHE_TTL_SECONDS
    with _db_lock:
        sql = ("SELECT sig FROM sig_cache WHERE ns=? AND ts>? ORDER BY ts DESC LIMIT 1"
               if ns else "SELECT sig FROM sig_cache WHERE ts>? ORDER BY ts DESC LIMIT 1")
        row = _get_db().execute(sql, (ns, cutoff) if ns else (cutoff,)).fetchone()
    return row[0] if row else None

# ═══════════════════════════════════════════════════════════════
#  thought_signature 读写辅助
# ═══════════════════════════════════════════════════════════════
def _get_sig(tc: Dict) -> Optional[str]:
    g = (tc.get("extra_content") or {}).get("google") or {}
    return g.get("thought_signature") or tc.get("thought_signature")

def _set_sig(tc: Dict, sig: str) -> Dict:
    tc = {k: v for k, v in tc.items() if k != "thought_signature"}
    eg = tc.get("extra_content") or {}
    tc["extra_content"] = {**eg, "google": {**(eg.get("google") or {}), "thought_signature": sig}}
    return tc

# ═══════════════════════════════════════════════════════════════
#  消息工具函数
# ═══════════════════════════════════════════════════════════════
def _flatten(content: Any) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "\n".join(str(i.get("text","")) if isinstance(i,dict) else str(i) for i in content)
    return str(content or "")

def _normalize_user_content(content: Any) -> Any:
    """保留多模态 block 列表，纯文本时 flatten"""
    if isinstance(content, list):
        return [b if isinstance(b,dict) else {"type":"text","text":str(b)} for b in content]
    return _flatten(content)

def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_CONTENT: return text
    h = MAX_TOOL_CONTENT // 2
    return text[:h] + f"\n\n[...Truncated {len(text)-MAX_TOOL_CONTENT} chars...]\n\n" + text[-h:]

def _chain_summary(messages: List[Dict]) -> List[str]:
    out = []
    for m in messages:
        tcs = m.get("tool_calls") or []
        if tcs:
            miss = sum(1 for tc in tcs if not _get_sig(tc))
            out.append(f"{m.get('role','?')}[{'sig=OK' if miss==0 else f'sig=MISSING{miss}/{len(tcs)}'}]")
        else:
            out.append(m.get("role","?"))
    return out

# ═══════════════════════════════════════════════════════════════
#  签名恢复（Vertex 路径）
# ═══════════════════════════════════════════════════════════════
def _restore_tool_calls(tcs: List[Dict], req_id: str, ns: str) -> List[Dict]:
    # 第1步：按 id 从缓存恢复
    restored = []
    for tc in tcs:
        if not _get_sig(tc):
            sig = sig_cache_get(ns, tc.get("id",""))
            if sig:
                tc = _set_sig(tc, sig)
                print(f"[{req_id}] [SIG] ✓ Restored id={tc.get('id')}")
            else:
                print(f"[{req_id}] [SIG] ✗ Miss id={tc.get('id')}")
        restored.append(tc)
    # 第2步：批内广播
    any_sig = next((_get_sig(tc) for tc in restored if _get_sig(tc)), None)
    if any_sig:
        restored = [(_set_sig(tc,any_sig) if not _get_sig(tc) else tc) for tc in restored]
    return restored

def sanitize_and_restore(body: Dict[str,Any], req_id: str, ns: str) -> Dict[str,Any]:
    if "messages" not in body: return body
    cleaned, found_sys, injected = [], False, False

    for msg in body["messages"]:
        role = msg.get("role","")
        raw  = msg.get("content")

        if role in ("system","developer"):
            found_sys = True
            text = _flatten(raw)
            if not injected:
                text = text.rstrip() + "\n\n" + AGENT_SYSTEM_INSTRUCTION
                injected = True
            cleaned.append({"role":"system","content":text})

        elif role == "assistant":
            m: Dict[str,Any] = {"role":"model","content":_flatten(raw)}
            tcs = msg.get("tool_calls")
            if tcs: m["tool_calls"] = _restore_tool_calls(tcs, req_id, ns)
            cleaned.append(m)

        elif role == "tool":
            m = {"role":"tool","content":_truncate(_flatten(raw))}
            if "tool_call_id" in msg: m["tool_call_id"] = msg["tool_call_id"]
            if "name"         in msg: m["name"]         = msg["name"]
            cleaned.append(m)

        elif role == "user":
            cleaned.append({"role":"user","content":_normalize_user_content(raw)})

        else:
            cleaned.append({**msg,"content":_flatten(raw)})

    if not found_sys:
        cleaned.insert(0,{"role":"system","content":AGENT_SYSTEM_INSTRUCTION})

    body["messages"] = cleaned
    _broadcast_signatures(cleaned, req_id, ns)
    return body

def _broadcast_signatures(cleaned: List[Dict], req_id: str, ns: str) -> None:
    """全局广播：链内找 → 自己DB → 全局DB"""
    sig = None
    for msg in cleaned:
        sig = next((_get_sig(tc) for tc in (msg.get("tool_calls") or []) if _get_sig(tc)), None)
        if sig: break
    if not sig:
        sig = sig_cache_latest(ns)
        if sig: print(f"[{req_id}] [SIG] ↗ Fallback: own DB ns={ns}")
    if not sig:
        sig = sig_cache_latest()
        if sig: print(f"[{req_id}] [SIG] ↗ Fallback: global DB")
    if sig:
        filled = 0
        for msg in cleaned:
            if msg.get("role") != "model" or not msg.get("tool_calls"): continue
            new_tcs = []
            for tc in msg["tool_calls"]:
                if not _get_sig(tc): tc = _set_sig(tc, sig); filled += 1
                new_tcs.append(tc)
            msg["tool_calls"] = new_tcs
        if filled: print(f"[{req_id}] [SIG] ↗ Global broadcast: filled {filled} orphan signature(s)")

# ═══════════════════════════════════════════════════════════════
#  Vertex 辅助
# ═══════════════════════════════════════════════════════════════
def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    for s in REASONING_LEVELS:
        if model_id.endswith(f"-{s}"):
            return model_id.rsplit(f"-{s}",1)[0], s
    return model_id, None

def get_endpoint_url(base_model: str) -> str:
    if "gemini-3" in base_model:
        return f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}/locations/global/endpoints/openapi"
    return f"https://{VERTEX_AI_REGION}-aiplatform.googleapis.com/v1/projects/{VERTEX_AI_PROJECT}/locations/{VERTEX_AI_REGION}/endpoints/openapi"

# ═══════════════════════════════════════════════════════════════
#  流式透传 + 实时签名缓存（race condition 修复）
# ═══════════════════════════════════════════════════════════════
async def stream_and_cache(resp: httpx.Response, req_id: str, start: float, ns: str):
    count = total = 0
    first = False
    live_sigs: Dict[str,str]  = {}
    live_ids:  List[str]      = []
    live_map:  Dict[int,Dict] = {}   # index → {id, sig}

    try:
        async for chunk in resp.aiter_bytes():
            if not first:
                print(f"[{req_id}] [TTFT] {int((time.time()-start)*1000)}ms"); first = True
            yield chunk
            total += len(chunk); count += 1
            if count % 100 == 0:
                print(f"[{req_id}] [STREAM] {count} chunks / {total/1024:.1f} KB")
            # 实时解析签名，立即写 DB
            for line in chunk.decode("utf-8","replace").splitlines():
                if not line.startswith("data:"): continue
                ds = line[5:].strip()
                if not ds or ds == "[DONE]": continue
                try: data = json.loads(ds)
                except: continue
                for tcd in (((data.get("choices") or [{}])[0]).get("delta") or {}).get("tool_calls") or []:
                    idx = tcd.get("index",0)
                    if idx not in live_map: live_map[idx] = {"id":"","sig":""}
                    e = live_map[idx]
                    if tcd.get("id") and not e["id"]:
                        e["id"] = tcd["id"]
                        if e["id"] not in live_ids: live_ids.append(e["id"])
                    g   = (tcd.get("extra_content") or {}).get("google") or {}
                    sig = (g.get("thought_signature")
                           or tcd.get("thought_signature")
                           or (tcd.get("function") or {}).get("thought_signature") or "")
                    if sig:
                        e["sig"] += sig
                        if e["id"] and e["id"] not in live_sigs:
                            live_sigs[e["id"]] = e["sig"]
                            sig_cache_put(ns, e["id"], e["sig"])  # 立即写

    except Exception as ex:
        print(f"[{req_id}] [ERROR] Stream: {ex}"); raise
    finally:
        await resp.aclose()
        print(f"[{req_id}] [DONE] {total/1024:.1f} KB in {time.time()-start:.2f}s")
        # finally：用完整签名覆盖写（修复分片残缺）
        for e in live_map.values():
            if e["id"] and e["sig"]:
                live_sigs[e["id"]] = e["sig"]
                sig_cache_put(ns, e["id"], e["sig"])
        # 广播给无签名的同批 id
        if live_sigs and live_ids:
            any_sig = next(iter(live_sigs.values()))
            for tc_id in live_ids:
                if tc_id not in live_sigs:
                    sig_cache_put(ns, tc_id, any_sig)
                    print(f"[{req_id}] [SIG] ↗ Broadcast to id={tc_id}")
        if live_sigs: print(f"[{req_id}] [SIG] Cached {len(live_sigs)} signature(s)")

# ═══════════════════════════════════════════════════════════════
#  主路由
# ═══════════════════════════════════════════════════════════════
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_id = str(int(time.time()*1000))[-6:]
    print(f"[{req_id}] ← Request")
    auth = request.headers.get("Authorization","")
    ns   = _user_ns(auth) if auth else "anonymous"
    try:
        raw  = await request.json()
        mid  = raw.get("model","")
        if mid.startswith("anthropic/") or mid.startswith("claude-"):
            return await _forward_anthropic(raw, req_id)

        body = sanitize_and_restore(raw, req_id, ns)
        if "tools" in body and "tool_choice" not in body:
            body["tool_choice"] = "auto"
        base, _ = parse_model_id(body.get("model",""))
        vbody   = {**body, "model": base}
        vbody.pop("reasoning_effort", None)

        print(f"[{req_id}] Chain({len(vbody.get('messages',[]))}) ns={ns}: {_chain_summary(vbody.get('messages',[]))}")

        token = get_vertex_token()
        if not token: raise HTTPException(500, "Failed to obtain Vertex AI access token.")

        stream  = bool(vbody.get("stream", False))
        req_obj = http_client.build_request(
            "POST", get_endpoint_url(base) + "/chat/completions", json=vbody,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        print(f"[{req_id}] → Vertex ({base})")
        start = time.time()
        resp  = await http_client.send(req_obj, stream=stream)

        if resp.status_code >= 400:
            err = await resp.aread()
            print(f"[{req_id}] [ERROR] Vertex {resp.status_code}: {err[:500]}")
            return Response(content=err, status_code=resp.status_code, media_type="application/json")

        if not stream:
            data = await resp.aread(); await resp.aclose()
            print(f"[{req_id}] [DONE] Vertex non-stream in {time.time()-start:.2f}s")
            return Response(content=data, status_code=resp.status_code, media_type="application/json")

        return StreamingResponse(stream_and_cache(resp, req_id, start, ns),
                                 status_code=resp.status_code, media_type="text/event-stream")
    except HTTPException: raise
    except Exception as e:
        import traceback; traceback.print_exc(); raise HTTPException(500, str(e))

# ═══════════════════════════════════════════════════════════════
#  Anthropic：消息转换
# ═══════════════════════════════════════════════════════════════
def _convert_messages(messages: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """返回 (system_blocks, anthropic_msgs)"""
    system_blocks, filtered = [], []
    for m in messages:
        if m.get("role") in ("system","developer"):
            t = m.get("content","")
            if isinstance(t, list): t = " ".join(b.get("text","") for b in t if b.get("type")=="text")
            if t: system_blocks.append(t)
        else:
            filtered.append(m)

    # 合并相邻同 role
    merged = []
    for m in filtered:
        role, content = m.get("role"), m.get("content","")
        tcs, tcid = m.get("tool_calls"), m.get("tool_call_id")
        if merged and merged[-1]["_role"]==role and role in("user","assistant") and not tcid and not tcs:
            p = merged[-1]
            if isinstance(p["_content"], list): p["_content"].append({"type":"text","text":content or ""})
            else: p["_content"] = (p["_content"] or "") + ("\n"+(content or "") if content else "")
        else:
            merged.append({"_role":role,"_content":content,"_tcs":tcs,"_tcid":tcid,"_name":m.get("name")})

    # 转换格式
    out = []
    for m in merged:
        role, content, tcs, tcid = m["_role"], m["_content"], m["_tcs"], m["_tcid"]
        if role == "assistant":
            blocks = []
            if isinstance(content, list):
                blocks += [{"type":"text","text":b["text"]} for b in content if b.get("type")=="text" and b.get("text")]
            elif content:
                blocks.append({"type":"text","text":content})
            for tc in (tcs or []):
                fn = tc.get("function",{})
                args = fn.get("arguments","{}")
                try: args = json.loads(args) if isinstance(args,str) else args
                except: args = {}
                blocks.append({"type":"tool_use","id":tc.get("id",""),"name":fn.get("name",""),"input":args})
            if not blocks: blocks.append({"type":"text","text":""})
            out.append({"role":"assistant","content":blocks})

        elif role == "tool":
            tr = {"type":"tool_result","tool_use_id":tcid or "","content":content or ""}
            if (out and out[-1]["role"]=="user" and isinstance(out[-1]["content"],list)
                    and any(b.get("type")=="tool_result" for b in out[-1]["content"])):
                out[-1]["content"].append(tr)
            else:
                out.append({"role":"user","content":[tr]})

        else:  # user
            if isinstance(content, list):
                blocks = []
                for b in content:
                    if b.get("type") == "text":
                        blocks.append({"type":"text","text":b.get("text","")})
                    elif b.get("type") == "image_url":
                        url = b.get("image_url",{}).get("url","")
                        if url.startswith("data:"):
                            try:
                                hdr, dat = url.split(",",1)
                                blocks.append({"type":"image","source":{
                                    "type":"base64","media_type":hdr.split(":")[1].split(";")[0],"data":dat}})
                            except: pass
                        else:
                            blocks.append({"type":"image","source":{"type":"url","url":url}})
                    else:
                        blocks.append(b)
                out.append({"role":role,"content":blocks})
            else:
                out.append({"role":role,"content":content or ""})

    if out and out[0]["role"] != "user":
        out.insert(0,{"role":"user","content":"."})
    return system_blocks, out

def _build_anthropic_body(body: dict, model_id: str, msgs: List[Dict],
                           system_blocks: List[str], tools: List[Dict]) -> Tuple[Dict, bool]:
    """构建请求体，返回 (anthropic_body, thinking_enabled)"""
    tc_raw = body.get("tool_choice")
    tool_choice = None
    if tc_raw == "auto":      tool_choice = {"type":"auto"}
    elif tc_raw == "required": tool_choice = {"type":"any"}
    elif isinstance(tc_raw, dict) and tc_raw.get("type")=="function":
        tool_choice = {"type":"tool","name":tc_raw["function"]["name"]}
    # none → 不传 tools

    ab: Dict[str,Any] = {"model":model_id,"max_tokens":body.get("max_tokens",8192),
                          "messages":msgs,"stream":body.get("stream",False)}
    if system_blocks: ab["system"] = "\n".join(system_blocks).strip()
    if tools and tc_raw != "none":
        ab["tools"] = tools
        if tool_choice: ab["tool_choice"] = tool_choice

    thinking_enabled = False
    effort = body.get("reasoning_effort")
    if model_id in _CLAUDE4_ADAPTIVE and effort and effort != "none":
        ab["thinking"] = {"type":"adaptive"}
        ab["effort"]   = {"low":"low","medium":"medium","high":"high","max":"max"}.get(effort,"high")
        if ab["max_tokens"] < 16000: ab["max_tokens"] = 16000
        thinking_enabled = True
    elif model_id in _CLAUDE4_BUDGET and effort and effort != "none":
        budget = {"low":2000,"medium":8000,"high":16000}.get(effort,8000)
        ab["thinking"] = {"type":"enabled","budget_tokens":budget}
        if ab["max_tokens"] <= budget: ab["max_tokens"] = budget + 4096
        thinking_enabled = True

    if not thinking_enabled:
        if body.get("temperature") is not None: ab["temperature"] = body["temperature"]
        if body.get("top_p")       is not None: ab["top_p"]       = body["top_p"]
    if body.get("stop") is not None:
        ab["stop_sequences"] = [body["stop"]] if isinstance(body["stop"],str) else body["stop"]
    if body.get("speed") == "fast" and model_id == "claude-opus-4-6":
        ab["speed"] = "fast"
    return ab, thinking_enabled

def _build_anthropic_headers(model_id: str, thinking_enabled: bool, body: dict) -> Dict[str,str]:
    betas = []
    if model_id in _CLAUDE4_ADAPTIVE: betas.append("output-128k-2025-02-19")
    if model_id == "claude-sonnet-4-6" and thinking_enabled:
        betas.append("interleaved-thinking-2025-05-14")
    if body.get("max_context")=="1m" or body.get("contextWindow",0)>=500_000 or body.get("enable_1m_context"):
        betas.append("context-1m-2025-08-07")
    h = {"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":ANTHROPIC_VERSION,"Content-Type":"application/json"}
    if betas: h["anthropic-beta"] = ",".join(betas)
    return h

# ═══════════════════════════════════════════════════════════════
#  Anthropic：SSE 流转换
# ═══════════════════════════════════════════════════════════════
def _map_stop_reason(r: Optional[str]) -> str:
    return {"tool_use":"tool_calls","max_tokens":"length",
            "model_context_window_exceeded":"length"}.get(r or "", "stop")

async def _stream_anthropic(resp: httpx.Response, req_id: str, start: float, model_id: str):
    buffer = ""; msg_id = ""; created = int(time.time()); role_sent = False
    try:
        async for chunk in resp.aiter_bytes():
            buffer += chunk.decode("utf-8","replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n",1)
                line = line.strip()
                if not line.startswith("data:"): continue
                raw = line[5:].strip()
                if not raw: continue
                try:
                    ev = json.loads(raw); et = ev.get("type","")

                    if et == "message_start":
                        msg_id  = ev.get("message",{}).get("id","")
                        created = int(time.time())
                        yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                    "model":model_id,"choices":[{"delta":{"role":"assistant"},"index":0}]})
                        role_sent = True

                    elif et == "content_block_start":
                        b = ev.get("content_block",{})
                        if b.get("type") == "tool_use":
                            if not role_sent:
                                yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                            "model":model_id,"choices":[{"delta":{"role":"assistant"},"index":0}]})
                                role_sent = True
                            yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                        "model":model_id,"choices":[{"delta":{"tool_calls":[{
                                            "index":ev.get("index",0),"id":b.get("id",""),"type":"function",
                                            "function":{"name":b.get("name",""),"arguments":""}}]},
                                            "index":0}]})

                    elif et == "content_block_delta":
                        d = ev.get("delta",{})
                        if d.get("type") == "thinking_delta":
                            pass  # 过滤思考内容
                        elif d.get("type") == "text_delta":
                            if not role_sent:
                                yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                            "model":model_id,"choices":[{"delta":{"role":"assistant"},"index":0}]})
                                role_sent = True
                            yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                        "model":model_id,"choices":[{"delta":{"content":d.get("text","")},"index":0}]})
                        elif d.get("type") == "input_json_delta":
                            yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                        "model":model_id,"choices":[{"delta":{"tool_calls":[{
                                            "index":ev.get("index",0),
                                            "function":{"arguments":d.get("partial_json","")}}]},
                                            "index":0}]})

                    elif et == "message_delta":
                        finish = _map_stop_reason(ev.get("delta",{}).get("stop_reason"))
                        yield _sse({"id":msg_id,"object":"chat.completion.chunk","created":created,
                                    "model":model_id,"choices":[{"delta":{},"index":0,"finish_reason":finish}]})

                    elif et == "message_stop":
                        yield b"data: [DONE]\n\n"

                except Exception: pass
    finally:
        # 处理末尾残留
        if buffer.strip().startswith("data:"):
            try:
                ev = json.loads(buffer.strip()[5:].strip())
                if ev.get("type") == "message_stop": yield b"data: [DONE]\n\n"
            except: pass
        await resp.aclose()
        print(f"[{req_id}] [DONE] Anthropic stream in {time.time()-start:.2f}s")

def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()

# ═══════════════════════════════════════════════════════════════
#  Anthropic：主入口
# ═══════════════════════════════════════════════════════════════
async def _forward_anthropic(body: dict, req_id: str):
    if not ANTHROPIC_API_KEY: raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    model_id = body.get("model","").replace("anthropic/","")
    stream   = body.get("stream", False)

    system_blocks, msgs = _convert_messages(body.get("messages",[]))
    tools = [{"name":t["function"].get("name"),"description":t["function"].get("description",""),
              "input_schema":t["function"].get("parameters",{"type":"object","properties":{}})}
             for t in (body.get("tools") or []) if "function" in t]

    ab, thinking = _build_anthropic_body(body, model_id, msgs, system_blocks, tools)
    headers      = _build_anthropic_headers(model_id, thinking, body)

    print(f"[{req_id}] → Anthropic ({model_id}) msgs={len(msgs)} stream={stream} thinking={thinking}")
    start = time.time()

    req_obj = http_client.build_request("POST", ANTHROPIC_API_URL, json=ab, headers=headers)
    resp    = await http_client.send(req_obj, stream=True)

    if resp.status_code >= 400:
        err = await resp.aread()
        print(f"[{req_id}] [ERROR] Anthropic {resp.status_code}: {err[:500]}")
        return Response(content=err, status_code=resp.status_code, media_type="application/json")

    if not stream:
        data = await resp.aread(); await resp.aclose()  # Bug1: 释放连接
        try:
            ar = json.loads(data)
            blocks = ar.get("content",[])
            text   = "\n".join(b.get("text","") for b in blocks if b.get("type")=="text" and b.get("text"))
            tcs    = [{"id":b["id"],"type":"function","function":{
                           "name":b["name"],"arguments":json.dumps(b.get("input",{}),ensure_ascii=False)}}
                      for b in blocks if b.get("type")=="tool_use"]
            return Response(content=json.dumps({
                "id":ar.get("id",""),"object":"chat.completion","created":int(time.time()),
                "model":model_id,
                "choices":[{"index":0,"message":{"role":"assistant",
                    "content":text or None,"tool_calls":tcs or None},
                    "finish_reason":_map_stop_reason(ar.get("stop_reason"))}],
                "usage":{"prompt_tokens":ar.get("usage",{}).get("input_tokens",0),
                         "completion_tokens":ar.get("usage",{}).get("output_tokens",0),
                         "total_tokens":ar.get("usage",{}).get("input_tokens",0)
                                        +ar.get("usage",{}).get("output_tokens",0)}
            }, ensure_ascii=False), media_type="application/json")
        except Exception as e:
            print(f"[{req_id}] [ERROR] Anthropic parse: {e}")
            return Response(content=data, status_code=200, media_type="application/json")

    return StreamingResponse(_stream_anthropic(resp, req_id, start, model_id),
                             status_code=200, media_type="text/event-stream")

# ═══════════════════════════════════════════════════════════════
#  健康检查
# ═══════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    token_age = int(time.time()-_token_cache["ts"]) if _token_cache["ts"] else -1
    try:
        with _db_lock:
            db = _get_db()
            total_sigs = db.execute("SELECT COUNT(*) FROM sig_cache").fetchone()[0]
            active_ns  = db.execute("SELECT COUNT(DISTINCT ns) FROM sig_cache").fetchone()[0]
    except: total_sigs = active_ns = -1
    return {"status":"ok","version":"v29.1","active_namespaces":active_ns,
            "cached_signatures":total_sigs,"cache_db":CACHE_DB_PATH,
            "token_age_seconds":token_age,
            "token_valid": token_age < TOKEN_REFRESH_SECS if token_age >= 0 else False}

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")
