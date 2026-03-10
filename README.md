# VertexAI-Proxy：OpenClaw 接入 Gemini & Claude 安装手册

> [OpenClaw 是什么？访问地址](https://github.com/openclaw/openclaw)

把 Google Vertex AI 上的 Gemini 模型以及 Anthropic Claude 模型统一封装成 **OpenAI 兼容接口**，供 OpenClaw 及其他客户端使用。

---

##  项目背景

直接调用 Gemini 公共 API 在生产环境中存在 IP 封禁、速率严格、区域限制等问题。**VertexAI-Proxy** 基于 Google Cloud 企业级 Vertex AI 平台，提供稳定、低限制的代理接入，并内置以下能力：

| 特性 | 说明 |
|------|------|
| **OpenAI 协议兼容** | 标准 `/v1/chat/completions` 接口，无需改造客户端 |
| **双后端路由** | `google/gemini-*` 路由到 Vertex AI，`anthropic/claude-*` 路由到 Anthropic API |
| **thought_signature 自动补全** | Gemini 思考模型要求签名字段，代理自动缓存并补回，工具调用链不断裂 |
| **SQLite 持久化缓存** | 签名写入本地数据库，代理重启后无需重新获取，长会话不中断 |
| **多用户隔离** | 按 Authorization header 哈希分桶，用户间签名不污染 |
| **gcloud Token 缓存** | 30 分钟内复用令牌，不因重复 fork 子进程拖慢请求 |

### 方案对比

| 对比维度 | Gemini 公共 API | VertexAI-Proxy (Vertex AI) |
|----------|-----------------|---------------------------|
| IP 限制 | 严格，易触发封禁 | 无，基于服务账号认证 |
| 稳定性 | 一般 | 高，企业级基础设施 |
| 区域可用性 | 受地区政策限制 | 全球可用 |
| 速率限制 | 较严格 | 更宽松，基于配额管理 |
| 工具调用（多轮） | 客户端需自行处理签名 | 代理自动处理，透明无感 |

---

## 准备工作

- Linux 服务器
- Python 3.8+
- Google Cloud 项目，已开启 Vertex AI API 权限

>  **免费额度**：Google Cloud 提供 $300 新用户免费额度，可用于测试。[如何领取](https://zhuanlan.zhihu.com/p/2000528085997605187)

---

## 1. 安装 Google Cloud SDK 并授权

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
gcloud config set project <YOUR_GCP_PROJECT_ID>
```

登录后生成 ADC（应用默认凭据），代理启动时自动读取。

---

## 2. 安装 VertexAI-Proxy

```bash
cd ~
git clone https://github.com/wismyzhizi2018/vertexai-proxy.git
cd vertexai-proxy/vertexai-proxy/
pip install -r requirements.txt
cp .env.example .env
vim .env
```

**.env 配置**：

```env
VERTEX_AI_PROJECT=<YOUR_GCP_PROJECT_ID>
VERTEX_AI_REGION=us-west1
PROXY_HOST=127.0.0.1   # 仅本机访问；多人内网部署改为 0.0.0.0
PROXY_PORT=8000
# 签名缓存数据库路径（目录需提前创建）
CACHE_DB_PATH=/var/lib/vertexai-proxy/sig_cache.db
# Anthropic Claude 支持（可选，不填则只能用 Gemini）
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

创建缓存目录：

```bash
mkdir -p /var/lib/vertexai-proxy
```

---

## 3. 启动并验证

```bash
python3 proxy.py
curl http://localhost:8000/health
```

正常返回示例：

```json
{
  "status": "ok",
  "version": "v28.0",
  "active_namespaces": 1,
  "cached_signatures": 12,
  "cache_db": "/var/lib/vertexai-proxy/sig_cache.db",
  "token_age_seconds": 42,
  "token_valid": true
}
```

### 接口测试

```bash
# 基础对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3.1-flash-lite-preview",
    "messages": [{"role": "user", "content": "你好，请确认你的模型版本。"}]
  }'

# 流式输出
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [{"role": "user", "content": "请用中文讲一个关于 AI 的短故事"}],
    "stream": true
  }'

# 多轮对话
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3.1-flash-lite-preview",
    "messages": [
      {"role": "user", "content": "法国首都是哪里？"},
      {"role": "assistant", "content": "法国首都是巴黎。"},
      {"role": "user", "content": "德国呢？"}
    ]
  }'
```

---

## 4. systemd 守护运行

创建 `/etc/systemd/system/vertexai-proxy.service`：

```ini
[Unit]
Description=Vertex AI Reasoning Proxy
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/vertexai-proxy/vertexai-proxy
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/google-cloud-sdk/bin"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json"
ExecStart=/usr/bin/python3 /root/vertexai-proxy/vertexai-proxy/proxy.py
Restart=always
RestartSec=5
SyslogIdentifier=vertexai-proxy

[Install]
WantedBy=multi-user.target
```

启用：

```bash
systemctl daemon-reload
systemctl enable vertexai-proxy
systemctl start vertexai-proxy
systemctl status vertexai-proxy
```

---

## 5. OpenClaw 注册模型

编辑 `~/.openclaw/openclaw.json`：

```json
{
  "providers": {
    "vertexai-proxy": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "apiKey": "your-api-key",
      "api": "openai-completions",
      "models": [
        {
          "id": "google/gemini-3.1-flash-lite-preview",
          "name": "Gemini 3.1 Flash Lite",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 1048576,
          "maxTokens": 65536
        },
        {
          "id": "google/gemini-2.5-flash-low",
          "name": "Gemini 2.5 Flash (Low)",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 1048576,
          "maxTokens": 65536
        },
        {
          "id": "anthropic/claude-sonnet-4-5",
          "name": "Claude Sonnet 4.5",
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 200000,
          "maxTokens": 8192
        },
        {
          "id": "anthropic/claude-opus-4-5",
          "name": "Claude Opus 4.5",
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 200000,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

验证：

```bash
openclaw models list --provider vertexai-proxy
openclaw chat -m vertexai-proxy/google/gemini-3.1-flash-lite-preview "你好"
```

---

## 6. 接入 Anthropic Claude（可选）

如需同时使用 Claude 模型，在 `.env` 里加入 API Key 即可，无需其他改动：

```bash
echo "ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx" >> .env
systemctl restart vertexai-proxy
```

获取 API Key：[console.anthropic.com](https://console.anthropic.com/)

**路由规则**：代理根据 model id 前缀自动分流，无需手动切换：

| model id 前缀 | 路由目标 |
|---------------|---------|
| `google/gemini-*` | Vertex AI |
| `anthropic/claude-*` | Anthropic API |
| `claude-*` | Anthropic API |

**OpenClaw 注册 Claude 模型**时，model id 填 `anthropic/claude-sonnet-4-5` 这样的格式，代理会自动去掉 `anthropic/` 前缀再发给 Anthropic。

---

## 7. 签名缓存管理

代理重启后签名从 SQLite 自动恢复，无需手动操作。如需查看缓存状态：

```bash
# 缓存概览（也可直接访问 /health 接口）
sqlite3 /var/lib/vertexai-proxy/sig_cache.db \
  "SELECT ns, COUNT(*) as cnt FROM sig_cache GROUP BY ns;"

# 最近写入的签名
sqlite3 /var/lib/vertexai-proxy/sig_cache.db \
  "SELECT tool_call_id, datetime(ts,'unixepoch','localtime') FROM sig_cache ORDER BY ts DESC LIMIT 10;"

# 手动清空缓存（一般不需要）
sqlite3 /var/lib/vertexai-proxy/sig_cache.db "DELETE FROM sig_cache;"
```

---

## 安全建议

默认监听 `127.0.0.1`，仅本机访问。

**多人内网部署**：将 `PROXY_HOST` 改为 `0.0.0.0`，每人配置独立 API Key（代理按 Key 哈希隔离用户数据）。

**如需公网访问**：
- 优先通过内网 / VPN
- 在 Nginx 前置反向代理，添加 Basic Auth 或 JWT 鉴权
- 限制来源 IP

---

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| `health` 接口不通 | 检查代理是否启动，端口与 host 是否和配置一致 |
| 鉴权错误（Gemini） | 确认执行过 `gcloud auth application-default login`，项目已启用 `aiplatform.googleapis.com` |
| 鉴权错误（Claude） | 确认 `.env` 里 `ANTHROPIC_API_KEY` 已填写且有效 |
| 工具调用后 400 错误 | 查看日志 `Chain` 行是否有 `sig=MISSING`，升级到最新版本 |
| 重启后工具调用失败 | 确认 `CACHE_DB_PATH` 目录存在且有写权限 |
| Vertex 429 配额耗尽 | 稍等 1 分钟后重试；长期可在 Google Cloud 控制台申请提高配额 |
| OpenClaw 看不到模型 | 检查 `providers.vertexai-proxy` 配置，运行 `openclaw models list --provider vertexai-proxy` |

---

## 变更记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-02-10 | v1.0 | 初版 |
| 2026-02-13 | v1.1 | 清理项目文件，开源到 GitHub |
| 2026-02-14 | v19–22 | 防幻觉、Content Flattener、连接池、流监控、强制工具汇报 |
| 2026-03-10 | v23–24 | 优化代理逻辑，支持 Gemini 3.1 Flash Lite |
| 2026-03-10 | v25 | 修复 thought_signature 缓存 key 设计，补全签名写回位置 |
| 2026-03-10 | v26–27 | 多用户 ns 隔离，gcloud token 缓存，日志优化 |
| 2026-03-10 | v28 | **SQLite 持久化**：签名跨重启保留，解决长会话 400 问题 |
| 2026-03-10 | v29 | **双后端路由**：新增 Anthropic Claude API 支持，model 前缀自动分流 |

---

## 许可证

MIT License

