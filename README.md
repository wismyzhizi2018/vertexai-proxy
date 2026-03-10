# Openclaw 通过 VertexAI-Proxy 接入 Gemini 安装与使用手册

> [Openclaw是什么?访问地址](https://github.com/openclaw/openclaw)

**Linux 服务器（示例路径 /root）**  
**Google Cloud 项目（写作 `<YOUR_GCP_PROJECT_ID>`）**

> 注意：本文默认绑定 `127.0.0.1:8000`，仅本机访问更安全。

---

## VertexAI-Proxy 接入 Gemini 2.5 和 Gemini 3

项目有 Vertex AI API 权限

### 📖 项目背景

随着 Google Gemini 系列模型的广泛应用，开发者普遍通过 Gemini CLI 或直接调用公共 API 来接入模型能力。然而，在生产环境及服务器端部署中，这类接入方式逐渐暴露出稳定性差、可用性受限等问题。

为解决上述痛点，**VertexAI-Proxy** 应运而生。它基于 Google Cloud 企业级 AI 平台 Vertex AI，为 Gemini 模型提供稳定、高效、低限制的代理接入方案，显著提升服务可靠性与部署灵活性。

**核心特性：**
- **SQLite 持久化**：内置 SQLite 数据库，用于持久化存储签名与状态，彻底解决代理重启后签名丢失的问题，保障服务连续性。

### 📊 方案对比

| 对比维度 | Gemini CLI / 公共 API | VertexAI-Proxy (Vertex AI) |
|-----------|------------------------|---------------------------|
| **IP 限制** | 严格，易触发封禁 | 无限制，基于服务账号认证 |
| **稳定性** | 一般，受公共资源影响 | 高，企业级基础设施 |
| **可用性** | 受地区政策限制 | 全球可用，基于 Google Cloud |
| **速率限制** | 较严格 | 更宽松，基于配额管理 |
| **部署方式** | 需要处理 API Key 安全 | 服务账号认证，更安全 |
| **扩展性** | 有限 | 优秀，支持企业级扩展 |

### 准备工作

**目标**：在 Linux 服务器上部署 vertexai-proxy，把 Google 的 Gemini 2.5 Flash 封装成 OpenAI 兼容接口，供 OpenClaw 使用。

**Python 3**

---

## 1. 安装 Google Cloud SDK 并登录

> 💡 **免费额度提示**：Google Cloud 提供 300 美元的新用户免费额度，可用于测试 Vertex AI API。
> 📖 [如何获取 Google Cloud 300 美元免费额度](https://zhuanlan.zhihu.com/p/2000528085997605187)

执行：

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
gcloud config set project <YOUR_GCP_PROJECT_ID>
```

**说明**：登录后生成 ADC（应用默认凭据），供后续代理调用。

---

## 2. 安装 vertexai-proxy

```bash
cd ~
mkdir -p vertexai-proxy
cd vertexai-proxy/
git clone https://github.com/wismyzhizi2018/vertexai-proxy.git
cd vertexai-proxy/
pip install -r requirements.txt
cp .env.example .env
vim .env
```

**.env 示例**：

```env
VERTEX_AI_PROJECT=<YOUR_GCP_PROJECT_ID>
VERTEX_AI_REGION=us-west1
PROXY_HOST=127.0.0.1
PROXY_PORT=8000
```

---

## 3. 启动并验证

```bash
python3 proxy.py
curl http://localhost:8000/health
```

**测试 OpenAI 兼容接口**：

### 基础测试（非流式）

```bash
# Gemini 3.1 Flash Lite - 推荐模型测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3.1-flash-lite-preview",
    "messages": [{"role": "user", "content": "你好，请确认你的模型版本。"}]
  }'
```

### 流式输出测试

```bash
# Gemini 2.5 Flash - 流式输出
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [{"role": "user", "content": "Tell me a short story about a robot."}],
    "stream": true
  }'

# Gemini 3 Flash - 流式输出（中文）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "请用中文讲一个简短的关于AI的故事"}],
    "stream": true
  }'
```

### 系统消息测试

```bash
# 带系统消息的测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant specialized in mathematics."},
      {"role": "user", "content": "What is the derivative of x²?"}
    ]
  }'
```

### 多轮对话测试

```bash
# 多轮对话（带历史消息）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-flash-preview",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."},
      {"role": "user", "content": "What about Germany?"}
    ]
  }'
```

### 参数测试

```bash
# 温度和最大 token 测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [{"role": "user", "content": "Write a creative poem."}],
    "temperature": 0.8,
    "max_tokens": 100
  }'

# 低温度测试（更确定性）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "What is 10 * 10?"}],
    "temperature": 0.1,
    "max_tokens": 50
  }'
```

### 高级测试

```bash
# JSON 格式输出
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [{"role": "user", "content": "Return the answer in JSON format: What are the primary colors?"}]
  }'

# 代码生成测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "Write a Python function to calculate factorial."}]
  }'

# 长文本处理
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [{"role": "user", "content": "Summarize the following text in 3 bullet points: The history of artificial intelligence dates back to antiquity. Philosophers described the process of human thinking as the symbolic manipulation of the mind. In the 1940s, the invention of the programmable digital computer gave birth to the field of AI. Over the decades, AI has evolved from simple rule-based systems to complex machine learning models that can perform tasks ranging from image recognition to natural language processing."}]
  }'
```

---

## 4. OpenClaw 注册模型

**检查 provider 基本配置**（`~/.openclaw/openclaw.json`）：

```json
{
  "providers": {
    "vertexai-proxy": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "apiKey": "dummy-key-not-used",
      "api": "openai-completions",
      "models": [
        {
          "id": "google/gemini-2.5-flash-low",
          "name": "Gemini 2.5 Flash (Low)",
          "reasoning": true,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 1048576,
          "maxTokens": 65536
        },
        {
          "id": "google/gemini-3-flash-preview",
          "name": "Gemini 3 Flash Preview (vertexai-proxy)",
          "reasoning": true,
          "input": ["text"],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 1048576,
          "maxTokens": 65536
        }
      ]
    }
  }
}
```

**验证**：

```bash
openclaw models list --provider vertexai-proxy
```

---

## 5. systemd 守护运行

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
curl http://localhost:8000/health
```

---

## 6. 使用示例

```bash
openclaw chat -m vertexai-proxy/google/gemini-2.5-flash-high "你好，请确认你的模型版本。"

openclaw run --model vertexai-proxy/google/gemini-2.5-flash-high --prompt "你好"
```

---

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| **health 不通** | 检查代理是否启动、端口和 host 是否一致。 |
| **鉴权错误** | 确认做过 `gcloud auth application-default login`，项目已启用 `aiplatform.googleapis.com`。 |
| **OpenClaw 看不到模型** | 检查 `providers.vertexai-proxy` 配置是否完整，运行 `openclaw models list --provider vertexai-proxy`。 |

---

## 安全建议

默认监听 `127.0.0.1`，仅本机访问更安全。

**如需外网访问**：
- 优先内网/VPN
- 加反向代理鉴权（Nginx + Basic Auth / JWT）
- 限制来源 IP
- 监控请求量和错误率

---

## 变更记录

| 日期 | 版本 | 作者 | 说明 |
|------|------|------|------|
| 2026-02-10 | v1.0 | jack-bot | 初版 |
| 2026-02-13 | v1.1 | jack-bot | 清理项目文件，开源到 GitHub |
| 2026-02-13 | v1.2 | jack-bot | 添加项目背景介绍和方案对比表格 |
| 2026-02-14 | v19.2 | jack-bot | 修复 SyntaxError，增加 Role Flip (防幻觉) 与 Content Flattener (防400错误)；增强系统指令。 |
| 2026-02-14 | v20.0 | jack-bot | **Production Ready**: 引入全局 HTTP 连接池提升并发；增加日志截断功能防止 Token 溢出；优化错误处理。 |
| 2026-02-14 | v21.0 | jack-bot | **Live Stream Monitor**: 新增实时流监控 (首包延迟 TTFT + 传输速率 + 流量统计)；优化连接池配置。 |
| 2026-02-14 | v22.0 | jack-bot | **Active Reporter**: 修复"沉默代理"问题；增强 System Prompt 强制模型在工具执行后汇报结果；优化日志截断提示。 |
| 2026-03-10 | v23.0 | 用户 | 最新版本更新：优化代理逻辑与稳定性。 |
| 2026-03-10 | v24.0 | 用户 | 增加对 Google Gemini 3.1 Flash Lite 模型的全面支持与测试示例。 |

---

## 许可证

MIT License
