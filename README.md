# VertexAI-Proxy 接入 Gemini 安装与使用手册

**Linux 服务器（示例路径 /root）**  
**Google Cloud 项目（写作 `<YOUR_GCP_PROJECT_ID>`）**

> 注意：本文默认绑定 `127.0.0.1:8000`，仅本机访问更安全。

---

## VertexAI-Proxy 接入 Gemini 2.5 和 Gemini 3

项目有 Vertex AI API 权限

### 准备工作

**目标**：在 Linux 服务器上部署 vertexai-proxy，把 Google 的 Gemini 2.5 Flash 封装成 OpenAI 兼容接口，供 OpenClaw 使用。

**Python 3**

---

## 1. 安装 Google Cloud SDK 并登录

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
# Gemini 2.5 Flash - 简单问答
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-2.5-flash-low",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# Gemini 3 Flash - 简单问候
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-flash-preview",
    "messages": [{"role": "user", "content": "Hello"}]
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

---

## 许可证

MIT License
