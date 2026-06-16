# 内置模型目录（PROVIDER_CATALOG）

Web UI「环境配置 → 模型预设」中，选择 **Provider** 后下拉模型来自 `seed_model_providers/model_providers.py` 的 `PROVIDER_CATALOG`。

运行时以磁盘上的 `config/seed.models.json` 为准；本文档是 **2026-06-16** catalog 快照，便于对照与排查。若代码中列表已更新，以源码 `PROVIDER_CATALOG` 为准。

API：`GET /api/ui/llm/providers` 返回 `list_provider_catalog()` 的 JSON。

---

## deepseek

| use_type | model id | 说明 |
|----------|----------|------|
| chat | `deepseek-v4-flash` | 默认 |
| chat | `deepseek-v4-pro` | |

- Base URL：`https://api.deepseek.com/v1`
- 协议：`deepseek`
- 别名：`deepseek-chat` / `deepseek-reasoner` → `deepseek-v4-flash`（materialize 时归一化）

---

## volcengine（火山方舟）

| use_type | model id | 说明 |
|----------|----------|------|
| chat | `doubao-seed-2-0-pro-260215` | Seed 2.0 Pro |
| chat | `doubao-seed-2-0-lite-260215` | 默认对话 |
| chat | `doubao-seed-1-8-251228` | Seed 1.8 |
| chat | `doubao-1-5-pro-32k-250115` | |
| chat | `deepseek-v3-2-251218` | 方舟接入 DeepSeek |
| image | `doubao-seedream-5-0-lite-260128` | Seedream |
| image | `doubao-seedream-5-0-260128` | |
| image | `doubao-seedream-4-5-251128` | |
| image | `doubao-seedream-4-0-250828` | |

- Base URL：`https://ark.cn-beijing.volces.com/api/v3`
- 对话 model 也可填控制台 **ep-** 接入点 ID
- 生图短名别名见 `VOLCENGINE_IMAGE_MODEL_ALIASES`

---

## minimax

| use_type | model id（节选） |
|----------|------------------|
| chat | `MiniMax-M3`, `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, …, `MiniMax-Text-01`, `MiniMax-VL-01` |
| image | `image-01`, `image-01-live` |
| speech | `speech-2.8-turbo`, `speech-2.8-hd`, … |
| music | `music-2.6`, `music-2.6-free`, … |
| video_gen | `hailuo-2.3`, `hailuo-2.3-fast`, `hailuo-02` |

- Base URL：`https://api.minimaxi.com/v1`
- Anthropic 兼容：`base_url` 含 `/anthropic` → 协议 `minimax_anthropic`

---

## openai

| use_type | model id |
|----------|----------|
| chat | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `o3-mini` |
| image | `dall-e-3`, `dall-e-2`, `gpt-image-1` |
| vision | `gpt-4.1`, `gpt-4o` |

- Base URL：`https://api.openai.com/v1`

---

## anthropic

| use_type | model id |
|----------|----------|
| chat | `claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `claude-3-7-sonnet-20250219`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` |
| vision | `claude-sonnet-4-20250514`, `claude-3-7-sonnet-20250219` |

- OpenAI 兼容层 Base URL：`https://api.anthropic.com/v1`
- 协议：`openai`（原生 Messages API 未内置）

---

## google

| use_type | model id |
|----------|----------|
| chat | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-2.5-flash-lite` |
| vision | `gemini-2.5-flash`, `gemini-2.5-pro` |

- Base URL：`https://generativelanguage.googleapis.com/v1beta/openai`

---

## dashscope（阿里云百炼 / Qwen）

| use_type | model id |
|----------|----------|
| chat | `qwen-plus`, `qwen-max`, `qwen-turbo`, `qwen3-max`, `qwen3.5-plus`, `qwq-plus`, `qwq-32b` |
| vision | `qwen-vl-max`, `qwen-vl-plus`, `qwen3-vl-plus` |

- Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 协议：`dashscope`

---

## moonshot（Kimi）

| use_type | model id |
|----------|----------|
| chat | `kimi-k2.5`, `kimi-k2-turbo-preview`, `kimi-k2.6`, `moonshot-v1-128k`, `moonshot-v1-32k` |
| vision | `moonshot-v1-128k-vision-preview` |

- Base URL：`https://api.moonshot.cn/v1`
- 协议：`moonshot`

---

## zhipu（智谱）

| use_type | model id |
|----------|----------|
| chat | `glm-5`, `glm-4.7`, `glm-4.6`, `glm-4-plus`, `glm-4-flash` |
| vision | `glm-4.5v`, `glm-4v-plus` |

- Base URL：`https://open.bigmodel.cn/api/paas/v4`
- 默认 chat：`glm-4.7`
- 协议：`zhipu`（非 generic `openai`）
- Coding Plan 端点：`/api/coding/paas/v4`（手填 advanced 模式）

---

## openrouter

| use_type | model id（slug） |
|----------|------------------|
| chat | `openai/gpt-4.1`, `openai/gpt-4o`, `anthropic/claude-sonnet-4`, `google/gemini-2.5-flash`, `deepseek/deepseek-chat`, `qwen/qwen-plus` |

- Base URL：`https://openrouter.ai/api/v1`
- 自动 HTTP 归因头，见 [`PROVIDER_PROTOCOLS.md`](PROVIDER_PROTOCOLS.md) §4

---

## groq

| use_type | model id |
|----------|----------|
| chat | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `meta-llama/llama-4-maverick-17b-128e-instruct`, `meta-llama/llama-4-scout-17b-16e-instruct`, `qwen-qwq-32b`, `openai/gpt-oss-120b` |

- Base URL：`https://api.groq.com/openai/v1`

---

## 其他 provider

| id | 说明 |
|----|------|
| `agnes` | 仅 `video_gen`：`agnes-video-v2.0` |
| `azure_openai` | 手填 deployment URL / 模型名 |
| `ollama` | 本地 `http://127.0.0.1:11434/v1`，模型名自定 |
| `sglang` | 本地 SGLang，协议 `sglang` |
| `openai_compatible` | 通用网关 |
| `custom` | 完全手填，协议按 URL 推断 |

---

## Provider 别名（normalize_provider_id）

| 输入 | 归一 id |
|------|---------|
| `qwen`, `aliyun`, `bailian` | `dashscope` |
| `kimi` | `moonshot` |
| `glm`, `bigmodel`, `zhipuai` | `zhipu` |
| `claude` | `anthropic` |
| `gemini`, `google_ai` | `google` |
| `ark`, `doubao`, `volc_engine` | `volcengine` |

---

## 修改 catalog 后

1. 编辑 `PROVIDER_CATALOG` + 必要时 `apply_chat_thinking_extra_body` / 测试
2. `pip install -e seed-model-providers`
3. 重启 CodeAgent
4. 更新本文档与 [`PROVIDER_PROTOCOLS.md`](PROVIDER_PROTOCOLS.md) 中的变更摘要
