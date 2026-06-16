# 模型提供商协议与上下文对齐（2026-06-16）

本文档描述 `seed-model-providers` 中 **Chat 协议**、**usage 归一化**、**OpenRouter 请求头**、**思考模式多轮回传** 的实现，以及与 CodeAgent **上下文指示器 / compact** 的关系。

实现源码：`seed_model_providers/model_providers.py`；运行时接入：`seed/seed/core/llm_exec.py`（`LLMAPIExecutor`）。

---

## 1. 设计目标

| 问题 | 对策 |
|------|------|
| 部分网关不返回 `usage.prompt_tokens` | `normalize_chat_usage()` 将 `input_tokens`、`promptTokenCount` 等映射为 OpenAI 形字段 |
| OpenRouter 缺 `HTTP-Referer` 导致空 content | `apply_provider_chat_headers()` 自动附带归因头 |
| DashScope / Kimi / 智谱 思考模式多轮 tool 链 | 各协议写入 `preserve_thinking` / `thinking.keep` / `clear_thinking: false` |
| Kimi 同时传 `thinking` + `reasoning_effort` 报 400 | Moonshot / 智谱 协议内 `params.pop("reasoning_effort")` |
| 流式最后一 chunk 无 usage | `apply_chat_stream_options()` → `stream_options.include_usage: true` |

**与上下文指示器的关系**：指示器只认 API 归一化后的 `prompt_tokens` + `peak_prompt_tokens`（见 `codeagent/docs/DEBUG_CONTEXT_INDICATOR.md`）。providers 层负责让各厂商响应尽量产出可用的 `prompt_tokens`；若网关完全不返 usage，指示器仍为 0（API-only 设计，非 UI bug）。

---

## 2. Chat 协议一览

Preset 的 `provider` + `base_url` 经 `resolve_chat_protocol()` 解析为下列协议之一：

| 协议 id | 典型 provider | 思考模式请求体 | 多轮 reasoning 回传 |
|---------|---------------|----------------|---------------------|
| `deepseek` | deepseek | `extra_body.thinking.type` + `params.reasoning_effort` | ✅ `uses_full_reasoning_content_echo` |
| `dashscope` | dashscope | `enable_thinking` + `preserve_thinking: true` | ✅ |
| `moonshot` | moonshot | `thinking.type`；K2.6/K2.7 加 `thinking.keep: all` | ✅ |
| `zhipu` | zhipu | `thinking.type` + `clear_thinking: false` | ✅ |
| `minimax` | minimax | `reasoning_split: true`（M3 等） | 走 `reasoning_details` / 内联 think 标签 |
| `minimax_anthropic` | minimax + URL 含 `/anthropic` | Anthropic Messages API | llm_exec 专用路径 |
| `sglang` | sglang | `separate_reasoning` + `chat_template_kwargs.enable_thinking` | 视网关 |
| `openai` | openai, anthropic, google, openrouter, groq, volcengine, ollama, … | 无内置 thinking 字段（可用 `SEED_LLM_EXTRA_BODY`） | 视模型 |

### 2.1 DeepSeek

- 文档：[思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)
- `apply_chat_thinking_extra_body(chat_protocol="deepseek")`：
  - `extra_body.thinking.type`: `enabled` / `disabled`
  - 开启时 `params.reasoning_effort`: `high` | `max`（`xhigh` → `max`）

### 2.2 阿里云百炼（DashScope / Qwen）

- 文档：[深度思考](https://help.aliyun.com/zh/model-studio/deep-thinking)
- 兼容 Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 开启思考时：
  - `extra_body.enable_thinking: true`
  - `extra_body.preserve_thinking: true`（多轮 Agent 回传历史 `reasoning_content`）

### 2.3 Moonshot / Kimi

- 文档：[Using Thinking Models](https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model)
- `extra_body.thinking.type`: `enabled`（默认）/ `disabled`
- **K2.6 / K2.7**：`thinking.keep: "all"`（Preserved Thinking）
- **禁止**与 `reasoning_effort` 同传；协议层会移除 `params["reasoning_effort"]`

### 2.4 智谱 GLM（`zhipu` 协议）

- 文档：[深度思考](https://docs.bigmodel.cn/cn/guide/capabilities/thinking)
- Base URL：`https://open.bigmodel.cn/api/paas/v4`
- `extra_body.thinking.type` + 开启时 **`clear_thinking: false`**（Agent 工具链保留 reasoning）
- 同样移除 `reasoning_effort`，避免与部分代理冲突

### 2.5 MiniMax

- M3：`reasoning_split: true` → `reasoning_details`
- M2.x：content 内 `<think>`，由 `llm_exec` 剥离
- usage：`normalize_chat_usage` 内处理 `prompt_tokens_details.cached_tokens` 等

### 2.6 OpenRouter

- 文档：[App Attribution](https://openrouter.ai/docs/app-attribution)
- 非可选场景下缺 header 可能导致空 `choices[0].message.content`
- 见下文 §4

---

## 3. Usage 归一化：`normalize_chat_usage()`

每次 LLM 响应（含流式末 chunk）在 `llm_exec` 中调用：

```python
normalize_chat_usage(usage, chat_protocol=..., provider=...)
```

### 3.1 字段映射

| 原始字段（部分网关） | 归一化后 |
|----------------------|----------|
| `input_tokens` + `cache_*` | `prompt_tokens` |
| `promptTokenCount` / `total_input_tokens` | `prompt_tokens` |
| `output_tokens` / `candidatesTokenCount` | `completion_tokens` |
| MiniMax `prompt_tokens_details.cached_tokens` | `prompt_cache_hit_tokens` 等（内部统计） |

### 3.2 仍无 `prompt_tokens` 时

- 运行中：WS `context_usage` 可能为 0
- compact：不触发（API-only，`maybe_compact_context_messages` 需要 API 分子）
- 对策：换官方兼容端点；流式确认 `stream_options.include_usage` 生效；检查网关文档

### 3.3 持久化

整轮对话结束后，`build_context_usage_from_run()` 将 peak 写入 `session.metadata.context_usage`（**不是**逐轮原始 `usage` 对象）。详见 `codeagent/docs/DEBUG_CONTEXT_INDICATOR.md`。

---

## 4. HTTP 请求头：`apply_provider_chat_headers()`

在 `LLMAPIExecutor.__init__` 中合并到 `self.headers`。

### OpenRouter

| Header | 环境变量（优先级从左到右） | 默认值（未设置 env 时） |
|--------|---------------------------|-------------------------|
| `HTTP-Referer` | `SEED_LLM_HTTP_REFERER`, `SEED_OPENROUTER_HTTP_REFERER`, `OPENROUTER_HTTP_REFERER` | `https://github.com/seed-agent/codeagent` |
| `X-OpenRouter-Title` | `SEED_LLM_APP_TITLE`, `SEED_OPENROUTER_APP_TITLE`, `OPENROUTER_APP_TITLE`, `X_TITLE` | `Seed CodeAgent` |
| `X-Title` | （同上，legacy 别名） | 同上 |

Code Agent 可通过 bridge 设置 `CODEAGENT_LLM_HTTP_REFERER` / `CODEAGENT_LLM_APP_TITLE`（映射为 `SEED_*`）。

---

## 5. 流式 Usage：`apply_chat_stream_options()`

在 `generate_stream()` 合并 `extra_body` 之后调用：

```json
{ "stream": true, "stream_options": { "include_usage": true } }
```

确保 DashScope / OpenAI 兼容层在**最后一个 SSE chunk** 返回 `usage`，供 `peak_prompt_tokens` 与指示器更新。

---

## 6. 环境变量（新增 / 相关）

| 变量 | 说明 |
|------|------|
| `SEED_LLM_HTTP_REFERER` | OpenRouter 等需要的 Referer |
| `SEED_LLM_APP_TITLE` | OpenRouter 应用展示名 |
| `SEED_LLM_ENABLE_THINKING` | 全局思考开关（默认开）；传给 `apply_chat_thinking_extra_body` |
| `SEED_LLM_REASONING_EFFORT` | 主要作用于 **DeepSeek**；Moonshot/智谱 协议会忽略顶层 `reasoning_effort` |
| `SEED_LLM_EXTRA_BODY` | JSON 合并进请求体（各厂商扩展参数） |
| `SEED_LLM_SEND_REASONING_CONTENT` | 强制开/关 assistant `reasoning_content` 回传 |

完整列表：`seed/docs/ENV_REFERENCE.md`。

---

## 7. Preset → 运行时调用链

```
Web UI 选 provider + model
  → materialize_preset_from_form()     # 填 base_url、supports_*、provider
  → enrich_preset_defaults()           # 附加 _chat_protocol / _image_protocol
  → LLMAPIExecutor(provider, base_url, model)
       apply_provider_chat_headers()
       resolve_chat_protocol()
  → run_llm_tool_loop / generate / generate_stream
       apply_chat_thinking_extra_body(..., model=self.model)
       apply_chat_stream_options()      # 仅 stream
       normalize_chat_usage()           # 响应后
  → build_context_usage_from_run → metadata.context_usage
  → WS context_usage + 刷新恢复
```

多模态生图/音乐/视频走 `seed-tools` + `call_*_generations()`，与上下文 usage **无直接耦合**。

---

## 8. Web UI Catalog

内置模型下拉来自 `PROVIDER_CATALOG`（`list_provider_catalog()` → `GET /api/ui/llm/providers`）。

**2026-06-16 更新摘要**：

- 各云厂商补充当前常用 model id（OpenAI 4.1 系、Claude 3.7、Gemini 2.5、Qwen3.5、Kimi K2.6、GLM-5/4.7、Groq Llama4、OpenRouter 热门 slug、方舟 Seed 2.0 等）
- 智谱默认模型改为 `glm-4.7`，`chat_protocol` 改为 `zhipu`
- 火山方舟对话默认 `doubao-seed-2-0-lite-260215`；生图 Seedream 列表保持独立 bucket

详细 model id 表见 [`MODEL_CATALOG.md`](MODEL_CATALOG.md)。

---

## 9. 验证清单

1. 重装 `seed-model-providers` + 重启 `codeagent serve`
2. 选目标 preset，跑一轮带工具链的长对话
3. 观察 WS `context_usage.prompt_tokens` 是否随 LLM 轮次阶梯上涨
4. 检查 session JSON：`metadata.context_usage.peak_prompt_tokens`
5. 刷新页面，指示器应与结束前一致
6. OpenRouter：若 content 为空，确认 Referer/Title header 已发（抓包或开 debug 日志）

---

## 10. 相关文档

| 文档 | 内容 |
|------|------|
| [`MODEL_CATALOG.md`](MODEL_CATALOG.md) | 各 provider 内置 model 列表 |
| [`../README.md`](../README.md) | 包安装与 API 入口 |
| [`../../codeagent/docs/DEBUG_CONTEXT_INDICATOR.md`](../../codeagent/docs/DEBUG_CONTEXT_INDICATOR.md) | 上下文指示器排查 |
| [`../../codeagent/docs/CONTEXT_COMPACT.md`](../../codeagent/docs/CONTEXT_COMPACT.md) | compact 与 usage |
| [`../../codeagent/docs/IMAGE_GEN_PROVIDERS.md`](../../codeagent/docs/IMAGE_GEN_PROVIDERS.md) | 生图协议 |
| [`../../seed/docs/ENV_REFERENCE.md`](../../seed/docs/ENV_REFERENCE.md) | `SEED_*` 环境变量 |
