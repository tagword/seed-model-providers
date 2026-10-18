# Changelog

## 1.0.2 (2026-10-18)

- feat(param-compat): 新增 `param_compat` 模块，处理跨供应商请求参数兼容：
  - OpenAI 推理家族（`o1`/`o3`/`o4`、`gpt-5+`，含 `azure/gpt-5.6-luna`、`gpt-6-luna` 等网关别名）改用 `max_completion_tokens`，不再被 `unsupported_parameter` 拒绝；`gpt-4.x` 及 DeepSeek/MiniMax 等仍用 `max_tokens`。
  - 服务端能力缓存：按 `(base_url, model)` 记录实际接受的参数名，进程内只探测一次。
  - `repetition_penalty` 预处理：DeepSeek NInfer 仅接受中性值 `1`，非 `1` 自动剥离并告警（不再 400）。
  - 兜底修复：对 `400/422` 参数错误自动修正后重试一次（`max_tokens` 双向互换、剥离 `repetition_penalty`、剥离推理模型不支持的 `temperature`/`top_p`/`top_k` 等）；鉴权/限流/5xx 不重试。
## Unreleased

- fix(minimax): 思考模式控制失效 — 旧实现只传 `reasoning_split` 从不传 `thinking`，导致「关闭思考」完全无效（服务端始终思考）。现按模型能力传 `thinking={"type":"adaptive"|"disabled"}`。
- fix(minimax): 显式传 `reasoning_split`（`True`/`False`），避免服务端默认把 thinking 内联进 `content` 污染正文。
- fix(minimax): `reasoning_effort` 走顶层参数并归一为 MiniMax 档位 `low|medium|high|xhigh|max`（不支持 `none`）；新增 `normalize_minimax_reasoning_effort()`。
- feat(minimax): 模型能力探测 helper — `minimax_forces_thinking` / `minimax_forces_reasoning_split` / `minimax_supports_reasoning_effort` / `minimax_thinking_capable` / `minimax_can_disable_thinking`。M3.1-Flash-Preview 强制 thinking + 强制 split（传 disabled/false → 400），自动降级；M2.x 无法关闭；Text-01/VL-01 无 thinking 能力。
- feat(minimax): 新增 `MiniMax-M3.1-Flash-Preview`；默认聊天模型 `MiniMax-M2.7` → `MiniMax-M3`。

## 1.0.1 (2026-08-06)

- feat: 新增网络代理模块 — is_local_url / requests_proxies_for / httpx_trust_env_for
- feat: 新增 DashScope/Moonshot/智谱/Anthropic/Gemini/OpenRouter/Groq 供应商
- feat: 添加 DeepSeek V3 tokenizer 封装
- fix: model_providers 细化空字符串检查
- fix: count_messages 中 tool_calls JSON 在 content 非空时被跳过

## 1.0.0 (2026-06-17)

- feat: 初始公开发布 — 从 seed.core.model_providers 提取为独立包
- 供应商 catalog：DeepSeek, OpenAI, Anthropic, Google Gemini, 阿里云百炼/Qwen,
  Moonshot/Kimi, 智谱 GLM, OpenRouter, Groq, MiniMax, 火山方舟, SGLang, Ollama,
  Agnes AI 等
- Chat 协议自动路由（deepseek / dashscope / moonshot / zhipu / minimax / sglang / openai）
- usage 归一化（normalize_chat_usage）
- Preset 映射与 UI 联动（materialize / enrich）
- 多模态生成接口（call_image_generations）
- docs: README, PROVIDER_PROTOCOLS, MODEL_CATALOG
