# Changelog

## 1.0.1 (unreleased)

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
