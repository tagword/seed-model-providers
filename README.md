# seed-model-providers

统一的模型提供商目录（catalog）与协议调度层。

提取自 `seed.core.model_providers`，作为独立包供 seed、seed-tools、codeagent 等共同使用。

## 安装

```bash
pip install -e .
```

## 使用

```python
from seed_model_providers import (
    resolve_provider_for_preset,
    resolve_chat_protocol,
    list_provider_catalog,
    normalize_chat_usage,
    apply_provider_chat_headers,
    call_image_generations,
)
```

## 能力

- 供应商 catalog 注册（DeepSeek, OpenAI, Anthropic, Google Gemini, 阿里云百炼/Qwen, Moonshot/Kimi, 智谱 GLM, OpenRouter, Groq, MiniMax, 火山方舟, SGLang, Ollama, Agnes AI 等）
- Chat 协议自动路由（deepseek / dashscope / moonshot / **zhipu** / minimax / sglang / openai）
- **usage 归一化**（`normalize_chat_usage`）与 **OpenRouter 请求头**（`apply_provider_chat_headers`）
- 思考模式多轮回传（DashScope preserve_thinking、Kimi keep=all、智谱 clear_thinking=false）
- Preset 映射与 UI 联动（materialize / enrich）
- 多模态生成（图像/音乐/视频）的统一调用接口

## 文档

| 文档 | 说明 |
|------|------|
| [`docs/PROVIDER_PROTOCOLS.md`](docs/PROVIDER_PROTOCOLS.md) | Chat 协议、usage、OpenRouter、思考模式、与上下文指示器关系 |
| [`docs/MODEL_CATALOG.md`](docs/MODEL_CATALOG.md) | 内置 model 下拉列表（PROVIDER_CATALOG） |
| [`docs/audit-model-providers-2026-06-16.md`](docs/audit-model-providers-2026-06-16.md) | 历史审计记录 |

CodeAgent 侧入口：[`codeagent/docs/MODEL_PROVIDERS.md`](../codeagent/docs/MODEL_PROVIDERS.md)

## 测试

```bash
pip install -e ".[dev]"
PYTHONPATH="../seed:." pytest tests/ -q
```
