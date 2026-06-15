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
    call_image_generations,
)
```

## 能力

- 供应商 catalog 注册（DeepSeek, OpenAI, MiniMax, 火山方舟, SGLang, Ollama, Agnes AI, OpenRouter 等）
- 协议自动路由（chat / image / music / video 各走不同协议）
- Preset 映射与 UI 联动（materialize / enrich）
- 多模态生成（图像/音乐/视频）的统一调用接口
