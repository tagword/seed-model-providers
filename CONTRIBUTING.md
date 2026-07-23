# Contributing

## 开发环境

```bash
git clone https://github.com/tagword/seed-model-providers
cd seed-model-providers

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 测试

```bash
PYTHONPATH="../seed:." pytest tests/ -q
```

## 添加新模型提供商

1. 在 `seed_model_providers/catalog.py` 中新增供应商条目
2. 在 `seed_model_providers/chat_protocols.py` 中注册协议路由
3. 若需要使用新的认证方式，在 `seed_model_providers/network.py` 中补充
4. 更新 `docs/MODEL_CATALOG.md` 和 `docs/PROVIDER_PROTOCOLS.md`
5. 编写测试覆盖新供应商的协议分发

## PR 规范

- 分支名：`feat/xxx`、`fix/xxx`
- Commit message 遵循 Conventional Commits
- 新增供应商需附带基础路由测试
