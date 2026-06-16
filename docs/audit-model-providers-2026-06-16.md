# Audit: Seed Model Provider — 2026-06-16

## 审计范围

- `/home/u2/agent/seed-model-providers/seed_model_providers/model_providers.py`（1695 行）
- `/home/u2/agent/seed-model-providers/seed_model_providers/token_counter.py`（82 行）
- `/home/u2/agent/seed-model-providers/seed_model_providers/deepseek_tokenizer/deepseek_tokenizer.py`（12 行）
- `/home/u2/agent/seed-model-providers/tests/test_model_providers.py`

---

## 🚨 Bug #1（中风险）: `apply_chat_thinking_extra_body` — 引用未定义变量 `_ea`

**位置**: `model_providers.py:773,775`

**表现**: 使用 `sglang` chat_protocol 时触发 `NameError: name '_ea' is not defined`

**根因**: `_ea` 的本地 import 只在 `normalize_reasoning_effort`（line 745）和 `should_send_reasoning_content`（line 793）中存在，但 `apply_chat_thinking_extra_body` 函数体内没有。该函数在第 773/775 行用到了 `_ea.pick_default(...)`，而 `_ea` 未定义。

```python
def apply_chat_thinking_extra_body(
    *,
    chat_protocol: str,
    base_url: str,
    params: Dict[str, Any],
    extra_body: Dict[str, Any],
    resolved_thinking: bool,
    reasoning_effort: Optional[str] = None,
) -> None:
    _ = base_url                          # 无用语句
    ...
    if chat_protocol == "sglang":
        if _ea.pick_default(...) != "0":  # ❌ _ea is not defined
```

**修复**: 在函数体内加 `from seed.core import env_access as _ea`

---

## ⚠️ 问题 #2（架构）: 单文件 1695 行，远超 400 行上限

`model_providers.py` 包含 5 个功能各异的逻辑域：

| 功能域 | 行数 |
|--------|------|
| Provider 目录静态数据 + 查询函数 | ~240 |
| 预设辅助函数（materialize/enrich） | ~240 |
| Chat 协议行为控制（thinking/effort） | ~80 |
| 图片生成适配器（OpenAI/MiniMax/火山） | ~360 |
| 音乐生成适配器（MiniMax） | ~140 |
| 视频生成适配器（Agnes/MiniMax） | ~450 |

**建议拆分结构**:

```
seed_model_providers/
├── catalog.py        # PROVIDER_CATALOG + list_*/get_* 函数
├── presets.py        # materialize/enrich/preset helpers
├── protocols.py      # resolve_chat_protocol + thinking helpers
├── image_gen.py      # 3 家图片生成适配器
├── music_gen.py      # MiniMax 音乐生成
├── video_gen.py      # Agnes + MiniMax 视频生成
├── token_counter.py  # 不变
└── __init__.py       # 统一 re-export
```

---

## ⚠️ 问题 #3（低风险）: `_ = base_url` 死参数

**位置**: `model_providers.py:769`

```python
def apply_chat_thinking_extra_body(
    *,
    chat_protocol: str,
    base_url: str,         # 传了，但函数内未使用
    ...
) -> None:
    _ = base_url            # 仅用于绕过 lint 的未使用变量检查
```

`base_url` 在所有代码路径中均未被读取。要么删除参数，要么实际用于协议判断。

---

## ⚠️ 问题 #4（低风险）: 测试代码通过旧包路径导入

`test_model_providers.py` 中有多处：

```python
from seed.core.model_providers import infer_preset_use_type, ...
```

虽然 `seed.core.model_providers` 现在是 re-export shim（功能正常），但作为独立包 `seed-model-providers` 的测试，应优先从 `seed_model_providers` 直接导入。否则未来 `seed` 删除此 shim 时测试会断。

---

## ⚠️ 问题 #5（低风险）: `_ea` import 位置不一致

3 个使用 `_ea` 的函数，import 写法不统一：

| 函数 | 有 import? | 位置 |
|------|-----------|------|
| `normalize_reasoning_effort` | ✅ | 函数体内 |
| `apply_chat_thinking_extra_body` | ❌ 缺失 | — |
| `should_send_reasoning_content` | ✅ | 函数体内 |
| `default_max_request_body_bytes` | ✅ | 函数体内 |

建议统一提到文件顶部，避免遗漏。

---

## 总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 功能完整性 | ✅ | 4 家 provider、6 种 use_type、3 种媒体生成，覆盖面好 |
| **[dev] 代码质量** | ⚠️ | 1 个 NameError + 单文件超限 4 倍是主要问题 |
| **[arch] 模块边界** | ⚠️ | 5 个功能域揉在一个文件，拆开更合理 |
| **[des] 测试** | ⚠️ | 19 用例覆盖核心逻辑，但 import 路径未升级 |
| **[ops] 安全/配置** | ✅ | 无敏感泄露，环境变量有 fallback |

## 修复计划

1. **Bug #1**: 补 `_ea` import（预计 2 分钟）
2. **问题 #3**: 清理 `_ = base_url` + 参数（预计 3 分钟）
3. **问题 #4**: 统一测试 import 路径（预计 3 分钟）
4. **问题 #5**: 统一 `_ea` import 到文件顶部（预计 3 分钟）
5. **问题 #2**: 拆文件（可选，较大重构）
