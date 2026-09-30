"""Model pricing & cost calculation.

Single source of truth for LLM API pricing across all providers.

Currently includes:
- DeepSeek V4 (per-time-bucket pricing; see https://api-docs.deepseek.com/zh-cn/quick_start/pricing)

Design:
- PRICING_TABLE keyed by ``(provider, std_model, bucket)``; each bucket is a dict
  with ``input_cache_hit``, ``input_cache_miss``, ``output`` (CNY per 1M tokens).
- ``bucket`` is a pricing tier label (``"off_peak"`` / ``"peak"`` / ``"standard"``).
  Providers without time-based pricing use a single ``"standard"`` bucket.
- ``is_peak_time()`` is DeepSeek-specific (kept here because the bucket key comes
  from the pricing logic); other providers always return ``"standard"``.

When extending:
1. Add an entry under PRICING_TABLE[provider_id][std_model] with one or more buckets.
2. Add an alias entry under _MODEL_ALIASES so the normalize step picks it up.
3. Decide whether the provider needs time-based pricing (most don't).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# ── 定价表（CNY / 百万 tokens） ─────────────────────────────────
# DeepSeek V4（2026-09 现行）来源：https://api-docs.deepseek.com/zh-cn/quick_start/pricing
# 时段规则：高峰 = 北京时间工作日 9:00-12:00、14:00-18:00（不含法定节假日）；
#          空闲 = 高峰以外 + 周末 + 法定节假日全天（空闲价 = 高峰价的一半）

PRICING_TABLE: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "deepseek": {
        "deepseek-flash": {
            "peak": {
                "input_cache_hit": 0.04,
                "input_cache_miss": 2.0,
                "output": 8.0,
            },
            "off_peak": {
                "input_cache_hit": 0.02,
                "input_cache_miss": 1.0,
                "output": 4.0,
            },
        },
        "deepseek-v4-pro": {
            "peak": {
                "input_cache_hit": 0.30,
                "input_cache_miss": 9.0,
                "output": 27.0,
            },
            "off_peak": {
                "input_cache_hit": 0.15,
                "input_cache_miss": 4.5,
                "output": 13.5,
            },
        },
    },
}

# ── Provider 模型名归一化 ─────────────────────────────────────
# 与 model_providers.DEEPSEEK_MODEL_ALIASES 保持一致；如需修改两边同步。
_PROVIDER_MODEL_ALIASES: dict[str, dict[str, str]] = {
    "deepseek": {
        # V3 时代旧名（V3.2 文档里的别名）
        "deepseek-chat": "deepseek-flash",
        "deepseek-reasoner": "deepseek-flash",
        # V4 过渡期旧名（官方 2026-09 起推荐 deepseek-flash 新名）
        "deepseek-v4-flash": "deepseek-flash",
        "deepseek-v4-flash-vision-exp": "deepseek-flash",
        # 现行名（无别名需求，原样返回）
        "deepseek-flash": "deepseek-flash",
        "deepseek-v4-pro": "deepseek-v4-pro",
    },
}


def normalize_model(provider: str, model: str) -> str | None:
    """将 ``(provider, model)`` 标准化为定价表中的 key。未知组合返回 ``None``。"""
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()
    if not p or not m:
        return None
    aliases = _PROVIDER_MODEL_ALIASES.get(p)
    if not aliases:
        return None
    return aliases.get(m)


def is_peak_time(now: datetime | None = None) -> bool:
    """判断给定时刻是否处于 DeepSeek 计费高峰时段（北京时间）。

    高峰：周一至周五 9:00-12:00 / 14:00-18:00。
    法定节假日暂未接库，按工作日规则处理（节假日期间会被误判为高峰）。
    下游若需精确节假日，可在此处接 ``holidays`` 库或维护白名单。

    Args:
        now: 给定时刻（默认当前北京时间）；传 None 表示「现在」。

    Returns:
        True = 高峰（peak）；False = 空闲（off_peak）。
    """
    if now is None:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    else:
        now = now.astimezone(ZoneInfo("Asia/Shanghai"))

    # weekday(): 周一=0 ... 周日=6
    if now.weekday() >= 5:
        return False  # 周末全天 = 空闲

    h = now.hour
    return (9 <= h < 12) or (14 <= h < 18)


def resolve_bucket(provider: str, now: datetime | None = None) -> str:
    """根据 provider 与当前时刻选择定价时段 bucket。

    DeepSeek → "peak" / "off_peak"（按 is_peak_time 判定）；
    其他 provider → 永远 "standard"。
    """
    if provider == "deepseek":
        return "peak" if is_peak_time(now) else "off_peak"
    return "standard"


def calculate_cost(
    provider: str,
    model: str,
    usage: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """根据 provider/model/usage 计算费用（人民币元）。

    Args:
        provider: provider id（"deepseek" 等）。
        model: 模型名（自动匹配别名）。
        usage: API 返回的 usage 对象；DeepSeek 期望字段：
            - prompt_cache_hit_tokens (int)
            - prompt_cache_miss_tokens (int)
            - completion_tokens (int)
            其他 provider 暂未接定价表，传什么都会被忽略。
        now: 用于时段判断的时刻（默认当前北京时间）；测试时显式传入。

    Returns:
        {
            "total_cost": float,
            "cache_hit_cost": float,
            "cache_miss_cost": float,
            "output_cost": float,
            "currency": "CNY",
            "provider": str,
            "model": str,              # 标准化模型名
            "time_bucket": str,        # "peak" | "off_peak" | "standard"
            "is_peak_time": bool,      # DeepSeek 时段；其他 provider 永远 False
        }
        provider/model 不在表中或 usage 为空 → 全部 0，model 原样回填。
    """
    std_model = normalize_model(provider, model)
    bucket = resolve_bucket(provider, now)
    empty = {
        "total_cost": 0.0,
        "cache_hit_cost": 0.0,
        "cache_miss_cost": 0.0,
        "output_cost": 0.0,
        "currency": "CNY",
        "provider": provider,
        "model": std_model or model,
        "time_bucket": bucket,
        "is_peak_time": bucket == "peak",
    }
    if not std_model or not usage:
        return empty

    provider_pricing = PRICING_TABLE.get(provider, {})
    model_pricing = provider_pricing.get(std_model)
    if not model_pricing or bucket not in model_pricing:
        return empty

    prices = model_pricing[bucket]
    cache_hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    output = int(usage.get("completion_tokens", 0) or 0)

    cache_hit_cost = cache_hit * prices["input_cache_hit"] / 1_000_000
    cache_miss_cost = cache_miss * prices["input_cache_miss"] / 1_000_000
    output_cost = output * prices["output"] / 1_000_000
    total = cache_hit_cost + cache_miss_cost + output_cost

    return {
        "total_cost": round(total, 6),
        "cache_hit_cost": round(cache_hit_cost, 6),
        "cache_miss_cost": round(cache_miss_cost, 6),
        "output_cost": round(output_cost, 6),
        "currency": "CNY",
        "provider": provider,
        "model": std_model,
        "time_bucket": bucket,
        "is_peak_time": bucket == "peak",
    }


def format_cost(cost: float, currency: str = "CNY") -> str:
    """将费用格式化为可读字符串。"""
    if cost <= 0:
        return "¥0"
    if cost < 0.01:
        return f"¥{cost:.4f}"
    if cost < 1:
        return f"¥{cost:.3f}"
    return f"¥{cost:.2f}"
