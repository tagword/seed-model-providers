"""Pricing & cost calculation tests (moved from codeagent/.codeagent/default/tests/test_pricing.py).

Canonical pricing lives in :mod:`seed_model_providers.pricing`.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from seed_model_providers.pricing import (
    PRICING_TABLE,
    calculate_cost,
    format_cost,
    is_peak_time,
    normalize_model,
    resolve_bucket,
)


# ── 别名映射 ─────────────────────────────────────────────
def test_normalize_model_aliases():
    """V3 + V4 过渡期旧名一律指向 deepseek-flash。"""
    assert normalize_model("deepseek", "deepseek-chat") == "deepseek-flash"
    assert normalize_model("deepseek", "deepseek-reasoner") == "deepseek-flash"
    assert normalize_model("deepseek", "deepseek-v4-flash") == "deepseek-flash"
    assert normalize_model("deepseek", "deepseek-v4-flash-vision-exp") == "deepseek-flash"
    assert normalize_model("deepseek", "deepseek-flash") == "deepseek-flash"
    assert normalize_model("deepseek", "deepseek-v4-pro") == "deepseek-v4-pro"
    assert normalize_model("deepseek", "unknown-model") is None
    assert normalize_model("unknown-provider", "x") is None
    assert normalize_model("deepseek", "") is None
    assert normalize_model("", "deepseek-flash") is None


# ── 时段判定 ─────────────────────────────────────────────
def test_is_peak_time_workday_morning():
    """周三 10:30 = 高峰（工作日上午 9:00-12:00）"""
    assert is_peak_time(datetime(2026, 9, 30, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))) is True


def test_is_peak_time_workday_lunch_break():
    """周三 13:00 = 空闲（午休 12:00-14:00）"""
    assert is_peak_time(datetime(2026, 9, 30, 13, 0, tzinfo=ZoneInfo("Asia/Shanghai"))) is False


def test_is_peak_time_workday_evening_boundary():
    """18:00 边界 = 空闲"""
    assert is_peak_time(datetime(2026, 9, 30, 17, 59, tzinfo=ZoneInfo("Asia/Shanghai"))) is True
    assert is_peak_time(datetime(2026, 9, 30, 18, 0, tzinfo=ZoneInfo("Asia/Shanghai"))) is False


def test_is_peak_time_weekend_all_day():
    """周末全天 = 空闲"""
    assert is_peak_time(datetime(2026, 10, 3, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))) is False  # 周六
    assert is_peak_time(datetime(2026, 10, 4, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai"))) is False  # 周日


def test_is_peak_time_naive_datetime_treated_as_beijing():
    """无 tzinfo 的 datetime 默认按北京时间。"""
    assert is_peak_time(datetime(2026, 9, 30, 10, 30)) is True


def test_resolve_bucket():
    """resolve_bucket 返回 peak / off_peak / standard。"""
    peak = datetime(2026, 9, 30, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    offpeak = datetime(2026, 9, 30, 22, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert resolve_bucket("deepseek", peak) == "peak"
    assert resolve_bucket("deepseek", offpeak) == "off_peak"
    assert resolve_bucket("openai", peak) == "standard"
    assert resolve_bucket("anthropic", peak) == "standard"


# ── 计算费用 ─────────────────────────────────────────────
def test_calculate_cost_flash_peak():
    """高峰 flash：1M cache_hit = ¥0.04；1M miss = ¥2；1M out = ¥8。"""
    peak = datetime(2026, 9, 30, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    # 1M cache_hit
    out = calculate_cost(
        "deepseek", "deepseek-flash",
        {"prompt_cache_hit_tokens": 1_000_000}, now=peak,
    )
    assert out["model"] == "deepseek-flash"
    assert out["time_bucket"] == "peak"
    assert out["total_cost"] == 0.04
    # 1M miss
    out = calculate_cost(
        "deepseek", "deepseek-flash",
        {"prompt_cache_miss_tokens": 1_000_000}, now=peak,
    )
    assert out["total_cost"] == 2.0
    # 1M out
    out = calculate_cost(
        "deepseek", "deepseek-flash",
        {"completion_tokens": 1_000_000}, now=peak,
    )
    assert out["total_cost"] == 8.0
    # 综合
    out = calculate_cost(
        "deepseek", "deepseek-flash",
        {"prompt_cache_hit_tokens": 1_000_000, "prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000},
        now=peak,
    )
    assert out["total_cost"] == 0.04 + 2.0 + 8.0


def test_calculate_cost_flash_off_peak():
    """空闲 flash：1M miss = ¥1；1M out = ¥4（高峰的一半）。"""
    offpeak = datetime(2026, 9, 30, 22, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    out = calculate_cost(
        "deepseek", "deepseek-flash",
        {"prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000},
        now=offpeak,
    )
    assert out["time_bucket"] == "off_peak"
    assert out["cache_miss_cost"] == 1.0
    assert out["output_cost"] == 4.0
    assert out["total_cost"] == 5.0


def test_calculate_cost_pro_peak():
    """高峰 v4-pro：1M miss = ¥9；1M out = ¥27。"""
    peak = datetime(2026, 9, 30, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    out = calculate_cost(
        "deepseek", "deepseek-v4-pro",
        {"prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000},
        now=peak,
    )
    assert out["model"] == "deepseek-v4-pro"
    assert out["time_bucket"] == "peak"
    assert out["cache_miss_cost"] == 9.0
    assert out["output_cost"] == 27.0
    assert out["total_cost"] == 36.0


def test_calculate_cost_alias_resolution():
    """deepseek-chat alias → deepseek-flash pricing."""
    peak = datetime(2026, 9, 30, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    out = calculate_cost(
        "deepseek", "deepseek-chat",
        {"prompt_cache_hit_tokens": 1_000_000}, now=peak,
    )
    assert out["model"] == "deepseek-flash"
    assert out["total_cost"] == 0.04


def test_calculate_cost_unknown_provider_returns_zero():
    """未知 provider → total_cost=0（暂未接定价表）。"""
    out = calculate_cost(
        "openai", "gpt-4o",
        {"prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 1_000_000},
    )
    assert out["total_cost"] == 0.0
    assert out["time_bucket"] == "standard"
    assert out["provider"] == "openai"


def test_calculate_cost_empty_usage():
    """usage 为空 → total_cost=0。"""
    out = calculate_cost("deepseek", "deepseek-flash", {})
    assert out["total_cost"] == 0.0
    assert out["currency"] == "CNY"


# ── 格式化 ─────────────────────────────────────────────
def test_format_cost_zero():
    assert format_cost(0) == "¥0"
    assert format_cost(-1) == "¥0"  # 负数归零


def test_format_cost_small():
    s = format_cost(0.005)
    assert s.startswith("¥")
    assert len(s.split(".")[1]) == 4  # 0.0050


def test_format_cost_medium():
    assert format_cost(0.5) == "¥0.500"


def test_format_cost_large():
    assert format_cost(5.0) == "¥5.00"


# ── PRICING_TABLE 结构 ─────────────────────────────────────────
def test_pricing_table_structure():
    """PRICING_TABLE 必须按 provider → model → bucket 三层。"""
    assert "deepseek" in PRICING_TABLE
    assert "deepseek-flash" in PRICING_TABLE["deepseek"]
    assert "deepseek-v4-pro" in PRICING_TABLE["deepseek"]
    for model in ("deepseek-flash", "deepseek-v4-pro"):
        buckets = PRICING_TABLE["deepseek"][model]
        assert "peak" in buckets
        assert "off_peak" in buckets
        for bucket in buckets.values():
            assert "input_cache_hit" in bucket
            assert "input_cache_miss" in bucket
            assert "output" in bucket
            # 空闲价 = 高峰价的一半（V4 文档明示）
            if model == "deepseek-flash":
                assert buckets["off_peak"]["input_cache_hit"] == buckets["peak"]["input_cache_hit"] / 2
                assert buckets["off_peak"]["input_cache_miss"] == buckets["peak"]["input_cache_miss"] / 2
                assert buckets["off_peak"]["output"] == buckets["peak"]["output"] / 2
