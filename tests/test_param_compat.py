"""Tests for chat request parameter compatibility (max_tokens / samplers)."""

import pytest

from seed_model_providers import param_compat
from seed_model_providers.param_compat import (
    MAX_COMPLETION_TOKENS,
    MAX_TOKENS,
    apply_max_tokens_param,
    fix_params_from_error,
    get_max_tokens,
    is_param_error,
    max_tokens_param_for,
    reset_param_cache,
    sanitize_chat_params,
    set_max_tokens,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_param_cache()
    yield
    reset_param_cache()


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-6-luna", MAX_COMPLETION_TOKENS),
        ("azure/gpt-5.6-luna", MAX_COMPLETION_TOKENS),
        ("gpt-5", MAX_COMPLETION_TOKENS),
        ("o1-mini", MAX_COMPLETION_TOKENS),
        ("o3", MAX_COMPLETION_TOKENS),
        ("gpt-4.1", MAX_TOKENS),
        ("gpt-4o", MAX_TOKENS),
        ("deepseek-4.1-flash", MAX_TOKENS),
        ("MiniMax-M3", MAX_TOKENS),
        ("", MAX_TOKENS),
    ],
)
def test_max_tokens_param_for(model, expected):
    assert max_tokens_param_for(model) == expected


def test_apply_renames_to_max_completion_tokens():
    params = {"model": "gpt-6-luna", "max_tokens": 1024}
    apply_max_tokens_param(params, model="gpt-6-luna", base_url="https://gw.example/v1")
    assert params == {"model": "gpt-6-luna", MAX_COMPLETION_TOKENS: 1024}


def test_apply_keeps_max_tokens_for_deepseek():
    params = {"model": "deepseek-4.1-flash", "max_tokens": 1024}
    apply_max_tokens_param(params, model="deepseek-4.1-flash")
    assert params[MAX_TOKENS] == 1024
    assert MAX_COMPLETION_TOKENS not in params


def test_learned_capability_overrides_heuristic():
    # Gateway rejects max_completion_tokens; learning flips it back.
    params = {"model": "gpt-6-luna", MAX_COMPLETION_TOKENS: 512}
    apply_max_tokens_param(params, model="gpt-6-luna", base_url="https://gw.example/v1")
    param_compat.note_max_tokens_param(
        "gpt-6-luna", MAX_TOKENS, base_url="https://gw.example/v1"
    )
    params2 = {"model": "gpt-6-luna", MAX_COMPLETION_TOKENS: 512}
    apply_max_tokens_param(params2, model="gpt-6-luna", base_url="https://gw.example/v1")
    assert params2[MAX_TOKENS] == 512
    assert MAX_COMPLETION_TOKENS not in params2


def test_get_set_max_tokens_handles_both_keys():
    assert get_max_tokens({MAX_COMPLETION_TOKENS: 77}) == 77
    assert get_max_tokens({MAX_TOKENS: 55}) == 55
    assert get_max_tokens({}) == 0

    a = {MAX_COMPLETION_TOKENS: 1}
    set_max_tokens(a, 9)
    assert a[MAX_COMPLETION_TOKENS] == 9

    b = {MAX_TOKENS: 1}
    set_max_tokens(b, 9)
    assert b[MAX_TOKENS] == 9


def test_sanitize_drops_non_neutral_repetition_penalty_on_deepseek():
    params = {"model": "deepseek-4.1-flash", "repetition_penalty": 1.2}
    changes = sanitize_chat_params(params, provider="deepseek", base_url="https://api.deepseek.com/v1")
    assert "repetition_penalty" not in params
    assert changes


def test_sanitize_keeps_neutral_repetition_penalty():
    params = {"repetition_penalty": 1}
    assert sanitize_chat_params(params, provider="deepseek") == []
    assert params["repetition_penalty"] == 1


def test_sanitize_keeps_repetition_penalty_for_other_providers():
    params = {"repetition_penalty": 1.2}
    assert sanitize_chat_params(params, provider="vllm", base_url="http://127.0.0.1:8000/v1") == []
    assert params["repetition_penalty"] == 1.2


def test_fix_max_tokens_error_renames():
    params = {"model": "gpt-6-luna", MAX_TOKENS: 256}
    body = (
        '{"error":{"message":"Unsupported parameter: \'max_tokens\' is not supported '
        'with this model. Use \'max_completion_tokens\' instead.","type":'
        '"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}'
    )
    fixes = fix_params_from_error(params, 400, body, base_url="https://gw.example/v1")
    assert params[MAX_COMPLETION_TOKENS] == 256
    assert MAX_TOKENS not in params
    assert fixes


def test_fix_repetition_penalty_error_drops_param():
    params = {"model": "deepseek-4.1-flash", "repetition_penalty": 1.05}
    body = (
        '{"error":{"code":"repetition_penalty_not_supported","message":'
        '"a non-neutral repetition_penalty requires a sampler transform that '
        'NInfer does not provide; only repetition_penalty=1 is accepted",'
        '"param":"repetition_penalty","type":"invalid_request_error"}}'
    )
    fixes = fix_params_from_error(params, 400, body)
    assert "repetition_penalty" not in params
    assert fixes


def test_fix_drops_sampler_rejected_by_reasoning_model():
    params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 10}
    body = '{"error":{"message":"Unsupported parameter: \'temperature\' is not supported with this model."}}'
    fixes = fix_params_from_error(params, 400, body)
    assert "temperature" not in params
    assert params["top_p"] == 0.9
    assert fixes


def test_is_param_error_ignores_auth_and_server_errors():
    assert not is_param_error(401, '{"error":"invalid api key"}')
    assert not is_param_error(500, '{"error":"boom"}')
    assert not is_param_error(429, '{"error":"rate limit"}')
    assert is_param_error(400, '{"error":{"code":"unsupported_parameter"}}')


def test_fix_max_completion_tokens_rejected_falls_back():
    params = {"model": "legacy-gateway", MAX_COMPLETION_TOKENS: 128}
    body = (
        '{"error":{"message":"Unsupported parameter: \'max_completion_tokens\' is '
        'not supported with this model.","type":"invalid_request_error",'
        '"code":"unsupported_parameter"}}'
    )
    fixes = fix_params_from_error(params, 400, body)
    assert params[MAX_TOKENS] == 128
    assert MAX_COMPLETION_TOKENS not in params
    assert fixes
