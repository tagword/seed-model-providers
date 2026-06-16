"""Tests for proxy-bypass helpers in seed_model_providers.net."""

from __future__ import annotations

import pytest

from seed_model_providers import (
    httpx_trust_env_for,
    is_local_host,
    is_local_url,
    requests_proxies_for,
)


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "127.5.0.1", "::1", "[::1]", "0.0.0.0", "app.localhost", "", None],
)
def test_local_hosts(host):
    assert is_local_host(host) is True


@pytest.mark.parametrize("host", ["example.com", "api.openai.com", "10.0.0.5", "192.168.1.2"])
def test_remote_hosts(host):
    assert is_local_host(host) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1/chat/completions",
        "http://localhost:8000/sse",
        "https://[::1]:443/v1",
        "127.0.0.1:1234",  # bare host:port without scheme
    ],
)
def test_local_urls(url):
    assert is_local_url(url) is True
    assert requests_proxies_for(url) == {"http": None, "https": None}
    assert httpx_trust_env_for(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1/chat/completions",
        "https://api.deepseek.com/v1",
        "http://10.0.0.5:8080/sse",
    ],
)
def test_remote_urls(url):
    assert is_local_url(url) is False
    assert requests_proxies_for(url) is None
    assert httpx_trust_env_for(url) is True


def test_empty_url_is_not_local():
    # No URL → leave default proxy behaviour untouched.
    assert is_local_url("") is False
    assert is_local_url(None) is False
    assert requests_proxies_for(None) is None
