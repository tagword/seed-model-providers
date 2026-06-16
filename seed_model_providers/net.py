"""Network helpers shared across seed / seed-tools / codeagent.

The key concern here is **proxy handling**. Both ``requests`` and ``httpx``
pick up the operating-system proxy configuration by default (on macOS via
``urllib.request.getproxies()`` which reads the SystemConfiguration proxy
settings, on Linux/Windows via the ``HTTP_PROXY`` / ``ALL_PROXY`` env vars).

That is desirable for remote API calls (many users in restricted networks rely
on a proxy to reach OpenAI/Anthropic), but it is actively harmful for requests
to **local** endpoints such as a local Ollama / SGLang server or a localhost
MCP server: the proxy receives a request for ``127.0.0.1`` and either refuses
it or loops back to itself, producing confusing ``500`` / connection errors.

These helpers let callers keep the proxy for remote hosts while transparently
bypassing it for loopback / local addresses.
"""

from __future__ import annotations

from typing import Dict, Optional
from urllib.parse import urlsplit

__all__ = ["is_local_url", "is_local_host", "requests_proxies_for", "httpx_trust_env_for"]

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


def is_local_host(host: Optional[str]) -> bool:
    """Return True if *host* is a loopback / local address."""
    if not host:
        return True
    h = host.strip().lower()
    # Strip IPv6 brackets, e.g. "[::1]".
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if h in _LOCAL_HOSTS:
        return True
    # *.localhost is reserved for loopback (RFC 6761).
    if h.endswith(".localhost"):
        return True
    # Link-local / private loopback ranges that are never reachable via a proxy.
    if h.startswith("127."):
        return True
    return False


def is_local_url(url: Optional[str]) -> bool:
    """Return True if *url* points at a loopback / local address."""
    if not url:
        return False
    try:
        # urlsplit needs a scheme to populate ``hostname``; tolerate bare hosts.
        parsed = urlsplit(url if "://" in url else f"http://{url}")
    except (ValueError, TypeError):
        return False
    return is_local_host(parsed.hostname)


def requests_proxies_for(url: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """Proxy mapping to pass to ``requests`` for *url*.

    Returns ``{"http": None, "https": None}`` to disable proxying for local
    URLs, or ``None`` to let ``requests`` use its default (env/system) proxies
    for remote URLs.
    """
    if is_local_url(url):
        return {"http": None, "https": None}
    return None


def httpx_trust_env_for(url: Optional[str]) -> bool:
    """``trust_env`` value to pass to ``httpx.Client`` for *url*.

    ``False`` for local URLs (ignore env/system proxy), ``True`` otherwise.
    """
    return not is_local_url(url)
