"""Chat request parameter compatibility across providers.

Two independent classes of incompatibility are handled here:

1. ``max_tokens`` vs ``max_completion_tokens``
   OpenAI reasoning families (o1/o3/o4, gpt-5+ including Azure aliases such as
   ``azure/gpt-5.6-luna`` and gateway names such as ``gpt-6-luna``) reject
   ``max_tokens`` with ``unsupported_parameter`` and require
   ``max_completion_tokens``. Older models keep using ``max_tokens``.

2. Sampling parameters unsupported by the inference engine
   DeepSeek's NInfer engine only accepts ``repetition_penalty == 1``; any other
   value raises ``repetition_penalty_not_supported``. In general a provider may
   reject ``temperature`` / ``top_p`` / ``top_k`` for reasoning models. These
   parameters are dropped on demand and the request is retried once.

Design notes:
- Static model-name matching avoids a guaranteed-to-fail first request.
- A process-local capability cache records what the server accepted/rejected so
  a model is probed at most once per process.
- Only *parameter* errors trigger a retry; auth/quota/network errors never do.
- 400/422 responses are free (no tokens billed), so one corrective retry is
  cheaper than shipping a broken request.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TOKENS = "max_tokens"
MAX_COMPLETION_TOKENS = "max_completion_tokens"

# (normalized base_url, model basename) -> parameter name accepted by the server.
_MAX_TOKENS_PARAM_CACHE: Dict[tuple[str, str], str] = {}

# OpenAI reasoning family: o1/o3/o4/o5..., gpt-5/6/7..., incl. point releases
# such as ``gpt-5.6-luna``. ``gpt-4.1`` must NOT match.
_OPENAI_REASONING_MODEL_RE = re.compile(
    r"^(?:o[1-9](?:$|[-_.])|gpt-(?:[5-9])(?:$|[-_.]))",
    re.IGNORECASE,
)

# Samplers that reasoning models commonly reject.
_DROPPABLE_SAMPLERS = ("temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty")

_ERR_NOT_SUPPORTED = ("not supported", "unsupported", "not_supported", "does not support")


def _norm_base_url(base_url: Optional[str]) -> str:
    return (base_url or "").strip().rstrip("/").lower()


def model_basename(model: Optional[str]) -> str:
    """Return the model name without a routing prefix (``azure/gpt-5.6-luna`` → ``gpt-5.6-luna``)."""
    raw = (model or "").strip()
    return raw.rsplit("/", 1)[-1] if "/" in raw else raw


def looks_like_openai_reasoning_model(model: Optional[str]) -> bool:
    return bool(_OPENAI_REASONING_MODEL_RE.match(model_basename(model)))


def _cache_key(base_url: Optional[str], model: Optional[str]) -> tuple[str, str]:
    return (_norm_base_url(base_url), model_basename(model).lower())


def max_tokens_param_for(
    model: Optional[str], *, provider: Optional[str] = None, base_url: Optional[str] = None
) -> str:
    """Return the completion-budget parameter name for ``model``.

    Learned server behaviour wins over the static model-name heuristic.
    """
    cached = _MAX_TOKENS_PARAM_CACHE.get(_cache_key(base_url, model))
    if cached:
        return cached
    if provider == "minimax_anthropic":
        return MAX_TOKENS
    return MAX_COMPLETION_TOKENS if looks_like_openai_reasoning_model(model) else MAX_TOKENS


def note_max_tokens_param(
    model: Optional[str], param_name: str, *, base_url: Optional[str] = None
) -> None:
    """Record which parameter name the server actually accepted."""
    if param_name in (MAX_TOKENS, MAX_COMPLETION_TOKENS):
        _MAX_TOKENS_PARAM_CACHE[_cache_key(base_url, model)] = param_name


def reset_param_cache() -> None:
    """Clear learned capabilities (tests / provider switch)."""
    _MAX_TOKENS_PARAM_CACHE.clear()


def apply_max_tokens_param(
    params: Dict[str, Any],
    *,
    model: Optional[str],
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Rename the completion-budget key in ``params`` to the supported one."""
    target = max_tokens_param_for(model, provider=provider, base_url=base_url)
    if target == MAX_COMPLETION_TOKENS:
        if MAX_TOKENS in params:
            params[MAX_COMPLETION_TOKENS] = params.pop(MAX_TOKENS)
    elif MAX_COMPLETION_TOKENS in params and MAX_TOKENS not in params:
        params[MAX_TOKENS] = params.pop(MAX_COMPLETION_TOKENS)


def get_max_tokens(params: Dict[str, Any], default: int = 0) -> int:
    """Read the completion budget regardless of which key is in use."""
    for key in (MAX_COMPLETION_TOKENS, MAX_TOKENS):
        value = params.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
    return default


def set_max_tokens(params: Dict[str, Any], value: int) -> None:
    """Update whichever completion-budget key is present."""
    if MAX_COMPLETION_TOKENS in params:
        params[MAX_COMPLETION_TOKENS] = int(value)
    else:
        params[MAX_TOKENS] = int(value)


def _rejects_non_neutral_repetition_penalty(
    *, provider: Optional[str], base_url: Optional[str], model: Optional[str]
) -> bool:
    """DeepSeek's NInfer engine only accepts ``repetition_penalty == 1``."""
    prov = (provider or "").lower()
    if prov in ("deepseek",):
        return True
    if "api.deepseek.com" in _norm_base_url(base_url):
        return True
    return "deepseek" in model_basename(model).lower()


def sanitize_chat_params(
    params: Dict[str, Any],
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> List[str]:
    """Drop parameters known to be rejected by the target provider.

    Returns human-readable descriptions of the changes (for logging).
    """
    changes: List[str] = []
    # Only the YAML/extra-body path can add repetition_penalty today, but guard
    # anyway so a stale preset cannot break every request.
    rp = params.get("repetition_penalty")
    if rp is not None and _rejects_non_neutral_repetition_penalty(
        provider=provider, base_url=base_url, model=model
    ):
        try:
            neutral = float(rp) == 1.0
        except (TypeError, ValueError):
            neutral = False
        if not neutral:
            params.pop("repetition_penalty", None)
            changes.append(f"drop repetition_penalty={rp} (engine accepts only 1)")
    return changes


def is_param_error(status: int, body: str) -> bool:
    """True when a request failed because of an unsupported/invalid *parameter*."""
    if status not in (400, 422):
        return False
    text = (body or "").lower()
    if not text:
        return False
    if "unsupported_parameter" in text:
        return True
    if any(k in text for k in ("invalid_request_error", "not supported", "unsupported")):
        return any(
            k in text
            for k in _DROPPABLE_SAMPLERS
            + ("max_tokens", "max_completion_tokens", "repetition_penalty")
        )
    return False


def fix_params_from_error(
    params: Dict[str, Any], status: int, body: str, *, base_url: Optional[str] = None
) -> List[str]:
    """Mutate ``params`` to correct a known parameter incompatibility.

    Returns descriptions of applied fixes. An empty list means the error could
    not be attributed to a fixable parameter (caller must not retry blindly).
    """
    if not is_param_error(status, body):
        return []

    text = body or ""
    low = text.lower()
    fixes: List[str] = []

    # --- completion budget key -------------------------------------------
    if MAX_TOKENS in params and ("use 'max_completion_tokens'" in low or "max_completion_tokens" in low):
        params[MAX_COMPLETION_TOKENS] = params.pop(MAX_TOKENS)
        fixes.append("max_tokens -> max_completion_tokens")
        note_max_tokens_param(str(params.get("model") or ""), MAX_COMPLETION_TOKENS, base_url=base_url)
    elif MAX_COMPLETION_TOKENS in params and (
        "use 'max_tokens'" in low
        or ("not supported" in low and "max_completion_tokens" in low)
    ):
        params[MAX_TOKENS] = params.pop(MAX_COMPLETION_TOKENS)
        fixes.append("max_completion_tokens -> max_tokens")
        note_max_tokens_param(str(params.get("model") or ""), MAX_TOKENS, base_url=base_url)

    # --- repetition penalty ----------------------------------------------
    if "repetition_penalty" in low and params.pop("repetition_penalty", None) is not None:
        fixes.append("drop repetition_penalty")

    # --- samplers rejected by reasoning models ---------------------------
    if any(k in low for k in _ERR_NOT_SUPPORTED):
        for key in _DROPPABLE_SAMPLERS:
            if key in low and key in params:
                params.pop(key, None)
                fixes.append(f"drop {key}")

    return fixes
