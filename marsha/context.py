from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING, Any, TypedDict

import openai

from marsha.config import (
    DEFAULT_API_BASE,
    resolve_api_base,
    resolve_model,
    resolve_provider,
)

if TYPE_CHECKING:
    import anthropic

# Conservative chars->tokens estimate. We overestimate on purpose (real text is closer to
# 4 chars/token) so the budget gate compacts early rather than risk overflowing.
CHARS_PER_TOKEN = 3
DEFAULT_CONTEXT_CAP = 0.5
DEFAULT_CONTEXT_WINDOW = 200_000

# Documented context windows for models whose API does not expose the value (Anthropic reports
# only the max OUTPUT tokens). Keyed by longest-prefix model name.
_KNOWN_CONTEXT: dict[str, int] = {
    'claude': 200_000,
    'gpt-6': 1_050_000,
    'gpt-5.6': 1_050_000,
    'gpt-5': 400_000,
    'o1': 200_000,
    'o3': 200_000,
    'o4': 200_000,
}

_cache: dict[tuple[str, str, str], int] = {}

# A model descriptor as marsha normalizes it from a backend's /models entry.


class ModelDescriptor(TypedDict):
    id: str
    context: int | None


def estimate_tokens(text: str | None) -> int:
    # Rough token count from length. Deliberately conservative (see CHARS_PER_TOKEN).
    return max(len(text or '') // CHARS_PER_TOKEN, 1)


def budget_tokens(context_window: int, cap: float = DEFAULT_CONTEXT_CAP) -> int:
    # The fraction of the context window we allow a single prompt to occupy, leaving room for
    # the model's output and a safety margin.
    return int(context_window * cap)


def fits(prompt_text: str, context_window: int, cap: float = DEFAULT_CONTEXT_CAP) -> bool:
    return estimate_tokens(prompt_text) <= budget_tokens(context_window, cap)


def known_context(model: str) -> int:
    for prefix, window in sorted(_KNOWN_CONTEXT.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(prefix):
            return window
    return DEFAULT_CONTEXT_WINDOW


_models_cache: dict[str, list[dict[str, Any]] | None] = {}


def _fetch_models_data(api_base: str) -> list[dict[str, Any]] | None:
    # Fetch (and cache) the raw model list from an OpenAI-compatible backend's /models endpoint.
    # Returns the list of model dicts, or None if the endpoint is unreachable or malformed.
    if api_base in _models_cache:
        return _models_cache[api_base]
    url = api_base.rstrip('/') + '/models'
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
    except Exception:
        _models_cache[api_base] = None
        return None
    arr: list[dict[str, Any]] = data.get('data') or data.get('models') or []
    _models_cache[api_base] = arr
    return arr


def _model_id(m: dict[str, Any]) -> str | None:
    mid = m.get('id') or m.get('name')
    return mid if isinstance(mid, str) else None


def _model_context(m: dict[str, Any]) -> int | None:
    # The context size (in tokens) a /models entry exposes, or None. llama.cpp nests it under
    # meta.n_ctx; some servers use details.n_ctx; newer OpenAI-style APIs expose context_window.
    nctx = (m.get('meta') or {}).get('n_ctx') or (
        m.get('details') or {}).get('n_ctx')
    if nctx:
        return int(nctx)
    cw = m.get('context_window')
    return int(cw) if cw else None


def discover_models(api_base: str) -> list[ModelDescriptor] | None:
    """Return the models served by an OpenAI-compatible backend as a list of descriptors
    {'id': <name>, 'context': <context window in tokens, or None>}, or None if the endpoint is
    unavailable. Used to log (and, on local servers, remap to) the model that will actually be
    used, since a local server serves whatever is loaded rather than a named model."""
    data = _fetch_models_data(api_base)
    if not data:
        return None
    models: list[ModelDescriptor] = []
    for m in data:
        mid = _model_id(m)
        if mid:
            models.append({'id': mid, 'context': _model_context(m)})
    return models


def _query_llama_context(api_base: str, model: str | None = None) -> int | None:
    # llama.cpp (and most OpenAI-compatible servers) expose the context size at /models.
    # llama.cpp nests it under data[].meta.n_ctx; other servers may use data[].details.n_ctx.
    # With a model name, look it up by id (multi-model backends); otherwise fall back to the
    # first entry that reports one (the single-model case).
    data = _fetch_models_data(api_base)
    if not data:
        return None
    if model is not None:
        for m in data:
            if _model_id(m) == model:
                ctx = _model_context(m)
                if ctx:
                    return ctx
    for m in data:
        ctx = _model_context(m)
        if ctx:
            return ctx
    return None


async def _query_openai_context(client: openai.AsyncOpenAI, model: str) -> int | None:
    try:
        m = await client.models.retrieve(model)
    except Exception:
        return None
    cw = getattr(m, 'context_window', None)
    if cw:
        return int(cw)
    extra = getattr(m, 'model_extra', None) or {}
    cw2 = extra.get('context_window')
    return int(cw2) if cw2 else None


async def resolve_context_window(model: str | None = None, provider: str | None = None,
                                 api_base: str | None = None,
                                 client: openai.AsyncOpenAI | anthropic.AsyncAnthropic | None = None,
                                 override: int | None = None) -> int:
    # Resolve the context window (in tokens) for the given backend/model, preferring a value
    # fetched from the service and falling back to documented defaults. Cached per
    # (provider, base, model) since it is stable for a run.
    if override:
        return int(override)
    model = model or resolve_model()
    provider = provider or resolve_provider()
    api_base = api_base or resolve_api_base()
    key = (provider, api_base, model)
    if key in _cache:
        return _cache[key]
    window: int | None = None
    if provider == 'openai':
        if api_base != DEFAULT_API_BASE:
            window = _query_llama_context(api_base, model)
        if window is None and isinstance(client, openai.AsyncOpenAI):
            window = await _query_openai_context(client, model)
    if window is None:
        window = known_context(model)
    _cache[key] = int(window)
    return _cache[key]


def reset_cache() -> None:
    # For tests.
    global _cache
    _cache = {}
    _models_cache.clear()
