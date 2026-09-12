import json
import urllib.request

from marsha.config import (
    DEFAULT_API_BASE,
    resolve_api_base,
    resolve_model,
    resolve_provider,
)

# Conservative chars->tokens estimate. We overestimate on purpose (real text is closer to
# 4 chars/token) so the budget gate compacts early rather than risk overflowing.
CHARS_PER_TOKEN = 3
DEFAULT_CONTEXT_CAP = 0.5
DEFAULT_CONTEXT_WINDOW = 200_000

# Documented context windows for models whose API does not expose the value (Anthropic reports
# only the max OUTPUT tokens). Keyed by longest-prefix model name.
_KNOWN_CONTEXT = {
    'claude': 200_000,
    'gpt-5': 400_000,
    'o1': 200_000,
    'o3': 200_000,
    'o4': 200_000,
}

_cache = {}


def estimate_tokens(text):
    # Rough token count from length. Deliberately conservative (see CHARS_PER_TOKEN).
    return max(len(text or '') // CHARS_PER_TOKEN, 1)


def budget_tokens(context_window, cap=DEFAULT_CONTEXT_CAP):
    # The fraction of the context window we allow a single prompt to occupy, leaving room for
    # the model's output and a safety margin.
    return int(context_window * cap)


def fits(prompt_text, context_window, cap=DEFAULT_CONTEXT_CAP):
    return estimate_tokens(prompt_text) <= budget_tokens(context_window, cap)


def _known_context(model):
    for prefix, window in sorted(_KNOWN_CONTEXT.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(prefix):
            return window
    return DEFAULT_CONTEXT_WINDOW


_models_cache = {}


def _fetch_models_data(api_base):
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
    arr = data.get('data') or data.get('models') or []
    _models_cache[api_base] = arr
    return arr


def _model_id(m):
    return m.get('id') or m.get('name')


def _model_context(m):
    # The context size (in tokens) a /models entry exposes, or None. llama.cpp nests it under
    # meta.n_ctx; some servers use details.n_ctx; newer OpenAI-style APIs expose context_window.
    nctx = (m.get('meta') or {}).get('n_ctx') or (
        m.get('details') or {}).get('n_ctx')
    if nctx:
        return int(nctx)
    cw = m.get('context_window')
    return int(cw) if cw else None


def discover_models(api_base):
    """Return the models served by an OpenAI-compatible backend as a list of descriptors
    {'id': <name>, 'context': <context window in tokens, or None>}, or None if the endpoint is
    unavailable. Used to log (and, on local servers, remap to) the model that will actually be
    used, since a local server serves whatever is loaded rather than a named model."""
    data = _fetch_models_data(api_base)
    if not data:
        return None
    models = []
    for m in data:
        mid = _model_id(m)
        if mid:
            models.append({'id': mid, 'context': _model_context(m)})
    return models


def _query_llama_context(api_base, model=None):
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


async def _query_openai_context(client, model):
    try:
        m = await client.models.retrieve(model)
    except Exception:
        return None
    cw = getattr(m, 'context_window', None)
    if cw:
        return int(cw)
    return (getattr(m, 'model_extra', None) or {}).get('context_window')


async def resolve_context_window(model=None, provider=None, api_base=None, client=None,
                                 override=None):
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
    window = None
    if provider == 'openai':
        if api_base != DEFAULT_API_BASE:
            window = _query_llama_context(api_base, model)
        if window is None and client is not None:
            window = await _query_openai_context(client, model)
    if window is None:
        window = _known_context(model)
    _cache[key] = int(window)
    return _cache[key]


def reset_cache():
    # For tests.
    global _cache
    _cache = {}
    _models_cache.clear()
