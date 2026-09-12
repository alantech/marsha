"""Role-driven model auto-matching for OpenAI-compatible backends.

Given the models a backend actually serves (discovered via /models), pick the closest
match for each marsha model role. Context size is the reliable cross-backend signal and
drives the choice; a known price (from the stats.py table) only breaks ties, and an
unknown price is neutral. Roles are profiled in ROLE_PROFILES, so adding a role is a
config change, not a code change.
"""

from marsha.config import (
    resolve_model,
    resolve_strong_model,
    set_cli_model,
    set_cli_strong_model,
    model_is_pinned,
    strong_model_is_pinned,
)
from marsha.stats import price_for, price_known

# Smallest context window (tokens) the standard role considers "still fits". The standard
# role lands on the cheapest, smallest model that clears this bar; below it marsha still
# works but compacts prompts more aggressively.
STANDARD_TARGET_CONTEXT = 65_536

# Per-role target profiles for the ranker:
#   strategy: 'smallest-fitting' -> the smallest context that is still >= target_context
#             (the cheapest, smallest model that still fits); 'largest' -> the largest
#             context, i.e. the most capable model.
#   target_context: the context floor for 'smallest-fitting' roles (ignored by 'largest').
#   price_weight: 0 disables the price tie-break; any positive value enables it. Context is
#             the primary signal; price is a tiebreaker at best.
ROLE_PROFILES = {
    'model': {
        'strategy': 'smallest-fitting',
        'target_context': STANDARD_TARGET_CONTEXT,
        'price_weight': 1.0,
    },
    'model_strong': {
        'strategy': 'largest',
        'target_context': None,
        'price_weight': 1.0,
    },
}

# The roles auto-matching resolves, in display order. Adding a role is an entry here
# (with a profile in ROLE_PROFILES), not new code.
_ROLES = (
    {
        'label': 'model',
        'role': 'model',
        'resolver': resolve_model,
        'setter': set_cli_model,
        'pinned': model_is_pinned,
    },
    {
        'label': 'strong model',
        'role': 'model_strong',
        'resolver': resolve_strong_model,
        'setter': set_cli_strong_model,
        'pinned': strong_model_is_pinned,
    },
)


def _price_key(model_id, price_weight):
    # Price tie-break key: known prices sort cheapest first; an unknown price sorts after
    # any known price (it is neutral, never a winner) and ties among unknowns keep list
    # order. Returns an empty key when the profile disables the price signal.
    if price_weight <= 0:
        return ()
    if model_id and price_known(model_id):
        in_price, out_price = price_for(model_id)
        return (0, in_price + out_price)
    return (1, 0.0)


def rank_models(models, role, profile=None):
    """Pick the closest match for `role` from a list of discovered model descriptors
    ({'id': <name>, 'context': <tokens or None>}). Returns the winning model id, or None
    if there is nothing to pick. Context size drives the choice per the role's profile
    (ROLE_PROFILES, or an explicit `profile`); a known price only breaks ties."""
    profile = profile or ROLE_PROFILES.get(role)
    if not models or profile is None:
        return None
    weight = profile.get('price_weight', 0)
    if profile.get('strategy') == 'smallest-fitting':
        target = profile.get('target_context')
        fitting = [
            m for m in models
            if target is None or (m.get('context') or 0) >= target
        ]
        if fitting:
            # The smallest context that still fits the target wins.
            descending = False
        else:
            # Nothing clears the target: best effort is the largest context available.
            fitting = list(models)
            descending = True

        def key(m, i):
            ctx = m.get('context')
            if ctx is None:
                # An unknown context cannot be verified, so it sorts last.
                ctx_part = (1, 0)
            elif descending:
                ctx_part = (0, -ctx)
            else:
                ctx_part = (0, ctx)
            return (ctx_part, _price_key(m.get('id'), weight), i)

        return min(enumerate(fitting), key=lambda p: key(p[1], p[0]))[1]['id']
    else:
        # 'largest': the largest context (most capable) wins, ties broken by price, and an
        # unknown context sorts last (with no signal at all, list order decides).

        def key(m, i):
            ctx = m.get('context')
            ctx_part = (0, -ctx) if ctx is not None else (1, 0)
            return (ctx_part, _price_key(m.get('id'), weight), i)

        return min(enumerate(models), key=lambda p: key(p[1], p[0]))[1]['id']


def apply_available_models(models):
    """Given the models actually served by an (OpenAI-compatible, e.g. local) backend — a
    list of {'id', 'context'} descriptors, or None when discovery failed — remap the
    standard and strong models to the closest available match when the configured model
    isn't served, so on a multi-model backend the two roles may resolve to different
    models. A local server runs whatever is loaded and ignores the requested model name,
    so this makes marsha log and send the model that will actually be used. A pinned
    model (an explicit --model / model_strong choice) is never remapped. Returns a list
    of human-readable notes for each decision (empty if nothing changed)."""
    if not models:
        return []
    available = [m['id'] for m in models]
    notes = []
    for role in _ROLES:
        current = role['resolver']()
        if role['pinned']():
            if current not in available:
                notes.append(
                    f"{role['label']} {current!r} is pinned but not served by the "
                    f'backend; sending it as-is')
            continue
        if current in available:
            continue
        chosen = rank_models(models, role['role'])
        if chosen:
            role['setter'](chosen)
            notes.append(
                f"{role['label']} {current!r} is not served by the backend; "
                f'using {chosen!r}')
    return notes
