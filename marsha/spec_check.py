"""Shared spec-completeness analysis: the "spec sanity check" that `compile` runs as its
first stage and that `refine --check` runs headless. Given a spec (a `.mrsh` assignment, a
GitHub issue, or a Linear ticket), it decides whether the spec is implementable as written and
enumerates the significant ambiguities. This is the single source of truth for both call sites
so the analysis never diverges. Kept in its own module (rather than `refine.py`) because
`llm.py` also calls it and `refine.py` imports `review.py`, which imports `llm.py`.
"""
from __future__ import annotations

import json
from typing import Any

from marsha import tools
from marsha.config import resolve_model, resolve_provider
from marsha.log import log
from marsha.mappers import get_mapper
from marsha.mappers.chatgpt import uses_completion_tokens

# A bounded round budget for the codebase-grounded analysis (the read-only tool loop): enough to
# probe the codebase before answering, not so many that a wandering model exhausts it first.
SPEC_CHECK_MAX_TOOL_ROUNDS = 25

SPEC_CHECK_PROMPT = '''You are a senior software engineer assessing whether a specification is complete enough to implement. The specification may be a structured assignment (for example a `.mrsh` listing functions with their inputs, outputs, descriptions, and usage examples), a GitHub issue, or a Linear ticket — each with a title and a body/description and possibly comments.

First, decide whether the specification is implementable as written. Use this test: could at least one implementation exist that satisfies every part of the specification (its description, its inputs, its outputs, and all of its examples) at the same time? If such an implementation could exist, it is implementable. Underspecification is not a reason it is not implementable: whatever the specification leaves open is for the implementer to decide reasonably. If the description allows several outcomes (several valid orderings, several equivalent error messages, several formats) and the examples show one of them, an implementation that follows the examples satisfies the specification. One section adding more detail than another is not a contradiction: sections only conflict when they state opposing views on what the software should do. The specification is not implementable only when no implementation could satisfy it as written, for example the description says one thing while an example shows the opposite, two examples give different outputs for the same input, or an example is malformed or violates a stated requirement.

Second, list the significant ambiguities. An ambiguity is an underspecified or unclear area that could lead two competent implementers to build something that behaves differently, for example a missing exception type or message, missing edge cases, an ambiguous output precision or format, unclear error or failure behavior, or non-deterministic behavior that would make the result flaky. Be conservative: only list an ambiguity when it is significant enough that two reasonable implementers could plausibly produce different behavior. Do not list style preferences, and do not ask for more examples or more precision in areas that are merely unspecified but unlikely to change behavior.

Respond with a single JSON object and nothing else, in exactly this shape:
{"compilable": true, "ambiguities": ["...", "..."]}
When the specification is not implementable, include a third key, an "errors" array with one or more entries:
{"compilable": false, "ambiguities": ["..."], "errors": ["...", "..."]}
Each ambiguity and each error is a markdown-formatted string that cites the relevant part of the specification using short inline quotes of the specification's own words. Each error must quote the parts that conflict and explain why no implementation could satisfy both.
Do not wrap the JSON object in code fences.
'''

# Appended (only) when the analysis is grounded in a codebase: the spec belongs to a repository
# the model can inspect, and a term the codebase already settles is not an ambiguity.
SPEC_CHECK_GROUNDED_NOTE = '''
The specification belongs to a codebase you can inspect with the read-only tools below (git, file reads, and the web). Before listing an ambiguity, check whether the point is already settled by the codebase: a term, type, or behavior that the codebase already defines or implements is NOT an ambiguity, so do not list it. For example, if the specification refers to a concept the codebase already has, that concept is understood, not underspecified. Use the tools to confirm before you flag something, and list only what remains genuinely ambiguous once the codebase context is taken into account.
'''


def parse_spec_check(text: str) -> dict[str, Any]:
    """Parse the structured spec-check response; raise on anything malformed."""
    t = text.strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1] if '\n' in t else ''
        if t.rstrip().endswith('```'):
            t = t.rstrip()[:-3]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find('{'), t.rfind('}')
        if start == -1 or end <= start:
            raise Exception(
                f'No JSON object in spec check response: {text[:200]}')
        obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict) or not isinstance(obj.get('compilable'), bool):
        raise Exception(f'Invalid spec check response: {text[:200]}')
    ambiguities = obj.get('ambiguities', [])
    errors = obj.get('errors', [])
    if not isinstance(ambiguities, list) or not all(
            isinstance(a, str) for a in ambiguities):
        raise Exception(f'Invalid spec check response: {text[:200]}')
    if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
        raise Exception(f'Invalid spec check response: {text[:200]}')
    if not obj['compilable'] and len(errors) == 0:
        raise Exception(f'Not implementable without errors: {text[:200]}')
    return {'compilable': obj['compilable'],
            'ambiguities': ambiguities, 'errors': errors}


def _spec_check_kwargs() -> dict[str, Any]:
    # Reasoning models need a larger output budget for their chain of thought. GPT-6 has no
    # 'minimal' tier (its lowest is 'none' = no reasoning); 'low' is the cheapest tier that
    # still reasons. Local OpenAI-compatible servers: leave the output budget to the server.
    if resolve_provider() == 'openai' and uses_completion_tokens(resolve_model()):
        return {'max_tokens': 8192, 'reasoning_effort': 'low'}
    if resolve_provider() == 'anthropic':
        return {'max_tokens': 4096}
    return {}


async def analyze_spec(spec_text: str, *, tool_ctx: 'tools.ToolContext | None' = None,
                       debug: bool = False, retries: int = 2,
                       stats_stage: str | None = None) -> dict[str, Any]:
    """Run the spec-completeness analysis and return {compilable, ambiguities, errors}.

    With no tool context the spec is analyzed on its own (a bare `.mrsh` assignment or the
    compile first stage). With a tool context (refining a GitHub issue or Linear ticket inside
    its repository) the analysis is grounded in the codebase: the read-only tool loop lets the
    model inspect the repository before deciding what is genuinely ambiguous. A malformed
    response is retried up to `retries` more times.
    """
    system = SPEC_CHECK_PROMPT
    if tool_ctx is not None:
        system += SPEC_CHECK_GROUNDED_NOTE
        system += tools.tool_instructions(tool_ctx)
    mapper = get_mapper(system, n_results=1, stats_stage=stats_stage,
                        label='spec-check', **_spec_check_kwargs())
    try:
        if tool_ctx is not None:
            text = await tools.run_with_tools(
                mapper, spec_text, tool_ctx, debug=debug,
                max_rounds=SPEC_CHECK_MAX_TOOL_ROUNDS)
        else:
            text = await mapper.run(spec_text)
        return parse_spec_check(text)
    except Exception:
        if retries > 0:
            log(f'spec check: malformed or failed response; retrying '
                f'({retries} left)')
            return await analyze_spec(
                spec_text, tool_ctx=tool_ctx, debug=debug, retries=retries - 1,
                stats_stage=stats_stage)
        raise
