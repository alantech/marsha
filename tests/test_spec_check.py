"""Deterministic tests for the shared spec-completeness analysis (marsha.spec_check).

`parse_spec_check` is pure (strict structured-output parsing that the retry loop depends on),
and `analyze_spec` is exercised with a mocked mapper / tool loop so no LLM or network is needed.
"""

import asyncio
import json
from typing import Any

from unittest.mock import AsyncMock, patch

import pytest

from marsha import spec_check
from marsha import tools


class _ScriptedMapper:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.i = 0

    async def run(self, messages: Any) -> str:
        i = self.i
        self.i += 1
        if i < len(self.responses):
            return self.responses[i]
        return self.responses[-1]


def test_prompts_are_nonempty_and_grounded_note_mentions_codebase() -> None:
    assert spec_check.SPEC_CHECK_PROMPT.strip()
    assert 'implementable' in spec_check.SPEC_CHECK_PROMPT
    assert 'ambiguities' in spec_check.SPEC_CHECK_PROMPT
    assert spec_check.SPEC_CHECK_GROUNDED_NOTE.strip()
    assert 'codebase' in spec_check.SPEC_CHECK_GROUNDED_NOTE


# --- parse_spec_check (strict, retry-friendly) --------------------------------


def test_parse_spec_check_compilable() -> None:
    text = '{"compilable": true, "ambiguities": ["a1", "a2"]}'
    assert spec_check.parse_spec_check(text) == {
        'compilable': True, 'ambiguities': ['a1', 'a2'], 'errors': []}


def test_parse_spec_check_not_compilable_carries_errors() -> None:
    text = '{"compilable": false, "ambiguities": ["a"], "errors": ["e1", "e2"]}'
    got = spec_check.parse_spec_check(text)
    assert got['compilable'] is False
    assert got['ambiguities'] == ['a']
    assert got['errors'] == ['e1', 'e2']


def test_parse_spec_check_strips_code_fences() -> None:
    text = '```\n{"compilable": true, "ambiguities": []}\n```'
    assert spec_check.parse_spec_check(text) == {
        'compilable': True, 'ambiguities': [], 'errors': []}


def test_parse_spec_check_extracts_embedded_object() -> None:
    text = 'Here is the analysis:\n{"compilable": true, "ambiguities": ["x"]} thanks'
    got = spec_check.parse_spec_check(text)
    assert got['compilable'] is True and got['ambiguities'] == ['x']


def test_parse_spec_check_missing_compilable_raises() -> None:
    with pytest.raises(Exception):
        spec_check.parse_spec_check('{"ambiguities": []}')


def test_parse_spec_check_missing_ambiguities_raises() -> None:
    # A missing `ambiguities` field is malformed (the prompt requires it), not an empty list:
    # it must raise so the retry path runs instead of reading as a locked spec.
    with pytest.raises(Exception):
        spec_check.parse_spec_check('{"compilable": true}')


def test_parse_spec_check_non_dict_raises() -> None:
    with pytest.raises(Exception):
        spec_check.parse_spec_check('[1, 2, 3]')


def test_parse_spec_check_not_compilable_without_errors_raises() -> None:
    with pytest.raises(Exception):
        spec_check.parse_spec_check('{"compilable": false, "ambiguities": []}')


def test_parse_spec_check_bad_ambiguity_type_raises() -> None:
    with pytest.raises(Exception):
        spec_check.parse_spec_check('{"compilable": true, "ambiguities": [1]}')


def test_parse_spec_check_bad_error_type_raises() -> None:
    with pytest.raises(Exception):
        spec_check.parse_spec_check(
            '{"compilable": false, "ambiguities": [], "errors": [3]}')


def test_parse_spec_check_no_object_raises() -> None:
    with pytest.raises(Exception):
        spec_check.parse_spec_check('no json here at all')


# --- analyze_spec (mocked mapper / tool loop) ---------------------------------


def test_analyze_spec_bare_uses_mapper_not_tools() -> None:
    out = json.dumps({'compilable': True, 'ambiguities': ['a']})
    captured: dict[str, str] = {}

    def get_mapper(system: str, **kw: Any) -> Any:
        captured['system'] = system
        return _ScriptedMapper([out])

    with patch.object(spec_check, 'get_mapper', new=get_mapper), \
         patch.object(tools, 'run_with_tools', new=AsyncMock()) as rwt:
        got = asyncio.run(spec_check.analyze_spec('SPEC'))
    assert got == {'compilable': True, 'ambiguities': ['a'], 'errors': []}
    # A bare (ungrounded) analysis must not append the grounded note.
    assert spec_check.SPEC_CHECK_GROUNDED_NOTE not in captured['system']
    rwt.assert_not_awaited()


def test_analyze_spec_grounded_appends_note_and_uses_tools() -> None:
    out = json.dumps({'compilable': True, 'ambiguities': []})
    captured: dict[str, str] = {}

    def get_mapper(system: str, **kw: Any) -> Any:
        captured['system'] = system
        return _ScriptedMapper([out])

    async def fake_rwt(mapper: Any, request: str, ctx: Any, debug: bool = False,
                       max_rounds: int = 99) -> str:
        return out

    ctx = tools.ToolContext(phase='refine', workdir='.')
    with patch.object(spec_check, 'get_mapper', new=get_mapper), \
         patch.object(tools, 'run_with_tools', new=fake_rwt):
        got = asyncio.run(spec_check.analyze_spec('SPEC', tool_ctx=ctx))
    assert got == {'compilable': True, 'ambiguities': [], 'errors': []}
    # A grounded analysis appends the note and drives the read-only tool loop.
    assert spec_check.SPEC_CHECK_GROUNDED_NOTE in captured['system']


def test_analyze_spec_retries_then_succeeds() -> None:
    state = {'n': 0}

    def get_mapper(system: str, **kw: Any) -> Any:
        class _M:
            async def run(self, messages: Any) -> str:
                state['n'] += 1
                return 'not json' if state['n'] == 1 else json.dumps(
                    {'compilable': True, 'ambiguities': []})
        return _M()

    with patch.object(spec_check, 'get_mapper', new=get_mapper):
        got = asyncio.run(spec_check.analyze_spec('SPEC', retries=2))
    assert got == {'compilable': True, 'ambiguities': [], 'errors': []}
    assert state['n'] == 2  # one malformed attempt, then the retry


def test_analyze_spec_raises_when_retries_exhausted() -> None:
    def get_mapper(system: str, **kw: Any) -> Any:
        class _M:
            async def run(self, messages: Any) -> str:
                return 'never valid'
        return _M()

    with patch.object(spec_check, 'get_mapper', new=get_mapper):
        with pytest.raises(Exception):
            asyncio.run(spec_check.analyze_spec('SPEC', retries=1))
