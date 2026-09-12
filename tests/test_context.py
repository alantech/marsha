"""Deterministic tests for context-window budgeting and findings compaction.

The LLM and any network I/O (the llama.cpp /models probe, the OpenAI models call) are mocked,
so these verify the token budget gate, the label-preserving compaction parser, and the
deterministic severity trim without a server or LLM non-determinism.
"""

import asyncio
import json
import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import context
from marsha import llm
from marsha import personas
from marsha.personas import parse_compacted_findings


@pytest.fixture(autouse=True)
def _reset_context_caches():
    # The /models probe and the context-window probe cache per backend; clear them so a test's
    # mocked response can't leak into another test that reuses the same api_base.
    context.reset_cache()
    yield
    context.reset_cache()


def _finding(name, label, severity):
    return {'name': name, 'label': label, 'severity': severity, 'location': '', 'desc': 'd'}


# --- pure token / budget helpers -------------------------------------------

def test_estimate_tokens():
    assert context.estimate_tokens('') == 1
    assert context.estimate_tokens('x' * 3000) == 1000


def test_budget_tokens_and_fits():
    assert context.budget_tokens(250000, 0.5) == 125000
    # ~100k tokens <= 125k
    assert context.fits('x' * 300000, 250000, 0.5) is True
    # ~200k tokens > 125k
    assert context.fits('x' * 600000, 250000, 0.5) is False


def test_known_context_fallback():
    assert context.known_context('claude-opus-5') == 200000
    assert context.known_context('gpt-5-mini') == 400000
    assert context.known_context(
        'mystery-model') == context.DEFAULT_CONTEXT_WINDOW


def test_resolve_override_wins():
    async def go():
        return await context.resolve_context_window(override=12345)
    assert asyncio.run(go()) == 12345


# --- llama.cpp /models discovery (mocked urllib) ---------------------------

class _FakeResp:
    # A minimal file-like context manager standing in for urllib's urlopen() return value.
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(payload):
    return patch.object(context.urllib.request, 'urlopen', return_value=_FakeResp(payload))


def test_query_llama_context_meta():
    with _mock_urlopen({"data": [{"id": "m", "meta": {"n_ctx": 250112}}]}):
        assert context._query_llama_context('http://x/v1') == 250112


def test_query_llama_context_details_fallback():
    with _mock_urlopen({"models": [{"details": {"n_ctx": 8192}}]}):
        assert context._query_llama_context('http://x/v1') == 8192


def test_query_llama_context_error_is_none():
    with patch.object(context.urllib.request, 'urlopen', side_effect=Exception('boom')):
        assert context._query_llama_context('http://x/v1') is None


def test_query_llama_context_by_model():
    # On a multi-model backend the context of the requested model wins over the first entry.
    payload = {"data": [
        {"id": "small", "meta": {"n_ctx": 4096}},
        {"id": "large", "meta": {"n_ctx": 131072}},
    ]}
    with _mock_urlopen(payload):
        assert context._query_llama_context(
            'http://x/v1', model='large') == 131072
        # Unknown model falls back to the first entry that reports a context.
        assert context._query_llama_context(
            'http://x/v1', model='missing') == 4096


# --- local-backend model discovery (descriptors with context) --------------

def test_discover_models_descriptors():
    with _mock_urlopen({"data": [
        {"id": "a"},
        {"id": "b", "meta": {"n_ctx": 1000}},
        {"id": "c", "details": {"n_ctx": 2000}},
        {"id": "d", "context_window": 3000},
    ]}):
        assert context.discover_models('http://x/v1') == [
            {'id': 'a', 'context': None},
            {'id': 'b', 'context': 1000},
            {'id': 'c', 'context': 2000},
            {'id': 'd', 'context': 3000},
        ]


def test_discover_models_name_fallback_and_error():
    with _mock_urlopen({"models": [{"name": "only"}]}):
        assert context.discover_models(
            'http://y/v1') == [{'id': 'only', 'context': None}]
    with patch.object(context.urllib.request, 'urlopen', side_effect=Exception('boom')):
        assert context.discover_models('http://z/v1') is None


# --- label-preserving compaction parser ------------------------------------

def test_parse_compacted_preserves_labels_and_gaps():
    text = ("- [Sage-A3] MAJOR x.py:10 - the spec requires X\n"
            "- [Vera-B1] MINOR - a minor note with no location")
    fs = parse_compacted_findings(text)
    assert [(f['name'], f['label'])
            for f in fs] == [('Sage', 'A3'), ('Vera', 'B1')]
    assert fs[0]['severity'] == 'MAJOR'
    assert fs[0]['location'] == 'x.py:10'
    assert fs[1]['location'] == ''


def test_parse_compacted_skips_non_finding_lines():
    fs = parse_compacted_findings(
        "NO FINDINGS\nhere is prose\n- [Ada-A1] NIT z.py:2 - tiny")
    assert [(f['name'], f['label']) for f in fs] == [('Ada', 'A1')]


# --- deterministic severity trim -------------------------------------------

def test_trim_drops_lowest_severity_first():
    findings = [_finding('a', 'A1', 'MAJOR'), _finding('b', 'A2', 'MINOR'),
                _finding('c', 'A3', 'NIT')]
    out = llm._trim_findings_to_budget(findings, lambda cur: len(cur) <= 1)
    assert len(out) == 1
    assert out[0]['severity'] == 'MAJOR'


# --- budget gate + compaction integration (mocked) -------------------------

def test_budgeted_findings_unchanged_when_fits():
    args = types.SimpleNamespace(context_window=1000000, context_cap=0.5)
    findings = [_finding('Ada', 'A1', 'MAJOR')]

    async def go():
        with patch.object(llm, 'get_client'), \
                patch.object(llm, 'resolve_context_window', new=AsyncMock(return_value=1000000)):
            return await llm._budgeted_findings(
                None, findings, lambda fs: 'x' * 10, 'model', args)
    assert asyncio.run(go()) is findings


def test_budgeted_findings_compacts_when_over():
    args = types.SimpleNamespace(context_window=1000, context_cap=0.5)
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Vera', 'A2', 'MINOR')]

    async def go():
        with patch.object(llm, 'get_client'), \
                patch.object(llm, 'resolve_context_window', new=AsyncMock(return_value=1000)), \
                patch.object(llm, 'compact_findings', new=AsyncMock(return_value=[findings[0]])):
            return await llm._budgeted_findings(
                None, findings, lambda fs: 'x' * (1000 * len(fs)), 'model', args)
    assert asyncio.run(go()) == [findings[0]]


def test_compact_findings_returns_strictly_smaller_with_original_labels():
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Sage', 'A2', 'MAJOR')]
    compacted = "- [Ada-A1] MAJOR x.py:1 - missing cycle msg"

    class FakeMapper:
        async def run(self, req):
            return compacted

    async def go():
        with patch.object(llm, 'format_marsha_for_llm', return_value='SPEC'), \
                patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.compact_findings(types.SimpleNamespace(), findings, 'model')
    out = asyncio.run(go())
    assert [(f['name'], f['label']) for f in out] == [('Ada', 'A1')]


def test_compact_findings_falls_back_when_not_smaller():
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Sage', 'A2', 'MAJOR')]

    class FakeMapper:
        async def run(self, req):
            return "NO FINDINGS"  # parses to nothing -> not strictly smaller

    async def go():
        with patch.object(llm, 'format_marsha_for_llm', return_value='SPEC'), \
                patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.compact_findings(types.SimpleNamespace(), findings, 'model')
    assert asyncio.run(go()) is findings


# --- local-backend reviewer serialization ----------------------------------

def _reviewers():
    return [('Ada', 'body', 1), ('Bram', 'body', 2), ('Vera', 'body', 3)]


def _in_flight_mapper(counter):
    class FakeMapper:
        async def run(self, req):
            counter['in'] += 1
            counter['max'] = max(counter['max'], counter['in'])
            await asyncio.sleep(0.01)
            counter['in'] -= 1
            return "NO FINDINGS"
    return FakeMapper()


def test_run_personas_serial_on_local_backend():
    counter = {'in': 0, 'max': 0}
    reviewers = _reviewers()

    async def go():
        with patch.object(personas, 'get_mapper', new=lambda *a, **k: _in_flight_mapper(counter)), \
                patch.object(personas, 'is_local_backend', return_value=True):
            await personas.run_personas(reviewers, 'msg', 'model', 'first_stage')
    asyncio.run(go())
    assert counter['max'] == 1  # never more than one reviewer in flight


def test_run_personas_parallel_on_remote_backend():
    counter = {'in': 0, 'max': 0}
    reviewers = _reviewers()

    async def go():
        with patch.object(personas, 'get_mapper', new=lambda *a, **k: _in_flight_mapper(counter)), \
                patch.object(personas, 'is_local_backend', return_value=False):
            await personas.run_personas(reviewers, 'msg', 'model', 'first_stage')
    asyncio.run(go())
    assert counter['max'] >= 2  # reviewers overlap


# --- --trace-full transcript dump ------------------------------------------

def _run_mapper_once(level, body):
    from marsha.log import set_level, TRACE_OFF
    from marsha.mappers.base import BaseMapper

    class _M(BaseMapper):
        async def transform(self, i):
            return 'RESPONSE-BODY-123'

    m = _M()
    m.label = 'mycall'

    async def go():
        set_level(level)
        try:
            await m.run('PROMPT-BODY-456')
        finally:
            set_level(TRACE_OFF)
    asyncio.run(go())


def test_trace_full_dumps_request_and_response(capsys):
    from marsha.log import TRACE_FULL
    _run_mapper_once(TRACE_FULL, None)
    err = capsys.readouterr().err
    assert '=== mycall: request' in err
    assert 'PROMPT-BODY-456' in err
    assert '=== mycall: response' in err
    assert 'RESPONSE-BODY-123' in err


def test_trace_summary_does_not_dump_transcript(capsys):
    from marsha.log import TRACE_SUMMARY
    _run_mapper_once(TRACE_SUMMARY, None)
    err = capsys.readouterr().err
    # Summary level emits progress lines but never the full prompt/response body.
    assert '=== mycall: request' not in err
    assert 'PROMPT-BODY-456' not in err
    assert 'RESPONSE-BODY-123' not in err
