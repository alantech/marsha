"""Deterministic tests for context-window budgeting and findings compaction.

The LLM and any network I/O (the llama.cpp /models probe, the OpenAI models call) are mocked,
so these verify the token budget gate, the label-preserving compaction parser, and the
deterministic severity trim without a server or LLM non-determinism.
"""

from typing import Any, Generator, cast
import asyncio
import json
import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import context
from marsha import llm
from marsha.meta import MarshaMeta
from marsha import personas
from marsha.personas import parse_compacted_findings
from marsha.mappers.chatgpt import uses_completion_tokens
from marsha.stats import price_for


@pytest.fixture(autouse=True)
def _reset_context_caches() -> Generator[None, None, None]:
    # The /models probe and the context-window probe cache per backend; clear them so a test's
    # mocked response can't leak into another test that reuses the same api_base.
    context.reset_cache()
    yield
    context.reset_cache()


def _finding(name: str, label: str, severity: str) -> Any:
    return {'name': name, 'label': label, 'severity': severity, 'location': '', 'desc': 'd'}


# --- pure token / budget helpers -------------------------------------------

def test_estimate_tokens() -> None:
    assert context.estimate_tokens('') == 1
    assert context.estimate_tokens('x' * 3000) == 1000


def test_budget_tokens_and_fits() -> None:
    assert context.budget_tokens(250000, 0.5) == 125000
    # ~100k tokens <= 125k
    assert context.fits('x' * 300000, 250000, 0.5) is True
    # ~200k tokens > 125k
    assert context.fits('x' * 600000, 250000, 0.5) is False


def test_known_context_fallback() -> None:
    assert context.known_context('claude-opus-5') == 200000
    assert context.known_context('gpt-5-mini') == 400000
    # GPT-6 and GPT-5.6 both document a 1.05M window; the longer 'gpt-5.6' prefix must win over
    # the 'gpt-5' (400k) prefix for gpt-5.6-* models.
    assert context.known_context('gpt-6-luna') == 1050000
    assert context.known_context('gpt-5.6-terra') == 1050000
    assert context.known_context(
        'mystery-model') == context.DEFAULT_CONTEXT_WINDOW


def test_reasoning_models_require_completion_tokens() -> None:
    # GPT-5, GPT-5.6 and GPT-6 are all reasoning models: they reject max_tokens and require
    # max_completion_tokens. gpt-5.6-*/gpt-6-* do not match the 'gpt-5' prefix, so each family
    # must be detected explicitly (a miss silently sends max_tokens and the API rejects it).
    for model in ('gpt-5', 'gpt-5-mini', 'gpt-5.6-terra', 'gpt-6-luna', 'gpt-6-sol'):
        assert uses_completion_tokens(model) is True
    assert uses_completion_tokens('claude-sonnet-5') is False


def test_pricing_prefix_precedence() -> None:
    # Pricing is matched by longest model-name prefix, so a gpt-5.6-* model must resolve to its
    # own entry, not the shorter 'gpt-5' one.
    luna_in, luna_out = price_for('gpt-6-luna')
    assert (luna_in, luna_out) == (0.00009765625, 0.00048828125)  # $0.10 / $0.50 per 1M
    terra_in, terra_out = price_for('gpt-5.6-terra')
    assert (terra_in, terra_out) == (0.001953125, 0.01171875)  # $2 / $12 per 1M
    # The bare gpt-5 entry still applies to old gpt-5 (not shadowed by the gpt-5.6 prefix).
    assert price_for('gpt-5') == (0.001220703125, 0.009765625)


def test_resolve_override_wins() -> None:
    async def go() -> Any:
        return await context.resolve_context_window(override=12345)
    assert asyncio.run(go()) == 12345


# --- llama.cpp /models discovery (mocked urllib) ---------------------------

class _FakeResp:
    # A minimal file-like context manager standing in for urllib's urlopen() return value.
    def __init__(self, payload: Any) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> Any:
        return self._raw

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *a: Any) -> Any:
        return False


def _mock_urlopen(payload: Any) -> Any:
    return patch('urllib.request.urlopen', return_value=_FakeResp(payload))


def test_query_llama_context_meta() -> None:
    with _mock_urlopen({"data": [{"id": "m", "meta": {"n_ctx": 250112}}]}):
        assert context._query_llama_context('http://x/v1') == 250112


def test_query_llama_context_details_fallback() -> None:
    with _mock_urlopen({"models": [{"details": {"n_ctx": 8192}}]}):
        assert context._query_llama_context('http://x/v1') == 8192


def test_query_llama_context_error_is_none() -> None:
    with patch('urllib.request.urlopen', side_effect=Exception('boom')):
        assert context._query_llama_context('http://x/v1') is None


def test_query_llama_context_by_model() -> None:
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

def test_discover_models_descriptors() -> None:
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


def test_discover_models_name_fallback_and_error() -> None:
    with _mock_urlopen({"models": [{"name": "only"}]}):
        assert context.discover_models(
            'http://y/v1') == [{'id': 'only', 'context': None}]
    with patch('urllib.request.urlopen', side_effect=Exception('boom')):
        assert context.discover_models('http://z/v1') is None


# --- label-preserving compaction parser ------------------------------------

def test_parse_compacted_preserves_labels_and_gaps() -> None:
    text = ("- [Sage-A3] MAJOR x.py:10 - the spec requires X\n"
            "- [Vera-B1] MINOR - a minor note with no location")
    fs = parse_compacted_findings(text)
    assert [(f['name'], f['label'])
            for f in fs] == [('Sage', 'A3'), ('Vera', 'B1')]
    assert fs[0]['severity'] == 'MAJOR'
    assert fs[0]['location'] == 'x.py:10'
    assert fs[1]['location'] == ''


def test_parse_compacted_skips_non_finding_lines() -> None:
    fs = parse_compacted_findings(
        "NO FINDINGS\nhere is prose\n- [Ada-A1] NIT z.py:2 - tiny")
    assert [(f['name'], f['label']) for f in fs] == [('Ada', 'A1')]


def test_parse_compacted_accepts_missing_leading_dash() -> None:
    # The model sometimes omits the list bullet; a well-formed finding must still parse rather
    # than be silently dropped (that dropped a real MAJOR in a real review run).
    fs = parse_compacted_findings(
        "[Sage-A3] MAJOR x.py:10 - the spec requires X\n- [Vera-B1] MINOR y.py - note")
    assert [(f['name'], f['label']) for f in fs] == [('Sage', 'A3'), ('Vera', 'B1')]


# --- deterministic severity trim -------------------------------------------

def test_trim_drops_lowest_severity_first() -> None:
    findings = [_finding('a', 'A1', 'MAJOR'), _finding('b', 'A2', 'MINOR'),
                _finding('c', 'A3', 'NIT')]
    out = llm._trim_findings_to_budget(findings, lambda cur: len(cur) <= 1)
    assert len(out) == 1
    assert out[0]['severity'] == 'MAJOR'


# --- budget gate + compaction integration (mocked) -------------------------

def test_budgeted_findings_unchanged_when_fits() -> None:
    args = types.SimpleNamespace(context_window=1000000, context_cap=0.5)
    findings = [_finding('Ada', 'A1', 'MAJOR')]

    async def go() -> Any:
        with patch.object(llm, 'get_client'), \
                patch.object(llm, 'resolve_context_window', new=AsyncMock(return_value=1000000)):
            return await llm._budgeted_findings(
                cast(MarshaMeta, None), findings, lambda fs: 'x' * 10, 'model', args)
    assert asyncio.run(go()) is findings


def test_budgeted_findings_compacts_when_over() -> None:
    args = types.SimpleNamespace(context_window=1000, context_cap=0.5)
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Vera', 'A2', 'MINOR')]

    async def go() -> Any:
        with patch.object(llm, 'get_client'), \
                patch.object(llm, 'resolve_context_window', new=AsyncMock(return_value=1000)), \
                patch.object(llm, 'compact_findings', new=AsyncMock(return_value=[findings[0]])):
            return await llm._budgeted_findings(
                cast(MarshaMeta, None), findings, lambda fs: 'x' * (1000 * len(fs)), 'model', args)
    assert asyncio.run(go()) == [findings[0]]


def test_compact_findings_returns_strictly_smaller_with_original_labels() -> None:
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Sage', 'A2', 'MAJOR')]
    compacted = "- [Ada-A1] MAJOR x.py:1 - missing cycle msg"

    class FakeMapper:
        async def run(self, req: Any) -> Any:
            return compacted

    async def go() -> Any:
        with patch.object(llm, 'format_marsha_for_llm', return_value='SPEC'), \
                patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.compact_findings(
                cast(MarshaMeta, types.SimpleNamespace()), findings, 'model')
    out = asyncio.run(go())
    assert [(f['name'], f['label']) for f in out] == [('Ada', 'A1')]


def test_compact_findings_falls_back_when_not_smaller() -> None:
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Sage', 'A2', 'MAJOR')]

    class FakeMapper:
        async def run(self, req: Any) -> Any:
            return "NO FINDINGS"  # parses to nothing -> not strictly smaller

    async def go() -> Any:
        with patch.object(llm, 'format_marsha_for_llm', return_value='SPEC'), \
                patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.compact_findings(
                cast(MarshaMeta, types.SimpleNamespace()), findings, 'model')
    assert asyncio.run(go()) is findings


def test_consolidate_allow_empty_drops_everything() -> None:
    # allow_empty=True (the review path) lets the pass reduce the list to zero when every
    # finding is a non-defect, so a clean change posts nothing.
    findings = [_finding('Ada', 'A1', 'MAJOR'), _finding('Sage', 'A2', 'MINOR')]

    class FakeMapper:
        async def run(self, req: Any) -> Any:
            return "NO FINDINGS"

    async def go() -> Any:
        with patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.consolidate_findings(
                'ctx', findings, 'model', allow_empty=True)
    assert asyncio.run(go()) == []


def test_consolidate_allow_empty_survives_flaky_empty() -> None:
    # A single flaky empty response must not zero a review: if another attempt shrank to a
    # non-empty list, keep that rather than dropping everything.
    findings = [_finding('Ada', 'A1', 'MAJOR'),
                _finding('Sage', 'A2', 'MINOR'),
                _finding('Vera', 'A3', 'NIT')]
    replies = iter([
        "NO FINDINGS",  # flaky empty (this is what zeroed a real run)
        "- [Ada-A1] MAJOR x.py:1 - real defect",
        "NO FINDINGS",
    ])

    class FakeMapper:
        async def run(self, req: Any) -> Any:
            return next(replies)

    async def go() -> Any:
        with patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.consolidate_findings(
                'ctx', findings, 'model', allow_empty=True, retries=3)
    out = asyncio.run(go())
    assert [(f['name'], f['label']) for f in out] == [('Ada', 'A1')]


def test_consolidate_default_never_returns_empty() -> None:
    # Without allow_empty (budget compaction), an empty result is rejected so no work is lost.
    findings = [_finding('Ada', 'A1', 'MAJOR')]

    class FakeMapper:
        async def run(self, req: Any) -> Any:
            return "NO FINDINGS"

    async def go() -> Any:
        with patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.consolidate_findings('ctx', findings, 'model')
    assert asyncio.run(go()) is findings


# --- local-backend reviewer serialization ----------------------------------

def _reviewers() -> Any:
    return [('Ada', 'body', 1), ('Bram', 'body', 2), ('Vera', 'body', 3)]


def _in_flight_mapper(counter: Any) -> Any:
    class FakeMapper:
        async def run(self, req: Any) -> Any:
            counter['in'] += 1
            counter['max'] = max(counter['max'], counter['in'])
            await asyncio.sleep(0.01)
            counter['in'] -= 1
            return "NO FINDINGS"
    return FakeMapper()


def test_run_personas_serial_on_local_backend() -> None:
    counter = {'in': 0, 'max': 0}
    reviewers = _reviewers()

    async def go() -> Any:
        with patch.object(personas, 'get_mapper', new=lambda *a, **k: _in_flight_mapper(counter)), \
                patch.object(personas, 'is_local_backend', return_value=True):
            await personas.run_personas(reviewers, 'msg', 'model', 'first_stage')
    asyncio.run(go())
    assert counter['max'] == 1  # never more than one reviewer in flight


def test_run_personas_parallel_on_remote_backend() -> None:
    counter = {'in': 0, 'max': 0}
    reviewers = _reviewers()

    async def go() -> Any:
        with patch.object(personas, 'get_mapper', new=lambda *a, **k: _in_flight_mapper(counter)), \
                patch.object(personas, 'is_local_backend', return_value=False):
            await personas.run_personas(reviewers, 'msg', 'model', 'first_stage')
    asyncio.run(go())
    assert counter['max'] >= 2  # reviewers overlap


# --- --trace-full transcript dump ------------------------------------------

def _run_mapper_once(level: Any, body: Any) -> Any:
    from marsha.log import set_level, TRACE_OFF
    from marsha.mappers.base import BaseMapper

    class _M(BaseMapper):
        async def transform(self, i: Any) -> Any:
            return 'RESPONSE-BODY-123'

    m = _M()
    m.label = 'mycall'

    async def go() -> Any:
        set_level(level)
        try:
            await m.run('PROMPT-BODY-456')
        finally:
            set_level(TRACE_OFF)
    asyncio.run(go())


def test_trace_full_dumps_request_and_response(capsys: Any) -> None:
    from marsha.log import TRACE_FULL
    _run_mapper_once(TRACE_FULL, None)
    err = capsys.readouterr().err
    assert '=== mycall: request' in err
    assert 'PROMPT-BODY-456' in err
    assert '=== mycall: response' in err
    assert 'RESPONSE-BODY-123' in err


def test_trace_summary_does_not_dump_transcript(capsys: Any) -> None:
    from marsha.log import TRACE_SUMMARY
    _run_mapper_once(TRACE_SUMMARY, None)
    err = capsys.readouterr().err
    # Summary level emits progress lines but never the full prompt/response body.
    assert '=== mycall: request' not in err
    assert 'PROMPT-BODY-456' not in err
    assert 'RESPONSE-BODY-123' not in err
