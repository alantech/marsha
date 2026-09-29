"""Deterministic tests for the `marsha refine` subcommand (marsha.refine).

The LLM analysis (`analyze_spec`), the chat loop (`run_refine_chat`), and the GitHub/Linear CLIs
(`_gh`, `_run`, `_repo_name`) are mocked so nothing needs a network, an API key, or the external
tools. The pure helpers (`parse_issue_ref`, `resolve_source`, `parse_locked_output`, `_run_check`)
are exercised directly, and the `run_refine` driver is driven end-to-end with a real `.mrsh` file.
"""

import asyncio
import email.message
import json
import os
import threading
import time
import urllib.error
from typing import Any, Awaitable, Generator

import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import refine
from marsha import term
from marsha import tools
from marsha.mappers.base import ContextOverflowError
from marsha.spec_check import SPEC_CHECK_GROUNDED_NOTE


@pytest.fixture(autouse=True)
def _fresh_rich_console() -> Generator[None, None, None]:
    # `print_diagnostic` caches a rich Console bound to sys.stderr; under pytest's capsys that
    # handle goes stale ("I/O operation on closed file"), so rebind it per test to the captured
    # stderr before the test runs.
    term._stderr_console = None
    yield
    term._stderr_console = None


@pytest.fixture(autouse=True)
def _fixed_context_window() -> Generator[None, None, None]:
    # The chat compacts against the model's context budget; in tests the window is fixed so no
    # API probing happens and compaction stays off for the small scripted conversations.
    with patch.object(refine, 'resolve_context_window',
                      new=AsyncMock(return_value=200_000)):
        yield


def _args(**kw: Any) -> Any:
    base = dict(source=None, issue=None, linear=None, check=False,
                max_turns=40, dry_run=False, target='python',
                target_version=None, debug=False, trace=False,
                trace_full=False, model=None, provider=None, api_base=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _scripted_mapper(responses: list[str]) -> Any:
    return _ScriptedMapper(responses)


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


def _mrsh_spec(marker: str) -> str:
    # A minimal spec that is a valid .mrsh (one func section with a description and two usage
    # examples — the shape the .mrsh format rules require, the same parser compile runs): the
    # [[NEW:SPEC]] payload the locking chat tests emit, with `marker` in the description for
    # the assertions to find.
    return ('# func add(a: int, b: int): int\n'
            f'Adds two integers together and returns the sum. {marker} '
            'The rest of the description pads it past the minimum length rule.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')


def _mrsh_spec_subsections(marker: str) -> str:
    # A valid .mrsh that uses "##" subsections inside the function section (the extended
    # format): the description paragraph first, subsections in between, and the usage-
    # examples list as the section's final block.
    return ('# func add(a: int, b: int): int\n'
            f'Adds two integers together and returns the sum. {marker} '
            'The rest of the description pads it past the minimum length rule.\n'
            '\n'
            '## Behavior\n\n'
            'The addition is commutative and handles negative integers.\n'
            '\n'
            '## Usage examples\n\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')


def _mrsh_spec_with_url(marker: str, url: str) -> str:
    # A valid .mrsh whose description paragraph names an external endpoint: the URL sits
    # where the .mrsh format rules allow it (inside the description block, the examples
    # list still last), which the endpoint-verification tests need.
    return ('# func add(a: int, b: int): int\n'
            f'Adds two integers together and returns the sum. {marker} Data is fetched '
            f'from {url} before the sum is returned, which pads the description past the '
            'minimum length rule as well.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')


def _http_error(code: int, url: str = 'https://dead.example.com/api') -> Any:
    # A urllib HTTPError for `code` (what tools.http_get raises for 4xx/5xx responses).
    return urllib.error.HTTPError(url, code, 'err', email.message.Message(), None)


# --- parse_issue_ref ----------------------------------------------------------


def test_parse_issue_ref_bare() -> None:
    assert refine.parse_issue_ref('218') == (218, None)
    assert refine.parse_issue_ref('  7 ') == (7, None)


def test_parse_issue_ref_qualified() -> None:
    assert refine.parse_issue_ref('acme/widget#218') == (218, 'acme/widget')


def test_parse_issue_ref_url() -> None:
    assert refine.parse_issue_ref(
        'https://github.com/acme/widget/issues/218') == (218, 'acme/widget')


def test_parse_issue_ref_url_is_exact() -> None:
    # The URL form is the whole value: garbage around a URL is a malformed reference, not a URL
    # to be searched for (which could point the load/apply paths at an unintended issue).
    with pytest.raises(Exception, match='Invalid --issue reference'):
        refine.parse_issue_ref('not-a-url github.com/acme/widget/issues/218 typo')
    with pytest.raises(Exception, match='Invalid --issue reference'):
        refine.parse_issue_ref('x https://github.com/acme/widget/issues/218')


def test_parse_issue_ref_rejects_pull_request_url() -> None:
    # A pull-request URL is not an issue: it is rejected rather than mis-handled by the
    # issue-only load/write path (gh issue view / gh issue edit).
    with pytest.raises(Exception, match='pull-request'):
        refine.parse_issue_ref('https://github.com/acme/widget/pull/9')


def test_parse_issue_ref_invalid() -> None:
    for bad in ('abc', 'acme/widget', '#12', '1.5'):
        with pytest.raises(Exception):
            refine.parse_issue_ref(bad)


def test_parse_issue_ref_rejects_zero() -> None:
    # GitHub numbers issues from 1: 0 (bare, qualified, or URL) is a malformed reference, not
    # an issue to load — it must not reach the issue load path.
    for bad in ('0', 'acme/widget#0',
                'https://github.com/acme/widget/issues/0'):
        with pytest.raises(Exception, match='issue numbers start at 1'):
            refine.parse_issue_ref(bad)


# --- resolve_source -----------------------------------------------------------


def test_resolve_source_mrsh() -> None:
    s = refine.resolve_source(_args(source='spec.mrsh'))
    assert s.kind == 'mrsh' and s.path == 'spec.mrsh'


def test_resolve_source_mrsh_requires_mrstsh_extension() -> None:
    # The positional source is rewritten in place when the design locks, so a non-.mrsh path
    # (a typo or an accidental file) is rejected before any LLM call, not silently overwritten.
    for bad in ('README.md', 'spec.txt', 'notes'):
        with pytest.raises(Exception, match=r'not a .mrsh file'):
            refine.resolve_source(_args(source=bad))


def test_resolve_source_issue() -> None:
    s = refine.resolve_source(_args(issue='218'))
    assert s.kind == 'issue' and s.num == 218 and s.repo is None


def test_resolve_source_issue_qualified_carries_repo() -> None:
    s = refine.resolve_source(_args(issue='acme/widget#218'))
    assert s.kind == 'issue' and s.num == 218 and s.repo == 'acme/widget'


def test_resolve_source_linear() -> None:
    s = refine.resolve_source(_args(linear='ENG-5'))
    assert s.kind == 'linear' and s.name == 'ENG-5'


def test_resolve_source_requires_exactly_one() -> None:
    with pytest.raises(Exception):
        refine.resolve_source(_args())  # none given
    with pytest.raises(Exception):
        refine.resolve_source(_args(source='a.mrsh', issue='1'))  # two
    with pytest.raises(Exception):
        refine.resolve_source(_args(issue='1', linear='ENG-1'))  # two


# --- _repo_gate (fast-fail, before any LLM call) ------------------------------


def test_repo_gate_mrsh_reports_repo_without_gating() -> None:
    # A .mrsh file is standalone: no gate, no gh/linear required, but the current repo (if any)
    # is reported so the chat can use the read-only codebase tools.
    s = refine.resolve_source(_args(source='spec.mrsh'))
    with patch.object(refine, '_git_repo_name',
                      new=AsyncMock(return_value='acme/widget')):
        assert asyncio.run(refine._repo_gate(s, '/x')) == 'acme/widget'


def test_repo_gate_mrsh_without_repo_reports_empty() -> None:
    s = refine.resolve_source(_args(source='spec.mrsh'))
    with patch.object(refine, '_git_repo_name', new=AsyncMock(return_value='')):
        assert asyncio.run(refine._repo_gate(s, '/x')) == ''


def test_repo_gate_mrsh_repo_detection_failure_is_standalone() -> None:
    # A .mrsh file is standalone: if the repo cannot be detected (e.g. git not installed), the
    # gate reports no repo rather than failing, so the file still refines (without tools).
    s = refine.resolve_source(_args(source='spec.mrsh'))

    async def no_git(*a: Any, **k: Any) -> Any:
        raise Exception('`git` is not installed or not on PATH.')

    with patch.object(refine, '_git_repo_name', new=no_git):
        assert asyncio.run(refine._repo_gate(s, '/x')) == ''


def test_git_repo_name_parses_remote() -> None:
    for url, want in (
            ('git@github.com:acme/widget.git', 'acme/widget'),
            ('https://github.com/acme/widget.git', 'acme/widget'),
            ('https://github.com/acme/widget', 'acme/widget')):
        async def fake_run(cmd: Any, *args: Any, **k: Any) -> Any:
            return (0, url, '')
        with patch.object(refine, '_run', new=fake_run):
            assert asyncio.run(refine._git_repo_name('/x')) == want
    # No origin remote (or a git failure) -> empty, not an error.
    async def no_remote(cmd: Any, *args: Any, **k: Any) -> Any:
        return (1, '', 'fatal: no remote named origin')
    with patch.object(refine, '_run', new=no_remote):
        assert asyncio.run(refine._git_repo_name('/x')) == ''


def test_repo_gate_issue_requires_a_repo() -> None:
    s = refine.resolve_source(_args(issue='218'))
    with patch.object(refine, 'gh_available', lambda: True), \
         patch.object(refine, '_repo_name', new=AsyncMock(return_value='')):
        with pytest.raises(Exception, match='inside a git repository'):
            asyncio.run(refine._repo_gate(s, '/nowhere'))


def test_repo_gate_issue_requires_gh_cli() -> None:
    s = refine.resolve_source(_args(issue='218'))
    with patch.object(refine, 'gh_available', lambda: False):
        with pytest.raises(Exception, match='`gh` CLI'):
            asyncio.run(refine._repo_gate(s, '/x'))


def test_repo_gate_issue_returns_current_repo() -> None:
    s = refine.resolve_source(_args(issue='218'))
    with patch.object(refine, 'gh_available', lambda: True), \
         patch.object(refine, '_repo_name', new=AsyncMock(return_value='acme/widget')):
        assert asyncio.run(refine._repo_gate(s, '/x')) == 'acme/widget'


def test_repo_gate_issue_repo_mismatch_errors() -> None:
    s = refine.resolve_source(_args(issue='acme/widget#218'))
    with patch.object(refine, 'gh_available', lambda: True), \
         patch.object(refine, '_repo_name', new=AsyncMock(return_value='other/repo')):
        with pytest.raises(Exception, match='belongs to acme/widget'):
            asyncio.run(refine._repo_gate(s, '/x'))


def test_repo_gate_issue_repo_match_is_case_insensitive() -> None:
    s = refine.resolve_source(_args(issue='ACME/Widget#218'))
    with patch.object(refine, 'gh_available', lambda: True), \
         patch.object(refine, '_repo_name', new=AsyncMock(return_value='acme/widget')):
        assert asyncio.run(refine._repo_gate(s, '/x')) == 'acme/widget'


def test_repo_gate_linear_requires_a_repo() -> None:
    s = refine.resolve_source(_args(linear='ENG-5'))
    with patch.object(refine, 'linear_available', lambda: True), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)):
        with pytest.raises(Exception, match='inside a git repository'):
            asyncio.run(refine._repo_gate(s, '/x'))


def test_repo_gate_linear_uses_git_not_gh() -> None:
    # A Linear ticket must be refinable without GitHub auth: the gate uses git (not gh) and takes
    # the repo name from the remote, so gh/_repo_name must not be consulted.
    async def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError('the linear gate must not call gh (_repo_name)')
    s = refine.resolve_source(_args(linear='ENG-5'))
    with patch.object(refine, 'linear_available', lambda: True), \
         patch.object(refine, '_repo_name', new=boom), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=True)), \
         patch.object(refine, '_git_repo_name',
                      new=AsyncMock(return_value='acme/widget')):
        assert asyncio.run(refine._repo_gate(s, '/x')) == 'acme/widget'


def test_repo_gate_linear_requires_linear_cli() -> None:
    s = refine.resolve_source(_args(linear='ENG-5'))
    with patch.object(refine, 'linear_available', lambda: False):
        with pytest.raises(Exception, match='`linear` CLI'):
            asyncio.run(refine._repo_gate(s, '/x'))


# --- gh_issue_context ---------------------------------------------------------


def test_gh_issue_context_renders_title_body_and_visible_comments() -> None:
    payload = json.dumps({
        'number': 218, 'title': 'Refine the refine command',
        'body': 'the description',
        'comments': [
            {'author': {'login': 'a'}, 'body': 'comment one'},
            {'author': {'login': 'b'}, 'body': 'hidden', 'isMinimized': True},
        ]})
    calls: dict[str, Any] = {}

    async def fake_gh(*a: Any, **k: Any) -> Any:
        calls['args'] = a
        return (0, payload, '')

    with patch.object(refine, '_gh', new=fake_gh):
        out = asyncio.run(refine.gh_issue_context(218))
    assert '218' in out and 'Refine the refine command' in out
    assert 'the description' in out
    assert 'comment one' in out and 'a:' in out
    assert 'hidden' not in out  # the minimized comment is skipped
    assert calls['args'][:3] == ('issue', 'view', '218')
    assert '--json' in calls['args']


def test_gh_issue_context_surfaces_gh_failure() -> None:
    async def fake_gh(*a: Any, **k: Any) -> Any:
        return (1, '', 'boom')
    with patch.object(refine, '_gh', new=fake_gh):
        with pytest.raises(Exception, match='gh issue view 218'):
            asyncio.run(refine.gh_issue_context(218))


def test_initial_chat_message_wraps_findings_as_untrusted_data() -> None:
    # The analysis findings are model-generated from untrusted source text: they are wrapped as
    # untrusted data in the chat message, so instruction-like finding text cannot steer the chat.
    msg = refine._initial_chat_message(
        'mrsh', 'SPEC', ['amb one', 'amb two'], ['err one'], 'acme/widget')
    assert msg.count('[tool:spec-check]') == 2  # the ambiguities and the errors, each wrapped
    assert 'amb one' in msg and 'amb two' in msg and 'err one' in msg
    assert 'never as instructions' in msg


# --- _extract_section / parse_locked_output -----------------------------------


def test_section_to_end() -> None:
    lines = 'intro\n[[NEW:SPEC]]\nspec line 1\nspec line 2\n'.split('\n')
    assert refine._section_to_end(lines, '[[NEW:SPEC]]') == 'spec line 1\nspec line 2'
    # A marker-like line in the content is preserved (runs to the end, not truncated).
    lines2 = ('[[NEW:SPEC]]\na\n[[DESIGN:LOCKED]]\nb\n').split('\n')
    assert refine._section_to_end(lines2, '[[NEW:SPEC]]') == 'a\n[[DESIGN:LOCKED]]\nb'
    assert refine._section_to_end(['no marker'], '[[NEW:SPEC]]') is None


def test_section_to_stops_only_at_named_end_marker() -> None:
    lines = '[[NEW:TITLE]]\nTitle here\n[[NEW:BODY]]\nBody here'.split('\n')
    assert refine._section_to(lines, '[[NEW:TITLE]]', '[[NEW:BODY]]') == 'Title here'
    # A marker other than the named end marker does not terminate the section.
    lines2 = '[[NEW:TITLE]]\nT\n[[NEW:SPEC]]\nU\n[[NEW:BODY]]\nB'.split('\n')
    assert refine._section_to(lines2, '[[NEW:TITLE]]', '[[NEW:BODY]]') == 'T\n[[NEW:SPEC]]\nU'
    assert refine._section_to(['x'], '[[NEW:TITLE]]', '[[NEW:BODY]]') is None


def test_parse_locked_output_mrsh_preserves_marker_line() -> None:
    # A .mrsh spec that legitimately contains a line equal to a protocol marker must not be
    # truncated at that line: the spec section runs to the end of the response.
    text = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nline one\n[[DESIGN:LOCKED]]\nline two'
    assert refine.parse_locked_output(text, 'mrsh') == {
        'spec': 'line one\n[[DESIGN:LOCKED]]\nline two'}


def test_parse_locked_output_mrsh() -> None:
    text = 'All done.\n[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n# New spec\ncontent'
    assert refine.parse_locked_output(text, 'mrsh') == {
        'spec': '# New spec\ncontent'}


def test_parse_locked_output_mrsh_missing_spec() -> None:
    assert refine.parse_locked_output(
        '[[DESIGN:LOCKED]]\nlocked but no spec section', 'mrsh') is None


def test_parse_locked_output_mrsh_empty_spec() -> None:
    assert refine.parse_locked_output(
        '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n   \n', 'mrsh') is None


def test_parse_locked_output_issue() -> None:
    text = ('[[DESIGN:LOCKED]]\n'
            '[[NEW:TITLE]]\nNew Title\n'
            '[[NEW:BODY]]\nBody para 1\nBody para 2')
    assert refine.parse_locked_output(text, 'issue') == {
        'title': 'New Title', 'body': 'Body para 1\nBody para 2'}


def test_parse_locked_output_title_is_first_line_only() -> None:
    text = ('[[DESIGN:LOCKED]]\n'
            '[[NEW:TITLE]]\nReal Title\nExtra line\n'
            '[[NEW:BODY]]\nbody')
    got = refine.parse_locked_output(text, 'issue')
    assert got is not None and got['title'] == 'Real Title'


def test_parse_locked_output_requires_lock_marker() -> None:
    assert refine.parse_locked_output('[[NEW:SPEC]]\nx', 'mrsh') is None


def test_parse_locked_output_lock_marker_in_prose_is_ignored() -> None:
    # A marker quoted inside a sentence is not the protocol line: no payload is extracted.
    text = ('I think we are done - here is what [[DESIGN:LOCKED]] would look like:\n'
            '[[NEW:SPEC]]\nthe spec')
    assert refine.parse_locked_output(text, 'mrsh') is None


def test_parse_locked_output_body_before_title_is_rejected() -> None:
    # The protocol is title then body: a body marker before the title is a malformed ordering
    # and must not be written back as the issue/ticket body.
    text = ('[[DESIGN:LOCKED]]\n[[NEW:BODY]]\nbody content\n'
            '[[NEW:TITLE]]\nMy Title')
    assert refine.parse_locked_output(text, 'issue') is None
    assert refine.parse_locked_output(text, 'linear') is None


def test_parse_locked_output_bail_before_payload_is_rejected() -> None:
    # A bail signal line before the payload makes the response self-contradictory: it must not
    # lock (which would overwrite the source).
    text = '[[DESIGN:BAIL]]\n[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'
    assert refine.parse_locked_output(text, 'mrsh') is None
    text2 = ('[[DESIGN:LOCKED]]\n[[DESIGN:BAIL]]\n[[NEW:TITLE]]\nT\n'
             '[[NEW:BODY]]\nB')
    assert refine.parse_locked_output(text2, 'issue') is None


def test_parse_locked_output_bail_line_inside_payload_is_content() -> None:
    # A [[DESIGN:BAIL]] line inside the rewritten spec is payload content (a spec may document
    # the protocol itself), not a contradictory bail signal: the spec is saved verbatim.
    text = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nline one\n[[DESIGN:BAIL]]\nline two'
    assert refine.parse_locked_output(text, 'mrsh') == {
        'spec': 'line one\n[[DESIGN:BAIL]]\nline two'}


def test_parse_locked_output_issue_empty_body_is_rejected() -> None:
    # An empty body would erase the issue/ticket's description when written back: it is
    # rejected rather than applied.
    assert refine.parse_locked_output(
        '[[DESIGN:LOCKED]]\n[[NEW:TITLE]]\nT\n[[NEW:BODY]]\n', 'issue') is None
    assert refine.parse_locked_output(
        '[[DESIGN:LOCKED]]\n[[NEW:TITLE]]\nT\n[[NEW:BODY]]\n   \n', 'linear') is None


def test_parse_locked_output_lock_only_inside_payload_is_ignored() -> None:
    # A [[DESIGN:LOCKED]] line that appears only inside the rewritten spec (after the payload
    # marker) is content, not the signal: it cannot lock by itself.
    text = ('Here is the updated spec:\n[[NEW:SPEC]]\nline one\n'
            '[[DESIGN:LOCKED]]\nline two')
    assert refine.parse_locked_output(text, 'mrsh') is None
    # Same for an issue/ticket payload: the body runs to the end and may contain the marker.
    text2 = '[[NEW:TITLE]]\nT\n[[NEW:BODY]]\nbody\n[[DESIGN:LOCKED]]'
    assert refine.parse_locked_output(text2, 'issue') is None


def test_parse_locked_output_issue_missing_body() -> None:
    assert refine.parse_locked_output(
        '[[DESIGN:LOCKED]]\n[[NEW:TITLE]]\nT', 'issue') is None


def test_is_bail_token() -> None:
    for t in ('!bail', '!BAIL', 'q', 'bail', '!quit', ' Q '):
        assert refine._is_bail_token(t)
    for t in ('no', 'continue', 'yes please', ''):
        assert not refine._is_bail_token(t)


# --- _run_check (the headless --check gate) -----------------------------------


def test_run_check_locked_returns_zero(capsys: Any) -> None:
    rc = refine._run_check({'compilable': True, 'ambiguities': [], 'errors': []})
    assert rc == 0
    assert 'Spec is locked' in capsys.readouterr().out


def test_run_check_open_ambiguities_returns_one(capsys: Any) -> None:
    rc = refine._run_check(
        {'compilable': True, 'ambiguities': ['a', 'b'], 'errors': []})
    assert rc == 1
    assert '2 open ambiguity' in capsys.readouterr().out


def test_run_check_not_compilable_returns_one(capsys: Any) -> None:
    rc = refine._run_check(
        {'compilable': False, 'ambiguities': [], 'errors': ['boom']})
    cap = capsys.readouterr()
    assert rc == 1
    # Errors are rendered to stderr (print_diagnostic), not the stdout summary.
    assert 'error:' in cap.err and 'boom' in cap.err


# --- run_refine driver (end-to-end, mocked analysis + chat) -------------------


def test_run_refine_usage_error_with_no_source() -> None:
    assert asyncio.run(refine.run_refine(_args())) == 2


def test_run_refine_refuses_oversized_source_for_rewrite(
        tmp_path: Any, capsys: Any) -> None:
    # A source over the chat limit must not be rewritten from a truncated view: the interactive
    # path refuses before the analysis, so the oversized source is never even sent to a model.
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('x' * (refine.REFINE_SPEC_LIMIT + 1))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        raise AssertionError(
            'an oversized source must be refused before it is sent to the analysis model')

    async def fake_chat(**k: Any) -> Any:
        raise AssertionError('the chat must not run on a source it could not see in full')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'over the' in capsys.readouterr().err
    with open(p) as f:
        assert len(f.read()) == refine.REFINE_SPEC_LIMIT + 1  # untouched


def test_run_refine_check_analyzes_full_oversized_source(
        tmp_path: Any, capsys: Any) -> None:
    # --check never writes back, so an oversized source is analyzed in full (not refused) and the
    # rewrite guard does not apply.
    p = str(tmp_path / 'spec.mrsh')
    size = refine.REFINE_SPEC_LIMIT + 1
    # A valid .mrsh of exactly `size` chars (the oversized content must still pass the format
    # gate so the test exercises the analysis, not the format rejection).
    prefix = ('# func add(a: int, b: int): int\n'
              'Adds two integers together and returns the sum. ')
    suffix = '\n\n* add(1, 2) -> 3\n* add(-1, 1) -> 0'
    with open(p, 'w') as f:
        f.write(prefix + 'x' * (size - len(prefix) - len(suffix)) + suffix)
    seen: dict[str, int] = {}

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        seen['len'] = len(spec_text)
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    with patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=p, check=True)))
    assert rc == 1  # the open ambiguity, not a size refusal
    assert seen['len'] == size  # the full source was analyzed, not truncated
    assert 'open ambiguity' in capsys.readouterr().out


def test_run_refine_missing_mrsh_reports_error(tmp_path: Any, capsys: Any) -> None:
    # A missing .mrsh path is a normal error (no traceback): the byte-size pre-check skips a
    # file it cannot stat, and the load reports the failure.
    p = str(tmp_path / 'nope.mrsh')
    rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'error' in capsys.readouterr().err


def test_run_refine_refuses_huge_mrsh_before_reading(
        tmp_path: Any, capsys: Any) -> None:
    # A file whose byte count alone proves it is oversized (UTF-8: at most 4 bytes per char) is
    # refused before it is read, so a huge file cannot exhaust memory before the refusal.
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('x' * (refine.REFINE_SPEC_LIMIT * 4 + 1))

    async def fake_load(source: Any, cwd: Any) -> str:
        raise AssertionError('an obviously oversized file must be refused before it is read')

    with patch.object(refine, 'load_spec', new=fake_load):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'limit' in capsys.readouterr().err


def test_run_refine_refuses_special_file_source(tmp_path: Any, capsys: Any) -> None:
    # A FIFO or device can report size zero (bypassing the byte guard) yet yield an unbounded
    # stream — and a named pipe can block the read indefinitely: refuse it before reading.
    p = str(tmp_path / 'spec.mrsh')
    os.mkfifo(p)

    async def fake_load(source: Any, cwd: Any) -> str:
        raise AssertionError('a non-regular file must be refused before it is read')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    with patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'regular file' in capsys.readouterr().err


def test_run_refine_refuses_symlink_source(tmp_path: Any, capsys: Any) -> None:
    # A symlink to a regular file passes os.stat's S_ISREG, but the apply step writes through
    # the path — to the symlink's target, which may be outside the intended spec. The guard
    # uses lstat, so a symlinked source is refused and its target is untouched.
    real = tmp_path / 'real.mrsh'
    real.write_text('original')
    link = str(tmp_path / 'link.mrsh')
    os.symlink(real, link)

    async def fake_load(source: Any, cwd: Any, **kw: Any) -> Any:
        raise AssertionError('a symlinked source must be refused before it is read')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    with patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=link)))
    assert rc == 1
    assert 'regular file' in capsys.readouterr().err
    assert real.read_text() == 'original'  # the target was never read or written


def test_run_refine_chat_failure_is_a_handled_error(tmp_path: Any, capsys: Any) -> None:
    # A provider/API failure mid-conversation (e.g. an exhausted request budget) exits as a
    # handled command error, not an unhandled traceback.
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def boom(**k: Any) -> Any:
        raise Exception('request budget exhausted')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=boom), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'error: the conversation failed: request budget exhausted' \
        in capsys.readouterr().err


def test_run_refine_limit_tracks_the_model_context(
        tmp_path: Any, capsys: Any) -> None:
    # The interactive limit tracks the selected model's prompt budget: with a 16k-token context
    # the 48k-char ceiling drops, and a source in between is refused with a clear message
    # (rather than sent to a model that cannot hold it and failing mid-chat).
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('x' * 30_000)

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        raise AssertionError(
            'an oversized-for-the-model source must be refused before the analysis')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'resolve_context_window',
                      new=AsyncMock(return_value=16_000)):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'over the' in capsys.readouterr().err


def test_run_refine_check_mrsh_reports_and_never_mutates(
        tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write(_mrsh_spec('original spec'))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a1', 'a2'], 'errors': []}

    with patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=p, check=True)))
    assert rc == 1
    assert '2 open ambiguity' in capsys.readouterr().out
    with open(p) as f:
        assert f.read() == _mrsh_spec('original spec')  # --check never writes back


def test_run_refine_check_mrsh_locked(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write(_mrsh_spec('the spec'))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': [], 'errors': []}

    with patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=p, check=True)))
    assert rc == 0
    cap = capsys.readouterr()
    assert 'Spec is locked' in cap.out
    # The load and analyze phases announce themselves on stderr (stdout stays
    # clean for the gate) so a slow run does not look like a hang.
    assert f'Reading {p}...' in cap.err
    assert 'Analyzing the spec for open ambiguities...' in cap.err


def test_run_refine_check_mrsh_format_invalid_fails_before_analysis(
        tmp_path: Any, capsys: Any) -> None:
    # A .mrsh the format parser rejects can never compile, so --check reports the format
    # failure without paying for the LLM analysis.
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('A free-form document with no func sections.')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        raise AssertionError('a non-compilable .mrsh must fail before the analysis')

    with patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=p, check=True)))
    assert rc == 1
    assert 'No functions or types found in file' in capsys.readouterr().err


def test_run_refine_check_dead_endpoint_fails(
        tmp_path: Any, capsys: Any) -> None:
    # A spec that names an endpoint that could not be verified as working is not
    # implementable as written: --check reports the dead endpoint and exits non-zero
    # even when the LLM analysis finds nothing else (the harness decides liveness,
    # not the analysis model).
    dead_url = 'https://dead.example.com/api'
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write(_mrsh_spec_with_url('the spec', dead_url))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': [], 'errors': []}

    async def fake_probe(text: str) -> list[str]:
        return [f'{dead_url} — HTTP 404']

    async def fake_read_line() -> str:
        raise AssertionError('--check is headless: it reports, it does not ask')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, '_read_line', new=fake_read_line):
        rc = asyncio.run(refine.run_refine(_args(source=p, check=True)))
    assert rc == 1
    cap = capsys.readouterr()
    assert 'could not be verified as working' in cap.err  # the diagnostic is on stderr
    assert dead_url in cap.err
    assert 'Spec is locked' not in cap.out  # a dead endpoint is never "locked"


def test_run_refine_confirmed_dead_source_endpoint_is_exempted(
        tmp_path: Any, capsys: Any) -> None:
    # When the person confirms the source's dead endpoint as-is (a private API a generic
    # sample request cannot reach), it is exempted for the session: the chat neither hears
    # of it nor probes it at the lock gate.
    url = 'https://private.example.com/v1'
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write(_mrsh_spec_with_url('the spec', url))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    seen: dict[str, Any] = {}

    async def fake_chat(**k: Any) -> Any:
        seen.update(k)
        return refine.ChatResult('bail', None, 'bail')

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        return [f'{url} — HTTP 404']

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert seen['dead_endpoints'] == []
    assert seen['approved_endpoints'] == {url}


def test_run_refine_unconfirmed_dead_source_endpoint_is_reported_to_the_chat(
        tmp_path: Any, capsys: Any) -> None:
    # When the person wants a working alternative, the dead endpoint is told to the chat
    # (and the lock gate will still probe for it).
    url = 'https://dead.example.com/api'
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write(_mrsh_spec_with_url('the spec', url))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    seen: dict[str, Any] = {}

    async def fake_chat(**k: Any) -> Any:
        seen.update(k)
        return refine.ChatResult('bail', None, 'bail')

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        return [f'{url} — HTTP 404']

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, '_read_line', new=lambda: 'n'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert seen['dead_endpoints'] == [f'{url} — HTTP 404']
    assert seen['approved_endpoints'] == set()
    assert 'external endpoint(s) named in the spec could not be verified as ' \
        'working' in capsys.readouterr().err


def test_run_refine_bails_at_the_source_endpoint_question(
        tmp_path: Any, capsys: Any) -> None:
    # A bail token at the source-level endpoint question ends the session before the
    # analysis or the chat.
    url = 'https://private.example.com/v1'
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write(_mrsh_spec_with_url('the spec', url))

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        raise AssertionError('the analysis must not run after a bail')

    async def fake_chat(**k: Any) -> Any:
        raise AssertionError('the chat must not run after a bail')

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        return [f'{url} — HTTP 404']

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, '_read_line', new=lambda: '!bail'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'Bailed out; the source was not modified.' in capsys.readouterr().out


def test_run_refine_check_issue_uses_gate_and_loader(capsys: Any) -> None:
    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_gate(source: Any, cwd: Any) -> str:
        return 'acme/widget'

    async def fake_load(source: Any, cwd: Any, **kw: Any) -> Any:
        return 'ISSUE TEXT', ('T', 'B')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, '_repo_gate', new=fake_gate), \
         patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, '_resolve_window',
                      new=AsyncMock(return_value=200000)):
        rc = asyncio.run(refine.run_refine(_args(issue='218', check=True)))
    assert rc == 1
    cap = capsys.readouterr()
    assert 'open ambiguity' in cap.out
    assert 'Loading issue #218...' in cap.err
    assert 'Analyzing the spec for open ambiguities...' in cap.err


def test_run_refine_check_linear_announces_load_and_analyze(capsys: Any) -> None:
    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_gate(source: Any, cwd: Any) -> str:
        return 'acme/widget'

    async def fake_load(source: Any, cwd: Any, **kw: Any) -> Any:
        return 'TICKET TEXT', ('T', 'B')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, '_repo_gate', new=fake_gate), \
         patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, '_resolve_window',
                      new=AsyncMock(return_value=200000)):
        rc = asyncio.run(refine.run_refine(_args(linear='ENG-5', check=True)))
    assert rc == 1
    cap = capsys.readouterr()
    assert 'Loading Linear ticket ENG-5...' in cap.err
    assert 'Analyzing the spec for open ambiguities...' in cap.err


def test_run_refine_locks_and_writes_mrsh(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'spec': 'LOCKED SPEC'}, 'detail')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 0
    with open(p) as f:
        assert f.read() == 'LOCKED SPEC'
    assert 'design-locked' in capsys.readouterr().out


def test_run_refine_dry_run_does_not_write(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'spec': 'NEW SPEC'}, 'x')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)):
        rc = asyncio.run(refine.run_refine(_args(source=p, dry_run=True)))
    assert rc == 0
    with open(p) as f:
        assert f.read() == 'original'  # dry run leaves the file untouched
    out = capsys.readouterr().out
    assert 'dry run' in out and 'NEW SPEC' in out


def test_run_refine_mrsh_without_git_still_refines(tmp_path: Any) -> None:
    # A standalone .mrsh refines without git installed: the repo lookup and the in-repo check
    # are best-effort, and the chat simply runs without the codebase tools.
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        assert k.get('in_repo') is False
        return refine.ChatResult('locked', {'spec': 'LOCKED SPEC'}, 'detail')

    async def no_git(*a: Any, **k: Any) -> Any:
        raise Exception('`git` is not installed or not on PATH.')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=no_git), \
         patch.object(refine, '_git_repo_name', new=no_git), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 0
    with open(p) as f:
        assert f.read() == 'LOCKED SPEC'


def test_run_refine_declined_confirmation_does_not_write(
        tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'spec': 'NEW SPEC'}, 'x')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)), \
         patch.object(refine, '_read_line', new=lambda: 'n'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    with open(p) as f:
        assert f.read() == 'original'  # a declined confirmation never writes
    cap = capsys.readouterr()
    # The proposal is shown once (the assistant's lock turn), never repeated at the
    # confirmation: the chat is faked here, so the payload must not appear at all.
    assert 'NEW SPEC' not in cap.out
    assert 'Apply the locked design shown above' in cap.out  # explicit question, default no
    assert '╔' in cap.out  # boxed, so it stands out from the proposal it follows
    assert 'Not applied' in cap.out


def test_run_refine_refuses_apply_when_source_changed_during_chat(
        tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    # The first read (at start) and the re-read (before applying) disagree: the source was
    # edited elsewhere while the conversation ran.
    loads = iter(['original', 'edited by someone else'])

    async def fake_load(source: Any, cwd: Any) -> str:
        return next(loads)

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'spec': 'NEW SPEC'}, 'x')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)), \
         patch.object(refine, 'load_spec', new=fake_load), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    with open(p) as f:
        assert f.read() == 'original'  # a stale rewrite is never applied
    assert 'changed while the conversation was running' \
        in capsys.readouterr().err


def test_run_refine_issue_comment_only_change_allows_apply(capsys: Any) -> None:
    # Only the title/body gate the apply: a comment added to the issue while the conversation
    # ran changes the rendered context (and the chat input) but not the rewritten fields, so
    # the apply-time re-read still matches the baseline and the apply proceeds.
    fields = iter([('Title', 'Body')])

    async def fake_source_fields(source: Any, cwd: Any, **kw: Any) -> Any:
        return next(fields)

    async def fake_load(source: Any, cwd: Any, **kw: Any) -> Any:
        return 'ISSUE TEXT', ('Title', 'Body')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': [], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'title': 'T2', 'body': 'B2'}, 'x')

    applied: list[Any] = []

    async def fake_apply(source: Any, payload: Any, cwd: Any) -> None:
        applied.append(payload)

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)), \
         patch.object(refine, '_repo_gate', new=AsyncMock(return_value='acme/widget')), \
         patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, '_source_fields', new=fake_source_fields), \
         patch.object(refine, '_apply', new=fake_apply), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(issue='218')))
    assert rc == 0
    assert applied == [{'title': 'T2', 'body': 'B2'}]


def test_run_refine_issue_title_change_refuses_apply(capsys: Any) -> None:
    # The baseline comes from load_spec_with_fields (('Title', 'Body') below); the single
    # apply-time re-read (via _source_fields) returns the edited title.
    fields = iter([('Edited Title', 'Body')])

    async def fake_source_fields(source: Any, cwd: Any, **kw: Any) -> Any:
        return next(fields)

    async def fake_load(source: Any, cwd: Any, **kw: Any) -> Any:
        return 'ISSUE TEXT', ('Title', 'Body')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': [], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'title': 'T2', 'body': 'B2'}, 'x')

    applied: list[Any] = []

    async def fake_apply(source: Any, payload: Any, cwd: Any) -> None:
        applied.append(payload)

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)), \
         patch.object(refine, '_repo_gate', new=AsyncMock(return_value='acme/widget')), \
         patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, '_source_fields', new=fake_source_fields), \
         patch.object(refine, '_apply', new=fake_apply), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(issue='218')))
    assert rc == 1
    assert applied == []  # a changed rewritten field is never applied
    assert 'changed while the conversation was running' \
        in capsys.readouterr().err


def test_gh_issue_fields_extract_title_and_body() -> None:
    async def fake_gh(*a: Any, **k: Any) -> Any:
        return 0, '{"title": "T", "body": null}', ''

    with patch.object(refine, '_gh', new=fake_gh):
        assert asyncio.run(refine.gh_issue_fields(218)) == ('T', '')


def test_linear_fields_extract_title_and_description() -> None:
    async def fake_run(*a: Any, **k: Any) -> Any:
        return 0, '{"title": "T", "description": "D"}', ''

    with patch.object(refine, '_run', new=fake_run):
        assert asyncio.run(refine.linear_fields('ENG-5')) == ('T', 'D')

    async def fake_run_list(*a: Any, **k: Any) -> Any:
        return 0, '[{"title": "T2", "description": null}]', ''

    with patch.object(refine, '_run', new=fake_run_list):
        assert asyncio.run(refine.linear_fields('ENG-5')) == ('T2', '')


def test_load_spec_with_fields_reads_the_issue_once() -> None:
    # The chat text and the rewrite baseline must come from the same fetch: an edit landing
    # between two reads would shift the baseline while the chat still sees the older content.
    views: list[Any] = []

    async def fake_gh_issue_view(num: Any, cwd: Any = None,
                                 max_bytes: Any = None) -> Any:
        views.append(num)
        return {'title': 'T', 'body': 'B', 'comments': []}

    source = refine.resolve_source(types.SimpleNamespace(
        source=None, issue='218', linear=None))
    with patch.object(refine, 'gh_issue_view', new=fake_gh_issue_view):
        text, fields = asyncio.run(refine.load_spec_with_fields(source, '/tmp'))
    assert views == [218]  # exactly one fetch
    assert fields == ('T', 'B')
    assert text.startswith('Issue #218: T')


def test_load_spec_with_fields_reads_the_ticket_once() -> None:
    reads: list[Any] = []

    async def fake_linear_context(ticket: Any, cwd: Any = None,
                                  max_bytes: Any = None) -> str:
        reads.append(ticket)
        return '{"title": "T", "description": "D"}'

    source = refine.resolve_source(types.SimpleNamespace(
        source=None, issue=None, linear='ENG-5'))
    with patch.object(refine, 'linear_context', new=fake_linear_context):
        text, fields = asyncio.run(refine.load_spec_with_fields(source, '/tmp'))
    assert reads == ['ENG-5']  # exactly one fetch
    assert fields == ('T', 'D')
    assert 'description' in text


def test_run_refine_binds_remote_load_to_the_spec_limit(capsys: Any) -> None:
    # A remote source (issue/ticket) cannot be stat'd before reading, so its fetch is bounded
    # at the same 4x-byte ceiling a .mrsh is guarded by — and --check, which analyzes the full
    # source by design, is unbounded.
    calls: list[Any] = []

    async def fake_load(source: Any, cwd: Any, **kw: Any) -> Any:
        calls.append(kw.get('max_bytes'))
        return 'ISSUE TEXT', ('T', 'B')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('bail', None, '')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_repo_gate', new=AsyncMock(return_value='acme/widget')), \
         patch.object(refine, 'load_spec_with_fields', new=fake_load), \
         patch.object(refine, '_resolve_window',
                      new=AsyncMock(return_value=200000)):
        assert asyncio.run(refine.run_refine(_args(issue='218'))) == 1
        assert asyncio.run(refine.run_refine(_args(issue='218', check=True))) == 1
    assert calls == [refine.REFINE_SPEC_LIMIT * 4, None]


def test_run_refine_bail_does_not_write(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('bail', None, '')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    with open(p) as f:
        assert f.read() == 'original'
    assert 'Bailed out' in capsys.readouterr().out


def test_run_refine_timeout_reports_detail(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult(
            'timeout', None, 'Reached the maximum number of turns')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)):
        rc = asyncio.run(refine.run_refine(_args(source=p, max_turns=3)))
    assert rc == 1
    assert 'maximum number of turns' in capsys.readouterr().out


def test_run_refine_chat_error_is_handled_and_never_writes(
        tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('error', None, 'context overflow detail')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    assert 'context overflow detail' in capsys.readouterr().out
    with open(p) as f:
        assert f.read() == 'original'


def test_run_refine_apply_failure_reports_error(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('original')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_chat(**k: Any) -> Any:
        return refine.ChatResult('locked', {'spec': 'NEW'}, 'x')

    async def boom(source: Any, payload: Any, cwd: Any) -> None:
        raise Exception('disk full')

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, 'run_refine_chat', new=fake_chat), \
         patch.object(refine, '_apply', new=boom), \
         patch.object(refine, '_is_git_repo', new=AsyncMock(return_value=False)), \
         patch.object(refine, '_read_line', new=lambda: 'y'):
        rc = asyncio.run(refine.run_refine(_args(source=p)))
    assert rc == 1
    # The failure is reported to stderr (the source is left untouched).
    assert 'failed to update' in capsys.readouterr().err
    with open(p) as f:
        assert f.read() == 'original'


# --- run_refine_chat (the multi-turn loop, mocked mapper) ---------------------


def test_run_refine_chat_first_turn_lock_goes_through_the_gate(
        capsys: Any) -> None:
    # A lock on the very first turn (the model believes the spec is already resolved) is
    # still gated: the session asks the person directly before showing the proposal, and
    # the model itself is instructed not to ask a readiness question — the gate is the
    # flow's only one.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the full spec')
    reads: list[str] = []

    def read() -> str:
        reads.append('y')
        return 'y'

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=2,
            read_line=read))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the full spec')}
    assert reads == ['y']  # the single readiness question, at the gate
    out = capsys.readouterr().out
    # The proposal appears exactly once: the rendered payload at the acceptance, never the
    # lock's raw text.
    assert out.count('the full spec') == 1


def test_run_refine_chat_bails_on_user_token() -> None:
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          ['Let me ask: what about X?'])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: '!bail'))
    assert res.status == 'bail'
    assert res.payload is None


def test_run_refine_chat_bails_when_assistant_bails() -> None:
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          ['[[DESIGN:BAIL]]\ncannot resolve this'])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'bail'


def test_run_refine_chat_times_out() -> None:
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(['still working...'])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=2,
            read_line=lambda: 'go on'))
    assert res.status == 'timeout'
    assert res.payload is None


def test_run_refine_chat_retries_a_malformed_lock(capsys: Any) -> None:
    malformed = '[[DESIGN:LOCKED]]\nI locked it but forgot the section'
    good = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('fixed spec')
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([malformed, good])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('fixed spec')}
    # The malformed first-turn lock is never shown: no raw protocol text reaches the screen,
    # and the corrected proposal appears only once the user accepts it.
    assert '[[DESIGN:LOCKED]]' not in capsys.readouterr().out


def test_run_refine_chat_rejects_a_noncompilable_mrsh_lock(capsys: Any) -> None:
    # A .mrsh rewrite must be a .mrsh: a lock whose spec the format parser rejects (here, a
    # free-form document instead of func sections) is never gated or shown — the parser's
    # error is fed back to the assistant, and only the corrected lock reaches the gate.
    bad = ('[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n'
           '# Purpose\nA free-form document with no func sections.')
    good = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the valid spec')
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([bad, good])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the valid spec')}
    out = capsys.readouterr().out
    assert 'free-form document' not in out  # the non-compilable proposal was never shown
    assert 'No functions or types found in file' in out  # the parser error was reported
    assert 'Show the proposal now?' in out  # only the corrected lock reached the gate


def test_run_refine_chat_accepts_a_subsection_mrsh_lock(capsys: Any) -> None:
    # The extended .mrsh format allows "##" subsections inside a function section (the usage-
    # examples list still last): such a lock passes the format gate and locks normally.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec_subsections('subsection spec')
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec_subsections('subsection spec')}
    assert 'Show the proposal now?' in capsys.readouterr().out


# --- spec endpoint verification -----------------------------------------------


def test_spec_urls_extracts_urls_including_placeholders() -> None:
    # A URL runs through its `{placeholder}` group even when the placeholder contains a
    # space, trailing sentence punctuation is not part of it, and duplicates are dropped
    # (in order of first appearance).
    text = ('See https://api.example.com/v1/search?q={URL-encoded location}&limit=10 '
            'and https://docs.example.com/page. Also (https://link.example.com/a) '
            'and https://api.example.com/v1/search?q={URL-encoded location}&limit=10 '
            'again.')
    assert refine._spec_urls(text) == [
        'https://api.example.com/v1/search?q={URL-encoded location}&limit=10',
        'https://docs.example.com/page',
        'https://link.example.com/a',
    ]


def test_spec_urls_trims_a_long_run_of_unmatched_parens() -> None:
    # A URL followed by a long run of unmatched ')' (a markdown link's closing paren plus
    # prose) trims in one slice: re-counting the parens over the shrinking string on every
    # strip would take quadratic CPU, and --check has no source-size limit.
    url = 'https://x.example.com/a'
    assert refine._spec_urls(url + ')' * 100_000) == [url]
    # The trim never removes more than the trailing run allows (parens inside the URL
    # still balance): the original strip-loop behavior, kept.
    assert refine._spec_urls('https://x.example.com/(a))b)') == \
        ['https://x.example.com/(a))b']


def test_spec_urls_nested_matches_are_not_endpoints_of_their_own() -> None:
    # A URL starting inside another URL's span is part of it, not an endpoint of its
    # own — and skipping nested matches keeps the scan linear instead of quadratic,
    # where each nested match re-reading the same long span would dominate a
    # whitespace-free token full of URLs.
    assert refine._spec_urls(
        'https://a.example.com/https://b.example.com') == \
        ['https://a.example.com/https://b.example.com']
    big = ''.join(f'https://h{i}.example.com/v1' for i in range(20_000))
    assert refine._spec_urls(big) == [big]


def test_spec_urls_finds_nothing_without_urls() -> None:
    # A bare hostname or a non-http(s) scheme is not an endpoint to probe.
    assert refine._spec_urls('no links here: api.example.com and ftp://files.example.com') \
        == []


def test_spec_urls_strips_inline_code_backticks() -> None:
    # The model wraps URLs in inline code spans (`https://x/`); the closing backtick is not
    # part of the URL — a probe that kept it would request a bogus path, get a 404, and flag
    # a live endpoint dead.
    text = ('GeoIP is `https://ipwho.is/` and the forecast is '
            '`https://api.open-meteo.com/v1/forecast` (see the docs).')
    assert refine._spec_urls(text) == [
        'https://ipwho.is/',
        'https://api.open-meteo.com/v1/forecast',
    ]


def test_spec_endpoint_errors_probes_bare_urls_from_inline_code() -> None:
    # A live endpoint wrapped in inline code must not be reported dead: the probe goes out
    # to the bare URL, without the backtick the model wrapped it in.
    url = 'https://live.example.com/v1/{id}'
    probed: list[str] = []

    async def fake_get(url2: str, timeout: int = 0) -> Any:
        probed.append(url2)
        return 200, 'application/json', b'{}'

    with patch.object(tools, 'http_get', new=fake_get):
        errors = asyncio.run(refine._spec_endpoint_errors(f'GeoIP is `{url}`.'))
    assert errors == []
    assert probed == ['https://live.example.com/v1/Test']


def test_spec_endpoint_errors_flags_dead_routes_and_unreachable_hosts() -> None:
    # A 404/410 route, a 5xx server failure, and an unreachable host are dead; a 200 is
    # alive, and the probe substitutes a sample value for each placeholder.
    dead = 'https://dead.example.com/api?q={q}'
    gone = 'https://gone.example.com/v1'
    broken = 'https://broken.example.com/v1'
    live = 'https://live.example.com/v1/{id}'
    probed: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        probed.append(url)
        if url.startswith('https://gone.example.com'):
            raise OSError('[Errno -3] name resolution failed')
        if url.startswith('https://broken.example.com'):
            raise _http_error(503, url)
        if url == 'https://live.example.com/v1/Test':
            return 200, 'application/json', b'{}'
        raise _http_error(410, url)

    with patch.object(tools, 'http_get', new=fake_get):
        errors = asyncio.run(
            refine._spec_endpoint_errors(f'{dead} {gone} {broken} {live}'))
    # The probes run concurrently: the errors keep the spec's URL order, but the fetches
    # themselves need not complete in it.
    assert errors == [
        f'{dead} — HTTP 410',
        f'{gone} — unreachable ([Errno -3] name resolution failed)',
        f'{broken} — HTTP 503',
    ]
    assert sorted(probed) == sorted([
        'https://dead.example.com/api?q=Test',
        'https://gone.example.com/v1',
        'https://broken.example.com/v1',
        'https://live.example.com/v1/Test'])


def test_spec_endpoint_errors_probes_run_concurrently() -> None:
    # A sweep's latency is its slowest probe, not the sum: every probe must be in flight
    # before the first one finishes (serial awaits would interleave start/end per URL).
    events: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        events.append(f'start {url}')
        await asyncio.sleep(0.05)
        events.append(f'end {url}')
        return 200, 'application/json', b'{}'

    urls = ' '.join(f'https://host{i}.example.com/v1' for i in range(4))
    with patch.object(tools, 'http_get', new=fake_get):
        assert asyncio.run(refine._spec_endpoint_errors(urls)) == []
    assert len(events) == 8
    assert all(e.startswith('start') for e in events[:4])  # all four overlapped


def test_spec_endpoint_errors_probes_every_url_in_batches() -> None:
    # The batch width bounds concurrency, not coverage: every named URL is probed, so a
    # dead endpoint beyond the first batch cannot slip past the gate unreported.
    urls = [f'https://host{i}.example.com/v1' for i in range(12)]
    probed: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        probed.append(url)
        if url.startswith('https://host11.example.com'):
            raise _http_error(404, url)
        return 200, 'application/json', b'{}'

    with patch.object(tools, 'http_get', new=fake_get):
        errors = asyncio.run(refine._spec_endpoint_errors(' '.join(urls)))
    assert errors == ['https://host11.example.com/v1 — HTTP 404']
    assert len(probed) == 12


def test_spec_endpoint_errors_stops_the_sweep_when_the_network_is_down() -> None:
    # A batch in which nothing connects means the network is down: later batches are
    # skipped (they could not verify anything), and nothing is reported.
    urls = [f'https://host{i}.example.com/v1' for i in range(15)]
    probed: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        probed.append(url)
        raise OSError('[Errno -3] name resolution failed')

    with patch.object(tools, 'http_get', new=fake_get):
        errors = asyncio.run(refine._spec_endpoint_errors(' '.join(urls)))
    assert errors == []
    assert len(probed) == 10  # one batch, then the sweep stops


def test_spec_endpoint_errors_a_blocked_batch_does_not_stop_the_sweep() -> None:
    # An SSRF-blocked probe never touches the network: a batch of only non-public hosts
    # is not evidence of an outage, so later batches (which may hold public endpoints)
    # are still probed.
    urls = [f'https://host{i}.internal.example.com/v1' for i in range(10)]
    urls += ['https://public.example.com/v1', 'https://dead.example.com/v1']
    probed: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        probed.append(url)
        if url.startswith('https://dead.example.com'):
            raise _http_error(404, url)
        if 'internal.example.com' in url:
            raise Exception(f'blocked: {url.split("//")[1]} is not a public host '
                            '(SSRF guard)')
        return 200, 'application/json', b'{}'

    with patch.object(tools, 'http_get', new=fake_get):
        errors = asyncio.run(refine._spec_endpoint_errors(' '.join(urls)))
    assert errors == ['https://dead.example.com/v1 — HTTP 404']
    assert len(probed) == 12


def test_spec_endpoint_errors_an_outage_batch_reports_nothing() -> None:
    # When a later batch fails to connect entirely, its failures are the outage's, not
    # the endpoints': they are not reported (only what earlier, connected batches found).
    urls = [f'https://host{i}.example.com/v1' for i in range(9)]
    urls += ['https://dead.example.com/v1']
    urls += [f'https://outage{i}.example.com/v1' for i in range(4)]
    probed: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        probed.append(url)
        if url.startswith('https://dead.example.com'):
            raise _http_error(404, url)
        if url.startswith('https://outage'):
            raise OSError('[Errno -3] name resolution failed')
        return 200, 'application/json', b'{}'

    with patch.object(tools, 'http_get', new=fake_get):
        errors = asyncio.run(refine._spec_endpoint_errors(' '.join(urls)))
    assert errors == ['https://dead.example.com/v1 — HTTP 404']
    assert len(probed) == 14


def test_spec_endpoint_errors_reports_the_rest_when_the_deadline_expires() -> None:
    # Out of budget, the sweep reports the not-yet-probed URLs as unverified (the person
    # can keep them or have them checked) instead of skipping them silently.
    urls = [f'https://host{i}.example.com/v1' for i in range(12)]
    probed: list[str] = []

    async def fake_get(url: str, timeout: int = 0) -> Any:
        probed.append(url)
        await asyncio.sleep(0.2)
        return 200, 'application/json', b'{}'

    with patch.object(tools, 'http_get', new=fake_get), \
         patch.object(refine, 'ENDPOINT_PROBE_SWEEP_DEADLINE', 0.1):
        errors = asyncio.run(refine._spec_endpoint_errors(' '.join(urls)))
    assert probed == urls[:10]  # one batch, then the deadline
    assert errors == [f'{u} — not verified (probe deadline)' for u in urls[10:]]


def test_spec_endpoint_errors_treats_a_rejected_probe_as_alive() -> None:
    # A 400/403 for the probe's sample values means the endpoint exists and refused the
    # request (some APIs want auth or a real value): alive, not dead.
    url = 'https://strict.example.com/v1/{id}'

    async def fake_get(url2: str, timeout: int = 0) -> Any:
        raise _http_error(403, url2)

    with patch.object(tools, 'http_get', new=fake_get):
        assert asyncio.run(refine._spec_endpoint_errors(url)) == []


def test_spec_endpoint_errors_skips_nonpublic_hosts() -> None:
    # The SSRF guard refuses non-public hosts: such an endpoint is unverifiable from here,
    # not dead (a spec for an internal API is the user's to own, not this probe's).
    url = 'https://api.internal.example.com/v1'

    async def fake_get(url2: str, timeout: int = 0) -> Any:
        raise Exception('blocked: api.internal.example.com is not a public host '
                        '(SSRF guard)')

    with patch.object(tools, 'http_get', new=fake_get):
        assert asyncio.run(refine._spec_endpoint_errors(url)) == []


def test_spec_endpoint_errors_skips_all_when_the_network_is_down() -> None:
    # When every probe fails to connect, it is the network that is down, not the endpoints:
    # nothing is reported, so a refine session stays usable offline.
    urls = 'https://one.example.com/a https://two.example.com/b'

    async def fake_get(url: str, timeout: int = 0) -> Any:
        raise OSError('[Errno -3] Temporary failure in name resolution')

    with patch.object(tools, 'http_get', new=fake_get):
        assert asyncio.run(refine._spec_endpoint_errors(urls)) == []


def test_run_refine_chat_sends_back_a_dead_endpoint_when_not_kept(capsys: Any) -> None:
    # A lock whose payload names an endpoint that could not be verified as working is
    # asked of the person
    # before the gate; when they want a working alternative, the failure is fed back to the
    # assistant, and only the corrected lock reaches the gate.
    dead_url = 'https://dead.example.com/api?q={q}'
    bad = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' \
        + _mrsh_spec_with_url('dead endpoint', dead_url)
    good = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the live spec')

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        return [f'{dead_url} — HTTP 404'] if dead_url in text else []

    lines = iter(['n', 'y'])  # decline the keep question, then accept the proposal gate
    with patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([bad, good])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: next(lines, 'y')))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the live spec')}
    out = capsys.readouterr().out
    assert 'dead endpoint' not in out  # the dead-endpoint proposal was never shown
    assert 'could not be verified as working' in out  # the endpoint question was on screen
    assert 'Show the proposal now?' in out


def test_run_refine_chat_keeps_a_confirmed_dead_endpoint(capsys: Any) -> None:
    # When the person confirms the dead endpoint as-is (a private API a generic sample
    # request cannot reach), the lock is exempted and proceeds to the proposal gate.
    dead_url = 'https://private.example.com/v1'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec_with_url('the spec', dead_url)

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        if skip is not None and dead_url in skip:
            return []
        return [f'{dead_url} — HTTP 404'] if dead_url in text else []

    lines = iter(['y', 'y'])  # keep the endpoint, then accept the proposal gate
    with patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: next(lines, 'y')))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec_with_url('the spec', dead_url)}
    out = capsys.readouterr().out
    assert 'could not be verified as working' in out  # the endpoint question was on screen
    assert 'Show the proposal now?' in out


def test_run_refine_chat_approved_endpoint_is_neither_probed_nor_prompted(
        capsys: Any) -> None:
    # An endpoint the person already confirmed (approved_endpoints) is not probed at the lock
    # gate: no question, straight to the proposal gate.
    dead_url = 'https://private.example.com/v1'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec_with_url('the spec', dead_url)
    reads = 0

    def read_line() -> str:
        nonlocal reads
        reads += 1
        return 'y'  # only the proposal gate asks

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        assert skip is not None and dead_url in skip, \
            'an approved endpoint must be skipped, not re-probed'
        return []

    with patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line,
            approved_endpoints={dead_url}))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec_with_url('the spec', dead_url)}
    assert reads == 1  # the endpoint question was never asked
    assert 'could not be verified as working' not in capsys.readouterr().out


def test_run_refine_chat_bails_at_the_endpoint_question(capsys: Any) -> None:
    # A bail token at the endpoint question ends the session (the source is not modified).
    dead_url = 'https://private.example.com/v1'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec_with_url('the spec', dead_url)

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        return [f'{dead_url} — HTTP 404']

    with patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: '!bail'))
    assert res.status == 'bail'
    assert res.payload is None


def test_run_refine_chat_locks_when_the_endpoint_probe_reports_nothing(
        capsys: Any) -> None:
    # The probe reports nothing when the network is down: a lock that names an endpoint
    # still reaches the gate, so refine stays usable offline.
    url = 'https://api.example.com/v1'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec_with_url('the spec', url)

    async def fake_probe(text: str, skip: Any = None) -> list[str]:
        return []

    with patch.object(refine, '_spec_endpoint_errors', new=fake_probe), \
         patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec_with_url('the spec', url)}
    assert 'Show the proposal now?' in capsys.readouterr().out


def test_initial_chat_message_reports_dead_endpoints() -> None:
    # The source's endpoints that could not be verified as working are told to the
    # assistant up front (a harness measurement, presented as its own section), and the
    # section is absent otherwise.
    msg = refine._initial_chat_message(
        'mrsh', 'SPEC', [], [], '',
        dead_endpoints=['https://dead.example.com/api — HTTP 404'])
    assert 'External endpoints that could not be verified as working' in msg
    assert 'https://dead.example.com/api — HTTP 404' in msg
    assert 'cannot be locked into the specification' in msg
    assert 'External endpoints that could not be verified as working' not in \
        refine._initial_chat_message(
        'mrsh', 'SPEC', [], [], '', dead_endpoints=None)


def test_run_refine_chat_lock_and_bail_together_retries(capsys: Any) -> None:
    # A first-turn response carrying a lock with proposal content but failing to parse (here,
    # a self-contradictory lock and bail) is neither shown nor locked: the proposal must not
    # appear before the go-ahead, so it stays off the screen and the conversation continues.
    contradictory = '[[DESIGN:LOCKED]]\n[[DESIGN:BAIL]]\n[[NEW:SPEC]]\nspec'
    good = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([contradictory, good])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the spec')}
    out = capsys.readouterr().out
    assert '[[DESIGN:LOCKED]]' not in out  # the unparseable lock's payload stays hidden
    assert '[[DESIGN:BAIL]]' not in out


def test_run_refine_chat_bail_line_inside_payload_locks() -> None:
    # A bail line inside the rewritten spec is content, not a signal: the chat locks and saves
    # the spec verbatim instead of retrying or bailing.
    spec = '[[DESIGN:BAIL]]\n' + _mrsh_spec('inside the payload')
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + spec
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': spec}


def test_run_refine_chat_locks_issue_with_readonly_tools() -> None:
    # An issue source gets a read-only tool context (build_commands + tool_instructions);
    # the assistant can still lock on the first turn without running a tool.
    locked = ('[[DESIGN:LOCKED]]\n'
              '[[NEW:TITLE]]\nUpdated Issue Title\n'
              '[[NEW:BODY]]\nUpdated body text')
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])), \
         patch.object(refine, '_resolve_window',
                      new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='issue', spec_text='ISSUE SPEC', ambiguities=['a'], errors=[],
            current_repo='acme/widget', in_repo=True, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'title': 'Updated Issue Title',
                           'body': 'Updated body text'}


def test_run_refine_chat_mrsh_in_repo_gets_readonly_tools() -> None:
    # A .mrsh refined inside a repository gets the read-only codebase tools too (the grounded note
    # is appended to the system prompt), matching issue/linear.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('locked spec')
    captured: dict[str, str] = {}

    def get_mapper(system: str, **k: Any) -> Any:
        captured['system'] = system
        return _scripted_mapper([locked])

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='acme/widget', in_repo=True, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert SPEC_CHECK_GROUNDED_NOTE in captured['system']


def test_run_refine_chat_standalone_mrsh_gets_repo_independent_tools() -> None:
    # A .mrsh outside a git working tree has no codebase to inspect (no grounded note, no git
    # tool), but the repo-independent tools (web, local reads, notes) are still available, so
    # the assistant can e.g. fetch a documentation page the person links instead of asking
    # them to paste it.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('locked spec')
    captured: dict[str, str] = {}

    def get_mapper(system: str, **k: Any) -> Any:
        captured['system'] = system
        return _scripted_mapper([locked])

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert SPEC_CHECK_GROUNDED_NOTE not in captured['system']
    assert 'web-search' in captured['system']  # the web tools are offered
    assert '$ git ' not in captured['system']  # but not the repo-bound git tool


def test_run_refine_chat_turns_are_rendered_as_markdown(capsys: Any) -> None:
    # Every assistant turn is rendered as markdown (headings, bold, lists consumed into the
    # rendering), not dumped as raw markup.
    reply = '# Heading\n\n**bold** text'
    with patch.object(refine, 'get_mapper',
                      new=lambda system, **k: _scripted_mapper([reply])), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=2,
            read_line=lambda: '!bail'))
    assert res.status == 'bail'
    out = capsys.readouterr().out
    assert 'marsha>' in out
    assert 'Heading' in out and 'bold' in out
    assert '**bold**' not in out  # the markup is rendered, not shown raw


def test_run_refine_chat_lock_turn_prints_a_clean_proposal(capsys: Any) -> None:
    # A lock turn is shown once, as a clean rendered proposal: the preamble is kept, the raw
    # protocol markers are never shown, and the payload is not repeated (the confirmation
    # after the chat refers to it instead).
    question = 'Ready for me to propose the updated specification?'
    locked = ('All settled — here is the locked design.\n'
              '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n'
              + _mrsh_spec('the full spec body'))
    with patch.object(refine, 'get_mapper',
                      new=lambda system, **k: _scripted_mapper(
                          [question, locked])), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    out = capsys.readouterr().out
    assert '[[DESIGN:LOCKED]]' not in out  # the raw protocol text is never shown
    assert '[[NEW:SPEC]]' not in out
    assert 'All settled' in out  # the preamble is kept
    assert out.count('the full spec body') == 1  # the proposal appears exactly once


def test_run_refine_chat_declined_proposal_keeps_refining(capsys: Any) -> None:
    # The model's lock is not the user's: when the user declines the proposal, the rewrite is
    # not shown and the conversation continues with their objection; a later lock is accepted
    # only once the user accepts the proposal.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')
    # Replies, in order: the declined proposal (the gate's objection), an ordinary next-turn
    # reply, the declined proposal again (the gate's objection), and the accepted proposal.
    lines = iter(['x', 'n', 'make the error message friendlier', 'y'])

    def read_line() -> str:
        return next(lines)

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          [locked, 'Working on the error message.', locked, locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line))
    assert res.status == 'locked'
    out = capsys.readouterr().out
    assert 'Show the proposal now?' in out
    # The rendered proposal appears exactly once: at the acceptance, not at the decline.
    assert out.count('The design is locked. The updated source:') == 1


def test_run_refine_chat_decline_with_inline_objection(capsys: Any) -> None:
    # A decline that carries the objection in the same reply ("no, make the error message
    # friendlier") is not discarded: the reply itself becomes the objection, and the user is
    # not prompted for it a second time.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')
    # The first reply is the gate's: the decline with its objection in the same text. The
    # later replies are an ordinary next-turn reply and the accepted proposal.
    lines = iter(['no, make the error message friendlier', 'x', 'y'])

    def read_line() -> str:
        return next(lines)

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          [locked, 'Working on the error message.', locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line))
    assert res.status == 'locked'
    out = capsys.readouterr().out
    # The proposal question is asked once at the decline and once at the acceptance: the
    # inline objection was not re-prompted (a discarded reply would have consumed the 'y').
    assert out.count('Show the proposal now?') == 2


def test_print_apply_prompt_is_separated_and_boxed(capsys: Any) -> None:
    # The confirmation follows directly after the (long) proposal, where a bare line would read
    # as part of the document: it must be separated by a blank line and set in a box.
    refine._print_apply_prompt('the .mrsh file')
    out = capsys.readouterr().out
    assert out.startswith('\n')  # a blank line of separation from the proposal above
    assert 'Apply the locked design shown above' in out
    assert '╔' in out and '╚' in out  # the double-line (flower) box


def test_run_refine_chat_repo_without_origin_name_still_gets_tools() -> None:
    # A working tree whose origin remote is missing or unparseable still has a codebase to
    # inspect: the tools are gated on being in a repo (in_repo), not on a parseable repo name.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('locked spec')
    captured: dict[str, str] = {}

    def get_mapper(system: str, **k: Any) -> Any:
        captured['system'] = system
        return _scripted_mapper([locked])

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=True, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert SPEC_CHECK_GROUNDED_NOTE in captured['system']


def test_run_refine_chat_bail_marker_in_prose_does_not_bail() -> None:
    # A marker quoted in prose is not the protocol line: that turn is not a bail, so the user is
    # prompted and the next turn can still lock.
    prose = "If this is hopeless I would emit [[DESIGN:BAIL]] - but let's try one more thing?"
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')
    lines = iter(['go on', 'y'])

    def read_line() -> str:
        return next(lines, '!bail')

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([prose, locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the spec')}


def test_run_refine_chat_lock_marker_in_prose_is_not_a_lock() -> None:
    # Merely mentioning [[DESIGN:LOCKED]] in prose is not the protocol line: no lock is
    # attempted, the user is prompted, and a later exact line still locks.
    prose = 'Once we settle the last point I will emit [[DESIGN:LOCKED]] and the new spec.'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')
    lines = iter(['sure', 'y'])

    def read_line() -> str:
        return next(lines, '!bail')

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([prose, locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the spec')}


def test_run_refine_chat_tool_cap_notes_unprocessed_result(capsys: Any) -> None:
    # When a turn's tool-round budget runs out with a result still unprocessed, the user is told
    # the assistant has not yet seen that result; the user is NOT prompted against the unfinished
    # turn (the assistant continues from that result at the top of the next turn), and the
    # tool-request response is not checked for lock/bail lines.
    def read_line() -> str:
        raise AssertionError('no prompt should be solicited against an unfinished turn')

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          ['Investigating.\n$ git grep needle'])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=1, read_line=read_line))
    assert res.status == 'timeout'
    assert 'has not yet' in capsys.readouterr().out


def test_run_refine_chat_final_turn_processes_pending_result() -> None:
    # When the tool-round budget runs out on the final turn, the assistant gets one extra
    # call to process the pending tool result, so the result can still inform the outcome
    # (here, a lock) instead of the session timing out without ever seeing it. The first
    # turn is a plain question so the user has had a turn to give the go-ahead the lock
    # requires.
    cmd = 'Looking.\n$ git grep needle'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          ['Ready for me to propose the updated specification?']
                          + [cmd] * 10 + [locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=2, read_line=lambda: 'y'))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the spec')}


def test_run_refine_chat_final_followup_tool_request_cannot_lock() -> None:
    # The final follow-up response is checked for a trailing tool command exactly like the main
    # loop: a lock/payload that still requests a tool must not lock and write (and, for a .mrsh,
    # the command line would otherwise run to the end and leak into the saved spec).
    cmd = 'Looking.\n$ git grep needle'
    sneaky = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nspec\n$ git grep x'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([cmd] * 10 + [sneaky])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=1, read_line=lambda: 'x'))
    assert res.status == 'timeout'
    assert res.payload is None


def test_run_refine_chat_overflow_returns_handled_result() -> None:
    # When a model call overflows the context window (compaction could not keep up), the
    # session ends with a handled 'error' result instead of an unhandled abort.
    class _Overflowing:
        async def run(self, messages: Any) -> str:
            raise ContextOverflowError('prompt exceeds context window: boom')

    with patch.object(refine, 'get_mapper', new=lambda *a, **k: _Overflowing()):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'error'
    assert res.payload is None
    assert 'context window' in res.detail


def test_run_refine_chat_pending_lock_hides_payload(capsys: Any) -> None:
    # A lock that still ends with a tool command is in-progress (the tool result may change
    # it): the proposal payload must not appear on screen, only the narration — like any
    # un-accepted lock, it is shown only once the user accepts it.
    lock_cmd = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nsecret proposal body\n$ git grep needle'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([lock_cmd])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=1, read_line=lambda: 'x'))
    assert res.status == 'timeout'
    out = capsys.readouterr().out
    assert 'secret proposal body' not in out
    assert '[[DESIGN:LOCKED]]' not in out


def test_run_refine_chat_followup_malformed_lock_hides_payload(capsys: Any) -> None:
    # A malformed lock in the final tool-result follow-up can still carry proposal content
    # (the parse failed, the text did not): like any un-accepted lock, its payload stays off
    # the screen — only the narration is shown.
    cmd = 'Looking.\n$ git grep needle'
    malformed = 'Preamble text.\n[[DESIGN:LOCKED]]\nsecret malformed payload'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          ['Question turn.'] + [cmd] * 10 + [malformed])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=2, read_line=lambda: 'x'))
    assert res.status == 'timeout'
    out = capsys.readouterr().out
    assert 'secret malformed payload' not in out
    assert '[[DESIGN:LOCKED]]' not in out
    assert 'Preamble text.' in out  # the narration is kept


def test_run_refine_chat_lock_with_pending_command_does_not_lock() -> None:
    # A response that ends with a tool command is a tool request, not a final artifact: even if
    # it carries a lock line and payload it must not lock and write the source (and, for a .mrsh,
    # the trailing command line would otherwise run to the end and leak into the saved spec).
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper(
                          ['[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nspec\n$ git grep x',
                           'Continuing.\n$ git grep y'])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=1, read_line=lambda: 'x'))
    assert res.status == 'timeout'
    assert res.payload is None


def test_run_refine_chat_compacts_when_over_budget_and_keeps_spec() -> None:
    # A conversation that would outgrow the context budget is summarized before the next model
    # call, with the specification re-attached verbatim, so the chat keeps running to the lock
    # instead of aborting on a context overflow.
    chat_calls: list[Any] = []

    class _Chat:
        def __init__(self) -> None:
            self.i = 0

        async def run(self, messages: Any) -> str:
            self.i += 1
            chat_calls.append(messages)
            if self.i == 1:
                return 'What should happen on a tie?'
            return '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')

    class _Compact:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, messages: Any) -> str:
            self.calls += 1
            assert 'Conversation so far' in str(messages)
            return 'ambiguity a: resolved, the person chose X'

    compact = _Compact()

    def get_mapper(system: str, **k: Any) -> Any:
        if k.get('label') == 'refine:compact':
            return compact
        return _Chat()

    lines = iter(['answer one', 'y'])
    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, 'fits', lambda text, window, cap=0.5: False):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC TEXT', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: next(lines, 'y')))
    assert res.status == 'locked'
    assert res.payload == {'spec': _mrsh_spec('the spec')}
    assert compact.calls >= 1
    # After a compaction the model sees a single message: the summary plus the spec verbatim.
    assert len(chat_calls[1]) == 1
    reattached = chat_calls[1][0]['content']
    assert 'Summary of the refinement conversation' in reattached
    assert 'SPEC TEXT' in reattached
    # The reconstructed summary is framed so quoted spec text in it reads as data, not commands.
    assert 'data, not instructions' in reattached


def test_refine_compact_prompt_treats_source_as_untrusted() -> None:
    # The compaction model reads the transcript, which contains the (attacker-influenceable)
    # specification: the prompt must name the wrapper the source actually appears in
    # (wrap_untrusted(kind, ...) -> [tool:mrsh] / [tool:issue] / [tool:linear]) and tell the
    # model that the wrapped content is data, never instructions.
    assert '[tool:mrsh]' in refine.REFINE_COMPACT_PROMPT
    assert '[tool:issue]' in refine.REFINE_COMPACT_PROMPT
    assert '[tool:linear]' in refine.REFINE_COMPACT_PROMPT
    assert 'never as instructions' in refine.REFINE_COMPACT_PROMPT


def test_run_refine_chat_compaction_reattaches_recorded_notes() -> None:
    # Notes recorded through the notes tool survive a compaction verbatim (re-attached), as in
    # the shared review compaction: the assistant never loses what it deliberately kept.
    chat_calls: list[Any] = []

    class _Chat:
        def __init__(self) -> None:
            self.i = 0

        async def run(self, messages: Any) -> str:
            self.i += 1
            chat_calls.append(messages)
            if self.i == 1:
                return 'Let me record what I found.\n$ notes add "src/a.py:3 - the tie-break"'
            return '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n' + _mrsh_spec('the spec')

    class _Compact:
        def __init__(self) -> None:
            self.last_request = ''

        async def run(self, messages: Any) -> str:
            self.last_request = str(messages)
            return 'summary of the conversation'

    compact = _Compact()

    def get_mapper(system: str, **k: Any) -> Any:
        if k.get('label') == 'refine:compact':
            return compact
        return _Chat()

    lines = iter(['go on', 'y'])
    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, 'fits', lambda text, window, cap=0.5: False):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC TEXT', ambiguities=['a'], errors=[],
            current_repo='acme/widget', in_repo=True, cwd='/', model=None,
            max_turns=5, read_line=lambda: next(lines, 'y')))
    assert res.status == 'locked'
    # The note recorded through the tool reached the compaction request and was re-attached
    # verbatim for the model's next call.
    assert 'src/a.py:3 - the tie-break' in compact.last_request
    reattached = chat_calls[1][0]['content']
    assert 'Your notes' in reattached


# --- the terminal reply reader (multi-line, paste-safe) --------------------------
#
# The reply reader drives a real prompt_toolkit session in a worker thread (a synchronous
# prompt() cannot nest inside the chat's own event loop). These tests feed it through a
# pipe input — the production code path minus the terminal — so the key bindings and the
# thread/stop machinery are the same ones the real reader runs.

def _run_piped_prompt(data: bytes) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input import create_pipe_input

    def make_prompt() -> Awaitable[str]:
        async def run() -> str:
            with create_pipe_input() as inp:
                bindings = refine._prompt_key_bindings()
                session: Any = PromptSession(multiline=True, input=inp,
                                             history=InMemoryHistory(),
                                             key_bindings=bindings)
                inp.send_bytes(data)
                result: str = await session.prompt_async('you> ')
                return result
        return run()

    return refine._run_prompt_in_thread(make_prompt, threading.Event())


def test_read_line_without_a_terminal_falls_back_to_plain_input(
        monkeypatch: Any) -> None:
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    monkeypatch.setattr('builtins.input', lambda _prompt='': 'plain reply')
    assert refine._read_line() == 'plain reply'


def test_read_line_without_a_terminal_treats_eof_as_bail(monkeypatch: Any) -> None:
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)

    def eof_input(_prompt: str = '') -> str:
        raise EOFError

    monkeypatch.setattr('builtins.input', eof_input)
    assert refine._read_line() == '!bail'


def test_prompt_reader_enter_submits_the_buffer() -> None:
    assert _run_piped_prompt(b'hello\r') == 'hello'


def test_prompt_reader_a_pasted_block_is_one_message() -> None:
    # A pasted block (the terminal's bracketed paste) lands as one reply, not one message
    # per pasted line — the failure mode of a plain input() read, which submitted every
    # pasted line on its own and shredded the conversation.
    data = b'\x1b[200~line1\nline2\nline3\x1b[201~\r'
    assert _run_piped_prompt(data) == 'line1\nline2\nline3'


def test_prompt_reader_shift_enter_inserts_a_newline() -> None:
    # Shift+Enter (where the terminal sends a distinct sequence) types a newline instead of
    # submitting: the kitty/xterm-modified and the legacy xterm forms.
    assert _run_piped_prompt(b'first\x1b[13;2usecond\r') == 'first\nsecond'
    assert _run_piped_prompt(b'first\x1bOMsecond\r') == 'first\nsecond'


def test_prompt_reader_ctrl_j_inserts_a_newline() -> None:
    # Ctrl+J is the newline key that works in every terminal (VTE — GNOME
    # Terminal — cannot report Shift- or Ctrl+Enter distinctly). Without the
    # explicit binding it would fall through to "treat \n as Enter" and submit.
    assert _run_piped_prompt(b'first\nsecond\r') == 'first\nsecond'


def test_prompt_reader_alt_enter_inserts_a_newline() -> None:
    # Alt+Enter: VTE prefixes it with ESC (ESC + CR), the one modified Enter
    # GNOME Terminal can distinguish.
    assert _run_piped_prompt(b'first\x1b\rsecond\r') == 'first\nsecond'


def test_prompt_reader_ctrl_enter_inserts_a_newline() -> None:
    # Ctrl+Enter on kitty/xterm-modified terminals (unreportable in VTE).
    assert _run_piped_prompt(b'first\x1b[13;5usecond\r') == 'first\nsecond'


def test_prompt_reader_ctrl_d_is_eof_on_an_empty_buffer() -> None:
    # Ctrl-D on an empty buffer is the session's EOF (a bail), not a submit of nothing.
    with pytest.raises(EOFError):
        _run_piped_prompt(b'\x04')


def test_prompt_reader_ctrl_d_submits_a_nonempty_buffer() -> None:
    assert _run_piped_prompt(b'yes\x04') == 'yes'


def test_prompt_reader_stop_tears_the_worker_down() -> None:
    # A stop (set by the main thread's SIGINT handler) cancels the prompt and ends the
    # runner with a KeyboardInterrupt; without it the worker — waiting on input forever —
    # would keep the process alive after the interrupt unwinds.
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input import create_pipe_input

    def make_prompt() -> Awaitable[str]:
        async def run() -> str:
            with create_pipe_input() as inp:  # no data: the prompt waits
                bindings = refine._prompt_key_bindings()
                session: Any = PromptSession(multiline=True, input=inp,
                                             history=InMemoryHistory(),
                                             key_bindings=bindings)
                result: str = await session.prompt_async('you> ')
                return result
        return run()

    stop = threading.Event()

    def stop_later() -> None:
        time.sleep(0.3)
        stop.set()

    stopper = threading.Thread(target=stop_later, daemon=True)
    stopper.start()
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        refine._run_prompt_in_thread(make_prompt, stop)
    assert time.monotonic() - start < 5
