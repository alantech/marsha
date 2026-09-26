"""Deterministic tests for the `marsha refine` subcommand (marsha.refine).

The LLM analysis (`analyze_spec`), the chat loop (`run_refine_chat`), and the GitHub/Linear CLIs
(`_gh`, `_run`, `_repo_name`) are mocked so nothing needs a network, an API key, or the external
tools. The pure helpers (`parse_issue_ref`, `resolve_source`, `parse_locked_output`, `_run_check`)
are exercised directly, and the `run_refine` driver is driven end-to-end with a real `.mrsh` file.
"""

import asyncio
import json
from typing import Any, Generator

import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import refine
from marsha import term
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
    with open(p, 'w') as f:
        f.write('x' * size)
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
        f.write('original spec')

    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a1', 'a2'], 'errors': []}

    with patch.object(refine, 'analyze_spec', new=fake_analyze):
        rc = asyncio.run(refine.run_refine(_args(source=p, check=True)))
    assert rc == 1
    assert '2 open ambiguity' in capsys.readouterr().out
    with open(p) as f:
        assert f.read() == 'original spec'  # --check never writes back


def test_run_refine_check_mrsh_locked(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path / 'spec.mrsh')
    with open(p, 'w') as f:
        f.write('spec')

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


def test_run_refine_check_issue_uses_gate_and_loader(capsys: Any) -> None:
    async def fake_analyze(spec_text: str, **k: Any) -> Any:
        return {'compilable': True, 'ambiguities': ['a'], 'errors': []}

    async def fake_gate(source: Any, cwd: Any) -> str:
        return 'acme/widget'

    async def fake_load(source: Any, cwd: Any) -> str:
        return 'ISSUE TEXT'

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, '_repo_gate', new=fake_gate), \
         patch.object(refine, 'load_spec', new=fake_load), \
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

    async def fake_load(source: Any, cwd: Any) -> str:
        return 'TICKET TEXT'

    with patch.object(refine, 'analyze_spec', new=fake_analyze), \
         patch.object(refine, '_repo_gate', new=fake_gate), \
         patch.object(refine, 'load_spec', new=fake_load), \
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
    assert 'NEW SPEC' in cap.out  # the rewrite is shown so the person can decide
    assert 'Apply this update? (y/N)' in cap.out  # explicit question, default no
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


def test_run_refine_chat_locks_on_first_turn() -> None:
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\n# Locked spec\nthe full spec'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'should not be read'))
    assert res.status == 'locked'
    assert res.payload == {'spec': '# Locked spec\nthe full spec'}


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


def test_run_refine_chat_retries_a_malformed_lock() -> None:
    malformed = '[[DESIGN:LOCKED]]\nI locked it but forgot the section'
    good = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nfixed spec'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([malformed, good])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'fixed spec'}


def test_run_refine_chat_lock_and_bail_together_retries() -> None:
    # A response carrying a bail signal before the payload is self-contradictory: it does not
    # lock (no overwrite), the assistant is asked to re-emit, and a clean follow-up locks.
    contradictory = '[[DESIGN:LOCKED]]\n[[DESIGN:BAIL]]\n[[NEW:SPEC]]\nspec'
    good = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([contradictory, good])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'the spec'}


def test_run_refine_chat_bail_line_inside_payload_locks() -> None:
    # A bail line inside the rewritten spec is content, not a signal: the chat locks and saves
    # the spec verbatim instead of retrying or bailing.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nline one\n[[DESIGN:BAIL]]\nline two'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'line one\n[[DESIGN:BAIL]]\nline two'}


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
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert res.payload == {'title': 'Updated Issue Title',
                           'body': 'Updated body text'}


def test_run_refine_chat_mrsh_in_repo_gets_readonly_tools() -> None:
    # A .mrsh refined inside a repository gets the read-only codebase tools too (the grounded note
    # is appended to the system prompt), matching issue/linear.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nlocked spec'
    captured: dict[str, str] = {}

    def get_mapper(system: str, **k: Any) -> Any:
        captured['system'] = system
        return _scripted_mapper([locked])

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='acme/widget', in_repo=True, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert SPEC_CHECK_GROUNDED_NOTE in captured['system']


def test_run_refine_chat_standalone_mrsh_has_no_tools() -> None:
    # A .mrsh with no repository has nothing to inspect, so it runs without the codebase tools
    # (no grounded note, and no context-window lookup).
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nlocked spec'
    captured: dict[str, str] = {}

    def get_mapper(system: str, **k: Any) -> Any:
        captured['system'] = system
        return _scripted_mapper([locked])

    with patch.object(refine, 'get_mapper', new=get_mapper):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert SPEC_CHECK_GROUNDED_NOTE not in captured['system']


def test_run_refine_chat_repo_without_origin_name_still_gets_tools() -> None:
    # A working tree whose origin remote is missing or unparseable still has a codebase to
    # inspect: the tools are gated on being in a repo (in_repo), not on a parseable repo name.
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nlocked spec'
    captured: dict[str, str] = {}

    def get_mapper(system: str, **k: Any) -> Any:
        captured['system'] = system
        return _scripted_mapper([locked])

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, '_resolve_window', new=AsyncMock(return_value=200000)):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=True, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert SPEC_CHECK_GROUNDED_NOTE in captured['system']


def test_run_refine_chat_bail_marker_in_prose_does_not_bail() -> None:
    # A marker quoted in prose is not the protocol line: that turn is not a bail, so the user is
    # prompted and the next turn can still lock.
    prose = "If this is hopeless I would emit [[DESIGN:BAIL]] - but let's try one more thing?"
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'
    lines = iter(['go on'])

    def read_line() -> str:
        return next(lines, '!bail')

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([prose, locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'the spec'}


def test_run_refine_chat_lock_marker_in_prose_is_not_a_lock() -> None:
    # Merely mentioning [[DESIGN:LOCKED]] in prose is not the protocol line: no lock is
    # attempted, the user is prompted, and a later exact line still locks.
    prose = 'Once we settle the last point I will emit [[DESIGN:LOCKED]] and the new spec.'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'
    lines = iter(['sure'])

    def read_line() -> str:
        return next(lines, '!bail')

    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([prose, locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=read_line))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'the spec'}


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
    # (here, a lock) instead of the session timing out without ever seeing it.
    cmd = 'Looking.\n$ git grep needle'
    locked = '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'
    with patch.object(refine, 'get_mapper',
                      new=lambda *a, **k: _scripted_mapper([cmd] * 10 + [locked])):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None,
            max_turns=1, read_line=lambda: 'x'))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'the spec'}


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
            return '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'

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

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, 'fits', lambda text, window, cap=0.5: False):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC TEXT', ambiguities=['a'], errors=[],
            current_repo='', in_repo=False, cwd='/', model=None, max_turns=5,
            read_line=lambda: 'answer one'))
    assert res.status == 'locked'
    assert res.payload == {'spec': 'the spec'}
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
            return '[[DESIGN:LOCKED]]\n[[NEW:SPEC]]\nthe spec'

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

    with patch.object(refine, 'get_mapper', new=get_mapper), \
         patch.object(refine, 'fits', lambda text, window, cap=0.5: False):
        res = asyncio.run(refine.run_refine_chat(
            kind='mrsh', spec_text='SPEC TEXT', ambiguities=['a'], errors=[],
            current_repo='acme/widget', in_repo=True, cwd='/', model=None,
            max_turns=5, read_line=lambda: 'go on'))
    assert res.status == 'locked'
    # The note recorded through the tool reached the compaction request and was re-attached
    # verbatim for the model's next call.
    assert 'src/a.py:3 - the tie-break' in compact.last_request
    reattached = chat_calls[1][0]['content']
    assert 'Your notes' in reattached
    assert 'src/a.py:3 - the tie-break' in reattached
