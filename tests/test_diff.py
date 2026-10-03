"""Deterministic tests for the `marsha diff` subcommand (marsha.diff).

The LLM-backed stages (`analyze_spec`, the implementor tool loop, the review gate) and the
validation runner are mocked so nothing needs a network or an API key. The pure helpers (title
shortening, branch slugging, commit-title detection, validation-command detection, findings/PR
rendering) are exercised directly, and the `run_diff` driver is driven end-to-end inside a real
throwaway git repository (with the git helpers running for real) to check branch creation, the
commit gate, and the exit codes.
"""
import asyncio
import contextlib
import os
import subprocess
import types
from typing import Any, Generator, Iterator

from unittest.mock import AsyncMock, patch

import pytest

from marsha import diff
from marsha import tools
from marsha.findings import Finding
from marsha.refine import ChatResult, SpecSource


def _args(**kw: Any) -> Any:
    base = dict(source=None, issue=None, linear=None, review_cycles=5,
                max_tool_failure=5, safe=False, target='python',
                target_version=None, debug=False, trace=False,
                trace_full=False, model=None, provider=None, api_base=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mrsh_spec() -> str:
    # A minimal .mrsh (the content is irrelevant to the driver tests — analyze_spec is mocked —
    # but it is a plausible spec so the real load_spec read succeeds).
    return ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')


@contextlib.contextmanager
def _chdir(path: str) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _git_repo(p: str, files: dict[str, str]) -> None:
    # A throwaway git repository with a `main` branch and one initial commit (clean tree).
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=p, check=True)
    subprocess.run(['git', 'config', 'user.email', 't@t'], cwd=p, check=True)
    subprocess.run(['git', 'config', 'user.name', 't'], cwd=p, check=True)
    for name, content in files.items():
        with open(os.path.join(p, name), 'w') as f:
            f.write(content)
    subprocess.run(['git', 'add', '-A'], cwd=p, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=p, check=True)


def _git_out(p: str, *args: str) -> str:
    r = subprocess.run(['git', *args], cwd=p, capture_output=True, text=True)
    return r.stdout.strip()


def _spec_path(p: str) -> str:
    return os.path.join(p, 'spec.mrsh')


# --- pure helpers ---------------------------------------------------------------

def test_short_title_first_sentence() -> None:
    assert diff._short_title('Add a marsha diff command. It does X.') == \
        'Add a marsha diff command.'


def test_short_title_empty_falls_back() -> None:
    assert diff._short_title('') == 'changes'
    assert diff._short_title('   ') == 'changes'


def test_short_title_truncates_long_titles() -> None:
    assert len(diff._short_title('a' * 200)) <= 60


def test_slugify_basic() -> None:
    assert diff._slugify('Add a Marsha Diff command') == 'add-a-marsha-diff-command'


def test_slugify_empty_and_symbol_only_fall_back() -> None:
    assert diff._slugify('') == 'spec'
    assert diff._slugify('!!!') == 'spec'


def test_slugify_truncates() -> None:
    assert len(diff._slugify('word ' * 30)) <= 40


def test_detect_commit_prefix_dominant() -> None:
    assert diff._detect_commit_prefix(
        ['feat: a', 'feat: b', 'feat: c', 'feat: d', 'feat: e']) == 'feat: '


def test_detect_commit_prefix_none() -> None:
    assert diff._detect_commit_prefix(['a', 'b', 'c', 'd', 'e']) == ''


def test_detect_commit_prefix_too_few() -> None:
    assert diff._detect_commit_prefix(['feat: a', 'feat: b', 'c']) == ''


def test_detect_commit_prefix_mixed_below_threshold() -> None:
    # 5 prefixed, but no single prefix reaches 60%: no convention is detected.
    assert diff._detect_commit_prefix(
        ['feat: a', 'fix: b', 'docs: c', 'feat: d', 'fix: e']) == ''


def test_commit_title_default() -> None:
    assert diff._commit_title([], 'My Feature', None) == 'Implement My Feature'


def test_commit_title_with_prefix() -> None:
    subjects = ['feat: a'] * 5
    assert diff._commit_title(subjects, 'My Feature', None) == \
        'feat: Implement My Feature'


def test_commit_title_with_ticket() -> None:
    assert diff._commit_title([], 'My Feature', 'ABC-123') == \
        'Implement My Feature (ABC-123)'


def test_commit_title_prefix_and_ticket() -> None:
    assert diff._commit_title(['feat: a'] * 5, 'My Feature', 'ABC-123') == \
        'feat: Implement My Feature (ABC-123)'


def test_validation_command_makefile_test_target(tmp_path: Any) -> None:
    with open(os.path.join(str(tmp_path), 'Makefile'), 'w') as f:
        f.write('build:\n\ntest:\n\tpytest -q\n')
    assert diff._validation_command(str(tmp_path)) == 'make test'


def test_validation_command_defaults_to_pytest(tmp_path: Any) -> None:
    assert diff._validation_command(str(tmp_path)) == 'pytest -q'


def test_findings_block_one_line_per_finding() -> None:
    finding: Finding = {'name': 'r', 'label': 'L', 'severity': 'major',
                        'location': 'a.py:1', 'desc': 'the desc'}
    assert diff._findings_block([finding]) == '- [L] major a.py:1: the desc'


def test_spec_summary_collapses_whitespace_and_truncates() -> None:
    assert diff._spec_summary('hello   world') == 'hello world'
    s = diff._spec_summary('x' * 600)
    assert len(s) == 501 and s.endswith('\u2026')


def test_pr_title_and_body_has_required_sections() -> None:
    title, body = diff._pr_title_and_body(
        'Implement X', 'A spec about X.', 'make test', 'clean (no actionable findings)')
    assert title == 'Implement X'
    assert '## Summary' in body and '## Validation' in body
    assert 'make test' in body


# --- mocked LLM/validation stages ----------------------------------------------

async def _locked_analyze(*a: Any, **k: Any) -> Any:
    return {'compilable': True, 'ambiguities': [], 'errors': []}


async def _unlocked_analyze(*a: Any, **k: Any) -> Any:
    return {'compilable': True, 'ambiguities': ['what about X?'], 'errors': []}


async def _impl_creates_file(*a: Any, **k: Any) -> str:
    with open('impl.txt', 'w') as f:
        f.write('implemented\n')
    return 'done'


async def _ok_validation(*a: Any, **k: Any) -> tuple[bool, str]:
    return (True, 'ok')


async def _clean_review(*a: Any, **k: Any) -> list[Any]:
    return []


async def _finding_review(*a: Any, **k: Any) -> list[Any]:
    return [{'name': 'r', 'label': 'L', 'severity': 'major',
             'location': 'a.py:1', 'desc': 'd'}]


async def _assert_not_called(*a: Any, **k: Any) -> str:
    raise AssertionError('this stage should not run')


@pytest.fixture(autouse=True)
def _no_llm() -> Generator[None, None, None]:
    # The design gate resolves the model's context window and model name: fix both so the driver
    # tests make no API probing or config lookups.
    with patch.object(diff, '_resolve_window', new=AsyncMock(return_value=200_000)), \
         patch.object(diff, 'resolve_model', return_value='test-model'):
        yield


# --- run_diff driver (end-to-end in a throwaway git repo) ----------------------

def test_run_diff_usage_error_with_no_source() -> None:
    assert asyncio.run(diff.run_diff(_args())) == 2


def test_run_diff_negative_review_cycles_is_usage_error() -> None:
    # A negative --review-cycles is a usage error (exit 2), not a silent clamp to 0 (which would
    # disable the review gate).
    assert asyncio.run(diff.run_diff(_args(source='x.mrsh', review_cycles=-1))) == 2


def test_run_diff_zero_max_tool_failure_is_usage_error() -> None:
    # A non-positive --max-tool-failure is a usage error (exit 2), not a silent clamp to 1.
    assert asyncio.run(diff.run_diff(_args(source='x.mrsh', max_tool_failure=0))) == 2


def test_run_diff_design_gate_not_locked(tmp_path: Any, capsys: Any) -> None:
    # An unlocked spec stops before any branch is created and before the implementor runs.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    with patch.object(diff, 'analyze_spec', new=_unlocked_analyze), \
         patch.object(diff, '_run_implementor', new=_assert_not_called):
        with _chdir(p):
            rc = asyncio.run(diff.run_diff(_args(source=_spec_path(p))))
    assert rc == 1
    assert 'NOT locked' in capsys.readouterr().out
    assert _git_out(p, 'branch', '--show-current') == 'main'  # no branch created


def test_run_diff_dirty_tree_refused(tmp_path: Any, capsys: Any) -> None:
    # A dirty working tree is refused before a branch is created or the implementor runs.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    with open(os.path.join(p, 'dirty.txt'), 'w') as f:
        f.write('x')
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_assert_not_called):
        with _chdir(p):
            rc = asyncio.run(diff.run_diff(_args(source=_spec_path(p))))
    assert rc == 1
    assert 'not clean' in capsys.readouterr().err
    assert _git_out(p, 'branch', '--show-current') == 'main'


def test_run_diff_review_not_clean_no_commit(tmp_path: Any, capsys: Any) -> None:
    # Findings that survive the fix-and-review budget stop the run with no commit.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_ensure_validated', new=_ok_validation), \
         patch.object(diff, '_review_gate', new=_finding_review):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), review_cycles=1)))
    assert rc == 1
    assert 'NOT clean' in capsys.readouterr().out
    # Only the initial commit exists: the implementation was never committed.
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_run_diff_review_gate_failed_no_commit(tmp_path: Any, capsys: Any) -> None:
    # A review gate that could not run (every reviewer failed) stops the run with exit 1 and no
    # commit, rather than being mistaken for a clean review.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})

    async def _failed_review(*a: Any, **k: Any) -> Any:
        raise diff.ReviewGateFailed('all reviewers failed')

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_ensure_validated', new=_ok_validation), \
         patch.object(diff, '_review_gate', new=_failed_review):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), review_cycles=1)))
    assert rc == 1
    assert 'could not run' in capsys.readouterr().out
    # Only the initial commit exists: nothing was committed on a review that never ran.
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_run_diff_safe_leaves_edits_uncommitted(tmp_path: Any, capsys: Any) -> None:
    # A successful --safe run (review skipped) leaves the edits uncommitted and offers no PR.
    # Safe mode prompts for approval before running the validation command; approve it here.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_ensure_validated', new=_ok_validation):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), safe=True, review_cycles=0),
                              read_line=lambda: 'y'))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Allow it?' in out  # the safe-mode validation approval prompt was shown
    assert 'UNCOMMITTED' in out
    assert '## Validation' not in out  # no PR proposal in safe mode
    # The file exists in the working tree but no commit was made beyond the initial one.
    assert os.path.isfile(os.path.join(p, 'impl.txt'))
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_run_diff_safe_declined_validation_skipped(tmp_path: Any, capsys: Any) -> None:
    # Declining the safe-mode validation prompt skips the validation (not a failure): the run
    # completes, leaving edits uncommitted, and _ensure_validated is never called.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    calls = {'n': 0}

    async def _counting_validation(*a: Any, **k: Any) -> Any:
        calls['n'] += 1
        return (True, 'ok')

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_ensure_validated', new=_counting_validation):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), safe=True, review_cycles=0),
                              read_line=lambda: 'n'))
    assert rc == 0
    assert calls['n'] == 0  # the validation was never run
    assert 'skipped' in capsys.readouterr().out


def test_review_gate_includes_untracked_files(tmp_path: Any) -> None:
    # An implementation that creates only NEW (untracked) files must still be reviewed: without
    # staging, `git diff <base>` reads as empty and the gate would skip it — yet the commit stages
    # the new files with `git add -A`, so an unreviewed file would reach the commit.
    p = str(tmp_path)
    _git_repo(p, {'a.py': 'x = 1\n'})
    with open(os.path.join(p, 'new.py'), 'w') as f:
        f.write('y = 2\n')
    finding: Finding = {'name': 'r', 'label': 'L', 'severity': 'major',
                        'location': 'new.py:1', 'desc': 'd'}

    async def fake_review_pass(*a: Any, **k: Any) -> list[Any]:
        return [dict(finding)]

    async def fake_evidence_gate(findings: Any, *a: Any, **k: Any) -> Any:
        return findings

    async def fake_consolidate(ctx: Any, findings: Any, *a: Any, **k: Any) -> Any:
        return findings

    with patch.object(diff, '_review_pass', new=fake_review_pass), \
         patch.object(diff, 'evidence_gate', new=fake_evidence_gate), \
         patch.object(diff, 'consolidate_findings', new=fake_consolidate):
        with _chdir(p):
            result = asyncio.run(
                diff._review_gate(p, 'main', 'main', 'spec', 'test-model', False))
    assert result  # the gate ran (not skipped on an empty diff) and returned the finding


def test_propose_refine_refreshes_staleness_baseline(tmp_path: Any) -> None:
    # Two consecutive reject->refine cycles: after the first refine is applied, the staleness
    # baseline must be refreshed to the refined fields, or the second refine would be mistaken for
    # a concurrent edit and dropped (leaving only one refinement applied).
    p = str(tmp_path)
    source = SpecSource(kind='issue', num=1, name='ISSUE-1')
    fields = {'current': ('Initial Title', 'Initial Body')}
    applied: list[Any] = []

    async def fake_source_fields(*a: Any, **k: Any) -> Any:
        return fields['current']

    async def fake_apply(src: Any, payload: Any, cwd: Any) -> None:
        applied.append(payload)
        fields['current'] = (f'Refined{len(applied)}', f'Body{len(applied)}')

    async def fake_refine_chat(*a: Any, **k: Any) -> Any:
        return ChatResult(
            status='locked', payload={'title': 'T', 'body': 'B', 'spec': 's'}, detail='')

    async def fake_implementor(*a: Any, **k: Any) -> str:
        return 'done'

    async def fake_validated(*a: Any, **k: Any) -> Any:
        return (True, 'ok')

    async def fake_review_gate(*a: Any, **k: Any) -> list[Any]:
        return []

    async def fake_commit(cwd: Any, title: Any, body: Any) -> str:
        return f'sha-{len(applied)}'

    async def fake_head_commit(cwd: Any) -> Any:
        return ('abc123', 'initial')

    async def fake_recent_subjects(cwd: Any) -> list[str]:
        return []

    answers = iter(['n', 'y', 'n', 'y', 'y'])  # reject,refine, reject,refine, accept
    with patch.object(diff, '_source_fields', new=fake_source_fields), \
         patch.object(diff, '_apply', new=fake_apply), \
         patch.object(diff, 'run_refine_chat', new=fake_refine_chat), \
         patch.object(diff, '_run_implementor', new=fake_implementor), \
         patch.object(diff, '_ensure_validated', new=fake_validated), \
         patch.object(diff, '_review_gate', new=fake_review_gate), \
         patch.object(diff, '_commit', new=fake_commit), \
         patch.object(diff, '_head_commit', new=fake_head_commit), \
         patch.object(diff, '_recent_commit_subjects', new=fake_recent_subjects), \
         patch.object(diff, '_conventions_text', return_value=''):
        rc, _title, _sha = asyncio.run(diff._propose_and_maybe_refine(
            lambda: next(answers), p, source, 'spec text',
            ('Initial Title', 'Initial Body'), 'repo', 'main', 'origin/main',
            tools.ToolContext(phase='implement'), 'Implement X', 'pytest -q',
            'clean', 'test-model', 5, False))
    assert rc == 0
    assert len(applied) == 2  # BOTH refinements applied (the second was not bailed as stale)


def test_run_diff_normal_commits_and_accepts(tmp_path: Any, capsys: Any) -> None:
    # A locked spec with passing validation and a clean review commits and the proposal is
    # accepted (no PR is pushed or created).
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    answers = iter(['y'])
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_ensure_validated', new=_ok_validation), \
         patch.object(diff, '_review_gate', new=_clean_review):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p)),
                              read_line=lambda: next(answers)))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Proposed PR title' in out and '## Validation' in out
    assert 'Accepted' in out
    # A commit was made on the marsha/<slug> branch (the slug is the .mrsh stem, 'spec').
    assert _git_out(p, 'branch', '--show-current') == 'marsha/spec'
    last = _git_out(p, 'log', '--oneline', '-1')
    assert 'Implement' in last
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 2
