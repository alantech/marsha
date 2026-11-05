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
import shlex
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
                max_tool_failure=10, safe=False, target='python',
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


def test_implementor_system_prompt_covers_checklist_and_response_protocol() -> None:
    prompt = diff.IMPL_SYSTEM_PROMPT
    assert 'every todo item is implemented and tested' in prompt
    assert 'short progress sentence before issuing tools' in prompt
    assert 'Never add prose or ellipses after a tool command' in prompt
    assert '... (or any other trailing text)' not in prompt
    assert 'Two incomplete finish attempts are bounced; a third incomplete attempt stops' in prompt
    assert 'Do not stop until every specification requirement' in prompt


def _spec_path(p: str) -> str:
    return os.path.join(p, 'spec.mrsh')


@pytest.fixture(autouse=True)
def _mock_repo_context_and_validation() -> Any:
    """Keep driver tests offline while exercising the outer review-cycle state machine."""
    from marsha.diff_validation import ValidationCheck, ValidationMetaResult

    async def validation(cwd: str, command: str, debug: bool,
                         safe: bool, allowed: bool) -> list[ValidationCheck]:
        if safe and not allowed:
            return [ValidationCheck(command, 'skipped', 'declined')]
        return [ValidationCheck(command, 'passed', 'ok')]

    with patch.object(diff, '_run_repo_context',
                      new=AsyncMock(return_value=('explorer map', 'rules'))), \
         patch.object(diff, '_run_validation_cycle', new=validation), \
         patch.object(diff, 'run_validation_meta_review',
                      new=AsyncMock(return_value=ValidationMetaResult([], [], 'clean'))):
        yield


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


def test_refined_mrsh_title_falls_back_to_source_filename() -> None:
    source = SpecSource(kind='mrsh', path='/repo/support-rust.mrsh')
    assert diff._refined_title(source, {'spec': 'refined spec'}, 'refined spec') \
        == 'support-rust'
    issue = SpecSource(kind='issue', num=7, name='ISSUE-7')
    assert diff._refined_title(
        issue, {'title': 'Refined title', 'body': 'body'}, 'rendered issue') == 'Refined title'


def test_validation_command_makefile_test_target(tmp_path: Any) -> None:
    with open(os.path.join(str(tmp_path), 'Makefile'), 'w') as f:
        f.write('build:\n\ntest:\n\tpytest -q\n')
    assert diff._validation_command(str(tmp_path)) == 'make test'


def test_validation_command_defaults_to_pytest(tmp_path: Any) -> None:
    assert diff._validation_command(str(tmp_path)) == 'pytest -q'


def test_validation_command_uses_node_test_script(tmp_path: Any) -> None:
    with open(os.path.join(str(tmp_path), 'package.json'), 'w') as f:
        f.write('{"scripts": {"test": "vitest run"}}')
    assert diff._validation_command(str(tmp_path)) == 'npm test'


def test_validation_command_uses_cargo_without_python_or_node(tmp_path: Any) -> None:
    with open(os.path.join(str(tmp_path), 'Cargo.toml'), 'w') as f:
        f.write('[package]\nname = "sample"\n')
    assert diff._validation_command(str(tmp_path)) == 'cargo test'


def test_validation_command_uses_go_module(tmp_path: Any) -> None:
    with open(os.path.join(str(tmp_path), 'go.mod'), 'w') as f:
        f.write('module example.test/app\n')
    assert diff._validation_command(str(tmp_path)) == 'go test ./...'


def test_validation_command_finds_project_venv_pytest(tmp_path: Any) -> None:
    repo = tmp_path / 'repo with spaces'
    pytest_bin = (repo / '.venv' / 'Scripts' / 'pytest.exe'
                  if os.name == 'nt' else repo / '.venv' / 'bin' / 'pytest')
    pytest_bin.parent.mkdir(parents=True)
    pytest_bin.write_text('#!/bin/sh\n')
    command = diff._validation_command(str(repo))
    assert command == shlex.quote(str(pytest_bin))
    assert shlex.split(command) == [str(pytest_bin)]


def test_run_whitelist_includes_primary_validator_but_not_in_safe_mode(
        tmp_path: Any) -> None:
    backend = types.SimpleNamespace(run_whitelist=lambda: [])
    repo = str(tmp_path)
    primary = shlex.quote(os.path.join(repo, '.venv', 'bin', 'pytest'))
    venv_python = (os.path.join(repo, '.venv', 'Scripts', 'python.exe')
                   if os.name == 'nt' else os.path.join(repo, '.venv', 'bin', 'python'))
    os.makedirs(os.path.dirname(venv_python), exist_ok=True)
    with open(venv_python, 'w') as f:
        f.write('')
    with patch('marsha.diff.backends.current', return_value=backend):
        normal = diff._build_run_whitelist(repo, primary, safe=False)
        safe = diff._build_run_whitelist(repo, primary, safe=True)
    assert any(rule.prefix == tuple(shlex.split(primary)) for rule in normal)
    assert any(rule.prefix == (venv_python, '-m', 'pip', 'install')
               and rule.network for rule in normal)
    assert safe == []


def test_unavailable_survey_checks_are_advisory(tmp_path: Any) -> None:
    from marsha.diff_validation import ValidationCheck

    async def missing_command(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError('tool is not installed')

    rules = [tools.RunRule(('mypy', 'src'), 'mypy src', network=True)]
    with patch.object(diff, '_introspect_run_commands', return_value=rules), \
            patch.object(asyncio, 'create_subprocess_exec', new=missing_command):
        checks = asyncio.run(diff._validation_survey(
            str(tmp_path), 'pytest -q', debug=False))

    assert len(checks) == 1
    assert checks[0].status == 'unavailable' and not checks[0].blocking
    assert 'ADVISORY' in checks[0].render()
    passing_primary = ValidationCheck('pytest -q', 'passed', 'ok')
    assert diff._validation_findings([passing_primary, *checks]) == []
    assert diff._validation_passed([passing_primary, *checks], False, True)
    assert 'advisory checks (mypy src unavailable)' in diff._validation_summary(
        [passing_primary, *checks], False, True)

    with patch.object(diff, '_introspect_run_commands', return_value=rules), \
            patch.object(asyncio, 'create_subprocess_exec', new=missing_command):
        after_failed_base = asyncio.run(diff._validation_survey(
            str(tmp_path), 'pytest -q', debug=False,
            baseline_checks=[ValidationCheck('mypy src', 'failed', 'base failure')]))
    assert after_failed_base[0].status == 'unavailable'
    assert after_failed_base[0].blocking

    failed_primary = ValidationCheck('pytest -q', 'failed', 'assertion failed')
    assert diff._validation_findings([failed_primary])
    assert not diff._validation_passed([failed_primary, *checks], False, True)


def test_survey_failures_already_on_base_are_advisory(tmp_path: Any) -> None:
    repo = str(tmp_path)
    _git_repo(repo, {'Makefile': 'lint:\n\t@printf base-failure; exit 1\n'})
    baseline = asyncio.run(diff._validation_baseline(
        repo, 'HEAD', 'pytest -q', debug=False))
    assert baseline is not None and len(baseline) == 1
    assert baseline[0].status == 'failed'

    current = asyncio.run(diff._validation_survey(
        repo, 'pytest -q', debug=False, baseline_checks=baseline))
    assert len(current) == 1
    assert current[0].status == 'failed' and not current[0].blocking
    assert 'clean base branch' in current[0].output
    assert diff._validation_findings(current) == []
    assert _git_out(repo, 'worktree', 'list', '--porcelain').count('worktree ') == 1


def test_introspect_run_commands_empty(tmp_path: Any) -> None:
    # No package.json, Makefile, or workflows: no introspected commands.
    assert diff._introspect_run_commands(str(tmp_path)) == []


def test_introspect_run_commands_makefile_targets(tmp_path: Any) -> None:
    # Only familiar validation targets are discovered; arbitrary recipes are not command access.
    p = os.path.join(str(tmp_path), 'Makefile')
    with open(p, 'w') as f:
        f.write('deploy:\n\trm -rf /tmp/x\n\nbuild:\n\techo build\n'
                '\ntest:\n\tpytest -q\n\nlint:\n\truff check .\n')
    rules = diff._introspect_run_commands(str(tmp_path))
    displays = [r.display for r in rules]
    assert 'make test' in displays
    assert 'make lint' in displays
    assert 'make build' not in displays
    assert 'make deploy' not in displays
    assert all(r.network for r in rules)


def test_introspect_run_commands_npm_scripts(tmp_path: Any) -> None:
    # package.json scripts become `npm run <script>` rules (network).
    p = os.path.join(str(tmp_path), 'package.json')
    with open(p, 'w') as f:
        f.write('{"scripts": {"build": "tsc", "test": "jest", "lint": "eslint ."}}')
    rules = diff._introspect_run_commands(str(tmp_path))
    displays = [r.display for r in rules]
    assert 'npm run build' in displays
    assert 'npm run test' in displays
    assert 'npm run lint' in displays
    assert all(r.network for r in rules)


def test_introspect_run_commands_ci_workflows(tmp_path: Any) -> None:
    # CI workflow `run:` steps become whitelisted commands (network).
    wf_dir = os.path.join(str(tmp_path), '.github', 'workflows')
    os.makedirs(wf_dir)
    with open(os.path.join(wf_dir, 'ci.yml'), 'w') as f:
        f.write('steps:\n  - run: pytest -q\n  - run: mypy marsha\n')
    rules = diff._introspect_run_commands(str(tmp_path))
    displays = [r.display for r in rules]
    assert 'pytest -q' in displays
    assert 'mypy marsha' in displays
    assert all(r.network for r in rules)


def test_validation_survey_uses_only_configured_argv_without_shell(tmp_path: Any) -> None:
    rules = [
        tools.RunRule(('mypy', 'marsha'), 'mypy marsha', network=True),
        tools.RunRule(('python', '-c', 'import os; os.remove("x")'),
                      'malicious', network=True),
        tools.RunRule(('make', 'deploy'), 'make deploy', network=True),
    ]
    proc = types.SimpleNamespace(returncode=0)
    spawn = AsyncMock(return_value=proc)
    with patch.object(diff, '_introspect_run_commands', return_value=rules), \
         patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
         patch.object(diff, 'run_subprocess', new=AsyncMock(return_value=('ok', ''))):
        checks = asyncio.run(diff._validation_survey(
            str(tmp_path), 'pytest -q', False))
    assert [check.command for check in checks] == ['mypy marsha']
    assert spawn.await_count == 1
    call = spawn.await_args
    assert call is not None
    assert call.args == ('mypy', 'marsha')
    assert call.kwargs['cwd'] == str(tmp_path)
    assert call.kwargs['env'] is not None
    assert not any(marker in key.upper() for key in call.kwargs['env']
                   for marker in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD',
                                  'CREDENTIAL', 'PRIVATE_KEY', 'ACCESS_KEY'))

    spawn.reset_mock()
    with patch.object(diff, '_introspect_run_commands', return_value=rules), \
         patch.object(asyncio, 'create_subprocess_exec', new=spawn):
        skipped = asyncio.run(diff._validation_survey(
            str(tmp_path), 'pytest -q', False, safe=True))
    assert skipped == []
    spawn.assert_not_awaited()


def test_introspect_run_commands_skips_shell_metacharacters(tmp_path: Any) -> None:
    # CI workflow commands with shell metacharacters or template variables are skipped.
    wf_dir = os.path.join(str(tmp_path), '.github', 'workflows')
    os.makedirs(wf_dir)
    with open(os.path.join(wf_dir, 'ci.yml'), 'w') as f:
        f.write('steps:\n  - run: echo ${{ secrets.TOKEN }}\n'
                '  - run: pytest && make build\n')
    rules = diff._introspect_run_commands(str(tmp_path))
    assert rules == []


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


def test_comments_section_extracts_comment_block() -> None:
    rendered = ('Issue #7: T\n\nThe body.\n\n# Comments so far\n'
                'alice: a note\nbob: another note')
    assert diff._comments_section(rendered) == 'alice: a note\nbob: another note'


def test_comments_section_absent() -> None:
    assert diff._comments_section('Issue #7: T\n\nThe body.') == ''


def test_impl_request_includes_comments_as_untrusted_block() -> None:
    req = diff._impl_request('the spec', '', 'main', False, comments='alice: a note')
    assert '[tool:spec]' in req and '[/tool:spec]' in req
    assert '[tool:comments]' in req and '[/tool:comments]' in req
    assert 'alice: a note' in req
    # The comments follow the spec and are labeled untrusted background, not part of the spec.
    assert req.index('[tool:spec]') < req.index('[tool:comments]')
    assert 'untrusted background context' in req


def test_impl_request_without_comments_has_no_comments_block() -> None:
    req = diff._impl_request('the spec', '', 'main', False)
    assert '[tool:comments]' not in req


def test_implementor_receives_repository_map_and_constraints_each_pass() -> None:
    ctx = tools.ToolContext(
        phase='implement', repository_map='src/core.py (main pipeline)',
        constraints='Use the configured pytest command.',
        run_whitelist=[tools.RunRule(('pytest',), 'pytest')])
    mapper = types.SimpleNamespace(n_results=1, model='test-model', system='')

    def make_mapper(system: str, **kwargs: Any) -> Any:
        mapper.system = system
        return mapper

    with patch.object(diff, 'get_mapper', side_effect=make_mapper), \
         patch.object(tools, 'run_with_tools', new=AsyncMock(return_value='done')):
        result = asyncio.run(diff._run_implementor(
            ctx, 'Implement the locked spec.', 'test-model', 5, False))
    assert result == 'done'
    assert 'src/core.py (main pipeline)' in mapper.system
    assert 'Use the configured pytest command.' in mapper.system
    assert 'locked spec wins' in mapper.system


# --- mocked LLM/validation stages ----------------------------------------------

async def _locked_analyze(*a: Any, **k: Any) -> Any:
    return {'compilable': True, 'ambiguities': [], 'errors': []}


async def _unlocked_analyze(*a: Any, **k: Any) -> Any:
    return {'compilable': True, 'ambiguities': ['what about X?'], 'errors': []}


def _complete_test_todos(ctx: tools.ToolContext) -> None:
    if not ctx.todo_items:
        tools.add_todo_item(ctx, 'Implement and test the locked specification.')
    for item in ctx.todo_items:
        item.completed = True


async def _impl_creates_file(ctx: tools.ToolContext, *a: Any, **k: Any) -> str:
    with open('impl.txt', 'w') as f:
        f.write('implemented\n')
    _complete_test_todos(ctx)
    return 'done'


async def _clean_review(*a: Any, **k: Any) -> list[Any]:
    return []


async def _finding_review(*a: Any, **k: Any) -> list[Any]:
    return [{'name': 'r', 'label': 'L', 'severity': 'major',
             'location': 'a.py:1', 'desc': 'd', 'support': 'grounded'}]


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


def test_run_diff_safe_refuses_remote_spec_source() -> None:
    # --safe is a no-network mode, but a --issue/--linear spec is loaded over the network inside
    # load_spec_with_fields (before any safe-mode restriction could apply), so a remote spec
    # source is a usage error (exit 2) in safe mode.
    assert asyncio.run(diff.run_diff(_args(issue='218', safe=True))) == 2
    assert asyncio.run(diff.run_diff(_args(linear='ABC-123', safe=True))) == 2


def test_cycle_fix_request_carries_findings_and_validation_evidence() -> None:
    # In safe mode the implementor has no network access (implement-safe filters the run
    # whitelist to local-only commands), so the fix requests must not tell it to rerun
    # validation with the run tool (the harness does that); the non-safe requests keep
    # the run-tool instruction (shown as `(run)` in the message).
    finding: Finding = {
        'name': 'n', 'label': 'A1', 'severity': 'MAJOR',
        'location': 'a.py:1', 'desc': 'fix this',
        'support': 'The new branch returns before the write.',
    }
    safe_req = diff._address_findings_request([finding], 'main', safe=True)
    assert '(run)' not in safe_req and 'harness reruns' in safe_req
    normal_req = diff._address_findings_request([finding], 'main')
    assert '(run)' in normal_req and 'harness reruns' not in normal_req
    from marsha.diff_validation import ValidationCheck
    request = diff._cycle_fix_request(
        [finding], [ValidationCheck('pytest -q', 'failed', 'assertion failed')],
        'main', False, 2)
    assert '[A1] MAJOR' in request
    assert 'CI checkpoint 2' in request
    assert 'pytest -q — FAILED' in request
    assert 'assertion failed' in request
    assert 'The new branch returns before the write.' in request
    assert '[tool:ci-review-findings]' in request
    assert '[tool:validation-results]' in request
    assert 'trace the violated contract to its root cause' in request
    assert 'focused regression test' in request


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
    # Findings still open at the review-cycle cap stop the run with no commit.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_review_gate', new=_finding_review):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), review_cycles=1)))
    assert rc == 1
    assert 'NOT clean' in capsys.readouterr().out
    # Only the initial commit exists: the implementation was never committed.
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_review_loop_repeats_panel_and_sends_prior_findings_to_meta() -> None:
    from marsha.diff_validation import ValidationCheck, ValidationMetaResult

    finding: Finding = {
        'name': 'Correctness', 'label': 'A1', 'severity': 'MAJOR',
        'location': 'src/core.py:10', 'desc': 'the update is not persisted',
        'support': 'The new branch returns before the write.',
    }
    panel = AsyncMock(side_effect=[[finding], []])
    meta = AsyncMock(return_value=ValidationMetaResult(
        [], [], 'Previous finding is resolved.'))

    initial_history = [
        {'role': 'user', 'content': 'Original locked specification.'},
        {'role': 'assistant', 'content': '$ read-file src/core.py 1..20'},
        {'role': 'user', 'content': '[tool:read-file] prior implementation context'},
        {'role': 'assistant', 'content': 'Implemented the first pass.\n$ finished'},
    ]
    implementor_history = [dict(message) for message in initial_history]
    repair_requests: list[str] = []

    async def implementor_pass(ctx: tools.ToolContext, request: str, *args: Any,
                               **kwargs: Any) -> str:
        assert any(item.source == 'review' and not item.completed
                   for item in ctx.todo_items)
        assert kwargs['history'] is implementor_history
        repair_requests.append(request)
        # Mirror run_with_tools: append review feedback and the repair response to the retained
        # conversation, rather than replacing the earlier implementation/tool turns.
        implementor_history.extend([
            {'role': 'user', 'content': request},
            {'role': 'assistant', 'content': 'Repaired the finding and reran tests.\n$ finished'},
        ])
        _complete_test_todos(ctx)
        return 'implemented'

    implementor = AsyncMock(side_effect=implementor_pass)
    checks = [ValidationCheck('pytest -q', 'passed', 'ok')]

    async def validated(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return checks

    ctx = tools.ToolContext(phase='implement')
    _complete_test_todos(ctx)
    with patch.object(diff, '_review_gate', new=panel), \
         patch.object(diff, 'run_validation_meta_review', new=meta), \
         patch.object(diff, '_run_implementor', new=implementor), \
         patch.object(diff, '_run_validation_cycle', new=validated):
        clean, remaining, final_checks = asyncio.run(diff._run_review_loop(
            ctx, '.', 'main', 'main', 'spec',
            'test-model', 5, False, False, 'pytest -q', True, 'constraints', 3,
            implementor_history=implementor_history))

    assert clean and remaining == [] and final_checks == checks
    assert panel.await_count == 2
    assert meta.await_count == 1
    assert meta.await_args_list[0].args[0] == [finding]
    assert implementor.await_count == 1
    assert implementor.await_args is not None
    assert implementor.await_args.kwargs['history'] is implementor_history
    assert implementor_history[:len(initial_history)] == initial_history
    assert 'CI checkpoint 1' in repair_requests[0]
    assert 'the update is not persisted' in repair_requests[0]
    assert implementor_history[len(initial_history)] == {
        'role': 'user', 'content': repair_requests[0]}
    review_todos = [item for item in ctx.todo_items if item.source == 'review']
    assert len(review_todos) == 1 and review_todos[0].completed


def test_reviewer_history_block_shows_state_and_dispositions() -> None:
    # Each reviewer sees its own prior findings from the past couple of cycles, with the
    # implementor's disposition of each: addressed, or rejected with the recorded reason. A
    # rejection no later cycle re-raises is settled; one that is re-raised stays contested.
    a1: Finding = {'name': 'Rosa', 'label': 'A1', 'severity': 'MAJOR',
                   'location': 'a.py:3', 'desc': 'first issue'}
    b2: Finding = {'name': 'Rosa', 'label': 'B2', 'severity': 'MINOR',
                   'location': 'b.py:9', 'desc': 'second issue'}
    c3: Finding = {'name': 'Rosa', 'label': 'C3', 'severity': 'MAJOR',
                   'location': 'c.py:1', 'desc': 'third issue'}
    block = diff._reviewer_history_block(
        [([a1, b2], {'Rosa-B2': 'the spec requires this behavior'}),
         ([c3], {})])
    assert '# Previous review cycles' in block
    assert '## Cycle 1' in block and '## Cycle 2' in block
    assert '- [Rosa-A1] MAJOR a.py:3 - first issue' in block
    assert '- [Rosa-B2] MINOR b.py:9 - second issue' in block
    assert '- [Rosa-C3] MAJOR c.py:1 - third issue' in block
    assert 'Disposition: addressed by the implementor' in block
    assert ('rejected by the implementor (not re-raised, settled) — '
            '"the spec requires this behavior"') in block
    contested = diff._reviewer_history_block(
        [([b2], {'Rosa-B2': 'the spec requires this behavior'}), ([b2], {})])
    assert ('rejected by the implementor (re-raised in a later cycle, '
            'still contested) — "the spec requires this behavior"') in contested


def test_review_gate_history_block_is_scoped_to_each_reviewer(tmp_path: Any) -> None:
    # Each reviewer's prior block contains only ITS OWN findings from the past cycles, with its
    # own rejections — another reviewer's findings never leak into its message.
    from marsha.personas import build_registry, resolve_loop_reviewers

    p = str(tmp_path)
    _git_repo(p, {'a.py': 'x = 1\n'})
    with open(os.path.join(p, 'new.py'), 'w') as f:
        f.write('y = 2\n')  # a change, so the gate does not skip an empty diff
    rosa: Finding = {'name': 'Rosa', 'label': 'A1', 'severity': 'MAJOR',
                     'location': 'a.py:1', 'desc': 'rosa\'s issue'}
    kit: Finding = {'name': 'Kit', 'label': 'B1', 'severity': 'MAJOR',
                    'location': 'a.py:2', 'desc': 'kit\'s issue'}
    history = [([rosa, kit], {'Kit-B1': 'kit\'s was rejected'}),
               ([rosa], {})]
    combined = (resolve_loop_reviewers('impl', None, build_registry())
                + resolve_loop_reviewers('review', None, build_registry()))
    number_by_name = {name: i + 1 for i, (name, _, _) in enumerate(combined)}
    captured: list[dict[int, str]] = []

    async def fake_review_pass(*a: Any, **k: Any) -> list[Any]:
        captured.append(dict(a[8]))
        return [dict(rosa)]

    async def fake_evidence_gate(findings: Any, *a: Any, **k: Any) -> Any:
        return findings

    with patch.object(diff, '_review_pass', new=fake_review_pass), \
         patch.object(diff, 'evidence_gate', new=fake_evidence_gate):
        with _chdir(p):
            asyncio.run(
                diff._review_gate(p, 'main', 'main', 'spec', 'test-model', False,
                                  constraints='', validation_report='',
                                  review_history=history))
    assert len(captured) == 1
    blocks = captured[0]
    # Both reviewers had own findings in the history, so both get a block — but only of their
    # own findings.
    assert set(blocks) == {number_by_name['Rosa'], number_by_name['Kit']}
    rosa_block = blocks[number_by_name['Rosa']]
    kit_block = blocks[number_by_name['Kit']]
    for block in (rosa_block, kit_block):
        assert '# Previous review cycles' in block
        assert '## Cycle 1' in block
    assert '## Cycle 2' in rosa_block
    assert '## Cycle 2' not in kit_block  # kit had no own finding in cycle 2
    assert '- [Rosa-A1] MAJOR a.py:1 - rosa\'s issue' in rosa_block
    assert '- [Kit-B1] MAJOR a.py:2 - kit\'s issue' not in rosa_block
    assert '- [Kit-B1] MAJOR a.py:2 - kit\'s issue' in kit_block
    assert '- [Rosa-A1] MAJOR a.py:1 - rosa\'s issue' not in kit_block
    # The rejection is shown only to the reviewer whose finding was rejected.
    assert ('rejected by the implementor (not re-raised, settled) — '
            '"kit\'s was rejected"') in kit_block
    assert 'rejected by the implementor' not in rosa_block


def test_review_loop_rejects_reach_review_history_and_settle() -> None:
    from marsha.diff_validation import ValidationCheck, ValidationMetaResult

    finding: Finding = {
        'name': 'Rosa', 'label': 'A1', 'severity': 'MAJOR',
        'location': 'src/core.py:10', 'desc': 'the update is not persisted',
        'support': 'The new branch returns before the write.',
    }
    panel = AsyncMock(side_effect=[[finding], []])
    meta = AsyncMock(return_value=ValidationMetaResult([], [], 'settled.'))
    checks = [ValidationCheck('pytest -q', 'passed', 'ok')]

    async def validated(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return checks

    async def implementor_pass(ctx: tools.ToolContext, *args: Any,
                               **kwargs: Any) -> str:
        item = next(entry for entry in ctx.todo_items
                    if entry.source == 'review' and not entry.completed)
        tools.reject_todo_item(ctx, item.id, 'the spec requires this behavior')
        return 'rejected the finding'

    implementor = AsyncMock(side_effect=implementor_pass)
    implementor_history: list[dict[str, str]] = [
        {'role': 'user', 'content': 'Original locked specification.'}]
    ctx = tools.ToolContext(phase='implement')
    _complete_test_todos(ctx)
    with patch.object(diff, '_review_gate', new=panel), \
         patch.object(diff, 'run_validation_meta_review', new=meta), \
         patch.object(diff, '_run_implementor', new=implementor), \
         patch.object(diff, '_run_validation_cycle', new=validated):
        clean, remaining, final_checks = asyncio.run(diff._run_review_loop(
            ctx, '.', 'main', 'main', 'spec',
            'test-model', 5, False, False, 'pytest -q', True, 'constraints', 3,
            implementor_history=implementor_history))

    assert clean and remaining == [] and final_checks == checks
    assert panel.await_count == 2
    # The re-review pass saw the first cycle's finding with the recorded rejection.
    assert panel.await_args_list[0].kwargs['review_history'] == []
    assert panel.await_args_list[1].kwargs['review_history'] == [
        ([finding], {'Rosa-A1': 'the spec requires this behavior'})]
    # The meta-review reconciles against the finding as before; the reviewer conceding means the
    # rejection settles: the run converges cleanly with no further repair.
    assert meta.await_count == 1
    assert meta.await_args_list[0].args[0] == [finding]
    item = next(entry for entry in ctx.todo_items if entry.source == 'review')
    assert item.completed and item.rejected
    assert 'rejected: the spec requires this behavior' in tools.format_todo_list(ctx)
    # The repair request told the implementor to reject, not silently skip.
    request = implementor.await_args_list[0].args[1]
    assert '$ todo reject <id> "<reason>"' in request
    assert 'never skipped silently' in request


def test_review_loop_defended_rejection_stays_open_and_reopens_todo() -> None:
    from marsha.diff_validation import ValidationCheck, ValidationMetaResult

    finding: Finding = {
        'name': 'Rosa', 'label': 'A1', 'severity': 'MAJOR',
        'location': 'src/core.py:10', 'desc': 'the update is not persisted',
        'support': 'The new branch returns before the write.',
    }
    panel = AsyncMock(return_value=[finding])
    meta = AsyncMock(return_value=ValidationMetaResult(
        [finding], [], 'Still open; the panel re-raised it.'))
    checks = [ValidationCheck('pytest -q', 'passed', 'ok')]

    async def validated(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return checks

    async def implementor_pass(ctx: tools.ToolContext, *args: Any,
                               **kwargs: Any) -> str:
        item = next(entry for entry in ctx.todo_items
                    if entry.source == 'review' and not entry.completed)
        tools.reject_todo_item(ctx, item.id, 'the spec requires this behavior')
        return 'rejected the finding'

    implementor = AsyncMock(side_effect=implementor_pass)
    implementor_history: list[dict[str, str]] = [
        {'role': 'user', 'content': 'Original locked specification.'}]
    ctx = tools.ToolContext(phase='implement')
    _complete_test_todos(ctx)
    with patch.object(diff, '_review_gate', new=panel), \
         patch.object(diff, 'run_validation_meta_review', new=meta), \
         patch.object(diff, '_run_implementor', new=implementor), \
         patch.object(diff, '_run_validation_cycle', new=validated):
        clean, remaining, final_checks = asyncio.run(diff._run_review_loop(
            ctx, '.', 'main', 'main', 'spec',
            'test-model', 5, False, False, 'pytest -q', True, 'constraints', 2,
            implementor_history=implementor_history))

    # The reviewer re-raised the rejected finding, so it is still open at the cycle cap: the run
    # is not clean, and the reopened item keeps the implementor's prior rejection visible.
    assert not clean
    assert len(remaining) == 1
    assert diff.finding_id(remaining[0]) == 'Rosa-A1'
    assert panel.await_count == 2
    assert panel.await_args_list[1].kwargs['review_history'] == [
        ([finding], {'Rosa-A1': 'the spec requires this behavior'})]
    item = next(entry for entry in ctx.todo_items if entry.source == 'review')
    assert not item.completed and item.rejected
    assert 'rejected before: the spec requires this behavior' \
        in tools.format_todo_list(ctx)


def test_review_loop_waits_for_completed_todo_checkpoint_before_review() -> None:
    from marsha.diff_validation import ValidationCheck

    ctx = tools.ToolContext(phase='implement')
    calls: list[str] = []
    panel = AsyncMock(return_value=[])

    async def implementor(ctx: tools.ToolContext, request: str, *args: Any,
                          **kwargs: Any) -> str:
        calls.append(request)
        if not ctx.todo_items:
            item = tools.add_todo_item(ctx, 'Implement and test each locked requirement.')
            item.completed = True
        else:
            for item in ctx.todo_items:
                item.completed = True
        return 'checklist updated'

    async def validated(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return [ValidationCheck('pytest -q', 'passed', 'ok')]

    with patch.object(diff, '_review_gate', new=panel), \
            patch.object(diff, '_run_validation_cycle', new=validated), \
            patch.object(diff, '_run_implementor', new=implementor):
        clean, remaining, checks = asyncio.run(diff._run_review_loop(
            ctx, '.', 'main', 'main', 'spec', 'test-model', 5, False, False,
            'pytest -q', True, '', 1))

    assert clean and remaining == []
    assert len(calls) == 1
    assert panel.await_count == 1
    assert tools.todo_list_ready(ctx)
    assert checks[0].passed


def test_review_loop_repairs_after_failed_validation_meta_review() -> None:
    from marsha.diff_validation import ValidationCheck, ValidationMetaResult

    finding: Finding = {
        'name': 'Correctness', 'label': 'A1', 'severity': 'MAJOR',
        'location': 'src/core.py:10', 'desc': 'the update is not persisted',
        'support': 'The new branch returns before the write.',
    }
    panel = AsyncMock(side_effect=[[finding], [], []])
    meta = AsyncMock(side_effect=[
        ValueError('meta-review returned no JSON object'),
        ValidationMetaResult([], [], 'Prior finding resolved.'),
    ])

    async def implementor_pass(ctx: tools.ToolContext, *args: Any,
                               **kwargs: Any) -> str:
        _complete_test_todos(ctx)
        return 'fixed'

    implementor = AsyncMock(side_effect=implementor_pass)
    implementor_history: list[dict[str, str]] = [
        {'role': 'user', 'content': 'Original locked specification.'}]
    checks = [ValidationCheck('pytest -q', 'passed', 'ok')]

    async def validated(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return checks

    ctx = tools.ToolContext(phase='implement')
    _complete_test_todos(ctx)
    with patch.object(diff, '_review_gate', new=panel), \
         patch.object(diff, 'run_validation_meta_review', new=meta), \
         patch.object(diff, '_run_implementor', new=implementor), \
         patch.object(diff, '_run_validation_cycle', new=validated):
        clean, remaining, final_checks = asyncio.run(diff._run_review_loop(
            ctx, '.', 'main', 'main', 'spec',
            'test-model', 5, False, False, 'pytest -q', True, 'constraints', 3,
            implementor_history=implementor_history))

    assert clean and remaining == [] and final_checks == checks
    assert panel.await_count == 3
    assert meta.await_count == 2
    assert implementor.await_count == 2
    assert all(call.kwargs['history'] is implementor_history
               for call in implementor.await_args_list)
    assert implementor.await_args is not None
    repair_request = implementor.await_args.args[1]
    assert 'the update is not persisted' in repair_request
    assert 'prior findings were conservatively kept open' in repair_request


def test_review_loop_repairs_validation_failures_before_review() -> None:
    from marsha.diff_validation import ValidationCheck

    failed_checks = [ValidationCheck(
        'make test', 'failed', 'SyntaxError during test collection')]
    passed_checks = [ValidationCheck('make test', 'passed', 'tests passed')]
    panel = AsyncMock(return_value=[])
    check_sets = iter([failed_checks, failed_checks, passed_checks])

    async def validated(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return next(check_sets)

    async def implementor_pass(ctx: tools.ToolContext, *args: Any,
                               **kwargs: Any) -> str:
        _complete_test_todos(ctx)
        return 'fixed'

    implementor = AsyncMock(side_effect=implementor_pass)

    ctx = tools.ToolContext(phase='implement')
    _complete_test_todos(ctx)
    with patch.object(diff, '_review_gate', new=panel), \
         patch.object(diff, '_run_validation_cycle', new=validated), \
         patch.object(diff, '_run_implementor', new=implementor):
        clean, remaining, final_checks = asyncio.run(diff._run_review_loop(
            ctx, '.', 'main', 'main', 'spec', 'test-model', 5, False, False,
            'make test', True, 'constraints', 2))

    assert clean and remaining == [] and final_checks == passed_checks
    assert panel.await_count == 1
    assert implementor.await_count == 2
    assert tools.todo_list_ready(ctx)
    assert all(item.completed for item in ctx.todo_items if item.source == 'validation')
    assert implementor.await_args is not None
    for call in implementor.await_args_list:
        repair_request = call.args[1]
        assert 'make test — FAILED' in repair_request
        assert 'SyntaxError during test collection' in repair_request


def test_run_diff_review_gate_failed_no_commit(tmp_path: Any, capsys: Any) -> None:
    # A review gate that could not run (every reviewer failed) stops the run with exit 1 and no
    # commit, rather than being mistaken for a clean review.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})

    async def _failed_review(*a: Any, **k: Any) -> Any:
        raise diff.ReviewGateFailed('all reviewers failed')

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_review_gate', new=_failed_review):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), review_cycles=1)))
    assert rc == 1
    assert 'could not run' in capsys.readouterr().out
    # Only the initial commit exists: nothing was committed on a review that never ran.
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_run_diff_implementor_exception_reports_summary(
        tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})

    async def failed_implementor(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError('provider transport failed')

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
            patch.object(diff, '_run_implementor', new=failed_implementor):
        with _chdir(p):
            result = asyncio.run(diff.run_diff(
                _args(source=_spec_path(p), review_cycles=1)))

    assert result == 1
    output = capsys.readouterr()
    assert 'implementation failed' in output.out
    assert 'provider transport failed' in output.out
    assert 'Traceback' not in output.err


def test_run_diff_primary_validation_unavailable_stops_early(
        tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})

    async def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise diff.PrimaryValidationUnavailable('pytest executable is missing')

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
            patch.object(diff, '_run_implementor', new=_impl_creates_file), \
            patch.object(diff, '_run_validation_cycle', new=unavailable):
        with _chdir(p):
            result = asyncio.run(diff.run_diff(
                _args(source=_spec_path(p), review_cycles=5)))

    assert result == 1
    output = capsys.readouterr().out
    assert 'UNAVAILABLE' in output
    assert 'pytest executable is missing' in output
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_run_diff_summaries_use_canonical_issue_spec(tmp_path: Any) -> None:
    p = str(tmp_path)
    _git_repo(p, {'README.md': 'base\n'})
    source = SpecSource(kind='issue', num=7, name='ISSUE-7')
    rendered = ('Issue #7: T\n\nBody\n\n# Comments so far\n'
                'alice: untrusted comment')
    captured: dict[str, str] = {}

    async def load_spec(*args: Any, **kwargs: Any) -> Any:
        return rendered, ('T', 'Body')

    async def repo_gate(*args: Any, **kwargs: Any) -> str:
        return 'repo'

    async def implementor(ctx: tools.ToolContext, *args: Any,
                          **kwargs: Any) -> str:
        with open('impl.txt', 'w') as f:
            f.write('implemented\n')
        _complete_test_todos(ctx)
        return 'done'

    async def commit(cwd: str, title: str, body: str) -> str:
        captured['commit_body'] = body
        return 'abc123'

    async def proposal(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
        captured['proposal_spec'] = args[3]
        return 0, args[9], 'abc123'

    with patch.object(diff, 'resolve_source', return_value=source), \
            patch.object(diff, '_repo_gate', new=repo_gate), \
            patch.object(diff, 'load_spec_with_fields', new=load_spec), \
            patch.object(diff, '_resolve_window', new=AsyncMock(return_value=200_000)), \
            patch.object(diff, 'analyze_spec', new=_locked_analyze), \
            patch.object(diff, '_run_implementor', new=implementor), \
            patch.object(diff, '_commit', new=commit), \
            patch.object(diff, '_propose_and_maybe_refine', new=proposal):
        with _chdir(p):
            result = asyncio.run(diff.run_diff(_args(issue='7', review_cycles=0)))

    assert result == 0
    assert captured['proposal_spec'] == 'T\nBody'
    assert 'Issue #7:' not in captured['commit_body']
    assert 'untrusted comment' not in captured['commit_body']
    assert 'Body' in captured['commit_body']


def test_review_loop_wraps_unexpected_repair_exception() -> None:
    from marsha.diff_validation import ValidationCheck

    async def failed_validation(*args: Any, **kwargs: Any) -> list[ValidationCheck]:
        return [ValidationCheck('pytest -q', 'failed', 'broken test')]

    async def failed_implementor(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError('repair model failed')

    ctx = tools.ToolContext(phase='implement')
    item = tools.add_todo_item(ctx, 'Implement and validate the requested work.')
    item.completed = True
    with patch.object(diff, '_run_validation_cycle', new=failed_validation), \
            patch.object(diff, '_run_implementor', new=failed_implementor):
        with pytest.raises(diff.ImplementorFailed, match='repair model failed'):
            asyncio.run(diff._run_review_loop(
                ctx, '.', 'main', 'main', 'spec',
                'test-model', 5, False, False, 'pytest -q', True, '', 0))


def test_zero_cycle_loop_wraps_unexpected_todo_checkpoint_exception() -> None:
    async def failed_implementor(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError('model unavailable')

    ctx = tools.ToolContext(phase='implement')
    tools.add_todo_item(ctx, 'Implement and validate the requested work.')
    with patch.object(diff, '_run_implementor', new=failed_implementor):
        with pytest.raises(diff.ImplementorFailed, match='todo checkpoint failed: model unavailable'):
            asyncio.run(diff._run_review_loop(
                ctx, '.', 'main', 'main', 'spec',
                'test-model', 5, False, False, 'pytest -q', True, '', 0))


def test_run_diff_no_changes_is_a_clear_failure(tmp_path: Any, capsys: Any) -> None:
    # A locked spec the implementer does not change is not implemented: the run stops with a clear
    # "no changes" report (including the implementer's own report) instead of running validation /
    # a "clean" review and then failing to commit an empty tree. No commit is made.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})

    async def _impl_noop(*a: Any, **k: Any) -> str:
        return 'I read the code but decided there was nothing to change.'

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_noop), \
         patch.object(diff, '_review_gate', new=_clean_review), \
         patch.object(diff, '_commit', new=_assert_not_called):
        with _chdir(p):
            rc = asyncio.run(diff.run_diff(_args(source=_spec_path(p), review_cycles=1)))
    assert rc == 1
    cap = capsys.readouterr()
    assert 'no changes' in cap.out  # the summary's no-changes lines (stdout)
    assert 'nothing to change' in cap.out  # the implementer's own report surfaced (stdout)
    assert 'git commit failed' not in (cap.out + cap.err)  # not a doomed commit failure
    # No commit was made beyond the initial one.
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_impl_request_canonical_spec_and_comments() -> None:
    # The implementer implements the canonical spec the design gate analyzed
    # (title + body — what the gate locked), not the rendered issue (whose
    # `Issue #N:` header is not part of the locked design); the comments, when
    # there are any, ride along as a separate untrusted context block.
    spec_text = 'T\n\nThe locked body.'
    comments = 'alice: context note'
    req = diff._impl_request(spec_text, '', 'main', False, comments)
    spec_block = req.split('[/tool:spec]')[0]
    assert 'The locked body.' in spec_block  # the locked spec (title + body)
    assert 'alice: context note' not in spec_block  # comments not in the spec
    assert 'Issue #7:' not in spec_block  # no rendered header
    comments_block = req.split('[tool:comments]')[1].split('[/tool:comments]')[0]
    assert 'alice: context note' in comments_block  # ...they ride along as untrusted context


def test_implementor_completion_instruction_uses_finished_marker() -> None:
    assert 'send `$ finished` alone' in diff.IMPL_SYSTEM_PROMPT
    assert 'incomplete `$ finished` is rejected' in diff.IMPL_SYSTEM_PROMPT
    assert 'Use `$ todo list`' not in diff.IMPL_SYSTEM_PROMPT
    assert 'Valid:' in diff.IMPL_SYSTEM_PROMPT
    assert 'Never add prose or ellipses after a tool command' in diff.IMPL_SYSTEM_PROMPT
    assert '... (or any other trailing text)' not in diff.IMPL_SYSTEM_PROMPT
    assert 'stop issuing commands' not in diff.IMPL_SYSTEM_PROMPT


def test_implementor_prompt_covers_review_dimensions_and_pre_finish_audit() -> None:
    prompt = diff.IMPL_SYSTEM_PROMPT
    for facet in (
            'Spec fidelity and completeness', 'Correctness and regression safety',
            'Error and execution contracts', 'Adversarial safety',
            'Architecture, integration, and scope', 'Dependencies and performance',
            'Test quality and maintainability'):
        assert facet in prompt
    assert 'audit before finishing' in prompt
    assert 'Report only checks that actually ran' in prompt


def test_run_diff_safe_leaves_edits_uncommitted(tmp_path: Any, capsys: Any) -> None:
    # A successful --safe run (review skipped) leaves the edits uncommitted and offers no PR.
    # Safe mode prompts for approval before running the validation command; approve it here.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})

    async def safe_implementor(ctx: Any, *args: Any, **kwargs: Any) -> str:
        assert ctx.run_whitelist == []
        return await _impl_creates_file(ctx, *args, **kwargs)

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=safe_implementor):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), safe=True, review_cycles=0),
                              read_line=lambda: 'y'))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Allow it?' in out  # the safe-mode validation approval prompt was shown
    assert 'executes code in the working tree' in out
    assert 'UNCOMMITTED' in out
    assert '## Validation' not in out  # no PR proposal in safe mode
    # The file exists in the working tree but no commit was made beyond the initial one.
    assert os.path.isfile(os.path.join(p, 'impl.txt'))
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_run_diff_safe_declined_validation_skipped(tmp_path: Any, capsys: Any) -> None:
    # Declining the safe-mode validation prompt skips the validation (not a failure): the run
    # completes, leaving edits uncommitted, and all checks are marked skipped.
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    calls = {'n': 0}

    async def _counting_validation(cwd: str, command: str, debug: bool,
                                   safe: bool, allowed: bool) -> Any:
        calls['n'] += 1
        assert safe and not allowed
        from marsha.diff_validation import ValidationCheck
        return [ValidationCheck(command, 'skipped', 'declined')]

    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_run_validation_cycle', new=_counting_validation):
        with _chdir(p):
            rc = asyncio.run(
                diff.run_diff(_args(source=_spec_path(p), safe=True, review_cycles=0),
                              read_line=lambda: 'n'))
    assert rc == 0
    assert calls['n'] == 1  # harness records the declined validation as skipped
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
        assert a[7].working_tree is True
        assert a[7].base_ref == 'main'
        return [dict(finding)]

    async def fake_evidence_gate(findings: Any, *a: Any, **k: Any) -> Any:
        return findings

    with patch.object(diff, '_review_pass', new=fake_review_pass), \
         patch.object(diff, 'evidence_gate', new=fake_evidence_gate):
        with _chdir(p):
            result = asyncio.run(
                diff._review_gate(p, 'main', 'main', 'spec', 'test-model', False,
                                  constraints='', validation_report='', review_history=[]))
    assert result  # the gate ran (not skipped on an empty diff) and returned the finding


def test_review_gate_propagates_no_network_to_review_tools(tmp_path: Any) -> None:
    p = str(tmp_path)
    _git_repo(p, {'a.py': 'x = 1\n'})
    with open(os.path.join(p, 'new.py'), 'w') as f:
        f.write('y = 2\n')
    seen: list[bool] = []

    async def fake_review_pass(*args: Any, **kwargs: Any) -> list[Any]:
        seen.append(args[7].no_network)
        return []

    async def fake_evidence_gate(findings: Any, *args: Any, **kwargs: Any) -> Any:
        return findings

    with patch.object(diff, '_review_pass', new=fake_review_pass), \
         patch.object(diff, 'evidence_gate', new=fake_evidence_gate):
        result = asyncio.run(diff._review_gate(
            p, 'main', 'main', 'spec', 'test-model', False,
            constraints='', validation_report='', review_history=[],
            no_network=True))
    assert result == []
    assert seen == [True]


def test_validation_subprocesses_scrub_llm_credentials(
        tmp_path: Any, monkeypatch: Any) -> None:
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'):
        monkeypatch.setenv(key, 'test-secret')
    process = types.SimpleNamespace(returncode=0)
    spawn = AsyncMock(return_value=process)
    run = AsyncMock(return_value=(b'ok', b''))
    with patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
            patch.object(diff, 'run_subprocess', new=run):
        passed, _output = asyncio.run(diff._run_validation(str(tmp_path), 'pytest -q'))
        assert passed
        first_call = spawn.await_args
        assert first_call is not None
        first_env = first_call.kwargs['env']
    assert not any(key in first_env for key in (
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'))
    primary_run = run.await_args
    assert primary_run is not None
    assert primary_run.args[1] == diff.VALIDATION_TIMEOUT
    assert primary_run.kwargs['tail_bytes'] == diff.VALIDATION_OUTPUT_TAIL_BYTES

    spawn.reset_mock()
    run.reset_mock()
    with patch.object(diff, '_introspect_run_commands', return_value=[
            tools.RunRule(('npm', 'run', 'test'), 'npm run test', network=True)]), \
            patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
            patch.object(diff, 'run_subprocess', new=run):
        checks = asyncio.run(diff._validation_survey(
            str(tmp_path), 'pytest -q', debug=False))
    assert len(checks) == 1 and checks[0].passed
    second_call = spawn.await_args
    assert second_call is not None
    second_env = second_call.kwargs['env']
    assert not any(key in second_env for key in (
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'))
    survey_run = run.await_args
    assert survey_run is not None
    assert survey_run.args[1] == diff.VALIDATION_TIMEOUT
    assert survey_run.kwargs['tail_bytes'] == diff.VALIDATION_OUTPUT_TAIL_BYTES


def test_validation_timeouts_are_blocking_failures(tmp_path: Any) -> None:
    process = types.SimpleNamespace(returncode=-9)
    spawn = AsyncMock(return_value=process)
    run = AsyncMock(side_effect=Exception('run_subprocess timeout...'))
    with patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
            patch.object(diff, 'run_subprocess', new=run):
        passed, output = asyncio.run(
            diff._run_validation(str(tmp_path), 'pytest -q'))
    assert not passed
    assert output.startswith('validation timed out after')
    assert not output.startswith('validation could not be run:')

    rule = tools.RunRule(('mypy', 'src'), 'mypy src', network=True)
    with patch.object(diff, '_introspect_run_commands', return_value=[rule]), \
            patch.object(asyncio, 'create_subprocess_exec', new=spawn), \
            patch.object(diff, 'run_subprocess', new=run):
        checks = asyncio.run(diff._validation_survey(
            str(tmp_path), 'pytest -q', debug=False))
    assert len(checks) == 1
    assert checks[0].status == 'failed' and checks[0].blocking
    assert 'timed out' in checks[0].output


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

    async def fake_review_loop(*a: Any, **k: Any) -> Any:
        from marsha.diff_validation import ValidationCheck
        return (True, [], [ValidationCheck('pytest -q', 'passed', 'ok')])

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
         patch.object(diff, '_run_review_loop', new=fake_review_loop), \
         patch.object(diff, '_commit', new=fake_commit), \
         patch.object(diff, '_head_commit', new=fake_head_commit), \
         patch.object(diff, '_recent_commit_subjects', new=fake_recent_subjects), \
         patch.object(diff, '_conventions_text', return_value=''):
        rc, _title, _sha = asyncio.run(diff._propose_and_maybe_refine(
             lambda: next(answers), p, source, 'spec text',
             ('Initial Title', 'Initial Body'), 'repo', 'main', 'origin/main',
             tools.ToolContext(phase='implement'), 'Implement X', 'pytest -q',
             'clean', 'test-model', 5, 5, False))
    assert rc == 0
    assert len(applied) == 2  # BOTH refinements applied (the second was not bailed as stale)


@pytest.mark.parametrize('failure_stage', ['review_gate', 'repo_context'])
def test_propose_refine_failure_leaves_commit_intact(
        tmp_path: Any, capsys: Any, failure_stage: str) -> None:
    # A repo-context or review failure after reject-and-refine must not escape as an uncaught
    # exception: the existing commit is left intact and no follow-up commit is made.
    p = str(tmp_path)
    source = SpecSource(kind='issue', num=1, name='ISSUE-1')
    committed = {'follow': 0}

    async def fake_source_fields(*a: Any, **k: Any) -> Any:
        return ('Initial Title', 'Initial Body')

    async def fake_apply(*a: Any, **k: Any) -> None:
        pass

    async def fake_refine_chat(*a: Any, **k: Any) -> Any:
        return ChatResult(status='locked', payload={'title': 'T', 'body': 'B', 'spec': 's'},
                          detail='')

    async def fake_implementor(*a: Any, **k: Any) -> str:
        return 'done'

    async def fake_review_loop(*a: Any, **k: Any) -> Any:
        if failure_stage == 'review_gate':
            raise diff.ReviewGateFailed('all reviewers failed')
        return (True, [], [])

    async def fake_repo_context(*a: Any, **k: Any) -> Any:
        if failure_stage == 'repo_context':
            raise RuntimeError('temporary model transport error')
        return ('map', 'constraints')

    async def fake_commit(*a: Any, **k: Any) -> str:
        committed['follow'] += 1
        return 'follow-sha'

    async def fake_head_commit(cwd: Any) -> Any:
        return ('abc123', 'initial')

    async def fake_recent_subjects(cwd: Any) -> list[str]:
        return []

    answers = iter(['n', 'y'])  # reject the proposal, then refine (the follow-up review fails)
    with patch.object(diff, '_source_fields', new=fake_source_fields), \
         patch.object(diff, '_apply', new=fake_apply), \
         patch.object(diff, 'run_refine_chat', new=fake_refine_chat), \
         patch.object(diff, '_run_repo_context', new=fake_repo_context), \
         patch.object(diff, '_run_implementor', new=fake_implementor), \
         patch.object(diff, '_run_review_loop', new=fake_review_loop), \
         patch.object(diff, '_commit', new=fake_commit), \
         patch.object(diff, '_head_commit', new=fake_head_commit), \
         patch.object(diff, '_recent_commit_subjects', new=fake_recent_subjects), \
         patch.object(diff, '_conventions_text', return_value=''):
        rc, _title, sha = asyncio.run(diff._propose_and_maybe_refine(
             lambda: next(answers), p, source, 'spec text',
             ('Initial Title', 'Initial Body'), 'repo', 'main', 'origin/main',
             tools.ToolContext(phase='implement'), 'Implement X', 'pytest -q',
             'clean', 'test-model', 5, 5, False))
    assert rc == 0
    assert sha == 'abc123'  # the original commit is preserved
    assert committed['follow'] == 0  # no follow-up commit on the strength of a failed review
    error = capsys.readouterr().err
    if failure_stage == 'review_gate':
        assert 'could not run' in error
    else:
        assert 'Re-implementation failed' in error


def test_review_gate_fails_if_staging_fails(tmp_path: Any) -> None:
    # If staging the working tree (git add -A) fails, the gate must fail rather than review an
    # incomplete tree (whose untracked changes could later be committed unreviewed).
    p = str(tmp_path)

    async def fake_git(*a: Any, **k: Any) -> Any:
        return (1, '', 'boom')

    with patch.object(diff, '_git', new=fake_git):
        with pytest.raises(diff.ReviewGateFailed):
            asyncio.run(diff._review_gate(
                p, 'main', 'main', 'spec', 'test-model', False,
                constraints='', validation_report='', review_history=[]))


def test_changed_files_includes_untracked_new_files(tmp_path: Any) -> None:
    # A newly created, still-untracked file must appear in the changed-files list (git diff omits
    # it), so the summary does not report a change of only new files as "no changes".
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    with open(os.path.join(p, 'brand_new.py'), 'w') as f:
        f.write('print(1)\n')
    files = asyncio.run(diff._changed_files(p, 'main'))
    assert 'brand_new.py' in files


def test_run_diff_normal_commits_and_accepts(tmp_path: Any, capsys: Any) -> None:
    # A locked spec with passing validation and a clean review commits and the proposal is
    # accepted (no PR is pushed or created).
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    answers = iter(['y'])
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
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


def test_zero_review_cycles_still_runs_validation(tmp_path: Any, capsys: Any) -> None:
    from marsha.diff_validation import ValidationCheck

    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    answers = iter(['y'])
    validation = AsyncMock(return_value=[ValidationCheck('make test', 'passed', 'ok')])
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_run_validation_cycle', new=validation), \
         patch.object(diff, '_review_gate', new=_assert_not_called):
        with _chdir(p):
            rc = asyncio.run(diff.run_diff(
                _args(source=_spec_path(p), review_cycles=0),
                read_line=lambda: next(answers)))
    assert rc == 0
    validation.assert_awaited_once()
    assert 'make test' in capsys.readouterr().out


def test_zero_review_cycles_does_not_commit_failed_validation(tmp_path: Any) -> None:
    from marsha.diff_validation import ValidationCheck

    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    validation = AsyncMock(return_value=[
        ValidationCheck('make test', 'failed', 'assertion failed')])
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_run_validation_cycle', new=validation), \
         patch.object(diff, '_review_gate', new=_assert_not_called), \
         patch.object(diff, '_commit', new=_assert_not_called):
        with _chdir(p):
            rc = asyncio.run(diff.run_diff(
                _args(source=_spec_path(p), review_cycles=0)))
    assert rc == 1
    assert validation.await_count == diff.MAX_VALIDATION_FIX_PASSES + 1
    assert len(_git_out(p, 'log', '--oneline').splitlines()) == 1


def test_zero_review_cycles_still_validates_before_commit(tmp_path: Any, capsys: Any) -> None:
    p = str(tmp_path)
    _git_repo(p, {'spec.mrsh': _mrsh_spec()})
    answers = iter(['y'])
    with patch.object(diff, 'analyze_spec', new=_locked_analyze), \
         patch.object(diff, '_run_implementor', new=_impl_creates_file), \
         patch.object(diff, '_review_gate', new=_assert_not_called):
        with _chdir(p):
            rc = asyncio.run(diff.run_diff(
                _args(source=_spec_path(p), review_cycles=0),
                read_line=lambda: next(answers)))
    assert rc == 0
    output = capsys.readouterr().out
    assert 'pytest -q' in output
    assert _git_out(p, 'branch', '--show-current') == 'marsha/spec'
