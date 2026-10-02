"""The `marsha diff` subcommand: implement a design-locked spec in the current Git repository.

It combines the design gate from `marsha refine --check` (reusing that command's source
resolution and analysis functions, not a subprocess), repository-aware implementation (an
implementor agent that reads and edits the working tree and runs the project's validation),
and a review gate over the implementation's working-tree changes before committing. A clean
working tree is required, a fresh `marsha/<slug>` branch is created (never pushed), and a commit
is made only after validation and the review gate pass. `--safe` disables the network, dependency
installation, and automatic commits (a successful safe run leaves its edits uncommitted). See
issue #219 for the full contract.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Callable, cast

from marsha import backends, tools
from marsha.config import resolve_model
from marsha.findings import Finding
from marsha.llm import consolidate_findings
from marsha.mappers import get_mapper
from marsha.personas import (build_registry, dedup_by_location, dedup_findings,
                             resolve_loop_reviewers)
from marsha.refine import (SpecSource, _apply, _read_line, _repo_gate, _resolve_window,
                           _source_fields, load_spec_with_fields, resolve_source,
                           run_refine_chat)
from marsha.review import (REVIEW_CONTEXT_LIMIT, REVIEW_REASONING_EFFORT, REVIEW_SEED, _git,
                           _review_pass, build_review_message, default_branch, evidence_gate,
                           working_tree_clean, working_tree_diff_stat)
from marsha.spec_check import analyze_spec
from marsha.utils import run_subprocess

# The implementor's tool budget: it reads, edits, and validates over many commands, so it gets
# more rounds than the read-only reviewers (which cap at 150); 250 leaves room for a full
# implement-validate-diagnose-fix cycle without the budget being the limiting factor.
IMPL_MAX_TOOL_ROUNDS = 250
# How many times the harness will hand a failing validation back to the implementor to fix and
# rerun before it gives up (a hard cap so a genuinely broken build cannot loop forever).
MAX_VALIDATION_FIX_PASSES = 3
# How much of a spec / conventions doc is handed to a prompt (the rest is retrievable via tools).
DIFF_SOURCE_LIMIT = 48_000
# The timeout (seconds) for the harness's own re-verification run of the validation command.
VALIDATION_TIMEOUT = 600

VM_WARNING = (
    'Warning: marsha diff runs a tool-using agent that edits files and executes commands in this\n'
    'repository. Run it inside a VM (or another isolated environment) to limit the blast radius of\n'
    'a failure. Do not run it against a working tree you cannot afford to lose.')


# --- pure helpers (deterministic and unit-testable) -------------------------------

def _short_title(title: str) -> str:
    # A short form of the spec title for a commit/PR title: the first sentence, trimmed to a
    # reasonable length. Falls back to a generic word when the title is empty.
    t = ' '.join(title.strip().split())
    t = re.split(r'(?<=[.!?])\s+', t, maxsplit=1)[0].strip()
    if not t:
        return 'changes'
    return t[:60].rstrip()


def _slugify(title: str) -> str:
    # A branch-name slug from the spec title: lowercase, non-alphanumerics to single dashes,
    # trimmed to keep `marsha/<slug>` a sane branch name.
    s = re.sub(r'[^a-zA-Z0-9]+', '-', title).strip('-').lower()
    s = s[:40].rstrip('-')
    return s or 'spec'


def _conventions_text(cwd: str) -> str:
    # The repository's stated conventions (AGENTS.md / CLAUDE.md), bounded; '' when neither
    # exists. This is untrusted reference data the implementor is told to follow.
    parts: list[str] = []
    for name in ('AGENTS.md', 'CLAUDE.md'):
        path = os.path.join(cwd, name)
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as f:
                    parts.append(f'### {name}\n' + f.read()
                                 [:DIFF_SOURCE_LIMIT])
            except OSError:
                pass
    return '\n\n'.join(parts)


_COMMIT_PREFIX_RE = re.compile(r'^([A-Za-z][A-Za-z0-9-]*)\s*:')


def _detect_commit_prefix(subjects: list[str]) -> str:
    # A change-type prefix ("feat: ", "fix: ") the repository clearly and consistently uses. Only
    # a dominant single prefix (enough signal, and >= 60% of the subjects that carry a prefix)
    # counts as a convention; a mixed history yields no prefix so the default title is used.
    prefixed: list[str] = []
    for s in subjects:
        m = _COMMIT_PREFIX_RE.match(s)
        if m is not None:
            prefixed.append(m.group(1))
    if len(prefixed) < 5:
        return ''
    word, _count = Counter(prefixed).most_common(1)[0]
    if _count * 5 < len(prefixed) * 3:
        return ''
    return f'{word}: '


def _commit_title(subjects: list[str], short_title: str, ticket_id: str | None) -> str:
    # The commit title: a detected change-type prefix (if the repo clearly uses one) followed by
    # `Implement <short spec title>`; a Linear ticket id is appended after the title.
    title = f'{_detect_commit_prefix(subjects)}Implement {short_title}'
    if ticket_id:
        title = f'{title} ({ticket_id})'
    return title


def _validation_command(cwd: str) -> str:
    # The project's test command, best-effort: a Makefile `test` target if there is one,
    # otherwise the common Python default. The implementor (which reads the repo's own docs) is
    # the authority on what to run; this is the harness's own re-verification command.
    makefile = os.path.join(cwd, 'Makefile')
    if os.path.isfile(makefile):
        try:
            with open(makefile, encoding='utf-8') as f:
                if re.search(r'(?m)^test[ \t]*:', f.read()):
                    return 'make test'
        except OSError:
            pass
    return 'pytest -q'


def _findings_block(findings: list[Finding]) -> str:
    # The findings rendered for an implementor to address: one line per finding.
    lines: list[str] = []
    for f in findings:
        lines.append(f"- [{f['label']}] {f['severity']} {f.get('location', '')}: "
                     f"{f['desc'].strip()}")
    return '\n'.join(lines)


def _spec_summary(spec_text: str, limit: int = 500) -> str:
    # A single-line summary of the spec for the PR/commit body.
    s = ' '.join(spec_text.strip().split())
    return s if len(s) <= limit else s[:limit] + '…'


def _pr_title_and_body(commit_title: str, spec_text: str, validation_cmd: str,
                       review_result: str) -> tuple[str, str]:
    # The proposed PR title and body. The title follows the commit-title policy; the body has the
    # required ## Summary and ## Validation sections (validation reports what was actually run).
    validation = f'Ran `{validation_cmd}` (passed). Review gate: {review_result}.'
    body = f'## Summary\n\n{_spec_summary(spec_text, 400)}\n\n## Validation\n\n{validation}\n'
    return commit_title, body


# --- Git helpers (async) ----------------------------------------------------------

async def _recent_commit_subjects(cwd: str) -> list[str]:
    # The 20 most recent non-merge commit subjects, used to detect the local commit-title
    # convention; best-effort (an empty list means no convention is detectable).
    rc, out, _err = await _git('log', '--no-merges', '--pretty=%s', '-20', cwd=cwd)
    if rc != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


async def _head_commit(cwd: str) -> tuple[str, str]:
    # The current HEAD's (short sha, subject), for the final summary; best-effort.
    rc, out, _err = await _git('log', '-1', '--pretty=%h %s', cwd=cwd)
    if rc != 0 or not out.strip():
        return ('(unknown)', '(unknown)')
    parts = out.strip().split(' ', 1)
    return (parts[0], parts[1] if len(parts) > 1 else '')


async def _branch_exists(name: str, cwd: str) -> bool:
    rc, _o, _e = await _git('rev-parse', '--verify', f'refs/heads/{name}', cwd=cwd)
    return rc == 0


async def _unique_branch(base: str, cwd: str) -> str:
    # `marsha/<slug>`, or a `-<n>` suffixed variant if that branch already exists.
    if not await _branch_exists(base, cwd):
        return base
    n = 2
    while await _branch_exists(f'{base}-{n}', cwd):
        n += 1
    return f'{base}-{n}'


async def _create_branch(name: str, base_ref: str, cwd: str) -> None:
    # Base the new branch on the default branch (the review gate and the eventual PR both diff
    # against it), not on the arbitrary current HEAD: if the clean starting branch were not the
    # default, basing on HEAD would fold those extra commits into the reviewed/committed change.
    rc, _o, err = await _git('checkout', '-b', name, base_ref, cwd=cwd)
    if rc != 0:
        raise Exception(f'git checkout -b {name} {base_ref} failed: {err}')


async def _changed_files(cwd: str, base_ref: str) -> list[str]:
    # The relative paths changed against the base (uncommitted working-tree work included).
    rc, out, _err = await _git('diff', '--name-only', base_ref, cwd=cwd)
    if rc != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


async def _commit(cwd: str, title: str, body: str) -> str:
    # Stage the implementation and commit it on the current branch; returns the short sha.
    rc, _o, err = await _git('add', '-A', cwd=cwd)
    if rc != 0:
        raise Exception(f'git add -A failed: {err}')
    rc, _o, err = await _git('commit', '-m', title, '-m', body, cwd=cwd)
    if rc != 0:
        raise Exception(f'git commit failed: {err}')
    rc, sha, _e = await _git('rev-parse', '--short', 'HEAD', cwd=cwd)
    return sha.strip() if rc == 0 else '(unknown)'


# --- validation (async) -----------------------------------------------------------

async def _run_validation(cwd: str, cmd: str, safe: bool = False) -> tuple[bool, str]:
    # Run the validation command and report whether it passed (a clean exit). The tail of the
    # output is returned so the implementor can see a failure without buffering without bound.
    # In safe mode the subprocess runs in a network-isolated environment (proxies point at a
    # non-routable local port, no_proxy cleared, package managers offline), so a validation
    # command that would build the env or install deps (e.g. a Makefile `test` target that
    # depends on a venv) cannot reach the web — safe mode forbids installs even via the
    # harness's own re-verification run.
    try:
        env: dict[str, str] = dict(os.environ)
        if safe:
            # Safe mode forbids the network and dependency installation. Offline flags alone are
            # not enough (a Makefile `test` target that depends on a venv still runs its install
            # step), so isolate the network: route every proxy at a non-routable local port
            # (nothing listens on :9, the discard port) and clear no_proxy, so any network access
            # — an install, a download — fails rather than reaching the web. UV_OFFLINE and
            # PIP_NO_INDEX additionally stop the package managers from using their index.
            for _var in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
                env[_var] = 'http://127.0.0.1:9'
            env['no_proxy'] = ''
            env['NO_PROXY'] = ''
            env['UV_OFFLINE'] = '1'
            env['PIP_NO_INDEX'] = '1'
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _err = await run_subprocess(proc, VALIDATION_TIMEOUT,
                                         max_bytes=tools.EXEC_MAX_BYTES)
    except Exception as e:
        return False, f'validation could not be run: {e}'
    return (proc.returncode == 0, (out or '').strip()[-4000:])


# --- implementor prompts and loop (async) -----------------------------------------

IMPL_SYSTEM_PROMPT = (
    'You are a senior software engineer implementing a design-locked specification in the current '
    'Git repository. You work directly in the working tree: read the code and its conventions, '
    'edit files, and run the project\'s own validation.\n'
    '\n'
    'Follow the repository\'s real conventions (its AGENTS.md / CLAUDE.md instructions if '
    'present, the existing code, and recent git history). Keep changes scoped to the '
    'specification. Investigate before you change: read the relevant files and their callers, '
    'and update any affected call sites and tests.\n'
    '\n'
    'Use the tools: read files (list-tree, summarize, find-in-file, git), write files '
    '(write-file), and run the project\'s validation/build/lint commands and install dependencies '
    '(exec). A command that runs and reports failing tests, builds, or lint is feedback to act on, '
    'not an error: read the failure, fix the code, and rerun the command.\n'
    '\n'
    'When the implementation is complete and the project\'s validation passes, stop issuing '
    'commands and give a short final report: the files you changed, what each change does, and '
    'the validation you ran with its result. Do not commit — the harness commits after its own '
    'review gate.')


def _impl_request(spec_text: str, conventions: str, base_name: str, safe: bool) -> str:
    # The initial implementor request: the locked spec (untrusted), the conventions, and the task.
    parts = [
        'Implement the following design-locked specification in the current repository.',
        tools.wrap_untrusted('spec', tools.truncate(
            spec_text, limit=DIFF_SOURCE_LIMIT)),
    ]
    if conventions:
        parts.append(
            'The repository\'s stated conventions (untrusted reference data to follow):')
        parts.append(tools.wrap_untrusted('conventions', conventions))
    parts.append(
        f'The default branch is `{base_name}`. Work in the working tree; do not commit.')
    if safe:
        parts.append('Safe mode is on: make local edits only. You have no command tool (no '
                     'network, no dependency installation, no shell) — the harness runs the '
                     'project\'s validation for you; fix whatever it reports by editing files.')
    else:
        parts.append('You may look up documentation over the network and install dependencies as '
                     'needed.')
    return '\n\n'.join(parts)


def _address_findings_request(findings: list[Finding], base_name: str) -> str:
    # The request that sends review findings back to the implementor to address.
    return (
        'A review of the working-tree changes found the findings below. Address each one: fix the '
        'code (write-file), then rerun the project\'s validation (exec) until it passes. Keep '
        f'changes scoped to the default branch `{base_name}` and do not commit. When done, give a '
        'short report of what you changed and the validation result.\n\n# Findings to address\n\n'
        + _findings_block(findings))


def _validation_fix_request(validation_cmd: str, failure_output: str) -> str:
    # The request that hands a failing validation back to the implementor.
    return (
        f'The project\'s validation command `{validation_cmd}` failed. Read the failure below, fix '
        'the code (write-file), and rerun the command (exec) until it passes. Then give a short '
        'report.\n\n# Validation failure\n\n' + failure_output[:8000])


async def _run_implementor(ctx: tools.ToolContext, request: str, model: str,
                           max_failures: int, debug: bool) -> str:
    # One implementor pass: drive the tool loop with the implementor persona. Raises
    # tools.ToolFailureLimitExceeded when the consecutive-failure budget is exhausted.
    system = IMPL_SYSTEM_PROMPT + tools.tool_instructions(ctx)
    mapper = get_mapper(system, n_results=1, model=model, label='diff:implementor',
                        reasoning_effort='high')
    return cast(
        str, await tools.run_with_tools(
            mapper, request, ctx, debug=debug, max_rounds=IMPL_MAX_TOOL_ROUNDS,
            max_consecutive_failures=max_failures))


async def _ensure_validated(ctx: tools.ToolContext, cwd: str, validation_cmd: str,
                            model: str, max_failures: int,
                            debug: bool, safe: bool = False) -> tuple[bool, str]:
    # Run the validation; on a failure, hand it to the implementor to fix and rerun, up to
    # MAX_VALIDATION_FIX_PASSES. Returns (passed, last_output). A validation that cannot even be
    # run counts as a failure and is also handed back for a workaround. In safe mode the re-run
    # blocks dependency installation (see _run_validation).
    passed, out = await _run_validation(cwd, validation_cmd, safe)
    fix_passes = 0
    while not passed and fix_passes < MAX_VALIDATION_FIX_PASSES:
        if debug:
            print(f'[diff] validation `{validation_cmd}` failed; asking the implementor to fix '
                  f'(pass {fix_passes + 1}/{MAX_VALIDATION_FIX_PASSES})')
        try:
            await _run_implementor(ctx, _validation_fix_request(validation_cmd, out), model,
                                   max_failures, debug)
        except (tools.ToolFailureLimitExceeded, KeyboardInterrupt):
            break
        passed, out = await _run_validation(cwd, validation_cmd, safe)
        fix_passes += 1
    return passed, out


# --- review gate (async) ----------------------------------------------------------

class ReviewGateFailed(Exception):
    """Raised by _review_gate when the review could not run at all (every reviewer failed). That
    is not a clean review: a commit must not be made on the strength of a review that never
    happened, so the caller stops and reports a failure."""
    pass


async def _review_gate(cwd: str, base_name: str, base_ref: str, spec_text: str,
                       model: str, debug: bool) -> list[Finding]:
    # The review gate over the implementation's working-tree changes (uncommitted): the review
    # personas plus the deterministic evidence gate, over `git diff <base>` (which includes
    # uncommitted work). A clean gate has zero findings at any severity.
    #
    # `git diff <base>` shows only tracked files, so a change that consists solely of NEW files
    # (still untracked) would read as empty and skip the gate — yet `_commit` stages everything
    # with `git add -A`. Stage the working tree first so the review sees exactly what the commit
    # will capture (this is the same non-destructive staging the commit performs).
    await _git('add', '-A', cwd=cwd)
    stat_text = await working_tree_diff_stat(base_ref, cwd)
    if not stat_text.strip():
        return []
    context_blocks = [tools.wrap_untrusted(
        'spec', tools.truncate(spec_text, limit=REVIEW_CONTEXT_LIMIT))]
    message = build_review_message(
        stat_text, base_name, base_ref, context_blocks, working_tree=True)
    registry = build_registry()
    combined = (resolve_loop_reviewers('impl', None, registry)
                + resolve_loop_reviewers('review', None, registry))
    reviewers = [(n, b, i + 1) for i, (n, b, _) in enumerate(combined)]
    tool_ctx = tools.ToolContext(
        phase='review', workdir=cwd, notes=[], require_evidence=True)
    guidance = backends.current().persona_guidance()
    try:
        findings = await _review_pass(
            reviewers, message, model, base_name, base_ref, 1, guidance, tool_ctx,
            {}, {}, REVIEW_REASONING_EFFORT, REVIEW_SEED, debug, fail_on_all_errors=True)
    except Exception as e:
        # The review could not run (every reviewer failed). That is not a clean review: do not
        # proceed to a commit on the strength of a review that never happened.
        raise ReviewGateFailed(str(e)) from e
    findings = await evidence_gate(findings, cwd, base_ref, debug=debug, working_tree=True)
    if findings:
        # Collapse same-location findings first (model-independent), run the semantic pass on the
        # smaller list, then collapse again; the semantic pass also drops non-defects, so a change
        # with no real issue can reduce to zero (allow_empty).
        findings = dedup_by_location(findings)
        support = {(f['name'], f['label']): f.get('support', '')
                   for f in findings}
        evidence = {(f['name'], f['label']): list(
            f.get('evidence') or []) for f in findings}
        sources = {(f['name'], f['label']): list(
            f.get('sources') or []) for f in findings}
        context = (f'Consolidate findings from a code review of the working-tree changes '
                   f'against the default branch {base_name}.')
        consolidated = await consolidate_findings(
            context, findings, model, debug=debug, allow_empty=True,
            reasoning_effort=REVIEW_REASONING_EFFORT, seed=REVIEW_SEED)
        findings = dedup_findings(consolidated)
        # Re-attach the reviewer's support and evidence by (name, label) (the consolidator only
        # guarantees the [Name-Label]), then re-run the deterministic gate on the rewritten set.
        for f in findings:
            key = (f['name'], f['label'])
            f['support'] = support.get(key, '')
            f['evidence'] = evidence.get(key, [])
            f['sources'] = sources.get(key, [])
        findings = await evidence_gate(
            findings, cwd, base_ref, debug=debug, post_consolidation=True, working_tree=True)
        findings = dedup_findings(findings)
        findings = dedup_by_location(findings)
    return findings


# --- reporting --------------------------------------------------------------------

async def _report(cwd: str, base_ref: str, short: str, ticket_id: str | None, *,
                  design: str, validation: str, review: str, commit: str) -> None:
    # The human-readable summary: changed files plus the design/validation/review/commit results.
    print('\n' + '=' * 64)
    print('marsha diff — summary')
    print('=' * 64)
    changed = await _changed_files(cwd, base_ref)
    print(f'Spec: {short}' + (f' (ticket {ticket_id})' if ticket_id else ''))
    print(f'Changed files ({len(changed)}):')
    for path in (changed or ['(none)']):
        print(f'  {path}')
    print(f'Design gate: {design}')
    print(f'Validation: {validation}')
    print(f'Review gate: {review}')
    print(f'Commit: {commit}')


# --- PR proposal + refine-on-reject (async) ---------------------------------------

async def _propose_and_maybe_refine(
        read_line: Callable[[], str], cwd: str, source: SpecSource, spec_text: str,
        original_fields: tuple[str, str] | None, current_repo: str, base_name: str,
        base_ref: str, impl_ctx: tools.ToolContext, commit_title: str,
        validation_cmd: str, review_result: str, model: str, max_failures: int,
        debug: bool, safe: bool = False) -> tuple[int, str, str]:
    # Show the proposed PR title/body; on acceptance finish; on rejection optionally refine the
    # source (and, if it locks, resume implementation + validation + a clean review on the same
    # branch and commit a follow-up) before finishing. Marsha never pushes or creates a PR.
    # Returns (exit_code, final_commit_title, final_sha).
    sha = (await _head_commit(cwd))[0]
    while True:
        title, body = _pr_title_and_body(
            commit_title, spec_text, validation_cmd, review_result)
        print('\n' + '=' * 64)
        print('Proposed PR title:\n  ' + title)
        print('\nProposed PR body:\n' + body)
        print('\nMarsha will not push or create a PR — this is ready for you to use when you '
              'open one.\nAccept this proposal? [y/N] ')
        answer = read_line().strip().lower()
        while answer not in ('y', 'yes', 'n', 'no'):
            print('Please answer y or n. [y/N] ')
            answer = read_line().strip().lower()
        if answer in ('y', 'yes'):
            print('Accepted. No PR was pushed or created.')
            return (0, commit_title, sha)
        # Rejection: offer to refine the source (the existing interactive refine flow).
        print(
            'Proposal rejected. Refine the specification and re-implement? [y/N] ')
        refine_answer = read_line().strip().lower()
        while refine_answer not in ('y', 'yes', 'n', 'no'):
            print('Please answer y or n. [y/N] ')
            refine_answer = read_line().strip().lower()
        if refine_answer not in ('y', 'yes'):
            print('Leaving the existing commit intact; no further implementation pass.')
            return (0, commit_title, sha)
        # Run the interactive refine chat; a lock yields the updated source to re-implement from.
        try:
            result = await run_refine_chat(
                kind=source.kind, spec_text=spec_text, ambiguities=[], errors=[],
                current_repo=current_repo, in_repo=True, cwd=cwd, model=model,
                max_turns=40, read_line=read_line, debug=debug)
        except (Exception, KeyboardInterrupt) as e:
            print(f'error: the refinement failed: {e}. Leaving the existing commit intact.',
                  file=sys.stderr)
            return (0, commit_title, sha)
        if result.status != 'locked' or result.payload is None:
            print('Refinement did not lock; leaving the existing commit intact.')
            return (0, commit_title, sha)
        # Apply the refined source (a staleness re-read guards against a concurrent edit).
        try:
            if source.kind == 'mrsh':
                current, _fields = await load_spec_with_fields(source, cwd)
                if current != spec_text:
                    print('The source changed while refining; leaving the existing commit intact.',
                          file=sys.stderr)
                    return (0, commit_title, sha)
            else:
                if await _source_fields(source, cwd) != original_fields:
                    print('The source changed while refining; leaving the existing commit intact.',
                          file=sys.stderr)
                    return (0, commit_title, sha)
            await _apply(source, result.payload, cwd)
            if source.kind != 'mrsh':
                # The source now holds the refined fields; refresh the staleness baseline so a
                # later reject-and-refine cycle compares against them (not the pre-refinement
                # originals), or a legitimate second refinement would read as a concurrent edit.
                original_fields = await _source_fields(source, cwd)
        except (Exception, KeyboardInterrupt) as e:
            print(f'error: failed to update the source: {e}. Leaving the existing commit '
                  'intact.', file=sys.stderr)
            return (0, commit_title, sha)
        # Resume implementation + validation on the same branch (the source is design-locked, so
        # the follow-up is a normal run). A follow-up commit is made only if validation passes and
        # the review gate is clean.
        spec_text = (result.payload.get('spec', '') if source.kind == 'mrsh'
                     else (result.payload.get('title', '') + '\n\n'
                           + result.payload.get('body', '')))
        try:
            print('Re-implementing the refined specification...', file=sys.stderr)
            await _run_implementor(
                impl_ctx, _impl_request(
                    spec_text, _conventions_text(cwd), base_name, False),
                model, max_failures, debug)
            passed, _out = await _ensure_validated(
                impl_ctx, cwd, validation_cmd, model, max_failures, debug, safe)
            if not passed:
                print('Validation failed after refinement; leaving the existing commit intact.',
                      file=sys.stderr)
                return (0, commit_title, sha)
            remaining = await _review_gate(cwd, base_name, base_ref, spec_text, model, debug)
            if remaining:
                print(f'Review gate not clean after refinement ({len(remaining)} finding(s)); '
                      'leaving the existing commit intact.', file=sys.stderr)
                return (0, commit_title, sha)
        except (tools.ToolFailureLimitExceeded, KeyboardInterrupt):
            print('The follow-up implementation did not complete; leaving the existing commit '
                  'intact.', file=sys.stderr)
            return (0, commit_title, sha)
        review_result = 'clean (no actionable findings)'
        follow_title = _commit_title(
            await _recent_commit_subjects(cwd), _short_title(
                result.payload.get('title') or spec_text),
            source.name if source.kind == 'linear' else None)
        try:
            follow_sha = await _commit(
                cwd, follow_title,
                _pr_title_and_body(
                    follow_title, spec_text, validation_cmd, review_result)[1])
        except (Exception, KeyboardInterrupt) as e:
            print(f'error: the follow-up commit failed: {e}.', file=sys.stderr)
            return (0, commit_title, sha)
        commit_title, sha = follow_title, follow_sha
        # Present the updated proposal (loop back).


# --- orchestrator (async) ---------------------------------------------------------

async def run_diff(args: Any, read_line: Callable[[], str] | None = None) -> int:
    # The marsha diff orchestrator. Returns 0 on success (including a successful --safe run), 1
    # on a design/implementation/validation/review failure, 2 on a command-usage error.
    read_line = read_line or _read_line
    cwd = os.getcwd()
    debug = bool(getattr(args, 'debug', False)
                 or getattr(args, 'trace', False)
                 or getattr(args, 'trace_full', False))
    safe = bool(getattr(args, 'safe', False))
    # `--review-cycles` must be a non-negative integer and `--max-tool-failure` a positive one (the
    # CLI contract). An out-of-range value is a usage error (exit 2), never silently clamped:
    # clamping a negative --review-cycles to 0 would disable the review gate — a safety bypass.
    try:
        review_cycles = int(args.review_cycles)
        max_tool_failure = int(args.max_tool_failure)
    except (TypeError, ValueError):
        print('error: --review-cycles and --max-tool-failure must be integers.', file=sys.stderr)
        return 2
    if review_cycles < 0:
        print('error: --review-cycles must be a non-negative integer (0 disables the review gate).',
              file=sys.stderr)
        return 2
    if max_tool_failure < 1:
        print('error: --max-tool-failure must be a positive integer.', file=sys.stderr)
        return 2
    model = resolve_model()

    # --- Design gate: resolve the source and confirm the spec is locked (no edits yet). ---
    try:
        source = resolve_source(args)
    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    try:
        current_repo = await _repo_gate(source, cwd)
    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    try:
        spec_text, original_fields = await load_spec_with_fields(source, cwd)
    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    # A .mrsh has no title field; the branch slug falls back to its filename stem.
    title = (original_fields[0] if original_fields else '').strip()
    if not title and source.kind == 'mrsh' and source.path:
        title = os.path.splitext(os.path.basename(source.path))[0]
    short = _short_title(title)
    ticket_id = source.name if source.kind == 'linear' else None

    print('Analyzing the spec for open ambiguities...', file=sys.stderr)
    window = await _resolve_window(model)
    design_tool_ctx = None
    if source.kind != 'mrsh':
        # Safe mode disables the network: the design gate (which uses the `refine` phase, that has
        # the web category) must not get web tools, so drop them from its category set.
        design_categories = tools.PHASE_CATEGORIES['refine']
        if safe:
            design_categories = design_categories - {tools.CATEGORY_WEB}
        design_tool_ctx = tools.ToolContext(
            phase='refine', workdir=cwd, require_evidence=False, context_window=window,
            categories=design_categories)
    try:
        check = await analyze_spec(spec_text, tool_ctx=design_tool_ctx, debug=debug)
    except Exception as e:
        print(f'error: spec analysis failed: {e}', file=sys.stderr)
        return 1
    if not (bool(check['compilable']) and not check['ambiguities']):
        for error in check['errors']:
            print(f'error: {error}')
        for ambiguity in check['ambiguities']:
            print(f'warning: {ambiguity}')
        print(f'Design gate: NOT locked ({len(check["ambiguities"])} open ambiguity(ies)); '
              'no changes were made.')
        return 1
    print('Design gate: spec is locked (compilable, no open ambiguities).')

    # --- Working tree and branch. ---
    base_ref = 'HEAD'
    try:
        if not await working_tree_clean(cwd):
            print('The working tree is not clean; refusing to start. Commit or stash your '
                  'changes and re-run.', file=sys.stderr)
            return 1
        base_name, base_ref = await default_branch(cwd)
        branch = await _unique_branch(f'marsha/{_slugify(title)}', cwd)
        await _create_branch(branch, base_ref, cwd)
        print(
            f'Created and checked out branch `{branch}` (based on `{base_name}`).')
    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    # --- Implementation. ---
    phase = 'implement-safe' if safe else 'implement'
    impl_ctx = tools.ToolContext(
        phase=phase, workdir=cwd, backend=backends.current())
    conventions = _conventions_text(cwd)
    if not safe:
        print(VM_WARNING)
    try:
        print('Implementing the spec in the working tree...', file=sys.stderr)
        await _run_implementor(
            impl_ctx, _impl_request(spec_text, conventions, base_name, safe),
            model, max_tool_failure, debug)
    except tools.ToolFailureLimitExceeded as e:
        await _report(cwd, base_ref, short, ticket_id, design='locked',
                      validation='not run (implementation stopped)',
                      review='not run', commit=f'none (implementation stopped: {e})')
        return 1
    except KeyboardInterrupt:
        await _report(cwd, base_ref, short, ticket_id, design='locked',
                      validation='not run (interrupted)', review='not run',
                      commit='none (interrupted)')
        return 1

    # --- Validation (the project's own checks; fix-and-rerun until green or the cap is hit). ---
    validation_cmd = _validation_command(cwd)
    print(f'Running validation: `{validation_cmd}`', file=sys.stderr)
    passed, _out = await _ensure_validated(
        impl_ctx, cwd, validation_cmd, model, max_tool_failure, debug, safe)
    if not passed:
        await _report(cwd, base_ref, short, ticket_id, design='locked',
                      validation=f'FAILED (`{validation_cmd}`)', review='not run',
                      commit='none (validation failed)')
        return 1
    validation = f'passed (`{validation_cmd}`)'

    # --- Review gate (working-tree changes vs base). ---
    if review_cycles >= 1:
        try:
            remaining = await _review_gate(cwd, base_name, base_ref, spec_text, model, debug)
            cycles = 0
            while remaining and cycles < review_cycles:
                print(f'Review found {len(remaining)} finding(s); addressing (cycle '
                      f'{cycles + 1}/{review_cycles})...', file=sys.stderr)
                try:
                    await _run_implementor(
                        impl_ctx, _address_findings_request(
                            remaining, base_name),
                        model, max_tool_failure, debug)
                except (tools.ToolFailureLimitExceeded, KeyboardInterrupt):
                    break
                passed, _out = await _ensure_validated(
                    impl_ctx, cwd, validation_cmd, model, max_tool_failure, debug, safe)
                if not passed:
                    break
                remaining = await _review_gate(
                    cwd, base_name, base_ref, spec_text, model, debug)
                cycles += 1
            if remaining:
                await _report(cwd, base_ref, short, ticket_id, design='locked',
                              validation=validation,
                              review=f'NOT clean ({len(remaining)} finding(s) remain)',
                              commit='none (review not clean)')
                return 1
        except ReviewGateFailed as e:
            # The review could not run (every reviewer failed). Do not commit on the strength of
            # a review that never happened: stop and report a failure.
            await _report(cwd, base_ref, short, ticket_id, design='locked',
                          validation=validation,
                          review=f'FAILED (the review gate could not run: {e})',
                          commit='none (review gate failed to run)')
            return 1
        review_result = 'clean (no actionable findings)'
    else:
        review_result = 'skipped (--review-cycles 0)'

    # --- Safe mode: leave edits uncommitted; no PR proposal. ---
    if safe:
        print('Safe mode: the implementation is complete; edits are left UNCOMMITTED. '
              'No PR proposal is offered.')
        await _report(cwd, base_ref, short, ticket_id, design='locked',
                      validation=validation, review=review_result,
                      commit='none (--safe leaves edits uncommitted)')
        return 0

    # --- Commit (normal mode, after validation and a clean review). ---
    try:
        commit_title = _commit_title(
            await _recent_commit_subjects(cwd), short, ticket_id)
        _commit_body = _pr_title_and_body(
            commit_title, spec_text, validation_cmd, review_result)[1]
        commit_sha = await _commit(cwd, commit_title, _commit_body)
    except (Exception, KeyboardInterrupt) as e:
        await _report(cwd, base_ref, short, ticket_id, design='locked',
                      validation=validation, review=review_result, commit=f'FAILED: {e}')
        return 1

    # --- PR proposal + interactive accept/reject (with optional refine-on-reject). ---
    exit_code, commit_title, commit_sha = await _propose_and_maybe_refine(
        read_line, cwd, source, spec_text, original_fields, current_repo, base_name,
        base_ref, impl_ctx, commit_title, validation_cmd, review_result, model,
        max_tool_failure, debug, safe)
    await _report(cwd, base_ref, short, ticket_id, design='locked', validation=validation,
                  review=review_result, commit=f'{commit_sha} ({commit_title})')
    return exit_code
