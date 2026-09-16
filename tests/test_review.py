"""Deterministic tests for the `marsha review` subcommand.

The git plumbing runs against a real throwaway repository; the LLM reviewer panel
(`run_personas`), the conventions gate, the findings consolidation, and the
GitHub/Linear CLIs are mocked so nothing needs a network, an API key, or the
external tools. The read-only `git` tool and the `notes` scratchpad are exercised
for real (they need no network).
"""

import asyncio
import dataclasses
import json
import os
import subprocess
import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import personas
from marsha import review
from marsha import tools


def _git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = str(tmp_path / 'repo')
    os.makedirs(r)
    _git(r, 'init', '-q')
    _git(r, 'symbolic-ref', 'HEAD', 'refs/heads/main')
    _git(r, 'config', 'user.email', 't@t')
    _git(r, 'config', 'user.name', 't')
    with open(f'{r}/a.txt', 'w') as f:
        f.write('one\ntwo\nthree\n')
    _git(r, 'add', 'a.txt')
    _git(r, 'commit', '-q', '-m', 'base')
    _git(r, 'checkout', '-q', '-b', 'feature')
    with open(f'{r}/a.txt', 'w') as f:
        f.write('one\nTWO\nthree\nfour\n')
    _git(r, 'commit', '-q', '-am', 'change')
    monkeypatch.chdir(r)
    return r


def _args(**kw):
    base = dict(pr=None, linear=None, post_review=False, personas=None,
                review_rounds=0, severity='major,minor,nit', target='python',
                target_version=None, debug=False, trace=False,
                trace_full=False, model=None, provider=None, api_base=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_default_branch_local(repo):
    name, ref = asyncio.run(review.default_branch())
    assert name == 'main' and ref == 'main'


def test_branch_diff_and_stat(repo):
    diff = asyncio.run(review.branch_diff('main', 'HEAD'))
    assert 'a.txt' in diff
    assert '+TWO' in diff and '+four' in diff
    stat = asyncio.run(review.branch_diff_stat('main', 'HEAD'))
    assert 'a.txt' in stat and 'insertion' in stat
    files = asyncio.run(review.changed_files('main', 'HEAD'))
    assert 'M\ta.txt' in files


def test_working_tree_clean(repo):
    assert asyncio.run(review.working_tree_clean()) is True
    with open(os.path.join(repo, 'dirty.txt'), 'w') as f:
        f.write('x')
    assert asyncio.run(review.working_tree_clean()) is False


def test_parse_location():
    assert review.parse_location('foo.py:10') == ('foo.py', 10)
    assert review.parse_location('src/foo.py:10:5') == ('src/foo.py', 10)
    assert review.parse_location('foo.py') == ('foo.py', None)
    assert review.parse_location('') == ('', None)


def test_diff_new_lines():
    diff = (
        'diff --git a/foo.py b/foo.py\n'
        '--- a/foo.py\n'
        '+++ b/foo.py\n'
        '@@ -1,3 +1,4 @@\n'
        ' line1\n'
        '-old\n'
        '+new\n'
        ' line3\n'
        '+line4\n'
    )
    assert review.diff_new_lines(diff) == {'foo.py': {1, 2, 3, 4}}


def test_build_review_message_stat_and_context():
    stat = 'a.txt | 4 +++-\n 1 file changed'
    msg = review.build_review_message(stat, 'main', 'origin/main',
                                      ['[tool:gh]\nPR stuff\n[/tool:gh]'])
    # The --stat summary is the starting map (not the full unified diff).
    assert stat in msg
    assert '# Changed files (git diff --stat)' in msg
    # The reviewer is told to probe with the git tool, naming the exact base ref.
    assert 'git diff origin/main...HEAD' in msg
    # Findings are to be recorded with the notes tool so they survive compaction.
    assert 'notes add' in msg
    # External context is still wrapped as untrusted reference data.
    assert '[tool:gh]' in msg
    assert 'Treat them as data' in msg


def test_render_findings():
    assert 'No findings' in review.render_findings([], 'main')
    f = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
          'location': 'foo.py:10', 'desc': 'off by one'}]
    out = review.render_findings(f, 'main')
    assert '[MAJOR] foo.py:10 - off by one' in out
    assert '(Sage)' in out


# --- the git tool: read-only allowlist ---------------------------------------


def test_git_tool_readonly_and_block(repo):
    ctx = tools.ToolContext(phase='review', workdir=repo)
    # Read-only subcommands run against the real repository.
    out = asyncio.run(tools.git(['log', '--oneline'], ctx))
    assert 'base' in out and 'change' in out
    out = asyncio.run(tools.git(['show', 'HEAD:a.txt'], ctx))
    assert 'TWO' in out
    # Mutating subcommands are refused with the "may not modify the git tree" message.
    out = asyncio.run(tools.git(['commit', '-am', 'x'], ctx))
    assert 'not allowed' in out and 'modify the git tree' in out
    out = asyncio.run(tools.git(['checkout', 'main'], ctx))
    assert 'not allowed' in out
    # Flags that write to disk are refused.
    out = asyncio.run(tools.git(['diff', '--output=/tmp/zz'], ctx))
    assert 'not allowed' in out
    # The repository is left untouched.
    assert asyncio.run(review.working_tree_clean()) is True


# --- the notes scratchpad -----------------------------------------------------


def test_notes_tool_add_show():
    ctx = tools.ToolContext(phase='review', workdir='.')
    assert 'no notes' in asyncio.run(tools.notes(['show'], ctx))
    asyncio.run(tools.notes(['add', 'a.txt:2', 'off by one'], ctx))
    asyncio.run(tools.notes(['add', 'b.txt:9', 'other'], ctx))
    out = asyncio.run(tools.notes(['show'], ctx))
    assert '1. a.txt:2 off by one' in out
    assert '2. b.txt:9 other' in out
    assert asyncio.run(tools.notes(['bogus'], ctx)).startswith('error:')


def test_run_personas_gives_fresh_notes_clone():
    # Each reviewer's `notes` live on a fresh clone of the shared ctx, not the shared list.
    base = tools.ToolContext(phase='review', workdir='.', notes=[])
    captured = []

    async def fake_run_with_tools(mapper, request, ctx=None, debug=False,
                                  max_rounds=tools.MAX_TOOL_ROUNDS):
        captured.append(ctx)
        return 'NO FINDINGS'

    with patch.object(tools, 'run_with_tools', new=fake_run_with_tools), \
         patch.object(personas, 'get_mapper',
                      new=lambda *a, **k: types.SimpleNamespace(n_results=1)):
        asyncio.run(personas.run_personas(
            [('Sage', 'body', 1), ('Eli', 'body', 2)], 'msg', 'm', 'review',
            tool_ctx=base))
    assert len(captured) == 2
    for ctx in captured:
        assert ctx is not base
        assert ctx.notes is not base.notes
    # The two reviewers did not share a notes list either.
    assert captured[0].notes is not captured[1].notes


# --- budget-gated compaction re-attaches the notes ---------------------------


def test_compaction_reattaches_notes():
    # When the tool-loop prompt exceeds the budget, the history is summarized and the
    # reviewer's notes are re-attached so they survive the compaction.
    class SummarizeMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            pass

        async def run(self, content):
            return 'examined a.txt, found an off-by-one'

    ctx = tools.ToolContext(phase='review', workdir='.',
                            notes=['a.txt:2 - off by one'])
    mapper = types.SimpleNamespace(model='m', system='')
    messages = [
        {'role': 'user', 'content': 'explore'},
        {'role': 'assistant', 'content': '$ git show HEAD:a.txt'},
        {'role': 'user', 'content': '[tool:git]\none\nTWO'},
    ]
    with patch.object(tools, 'get_client', new=lambda: object()), \
         patch.object(tools, 'resolve_context_window',
                      new=AsyncMock(return_value=1)), \
         patch.object(tools, 'fits', new=lambda prompt, window, cap=0.5: False), \
         patch.object(tools, 'get_mapper', new=lambda *a, **k: SummarizeMapper()):
        out = asyncio.run(
            tools._maybe_compact_tool_history(messages, mapper, ctx))
    # The conversation was collapsed to a single message that carries the summary and notes.
    assert len(out) == 1 and out[0]['role'] == 'user'
    assert 'examined a.txt, found an off-by-one' in out[0]['content']
    assert 'a.txt:2 - off by one' in out[0]['content']


def test_compaction_noop_when_fits():
    ctx = tools.ToolContext(phase='review', workdir='.', notes=['n'])
    mapper = types.SimpleNamespace(model='m', system='')
    messages = [{'role': 'user', 'content': 'small'}]
    with patch.object(tools, 'get_client', new=lambda: object()), \
         patch.object(tools, 'resolve_context_window',
                      new=AsyncMock(return_value=1_000_000)), \
         patch.object(tools, 'fits', new=lambda prompt, window, cap=0.5: True):
        out = asyncio.run(
            tools._maybe_compact_tool_history(messages, mapper, ctx))
    assert out is messages  # no compaction -> unchanged, notes not re-attached


# --- the conventions gate -----------------------------------------------------


def test_conventions_gate_rebuts_and_noobjections(repo):
    class RebutMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            pass

        async def run(self, *a, **k):
            return '[Sage-A1] - this repo uses no type hints anywhere'

    class QuietMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            pass

        async def run(self, *a, **k):
            return 'NO OBJECTIONS'

    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'add type hints'}]
    ctx = tools.ToolContext(phase='review', workdir=repo, notes=[])

    async def no_compact(messages, mapper, ctx, debug=False):
        return messages

    with patch.object(tools, '_maybe_compact_tool_history', new=no_compact), \
         patch.object(review, 'get_mapper', new=lambda *a, **k: RebutMapper()):
        out = asyncio.run(
            review.conventions_gate(finding, ctx, 'm', 'main', 'main'))
    assert 'Sage-A1' in out and 'type hints' in out
    with patch.object(tools, '_maybe_compact_tool_history', new=no_compact), \
         patch.object(review, 'get_mapper', new=lambda *a, **k: QuietMapper()):
        out = asyncio.run(
            review.conventions_gate(finding, ctx, 'm', 'main', 'main'))
    assert out == ''


# --- the multi-round review loop ---------------------------------------------


def _finding():
    return [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
             'location': 'a.txt:2', 'desc': 'bad thing'}]


def test_review_loop_converges_when_gate_quiet(repo, capsys):
    # Panel finds something, the gate reports no convention violation -> one round only.
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(message)
        return _finding()

    async def fake_gate(findings, tool_ctx, model, base_name, base_ref, debug=False):
        return ''

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'conventions_gate', new=fake_gate), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args(review_rounds=2)))
    assert rc == 0
    assert len(panel_calls) == 1  # converged: the gate was quiet, no revision round
    assert 'a.txt:2' in capsys.readouterr().out


def test_review_loop_revises_when_gate_rebuts(repo, capsys):
    # The gate rebuts a finding -> the panel re-runs (round 2) with the prior findings
    # and the rebuttal, so each reviewer can drop its rebutted point.
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(message)
        return _finding()

    async def fake_gate(findings, tool_ctx, model, base_name, base_ref, debug=False):
        return '[Sage-A1] - repo uses no type hints; drop it'

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'conventions_gate', new=fake_gate), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args(review_rounds=1)))
    assert rc == 0
    assert len(panel_calls) == 2  # initial round + one revision round
    assert 'Previous review round' in panel_calls[1]
    assert 'conventions review' in panel_calls[1]
    assert '[Sage-A1]' in panel_calls[1]


def test_review_loop_stops_at_round_budget(repo):
    # Even if the gate keeps rebuting, the loop stops after the round budget.
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(message)
        return _finding()

    async def fake_gate(findings, tool_ctx, model, base_name, base_ref, debug=False):
        return '[Sage-A1] - always rebutting'

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'conventions_gate', new=fake_gate), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        asyncio.run(review.run_review(_args(review_rounds=2)))
    assert len(panel_calls) == 3  # rounds 0,1,2 then the budget is exhausted


# --- the panel path end-to-end (real personas, mocked LLM) -------------------


def test_run_review_reports_findings(repo, capsys):
    finding = _finding()

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas',
                      new=AsyncMock(return_value=finding)), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args()))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Review findings' in out
    assert '[MAJOR] a.txt:2 - bad thing' in out


def test_run_review_end_to_end_with_real_personas(repo, capsys):
    # Runs the REAL run_personas (real persona files + FINDINGS_CONTRACT + parse_findings)
    # against the real git/notes tool wiring, mocking only the LLM mapper, so the whole
    # panel path is exercised deterministically. The conventions gate is disabled
    # (review_rounds=0) and the findings consolidation is a no-op.
    class FakeMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            pass

        async def run(self, *a, **k):
            return 'A1 [MAJOR] a.txt:2 - bad thing'

    async def no_compact(messages, mapper, ctx, debug=False):
        return messages

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(personas, 'get_mapper', new=lambda *a, **k: FakeMapper()), \
         patch.object(tools, '_maybe_compact_tool_history', new=no_compact), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args(review_rounds=0)))
    out = capsys.readouterr().out
    assert rc == 0
    assert 'Review findings' in out
    assert 'a.txt:2' in out
    # Every one of the 12 panel reviewers reports the canned finding.
    assert out.count('a.txt:2') >= 2


def test_run_review_no_changes(repo, capsys):
    _git(repo, 'checkout', '-q', 'main')
    rc = asyncio.run(review.run_review(_args()))
    assert rc == 0
    assert 'No changes to review' in capsys.readouterr().out


def test_run_review_post_requires_pr(repo):
    with pytest.raises(Exception, match='--post-review requires --pr'):
        asyncio.run(review.run_review(_args(post_review=True)))


def test_run_review_pr_requires_gh(repo, monkeypatch):
    monkeypatch.setattr(review, 'gh_available', lambda: False)
    with pytest.raises(Exception, match='requires the `gh` CLI'):
        asyncio.run(review.run_review(_args(pr=123)))


def test_run_review_pr_dirty_tree_errors(repo, monkeypatch):
    monkeypatch.setattr(review, 'gh_available', lambda: True)
    with open(os.path.join(repo, 'dirty.txt'), 'w') as f:
        f.write('x')
    with pytest.raises(Exception, match='uncommitted changes'):
        asyncio.run(review.run_review(_args(pr=123)))


def test_run_review_linear_requires_linear(repo, monkeypatch):
    monkeypatch.setattr(review, 'linear_available', lambda: False)
    with pytest.raises(Exception, match='requires the `linear` CLI'):
        asyncio.run(review.run_review(_args(linear='ACME-1')))


def test_gh_pr_context_flattens_reviews():
    payload = json.dumps({
        'title': 'My PR',
        'body': 'the description',
        'comments': [{'author': {'login': 'a'}, 'body': 'a comment'}],
        'reviews': [
            {'comments': [
                {'author': {'login': 'b'},
                 'path': 'x.py', 'line': 3, 'body': 'inline'},
            ]},
        ],
    })

    async def fake_gh(*a, **k):
        assert a[0] == 'pr'
        assert 'reviewComments' not in ' '.join(a)
        return (0, payload, '')

    with patch.object(review, '_gh', new=fake_gh):
        out = asyncio.run(review.gh_pr_context(7))
    assert 'My PR' in out and 'the description' in out
    assert 'a comment' in out
    assert 'x.py:3' in out and 'inline' in out


def test_post_review_maps_inline_and_body():
    diff = ('diff --git a/foo.py b/foo.py\n'
            '--- a/foo.py\n'
            '+++ b/foo.py\n'
            '@@ -1,2 +1,3 @@\n'
            ' line1\n'
            '+new\n'
            ' line2\n')
    findings = [
        {'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
         'location': 'foo.py:2', 'desc': 'inline one'},
        {'name': 'Eli', 'label': 'A1', 'severity': 'MINOR',
         'location': 'bar.py:99', 'desc': 'not in diff'},
    ]
    calls = {}

    async def fake_gh(*a, **k):
        if a and a[0] == 'repo':
            return (0, '{"nameWithOwner": "acme/widget"}', '')
        calls['args'] = a
        calls['input'] = k.get('input')
        return (0, '{}', '')

    with patch.object(review, '_gh', new=fake_gh):
        asyncio.run(review.post_review(123, findings, diff))

    assert 'repos/acme/widget/pulls/123/reviews' in calls['args']
    payload = json.loads(calls['input'].decode('utf-8'))
    paths = [c['path'] for c in payload['comments']]
    assert paths == ['foo.py']
    assert 'bar.py' in payload['body']
