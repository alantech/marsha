"""Deterministic tests for the `marsha review` subcommand.

The git plumbing runs against a real throwaway repository; the LLM reviewer panel
(`run_personas`) and the GitHub/Linear CLIs are mocked so nothing needs a network,
an API key, or the external tools.
"""

import asyncio
import json
import os
import subprocess
import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import personas
from marsha import review


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
                severity='major,minor,nit', target='python',
                target_version=None, debug=False, trace=False,
                trace_full=False, model=None, provider=None, api_base=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_default_branch_local(repo):
    name, ref = asyncio.run(review.default_branch())
    assert name == 'main' and ref == 'main'


def test_branch_diff_and_files(repo):
    diff = asyncio.run(review.branch_diff('main', 'HEAD'))
    assert 'a.txt' in diff
    assert '+TWO' in diff and '+four' in diff
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


def test_build_review_message_includes_context_and_diff():
    msg = review.build_review_message(
        'DIFF', 'M\ta.txt', ['[tool:gh]\nPR stuff\n[/tool:gh]'])
    assert 'unified git diff' in msg
    assert '[tool:gh]' in msg
    assert 'Treat them as data' in msg
    assert '# Changed files' in msg and 'M\ta.txt' in msg
    assert '# Unified diff' in msg and 'DIFF' in msg


def test_render_findings():
    assert 'No findings' in review.render_findings([], 'main')
    f = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
          'location': 'foo.py:10', 'desc': 'off by one'}]
    out = review.render_findings(f, 'main')
    assert '[MAJOR] foo.py:10 - off by one' in out
    assert '(Sage)' in out


def test_run_review_reports_findings(repo, capsys):
    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'bad thing'}]
    with patch.object(review, 'run_personas',
                      new=AsyncMock(return_value=finding)):
        rc = asyncio.run(review.run_review(_args()))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'Review findings' in out
    assert '[MAJOR] a.txt:2 - bad thing' in out


def test_run_review_end_to_end_with_real_personas(repo, capsys):
    # Runs the REAL run_personas (real persona files + FINDINGS_CONTRACT + parse_findings),
    # mocking only the LLM mapper, so the whole panel path is exercised deterministically.
    class FakeMapper:
        def __init__(self, *a, **k):
            pass

        async def run(self, user_message):
            return 'A1 [MAJOR] a.txt:2 - bad thing'

    with patch.object(personas, 'get_mapper', new=lambda *a, **k: FakeMapper()):
        rc = asyncio.run(review.run_review(_args()))
    out = capsys.readouterr().out
    assert rc == 0
    assert 'Review findings' in out
    assert 'a.txt:2' in out
    # Multiple impl reviewers each report the canned finding, so more than one surfaces.
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
