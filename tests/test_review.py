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
import re
import subprocess
import types
from unittest.mock import AsyncMock, patch

import pytest

from marsha import personas
from marsha import review
from marsha import tools


@pytest.fixture(autouse=True)
def _clear_repo_name_cache():
    # `_repo_name` caches the `gh repo view` result per cwd; reset it so each test starts with a
    # cold cache (a warm cache would leak one test's repo into the next).
    review._repo_name_cache.clear()
    yield
    review._repo_name_cache.clear()


# The review loop runs a critic gate (Vera) alongside the conventions gate. The tests no-op it by
# default so no integration test makes a real critic LLM call (matching the "gates are mocked"
# intent); the critic's own tests call the real function, captured here before the fixture patches
# the module attribute.
_CRITIC_GATE = review.critic_gate


@pytest.fixture(autouse=True)
def _critic_quiet():
    async def _quiet(*a, **k):
        return ''
    with patch.object(review, 'critic_gate', new=_quiet):
        yield


def _git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _rev(cwd, ref):
    out = subprocess.run(['git', 'rev-parse', ref], cwd=cwd, check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    return out.stdout.strip()


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
    base = dict(pr=None, remote=False, consensus=0, reasoning_effort=None,
                linear=None,
                post_review=False, personas=None,
                review_rounds=0, target='python',
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


def test_render_findings_includes_support():
    # A finding's supporting paragraphs are rendered under its headline.
    f = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
          'location': 'foo.py:10', 'desc': 'off by one',
          'support': 'the evidence, why it matters, and the impact'}]
    out = review.render_findings(f, 'main')
    assert '[MAJOR] foo.py:10 - off by one' in out
    assert 'the evidence, why it matters, and the impact' in out
    # A finding with no support renders just its headline (no dangling support).
    out2 = review.render_findings(
        [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
          'location': 'foo.py:10', 'desc': 'off by one', 'support': ''}], 'main')
    assert 'off by one' in out2 and '\n\n' not in out2.split('\n', 2)[-1]


def test_order_findings_by_severity_then_location():
    # The final set leads with the most severe, then is ordered by location within a severity.
    fs = [
        {'name': 'E', 'label': 'E1', 'severity': 'NIT', 'location': 'b.py:2', 'desc': 'n'},
        {'name': 'B', 'label': 'B1', 'severity': 'MAJOR', 'location': 'z.py:9', 'desc': 'm'},
        {'name': 'C', 'label': 'C1', 'severity': 'MINOR', 'location': 'a.py:1', 'desc': 'i'},
        {'name': 'D', 'label': 'D1', 'severity': 'MAJOR', 'location': 'a.py:5', 'desc': 'm2'},
        {'name': 'F', 'label': 'F1', 'severity': 'MINOR', 'location': 'a.py:10', 'desc': 'i2'},
    ]
    out = review.order_findings(fs)
    assert [(f['severity'], f['location']) for f in out] == [
        ('MAJOR', 'a.py:5'), ('MAJOR', 'z.py:9'),
        ('MINOR', 'a.py:1'), ('MINOR', 'a.py:10'),
        ('NIT', 'b.py:2')]


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


def test_run_personas_gives_fresh_evidence_clone():
    # Each reviewer's git evidence ledger lives on a fresh clone of the shared ctx (not the shared
    # list), so one reviewer's retrieved code cannot leak into another reviewer's findings.
    base = tools.ToolContext(phase='review', workdir='.', notes=[], evidence=[])
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
        assert ctx.evidence is not base.evidence
    assert captured[0].evidence is not captured[1].evidence


def test_run_personas_attaches_evidence_to_findings():
    # A reviewer's findings carry the git evidence it actually retrieved, so the evidence gate can
    # later verify each finding against real tool output. The shared base ledger is not mutated.
    base = tools.ToolContext(phase='review', workdir='.', notes=[], evidence=[])
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree')]

    async def fake_run_with_tools(mapper, request, ctx=None, debug=False,
                                  max_rounds=tools.MAX_TOOL_ROUNDS):
        ctx.evidence.extend(ev)
        return 'A1 [MAJOR] a.txt:2 - bad thing\nI confirmed the defect with git show HEAD:a.txt.'

    with patch.object(tools, 'run_with_tools', new=fake_run_with_tools), \
         patch.object(personas, 'get_mapper',
                      new=lambda *a, **k: types.SimpleNamespace(n_results=1)):
        fs = asyncio.run(personas.run_personas(
            [('Sage', 'body', 1)], 'msg', 'm', 'review', tool_ctx=base))
    assert len(fs) == 1
    assert fs[0]['evidence'] == ev
    assert base.evidence == []


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


# --- the evidence gate (deterministic anti-hallucination filter) ---------------


def _gate_finding(desc, location, evidence, support=''):
    return {'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
            'location': location, 'desc': desc, 'support': support,
            'evidence': evidence}


def test_gate_keeps_finding_grounded_in_evidence(repo):
    # A named symbol that appears in the reviewer's git output grounds the finding.
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total ignores the x term', 'a.txt:2', ev)
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_drops_finding_whose_symbol_was_never_retrieved(repo):
    # The core case: a finding citing a symbol that appears NOWHERE in the reviewer's git
    # evidence is a guess, dropped even though the cited file exists and the line is in bounds.
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('phantomHandler leaks memory', 'a.txt:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_drops_finding_with_no_symbol_and_no_location(repo):
    # A finding that names no symbol and cites no file cannot be grounded in the code the
    # reviewer read, so it is dropped even when it carries (unrelated) git evidence such as a
    # bare `git status`.
    ev = [('$ git status', 'On branch main\nnothing to commit, working tree clean')]
    f = _gate_finding('the overall approach is flawed', '', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_drops_finding_grounded_only_in_grep_query(repo):
    # A reviewer greps for a symbol that does not exist (git grep returns nothing) and reads a
    # file that lacks it. The symbol appears only in the grep QUERY, not in any retrieved output,
    # so the finding is a fabrication and is dropped — the query string is not evidence.
    ev = [
        ('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour'),
        ('$ git grep -n "phantomHandler"', ''),
    ]
    f = _gate_finding('phantomHandler leaks memory', 'a.txt:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def _add_code_file(repo, name, content):
    with open(f'{repo}/{name}', 'w') as fh:
        fh.write(content)
    _git(repo, 'add', name)
    _git(repo, 'commit', '-q', '-m', f'add {name}')


def test_gate_drops_finding_falsified_by_present_symbol(repo):
    # A finding asserts a symbol is undefined, yet the symbol is present in the tree (and in the
    # reviewer's own evidence, so it clears the grounding check). The contradiction check drops it,
    # because the claim and the code cannot both be true.
    _add_code_file(repo, 'impl.py', 'def presentHelper():\n    return 1\n')
    ev = [('$ git show HEAD:impl.py', 'def presentHelper():\n    return 1')]
    f = _gate_finding('presentHelper is undefined and will raise a NameError',
                      'impl.py:1', ev,
                      support='git grep -n "presentHelper" returns no matches; '
                              'presentHelper is not defined')
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_keeps_absence_claim_not_falsified(repo):
    # "missingThing is undefined, though presentHelper is defined": the accused symbol is genuinely
    # absent from the tree, so nothing contradicts the claim; the co-cited present symbol (nearest
    # to its own, positive "is defined") is never mistaken for the one the finding says is absent.
    _add_code_file(repo, 'impl.py', 'def presentHelper():\n    return 1\n')
    ev = [('$ git show HEAD:impl.py', 'def presentHelper():\n    return 1')]
    f = _gate_finding('missingThing is undefined; add a guard', 'impl.py:1', ev,
                      support='missingThing is not defined, though presentHelper is defined')
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_keeps_absence_finding_on_read_file(repo):
    # "no maxItems": the missing symbol is absent from the evidence, but a co-cited real symbol
    # (minItems) that the reviewer read is present, so the finding survives on at least one anchor.
    ev = [('$ git show HEAD:schema.json', 'impressions: { minItems: 1 }')]
    f = _gate_finding('impressions has no maxItems cap', 'a.txt:1', ev,
                      support='schema.json declares only minItems, no maxItems')
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_drops_finding_with_invented_subject_co_cited(repo):
    # A real, grounded symbol (compute_total, which the reviewer read) sits next to an invented one
    # (expand_globs) that is in neither the evidence nor the tree. The "at least one" primary check
    # passes on compute_total, but the fabricated-subject tripwire trips on expand_globs.
    _add_code_file(repo, 'calc.py', 'def compute_total():\n    return x + y\n')
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total calls expand_globs which never resolves',
                      'calc.py:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_drops_finding_with_invented_camelcase_co_cited(repo):
    # A real, grounded symbol (compute_total, which the reviewer read) sits next to an invented
    # camelCase one (phantomHandler) in neither the evidence nor the tree; the tripwire trips on
    # camelCase identifiers too, not only underscored ones (wistful is Python + TypeScript).
    _add_code_file(repo, 'calc.py', 'def compute_total():\n    return x + y\n')
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total calls phantomHandler which never resolves',
                      'calc.py:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_symbol_match_is_whole_word_not_substring(repo):
    # The anchor check is a whole-word match, not a substring match: a fabricated symbol that is
    # a prefix of a symbol the reviewer actually read (phantomHand on phantomHandler) does not
    # ground the finding.
    _add_code_file(repo, 'calc.py', 'def phantomHandler():\n    return 1\n')
    ev = [('$ git show HEAD:calc.py', 'def phantomHandler():\n    return 1')]
    f = _gate_finding('phantomHand leaks memory', 'calc.py:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_post_consolidation_drops_invented_camelcase(repo):
    # Post-consolidation the primary evidence check is skipped, so the fabricated-subject tripwire
    # is the only symbol-level backstop; it must catch a camelCase symbol a consolidator rewords in
    # (phantomHandler) that is in neither the evidence nor the tree.
    _add_code_file(repo, 'calc.py', 'def compute_total():\n    return x + y\n')
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total routes through phantomHandler', 'calc.py:2', ev)
    assert asyncio.run(review.evidence_gate(
        [f], repo, 'main', post_consolidation=True)) == []


def test_gate_keeps_finding_grounded_on_opened_file_in_support(repo):
    # A finding with no code-symbol anchor whose support cites the file it opened (a.txt, a real
    # file in the repo) is grounded on the file being opened, not dropped because the filename
    # does not appear in the retrieved output — a.txt is a file, not a code symbol.
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('the value here is wrong', 'a.txt:2', ev,
                      support='confirmed the line via git show HEAD:a.txt')
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_keeps_finding_with_real_tree_symbol_not_in_evidence(repo):
    # Two real snake_case symbols: compute_total is in the reviewer's evidence, helper_fn is in the
    # tree but not in this finding's evidence. The tripwire grounds helper_fn on the tree (a real
    # symbol is never a fabrication), so the finding is kept rather than dropped.
    _add_code_file(repo, 'calc.py',
                   'def compute_total():\n    return x + y\n\n'
                   'def helper_fn():\n    return z\n')
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total and helper_fn disagree on z', 'calc.py:2', ev)
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_post_consolidation_keeps_rephrased_real_finding(repo):
    # The consolidator rewords a finding to name a symbol the reviewer did not literally retrieve
    # (compute_total is in the tree but not in this finding's git output). Pre-consolidation the
    # "at least one symbol in the reviewer's evidence" check drops it; post_consolidation skips that
    # check, so a real symbol (grounded in the tree) is kept rather than mistaken for ungrounded.
    _add_code_file(repo, 'calc.py', 'def compute_total():\n    return x + y\n')
    ev = [('$ git show HEAD:calc.py', 'def add_pair():\n    return x + y')]
    f = _gate_finding('compute_total drops an operand', 'calc.py:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []
    kept = asyncio.run(review.evidence_gate([f], repo, 'main', post_consolidation=True))
    assert kept == [f]


def test_gate_drops_finding_whose_file_was_never_opened(repo):
    # No distinctive symbol and a cited file the reviewer never opened -> not grounded -> dropped.
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('the config is wrong', 'zeta.conf:5', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_backstop_drops_nonexistent_file(repo):
    # Even grounded in retrieved evidence, a finding citing a file that does not exist anywhere
    # (HEAD or base) is fabricated and dropped by the file-data backstop.
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total ignores the x term', 'does_not_exist.py:3', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_backstop_drops_line_beyond_file(repo):
    # A cited line past the end of the real file is a fabricated line number -> dropped.
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('the value here is wrong', 'a.txt:999', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_backstop_drops_citation_into_empty_file(repo):
    # An existing but empty file has 0 lines, so a citation to any line is out of range ->
    # dropped. (Before the fix an empty file read as "unknown length" and the check was skipped.)
    _add_code_file(repo, 'empty.txt', '')
    ev = [('$ git show HEAD:empty.txt', '')]
    f = _gate_finding('the value here is wrong', 'empty.txt:1', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_counts_blank_boundary_lines_in_file(repo):
    # A file with blank lines at its start/end: _file_info counts them from the raw (unstripped)
    # content, so a citation to a real line is in range rather than mis-dropped as out of range.
    _add_code_file(repo, 'pad.txt', '\n\nvalue\n\n')  # 4 lines: blank, blank, value, blank
    ev = [('$ git show HEAD:pad.txt', 'value')]
    f = _gate_finding('the value line is wrong', 'pad.txt:3', ev)
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_git_line_count_matches_splitlines(repo):
    # _git_line_count streams the file (no whole-file buffer) and must agree with splitlines() —
    # counting blank boundary lines and a final line with no trailing newline.
    cases = {'plain.txt': 'a\nb\nc\n', 'lead.txt': '\n\na\n',
             'trail.txt': 'a\n\n\n', 'noeol.txt': 'a\nb', 'empty.txt': ''}
    for name, content in cases.items():
        _add_code_file(repo, name, content)
        got = asyncio.run(review._git_line_count('HEAD', name, repo))
        assert got == len(content.splitlines()), (name, got, content)


def test_gate_drops_finding_with_no_evidence(repo):
    # A finding reported without any git probe is unverified (mandatory probing failed to force one)
    # -> dropped, however plausible it looks. This is the backstop when the loop gave up.
    f = _gate_finding('compute_total ignores the x term', 'a.txt:2', [])
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_empty_input_is_empty(repo):
    assert asyncio.run(review.evidence_gate([], repo, 'main')) == []


def test_gate_keeps_finding_with_line_range_location(repo):
    # A line-range location (a.txt:1-4) is not a single "path:line", so the file-data backstop must
    # not treat "a.txt:1-4" as a filename (that would fabricate a "missing file" and drop a real,
    # grounded finding). The finding is grounded on the file its reviewer read.
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('TWO is uppercase, breaks parsing', 'a.txt:1-4', ev,
                      support='confirmed with git show that the block is wrong')
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_keeps_finding_with_multi_file_location(repo):
    # A multi-file location (a.txt + b.txt) cannot be checked as one file; a finding grounded on a
    # real symbol it read survives rather than being dropped as a bogus "missing file".
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total and its caller disagree', 'a.txt + b.txt', ev)
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_keeps_finding_with_line_spec_suffixes(repo):
    # ":12+", ":1-EOF", and multi-range specs all carry a real file; the backstop strips the line
    # spec (everything after the ':' that starts with a digit) and checks the file, so these are
    # not mistaken for nonexistent paths.
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    for loc in ('a.txt:2+', 'a.txt:1-EOF', 'a.txt:1-2,3-4'):
        f = _gate_finding('the value here is wrong', loc, list(ev))
        kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
        assert kept == [f], loc


def test_gate_drops_out_of_range_line_range_location(repo):
    # A line RANGE past the end of a real file is out of bounds: parse_location returns the
    # largest number in the range (the end), so the backstop bounds-checks it rather than skipping
    # ranges (which previously returned no line and let a fabricated range through).
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]  # a.txt has 4 lines
    f = _gate_finding('the value here is wrong', 'a.txt:3-9', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []
    # A range whose end is within the file is kept.
    f = _gate_finding('the value here is wrong', 'a.txt:2-4', ev)
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_gate_drops_citation_when_line_count_unknown(repo):
    # A file that exists but whose line count cannot be read: a cited line cannot be verified
    # against the file's end, so it is dropped (an out-of-range citation must not pass an
    # unverified backstop).
    async def fake_line_count(ref, path, cwd):
        return None  # counting failed
    with patch.object(review, '_git_line_count', new=fake_line_count):
        ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
        f = _gate_finding('the value here is wrong', 'a.txt:999', ev)
        assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_drops_zero_based_line_citation(repo):
    # Citations are 1-based; a :0 line is not a valid line and is dropped (the bounds check used
    # to reject only lines past the end, so a zero-based citation slipped through).
    ev = [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('the value here is wrong', 'a.txt:0', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_symbol_present_tri_state_on_grep_error(repo):
    # A git grep error (rc >= 2) is not proof of absence: _symbol_present distinguishes a clean
    # no-match (False) from a failed grep (None). A grep error must therefore neither fabricate a
    # symbol (fabricated-subject drops only on a definitive no-match) nor fail to falsify an
    # absence claim (contradiction drops only on a positive match).
    assert asyncio.run(review._symbol_present('TWO', repo, {})) is True
    assert asyncio.run(review._symbol_present('zzzabsent', repo, {})) is False
    with patch.object(review, '_git', new=AsyncMock(return_value=(2, '', 'fatal: bad'))):
        assert asyncio.run(review._symbol_present('whatever', repo, {})) is None


def test_gate_drops_cited_file_match_in_unrelated_command(repo):
    # The cited-file check matches a whole path component, not a substring: a command that read
    # a.txt.backup does not ground a finding that cites a.txt, because a.txt is a prefix of a
    # different file's name, not the file that command actually opened.
    ev = [('$ git show HEAD:a.txt.backup', 'one\nTWO\nthree\nfour')]
    f = _gate_finding('the value here is wrong', 'a.txt:2', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_drops_same_basename_different_directory(repo):
    # A full-path citation (src/foo.py, with a directory) matches only that exact path, not just
    # its basename: a command that read vendor/foo.py does not ground a finding cited at src/foo.py,
    # even though both files share the name foo.py (the cited file was never actually read).
    os.makedirs(f'{repo}/src', exist_ok=True)
    _add_code_file(repo, 'src/foo.py', 'def f():\n    return 1\n')
    ev = [('$ git show HEAD:vendor/foo.py', 'def v():\n    return 2')]
    f = _gate_finding('the value here is wrong', 'src/foo.py:1', ev)
    assert asyncio.run(review.evidence_gate([f], repo, 'main')) == []


def test_gate_post_consolidation_drops_invented_dotted_chain(repo):
    # Post-consolidation the primary evidence check is skipped, so the fabricated-subject tripwire
    # is the only symbol-level backstop. A consolidator can invent a dotted chain (svc.foo.bar); it
    # must be checked in full, not skipped as dotted and not grounded on its `bar` leaf (a real,
    # present symbol), so the finding is dropped when that exact chain is in neither the evidence
    # nor the tree.
    _add_code_file(repo, 'calc.py', 'def compute_total():\n    return x + y\n\nbar = 1\n')
    ev = [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]
    f = _gate_finding('compute_total routes through svc.foo.bar', 'calc.py:2', ev)
    assert asyncio.run(review.evidence_gate(
        [f], repo, 'main', post_consolidation=True)) == []


def test_gate_keeps_deleted_file_finding(repo):
    # A finding about a file the change deleted (present at the base, absent at HEAD): its dotted
    # filename must be recognized as a file (from the base tree, not just HEAD), so the gate falls
    # to the file-opened branch and grounds it on the `git show <base>:<path>` the reviewer ran,
    # instead of treating the filename as a code symbol and dropping it as "never retrieved".
    _git(repo, 'checkout', '-q', 'main')
    with open(f'{repo}/removed.txt', 'w') as fh:
        fh.write('alpha\nbeta\n')
    _git(repo, 'add', 'removed.txt')
    _git(repo, 'commit', '-q', '-m', 'add removed.txt')
    # Checking back out to feature drops removed.txt from the working tree (it is tracked on the
    # base, not on HEAD), so it is the deleted-file case: present at base, absent at HEAD.
    _git(repo, 'checkout', '-q', 'feature')
    ev = [('$ git show main:removed.txt', 'alpha\nbeta')]
    f = _gate_finding('removed.txt handling is wrong', 'removed.txt:1', ev,
                      support='read the deleted file via git show main:removed.txt')
    kept = asyncio.run(review.evidence_gate([f], repo, 'main'))
    assert kept == [f]


def test_review_pass_merges_evidence_across_rounds(repo):
    # A reviewer that verifies with git in round 1 and re-states the finding in round 2 (without
    # re-probing) must keep its round-1 evidence on the final finding, so the gate can verify it
    # against the code the reviewer actually read over the whole pass.
    round1 = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
               'location': 'a.txt:2', 'desc': 'bad thing in compute_total', 'support': '',
               'evidence': [('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]}]
    round2 = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
               'location': 'a.txt:2', 'desc': 'bad thing in compute_total', 'support': '',
               'evidence': []}]
    calls = {'n': 0}

    async def fake_personas(reviewers, user_message, model, stage, **k):
        calls['n'] += 1
        return round1 if calls['n'] == 1 else round2

    async def fake_gate(actionable, tool_ctx, model, base_name, base_ref, debug, **k):
        return '[Sage-A1] - a convention'  # non-empty preamble forces a second round

    with patch.object(review, 'run_personas', new=fake_personas), \
         patch.object(review, 'conventions_gate', new=fake_gate):
        out = asyncio.run(review._review_pass(
            [('Sage', 'body', 1)], 'msg', 'm', 'main', 'main', 1, '',
            tools.ToolContext(phase='review', workdir=repo, notes=[]),
            {}, {}, None, 0, False))
    assert len(out) == 1
    assert out[0]['evidence'] == [
        ('$ git show HEAD:calc.py', 'def compute_total():\n    return x + y')]


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


def test_critic_gate_refutes_and_noobjections(repo):
    # The critic refutes a finding whose claim the code contradicts (citing counter-evidence),
    # or reports NO OBJECTIONS when every finding holds up.
    class RefuteMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            pass

        async def run(self, *a, **k):
            return '[Sage-A1] - compute_total is called at calc.py:40'

    class QuietMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            pass

        async def run(self, *a, **k):
            return 'NO OBJECTIONS'

    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'compute_total is never called'}]
    ctx = tools.ToolContext(phase='review', workdir=repo, notes=[])

    async def no_compact(messages, mapper, ctx, debug=False):
        return messages

    with patch.object(tools, '_maybe_compact_tool_history', new=no_compact), \
         patch.object(review, 'get_mapper', new=lambda *a, **k: RefuteMapper()):
        out = asyncio.run(_CRITIC_GATE(finding, ctx, 'm', 'main', 'main'))
    assert 'Sage-A1' in out and 'calc.py:40' in out
    with patch.object(tools, '_maybe_compact_tool_history', new=no_compact), \
         patch.object(review, 'get_mapper', new=lambda *a, **k: QuietMapper()):
        out = asyncio.run(_CRITIC_GATE(finding, ctx, 'm', 'main', 'main'))
    assert out == ''


# --- the multi-round review loop ---------------------------------------------


def _finding():
    # Carries git evidence (a.txt was read) so it passes the strict evidence gate.
    return [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
             'location': 'a.txt:2', 'desc': 'bad thing',
             'evidence': [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]}]


def test_review_loop_converges_when_gate_quiet(repo, capsys):
    # Panel finds something, the gate reports no convention violation -> one round only.
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(message)
        return _finding()

    async def fake_gate(findings, tool_ctx, model, base_name, base_ref, debug=False, **k):
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

    async def fake_gate(findings, tool_ctx, model, base_name, base_ref, debug=False, **k):
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
    # The revision round tells the reviewer to drop rebutted findings unless very confident.
    assert 'Handling the conventions review' in panel_calls[1]
    assert 'very confident' in panel_calls[1]


def test_review_loop_revises_when_critic_refutes(repo, capsys):
    # The conventions gate is quiet but the critic refutes a finding -> the panel re-runs (round 2)
    # with the refutation, so each reviewer can drop the finding the code contradicts.
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(message)
        return _finding()

    async def fake_conv(findings, tool_ctx, model, base_name, base_ref, debug=False, **k):
        return ''  # conventions gate is quiet

    async def fake_critic(findings, tool_ctx, model, base_name, base_ref, debug=False, **k):
        return '[Sage-A1] - the claimed-missing call is present at a.txt:5'

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'conventions_gate', new=fake_conv), \
         patch.object(review, 'critic_gate', new=fake_critic), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args(review_rounds=1)))
    assert rc == 0
    assert len(panel_calls) == 2  # initial round + one revision round driven by the critic
    assert 'Previous review round' in panel_calls[1]
    assert 'critic' in panel_calls[1]
    assert '[Sage-A1]' in panel_calls[1]


def test_review_loop_stops_at_round_budget(repo):
    # Even if the gate keeps rebuting, the loop stops after the round budget.
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(message)
        return _finding()

    async def fake_gate(findings, tool_ctx, model, base_name, base_ref, debug=False, **k):
        return '[Sage-A1] - always rebutting'

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'conventions_gate', new=fake_gate), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        asyncio.run(review.run_review(_args(review_rounds=2)))
    assert len(panel_calls) == 3  # rounds 0,1,2 then the budget is exhausted


def test_per_persona_critique_revises_refuted_reviewer(repo):
    # A reviewer's finding is refuted by the critic; _per_persona_critique gives that reviewer one
    # revision pass and keeps the corrected findings, merging the code it read in round 0 so a
    # re-stated finding is still grounded by the evidence gate.
    findings = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR', 'location': 'a.txt:2',
                 'desc': 'phantomThing is undefined', 'support': '',
                 'evidence': [('$ git show HEAD:a.txt', 'one\nTWO\nthree')]}]

    async def fake_critic(findings, tool_ctx, model, base_name, base_ref, **k):
        return '[Sage-A1] - phantomThing is defined at a.txt:3'

    async def fake_panel(reviewers, message, model, stage, **k):
        assert 'the critic' in message and '[Sage-A1]' in message  # it saw the refutation
        return [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR', 'location': 'a.txt:2',
                 'desc': 'corrected finding', 'support': '',
                 'evidence': [('$ git show HEAD:a.txt', 'one\nTWO')]}]

    ctx = tools.ToolContext(phase='review', workdir='.', notes=[], require_evidence=True)
    with patch.object(review, 'critic_gate', new=fake_critic), \
         patch.object(review, 'run_personas', new=fake_panel):
        out = asyncio.run(review._per_persona_critique(
            [('Sage', 'body', 1)], findings, 'msg', 'm', 'main', 'main', '',
            ctx, None, None, None, False))
    assert [f['desc'] for f in out] == ['corrected finding']
    # The revision-round evidence is merged with what the reviewer read in round 0.
    assert ('$ git show HEAD:a.txt', 'one\nTWO') in out[0]['evidence']
    assert ('$ git show HEAD:a.txt', 'one\nTWO\nthree') in out[0]['evidence']


def test_per_persona_critique_keeps_unrefuted_reviewer(repo):
    # Nothing refuted -> the reviewer's findings pass through unchanged and no revision runs.
    findings = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR', 'location': 'a.txt:2',
                 'desc': 'a real finding', 'support': '', 'evidence': []}]

    async def fake_critic(findings, tool_ctx, model, base_name, base_ref, **k):
        return ''

    async def fake_panel(reviewers, message, model, stage, **k):
        raise AssertionError('no revision should run when nothing is refuted')

    ctx = tools.ToolContext(phase='review', workdir='.', notes=[], require_evidence=True)
    with patch.object(review, 'critic_gate', new=fake_critic), \
         patch.object(review, 'run_personas', new=fake_panel):
        out = asyncio.run(review._per_persona_critique(
            [('Sage', 'body', 1)], findings, 'msg', 'm', 'main', 'main', '',
            ctx, None, None, None, False))
    assert out == findings


def test_corroborated_keeps_only_recurring_concerns():
    # A concern restated across a majority of passes survives; a one-pass fluke is dropped.
    def f(label, loc, desc):
        return {'name': 'Sage', 'label': label, 'severity': 'MAJOR',
                'location': loc, 'desc': desc}

    p1 = [f('A1', 'a.txt:2', 'null pointer dereference'),
          f('B1', 'a.txt:9', 'unused variable foo')]
    p2 = [f('A2', 'a.txt:2', 'null pointer check missing')]
    p3 = [f('A3', 'a.txt:3', 'null pointer not checked')]
    passes = [p1, p2, p3]
    union = [f for p in passes for f in p]
    kept = review._corroborated(union, passes, 2)
    # The null-pointer concern (a.txt) recurs in all three passes; the other two are one-offs.
    assert len(kept) == 3
    assert all(k['location'].startswith('a.txt') for k in kept)
    # With no corroboration required (threshold 1) every candidate is kept.
    assert len(review._corroborated(union, passes, 1)) == len(union)


def test_run_review_consensus_runs_panel_n_times(repo):
    # --consensus N runs the full panel pass N times (one per independent sample).
    panel_calls = []

    async def fake_panel(reviewers, message, model, stage, **k):
        panel_calls.append(1)
        return _finding()

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args(consensus=3)))
    assert rc == 0
    assert len(panel_calls) == 3


def test_run_review_rejects_invalid_consensus(repo):
    # --consensus must be 0 (single pass) or >= 2; 1 and negatives are rejected rather than being
    # silently clamped to a single pass.
    for bad in (1, -2):
        with pytest.raises(Exception, match='--consensus must be 0'):
            asyncio.run(review.run_review(_args(consensus=bad)))


def test_run_review_reasoning_effort_flag(repo):
    # --reasoning-effort overrides the default; without it the default (medium) applies.
    seen = []

    async def fake_panel(reviewers, message, model, stage, **k):
        seen.append(k.get('reasoning_effort'))
        return _finding()

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        asyncio.run(review.run_review(_args(reasoning_effort='high')))
        asyncio.run(review.run_review(_args()))
    assert seen == ['high', review.REVIEW_REASONING_EFFORT]


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


def test_run_review_evidence_gate_drops_ungrounded(repo, capsys):
    # End to end: a finding whose named symbol never appears in the reviewer's git evidence is
    # dropped by the deterministic gate before consolidation, so it never reaches the output.
    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'phantomHandler leaks memory',
                'support': '',
                'evidence': [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]}]

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas',
                      new=AsyncMock(return_value=finding)), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args()))
    assert rc == 0
    assert 'phantomHandler' not in capsys.readouterr().out


def test_run_review_evidence_gate_keeps_grounded(repo, capsys):
    # The mirror case: a finding whose symbol IS in the reviewer's evidence survives the gate into
    # the final output.
    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'bad thing in compute_total',
                'support': '',
                'evidence': [('$ git show HEAD:calc.py',
                              'def compute_total():\n    return x + y')]}]

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    with patch.object(review, 'run_personas',
                      new=AsyncMock(return_value=finding)), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args()))
    assert rc == 0
    assert 'compute_total' in capsys.readouterr().out


def test_run_review_drops_symbol_invented_by_consolidator(repo, capsys):
    # The panel's finding is grounded and passes the pre-consolidation gate; consolidation then
    # rewrites its description to name a function the reviewers never read and that is not in the
    # tree. The post-consolidation evidence gate drops it, so the invented name is never posted.
    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'the value here is wrong',
                'support': '',
                'evidence': [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]}]

    async def fake_consolidate(context_block, findings, model, **k):
        return [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                 'location': 'a.txt:2', 'desc': 'ensure_tool_probing can spin forever',
                 'support': ''}]

    with patch.object(review, 'run_personas',
                      new=AsyncMock(return_value=finding)), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args()))
    assert rc == 0
    assert 'ensure_tool_probing' not in capsys.readouterr().out


def test_run_review_preserves_support_through_consolidation(repo, capsys):
    # The panel finding carries supporting paragraphs; the real consolidation re-emits a
    # one-line finding (support dropped), but the reviewer's evidence is re-attached by
    # (name, label) so it survives into the final output.
    finding = [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                'location': 'a.txt:2', 'desc': 'bad thing',
                'support': 'verified with git show; the oracle asserts X',
                'evidence': [('$ git show HEAD:a.txt', 'one\nTWO\nthree\nfour')]}]

    async def fake_consolidate(context_block, findings, model, **k):
        return [{'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
                 'location': 'a.txt:2', 'desc': 'bad thing', 'support': ''}]

    with patch.object(review, 'run_personas',
                      new=AsyncMock(return_value=finding)), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate):
        rc = asyncio.run(review.run_review(_args()))
    out = capsys.readouterr().out
    assert rc == 0
    assert 'verified with git show; the oracle asserts X' in out


def test_run_review_end_to_end_with_real_personas(repo, capsys):
    # Runs the REAL run_personas (real persona files + FINDINGS_CONTRACT + parse_findings)
    # against the real git/notes tool wiring, mocking only the LLM mapper, so the whole
    # panel path is exercised deterministically. The conventions gate is disabled
    # (review_rounds=0) and the findings consolidation is a no-op.
    class FakeMapper:
        system = ''
        model = 'm'

        def __init__(self, *a, **k):
            self.n = 0

        async def run(self, *a, **k):
            # Probe first (mandatory probing requires a git command before a finding is accepted),
            # then report.
            self.n += 1
            if self.n == 1:
                return '$ git show HEAD:a.txt'
            return 'A1 [MAJOR] a.txt:2 - bad thing\nI confirmed the defect with git show HEAD:a.txt.'

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
    # All 12 panel reviewers report the same a.txt:2 finding; the same-location dedup
    # collapses them into a single finding.
    assert out.count('a.txt:2') == 1


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

    async def fake_head(num, cwd=None):
        return ('feature', 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef')

    async def not_ahead(cwd=None, head_ref='', head_oid=''):
        return False

    monkeypatch.setattr(review, 'gh_pr_head', fake_head)
    monkeypatch.setattr(review, 'local_branch_ahead_of', not_ahead)
    with open(os.path.join(repo, 'dirty.txt'), 'w') as f:
        f.write('x')
    with pytest.raises(Exception, match='uncommitted changes'):
        asyncio.run(review.run_review(_args(pr=123)))


def test_gh_pr_head_parses():
    payload = json.dumps({'headRefName': 'feature/x', 'headRefOid': 'abc123'})

    async def fake_gh(*a, **k):
        return (0, payload, '')

    with patch.object(review, '_gh', new=fake_gh):
        ref, oid = asyncio.run(review.gh_pr_head(7))
    assert ref == 'feature/x' and oid == 'abc123'


def test_local_branch_ahead_of(repo):
    # The checked-out feature branch is one commit ahead of main, so main's commit is an ancestor
    # of HEAD. There is no `origin` remote in the fixture, so the fetch is best-effort (skipped
    # when the object is already local).
    main_oid = _rev(repo, 'main')
    assert asyncio.run(review.local_branch_ahead_of(repo, 'main', main_oid)) is True
    assert asyncio.run(review.commits_ahead(repo, main_oid)) == 1
    # A commit that is not an ancestor of HEAD reports False.
    assert asyncio.run(
        review.local_branch_ahead_of(repo, 'main', 'f' * 40)) is False


def _fake_review_downstream():
    # Shared fakes so a --pr review runs end-to-end without a network or the LLM.
    async def fake_panel(reviewers, message, model, stage, **k):
        return _finding()

    async def fake_consolidate(context_block, findings, model, **k):
        return findings

    async def fake_gh(*a, **k):
        if a[0] == 'repo':
            return (0, json.dumps({'nameWithOwner': ''}), '')
        return (0, json.dumps({'title': 't', 'body': 'b', 'comments': []}), '')

    return fake_panel, fake_consolidate, fake_gh


def test_run_review_prefers_local_when_ahead(repo, capsys, monkeypatch):
    # The checked-out branch contains the PR head (main's commit is an ancestor of HEAD), so the
    # local commits are reviewed instead of checking the PR out (which would drop unpushed fixes).
    main_oid = _rev(repo, 'main')
    checkout_calls = []

    async def fake_head(num, cwd=None):
        return ('feature', main_oid)

    async def fake_checkout(num, cwd=None):
        checkout_calls.append(num)

    fake_panel, fake_consolidate, fake_gh = _fake_review_downstream()
    monkeypatch.setattr(review, 'gh_pr_head', fake_head)
    monkeypatch.setattr(review, 'gh_pr_checkout', fake_checkout)
    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate), \
         patch.object(review, '_gh', new=fake_gh):
        rc = asyncio.run(review.run_review(_args(pr=123)))
    assert rc == 0
    assert checkout_calls == []  # did not check the PR out
    assert 'local branch' in capsys.readouterr().out


def test_run_review_checks_out_when_not_ahead(repo, capsys, monkeypatch):
    # The local branch does not contain the PR head, so the PR is checked out.
    checkout_calls = []

    async def fake_head(num, cwd=None):
        return ('feature', 'f' * 40)

    async def not_ahead(cwd=None, head_ref='', head_oid=''):
        return False

    async def fake_checkout(num, cwd=None):
        checkout_calls.append(num)

    fake_panel, fake_consolidate, fake_gh = _fake_review_downstream()
    monkeypatch.setattr(review, 'gh_pr_head', fake_head)
    monkeypatch.setattr(review, 'local_branch_ahead_of', not_ahead)
    monkeypatch.setattr(review, 'gh_pr_checkout', fake_checkout)
    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate), \
         patch.object(review, '_gh', new=fake_gh):
        rc = asyncio.run(review.run_review(_args(pr=123)))
    assert rc == 0
    assert checkout_calls == [123]
    assert 'Checked out PR' in capsys.readouterr().out


def test_run_review_remote_forces_checkout(repo, capsys, monkeypatch):
    # Even when the local branch is ahead of the PR head, --remote checks the remote head out.
    main_oid = _rev(repo, 'main')
    checkout_calls = []

    async def fake_head(num, cwd=None):
        return ('feature', main_oid)

    async def ahead(cwd=None, head_ref='', head_oid=''):
        return True

    async def fake_checkout(num, cwd=None):
        checkout_calls.append(num)

    fake_panel, fake_consolidate, fake_gh = _fake_review_downstream()
    monkeypatch.setattr(review, 'gh_pr_head', fake_head)
    monkeypatch.setattr(review, 'local_branch_ahead_of', ahead)
    monkeypatch.setattr(review, 'gh_pr_checkout', fake_checkout)
    with patch.object(review, 'run_personas', new=fake_panel), \
         patch.object(review, 'consolidate_findings', new=fake_consolidate), \
         patch.object(review, '_gh', new=fake_gh):
        rc = asyncio.run(review.run_review(_args(pr=123, remote=True)))
    assert rc == 0
    assert checkout_calls == [123]
    assert 'Checked out PR' in capsys.readouterr().out


def test_run_review_linear_requires_linear(repo, monkeypatch):
    monkeypatch.setattr(review, 'linear_available', lambda: False)
    with pytest.raises(Exception, match='requires the `linear` CLI'):
        asyncio.run(review.run_review(_args(linear='ACME-1')))


def test_linear_context_uses_issue_view_json(repo):
    # The linear CLI (v2.x) fetches one issue with `linear issue view <id> --json`, not the
    # older `linear issue <id> --output json`; pin the exact command we shell out to.
    calls = {}

    async def fake_run(cmd, *args, **k):
        calls['cmd'] = cmd
        calls['args'] = args
        return (0, '{"identifier": "ACME-1"}', '')

    with patch.object(review, '_run', new=fake_run):
        out = asyncio.run(review.linear_context('ACME-1'))
    assert calls['cmd'] == 'linear'
    assert calls['args'] == ('issue', 'view', 'ACME-1', '--json', '--no-pager')
    assert out == '{"identifier": "ACME-1"}'


def test_gh_pr_context_flattens_reviews():
    pr_payload = json.dumps({
        'title': 'My PR',
        'body': 'the description',
        'comments': [{'author': {'login': 'a'}, 'body': 'a comment'}],
    })
    repo_payload = json.dumps({'nameWithOwner': 'octo/repo'})
    threads_payload = json.dumps({'data': {'repository': {'pullRequest': {
        'reviewThreads': {'nodes': [
            {'comments': {'nodes': [
                {'isMinimized': False, 'path': 'x.py', 'line': 3,
                 'body': 'inline finding', 'author': {'login': 'b'}},
                {'isMinimized': False, 'path': 'x.py', 'line': 3,
                 'body': 'reply says invalid', 'author': {'login': 'c'}},
            ]}},
        ]}}}}})

    async def fake_gh(*a, **k):
        if a[0] == 'pr':
            return (0, pr_payload, '')
        if a[0] == 'repo':
            return (0, repo_payload, '')
        if a[0] == 'api':
            assert 'graphql' in a and 'pullRequest(number: 7)' in ' '.join(a)
            return (0, threads_payload, '')
        raise AssertionError(f'unexpected gh call: {a}')

    with patch.object(review, '_gh', new=fake_gh):
        out = asyncio.run(review.gh_pr_context(7))
    assert 'My PR' in out and 'the description' in out
    assert 'a comment' in out
    assert 'x.py:3' in out and 'inline finding' in out
    assert 'reply says invalid' in out and '(reply)' in out


def test_gh_pr_context_skips_hidden_comments():
    # A minimized (hidden) comment is left out of the reviewer's context: issue comments via the
    # `isMinimized` flag, inline comments via GraphQL `isMinimized`.
    pr_payload = json.dumps({
        'title': 'T', 'body': 'B',
        'comments': [
            {'author': {'login': 'a'}, 'body': 'visible issue comment'},
            {'author': {'login': 'a'}, 'body': 'hidden issue comment',
             'isMinimized': True, 'minimizedReason': 'RESOLVED'},
        ],
    })
    repo_payload = json.dumps({'nameWithOwner': 'octo/repo'})
    threads_payload = json.dumps({'data': {'repository': {'pullRequest': {
        'reviewThreads': {'nodes': [
            {'comments': {'nodes': [
                {'isMinimized': False, 'path': 'x.py', 'line': 3,
                 'body': 'visible inline', 'author': {'login': 'b'}},
                {'isMinimized': True, 'path': 'x.py', 'line': 4,
                 'body': 'hidden inline', 'author': {'login': 'c'}},
            ]}},
        ]}}}}})

    async def fake_gh(*a, **k):
        if a[0] == 'pr':
            return (0, pr_payload, '')
        if a[0] == 'repo':
            return (0, repo_payload, '')
        if a[0] == 'api':
            return (0, threads_payload, '')
        raise AssertionError(f'unexpected gh call: {a}')

    with patch.object(review, '_gh', new=fake_gh):
        out = asyncio.run(review.gh_pr_context(7))
    assert 'visible issue comment' in out
    assert 'hidden issue comment' not in out
    assert 'visible inline' in out
    assert 'hidden inline' not in out


def test_repo_name_is_cached_per_cwd():
    # `_repo_name` resolves the repo once per cwd and caches it, so the several helpers that each
    # need the owner/name don't each shell out to `gh repo view` in the same run.
    calls = {'n': 0}

    async def fake_gh(*a, **k):
        calls['n'] += 1
        return (0, '{"nameWithOwner": "acme/widget"}', '')

    with patch.object(review, '_gh', new=fake_gh):
        assert asyncio.run(review._repo_name(None)) == 'acme/widget'
        assert asyncio.run(review._repo_name(None)) == 'acme/widget'
    assert calls['n'] == 1  # the second call is served from the cache


def test_repo_name_does_not_cache_failure():
    # A failed resolution is not cached, so a later call retries instead of reusing an empty name.
    results = iter([(1, '', 'boom'), (0, '{"nameWithOwner": "acme/widget"}', '')])

    async def fake_gh(*a, **k):
        return next(results)

    with patch.object(review, '_gh', new=fake_gh):
        assert asyncio.run(review._repo_name(None)) == ''
        assert asyncio.run(review._repo_name(None)) == 'acme/widget'


def test_same_concern_matches_by_file_and_identifier():
    # Same file + a shared identifier (or enough word overlap) is the same concern, even on a
    # different line; a different file, or an unrelated description, is not.
    a = {'location': 'marsha/review.py:1',
         'desc': 'build_review_message embeds example-driven test instructions'}
    prior = {'path': 'marsha/review.py',
             'desc': 'build_review_message embeds detailed example-driven instructions'}
    assert review._same_concern(a, prior)
    assert not review._same_concern(
        a, {'path': 'other/file.py', 'desc': prior['desc']})
    assert not review._same_concern(
        {'location': 'marsha/review.py:5', 'desc': 'the retry backoff is too aggressive'},
        prior)


def test_filter_duplicate_findings_drops_reraises():
    threads = {'C9': {'path': 'marsha/review.py', 'line': 1,
                      'desc': 'build_review_message embeds example-driven instructions'}}
    body = [{'path': 'pyproject.toml',
             'desc': 'quickjs-ng is listed but the module is quickjs'}]
    findings = [
        {'label': 'C9', 'location': 'marsha/review.py:1',
         'desc': 'build_review_message embeds example-driven instructions'},
        {'label': 'D9', 'location': 'marsha/review.py:277',
         'desc': 'build_review_message embeds detailed example-driven instructions in tests'},
        {'label': 'A1', 'location': 'pyproject.toml:11',
         'desc': 'quickjs-ng is listed as a dependency but code imports quickjs'},
        {'label': 'B2', 'location': 'marsha/tools.py:50',
         'desc': 'the socket timeout default is too long'},
    ]
    kept, dropped = review._filter_duplicate_findings(findings, threads, body)
    assert dropped == 2
    assert [f['label'] for f in kept] == ['C9', 'B2']


def test_fetch_prior_body_findings_parses_marsha_bodies():
    reviews = [
        {'body': 'Marsha review \u2014 findings:\n\n'
                 '- **[A1] MAJOR** `pyproject.toml:9`: dev deps in runtime\n'
                 '- **[B2] MINOR** `marsha/tools.py:517`: quickjs-ng vs quickjs\n'},
        {'body': 'A human review with no findings.'},
    ]
    payload = json.dumps(reviews)

    async def fake_gh(*a, **k):
        return (0, payload, '')

    with patch.object(review, '_gh', new=fake_gh):
        out = asyncio.run(review._fetch_prior_body_findings('acme/widget', 123))
    assert [f['label'] for f in out] == ['A1', 'B2']
    assert out[0]['path'] == 'pyproject.toml'
    assert out[1]['path'] == 'marsha/tools.py'


def test_thread_settled_detects_user_reply_or_resolution():
    # A thread is settled if resolved, or if it has any human reply (a Marsha re-raise leads with
    # **[label] and does not count).
    assert review._thread_settled({'is_resolved': True, 'replies': []})
    assert review._thread_settled(
        {'is_resolved': False, 'replies': ['Rejected. already handled']})
    assert not review._thread_settled(
        {'is_resolved': False, 'replies': ['**[B9] MINOR**: still a problem']})
    assert not review._thread_settled({'is_resolved': False, 'replies': []})


def test_filter_duplicate_drops_reraise_of_settled_thread():
    # A re-raise that reuses the exact label of a thread the user already settled is dropped; an
    # unrelated new finding is kept.
    threads = {'B9': {'path': 'marsha/review.py', 'line': 169, 'desc': 'renumbering'}}
    findings = [
        {'label': 'B9', 'location': 'marsha/review.py:169',
         'desc': 'renumbering a custom panel'},
        {'label': 'A1', 'location': 'marsha/tools.py:5', 'desc': 'something new'},
    ]
    kept, dropped = review._filter_duplicate_findings(
        findings, threads, [], settled_labels={'B9'})
    assert dropped == 1
    assert [f['label'] for f in kept] == ['A1']


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
        {'name': 'Eli', 'label': 'B1', 'severity': 'MINOR',
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
    # The fresh in-diff finding is inlined and leads with its [label]; the out-of-diff one is
    # folded into the review body.
    assert [c['path'] for c in payload['comments']] == ['foo.py']
    assert payload['comments'][0]['body'] == '**[A1] MAJOR**: inline one'
    assert 'bar.py' in payload['body']
    assert '**[B1] MINOR**' in payload['body']


def test_post_review_all_clear_posts_note():
    # A no-finding --post-review still posts a short all-clear note to the PR.
    calls = {}

    async def fake_gh(*a, **k):
        if a and a[0] == 'repo':
            return (0, '{"nameWithOwner": "acme/widget"}', '')
        calls['args'] = a
        calls['input'] = k.get('input')
        return (0, '{}', '')

    with patch.object(review, '_gh', new=fake_gh):
        asyncio.run(review.post_review(123, [], ''))

    assert 'repos/acme/widget/pulls/123/reviews' in calls['args']
    payload = json.loads(calls['input'].decode('utf-8'))
    assert payload['event'] == 'COMMENT'
    assert 'no issues found' in payload['body']


def test_post_review_replies_on_existing_thread():
    # A finding re-raised under a label that already has a thread is posted as a reply on that
    # thread (not a new top-level comment); a fresh label at an in-diff line is a new inline.
    diff = ('diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n'
            '@@ -1,2 +1,3 @@\n line1\n+new\n line2\n'
            'diff --git a/baz.py b/baz.py\n--- a/baz.py\n+++ b/baz.py\n'
            '@@ -1,2 +1,4 @@\n line1\n+x\n+y\n line2\n')
    findings = [
        {'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
         'location': 'foo.py:2', 'desc': 're-raised point'},
        {'name': 'Eli', 'label': 'B1', 'severity': 'MINOR',
         'location': 'baz.py:3', 'desc': 'fresh point'},
    ]
    threads = json.dumps({'data': {'repository': {'pullRequest': {'reviewThreads': {
        'nodes': [
            {'id': 'PRRT_1', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 999, 'path': 'foo.py', 'line': 2,
                 'body': '**[A1] MAJOR**: earlier finding'}]}},
        ]}}}}})
    review_payload = None
    reply_payload = None

    async def fake_gh(*a, **k):
        if a and a[0] == 'repo':
            return (0, '{"nameWithOwner": "acme/widget"}', '')
        joined = ' '.join(a)
        if 'pulls/123/comments' in joined:           # reply: POST .../comments (in_reply_to)
            nonlocal reply_payload
            reply_payload = k.get('input')
            return (0, '{}', '')
        if 'reviewThreads' in joined:                # fetch prior threads (GraphQL)
            return (0, threads, '')
        if '/reviews' in joined:                     # new review: .../pulls/123/reviews
            nonlocal review_payload
            review_payload = k.get('input')
            return (0, '{}', '')
        return (0, '{}', '')

    with patch.object(review, '_gh', new=fake_gh):
        asyncio.run(review.post_review(123, findings, diff))

    posted = json.loads(review_payload.decode('utf-8'))
    # Only the fresh label (B1) opens a new inline comment; A1 replies on its existing thread.
    assert [c['path'] for c in posted['comments']] == ['baz.py']
    assert posted['comments'][0]['body'] == '**[B1] MINOR**: fresh point'
    reply = json.loads(reply_payload.decode('utf-8'))
    assert reply['in_reply_to'] == 999
    assert reply['body'] == '**[A1] MAJOR**: re-raised point'


def test_post_review_resolves_conceded_thread():
    # A prior thread whose label the reviewer no longer raises (and whose reviewer ran this pass)
    # is closed by resolving its thread; threads for reviewers not re-run are left alone.
    findings = [
        {'name': 'Sage', 'label': 'A1', 'severity': 'MAJOR',
         'location': 'foo.py:2', 'desc': 'still stands'},
    ]
    threads = json.dumps({'data': {'repository': {'pullRequest': {'reviewThreads': {
        'nodes': [
            {'id': 'PRRT_A1', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 11, 'path': 'foo.py', 'line': 2,
                 'body': '**[A1] MAJOR**: still stands'}]}},
            {'id': 'PRRT_C2', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 22, 'path': 'bar.py', 'line': 5,
                 'body': '**[C2] MINOR**: conceded point'}]}},
            {'id': 'PRRT_D3', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 33, 'path': 'baz.py', 'line': 9,
                 'body': '**[D3] MINOR**: reviewer not re-run'}]}},
            {'id': 'PRRT_HUMAN', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 44, 'path': 'foo.py', 'line': 1,
                 'body': 'a human comment with no label'}]}},
        ]}}}}})
    resolved = []

    async def fake_gh(*a, **k):
        if a and a[0] == 'repo':
            return (0, '{"nameWithOwner": "acme/widget"}', '')
        joined = ' '.join(a)
        if 'resolveReviewThread' in joined:
            m = re.search(r'threadId: "([^"]+)"', joined)
            resolved.append(m.group(1) if m else None)
            return (0, '{"data": {"resolveReviewThread": {"thread": {"isResolved": true}}}}', '')
        if 'reviewThreads' in joined:
            return (0, threads, '')
        if '/reviews' in joined:
            return (0, '{}', '')
        return (0, '{}', '')

    with patch.object(review, '_gh', new=fake_gh):
        # Reviewer #1 (Sage) and #2 ran this pass; #3 did not. A1 was re-raised (kept), so only
        # C2 (reviewer #2, conceded) is resolved — D3 (reviewer #3) and the human thread stay.
        asyncio.run(
            review.post_review(123, findings, '', active_numbers=[1, 2]))

    assert resolved == ['PRRT_C2']


def test_fetch_review_threads_parses_labels():
    # Only threads whose root comment leads with a [label] are matched; the label is the key and
    # the thread id / root databaseId / resolved flag are carried through for reply + resolve.
    payload = json.dumps({'data': {'repository': {'pullRequest': {'reviewThreads': {
        'nodes': [
            {'id': 'PRRT_1', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 999, 'path': 'foo.py', 'line': 2,
                 'body': '**[A1] MAJOR**: one'}]}},
            {'id': 'PRRT_2', 'isResolved': True, 'comments': {'nodes': [
                {'databaseId': 1000, 'path': 'foo.py', 'line': 3,
                 'body': '**[B2] MINOR**: two'}]}},
            {'id': 'PRRT_3', 'isResolved': False, 'comments': {'nodes': [
                {'databaseId': 1001, 'path': 'bar.py', 'line': 4,
                 'body': 'no label here'}]}},
        ]}}}}})

    async def fake_gh(*a, **k):
        assert a[:2] == ('api', 'graphql')
        return (0, payload, '')

    with patch.object(review, '_gh', new=fake_gh):
        threads = asyncio.run(review._fetch_review_threads('acme/widget', 123))

    assert set(threads) == {'A1', 'B2'}
    assert threads['A1'] == {'thread_id': 'PRRT_1', 'root_id': 999,
                             'is_resolved': False, 'path': 'foo.py', 'line': 2,
                             'desc': 'one'}
    assert threads['B2']['is_resolved'] is True


def test_resolve_thread_posts_mutation():
    sent = {}

    async def fake_gh(*a, **k):
        sent['args'] = a
        return (0, '{"data": {"resolveReviewThread": {"thread": {"isResolved": true}}}}', '')

    with patch.object(review, '_gh', new=fake_gh):
        ok = asyncio.run(review._resolve_thread('PRRT_1'))

    assert ok is True
    assert 'api' in sent['args'] and 'graphql' in sent['args']
    assert any('resolveReviewThread' in str(part) for part in sent['args'])
    assert any('PRRT_1' in str(part) for part in sent['args'])


def test_finding_matches_thread_same_line():
    # Same file + same line is a conclusive match; a different file is not, regardless of text.
    assert review._finding_matches_thread(
        {'location': 'foo.py:10', 'desc': 'x'},
        {'path': 'foo.py', 'line': 10, 'desc': 'y'}) is True
    assert review._finding_matches_thread(
        {'location': 'bar.py:10', 'desc': 'x'},
        {'path': 'foo.py', 'line': 10, 'desc': 'y'}) is False


def test_desc_similar_threshold():
    # A re-raised finding restates a prior one (high overlap); two near-siblings on the same file
    # share a common phrase but are distinct concerns and must NOT match (avoids the collision).
    assert review._desc_similar(
        'the run helper does not catch oserror',
        'the run helper does not catch oserror') is True
    assert review._desc_similar(
        'missing type hint on run_review function',
        'missing type hint on post_review function') is False


def test_verify_finding_labels_reassigns_collision():
    # A finding that wears a prior thread's label but is a different concern (different file) is
    # reassigned a fresh label that skips all the reviewer's prior labels, freeing the old label.
    threads = {
        'A1': {'path': 'foo.py', 'line': 10, 'desc': 'missing type hint'},
        'B1': {'path': 'bar.py', 'line': 5, 'desc': 'off-by-one'},
    }
    findings = [{'label': 'A1', 'location': 'baz.py:9', 'desc': 'wrong variable name'}]
    n = review._verify_finding_labels(findings, threads)
    assert n == 1
    assert findings[0]['label'] == 'C1'


def test_verify_finding_labels_keeps_genuine_reraise():
    # A finding that reuses a prior label and matches it (same file + line) is a genuine re-raise;
    # the label is kept so it replies on that thread.
    threads = {'A1': {'path': 'foo.py', 'line': 10, 'desc': 'missing type hint'}}
    findings = [{'label': 'A1', 'location': 'foo.py:10', 'desc': 'missing type hint again'}]
    n = review._verify_finding_labels(findings, threads)
    assert n == 0
    assert findings[0]['label'] == 'A1'


def test_verify_finding_labels_reraise_line_shifted():
    # Same file, shifted line, near-identical description: still the same concern (a line moved
    # between runs), so the label is kept rather than reassigned.
    threads = {'A1': {'path': 'foo.py', 'line': 10,
                      'desc': 'the run helper does not catch oserror for a missing binary'}}
    findings = [{'label': 'A1', 'location': 'foo.py:12',
                 'desc': 'the run helper does not catch oserror when a binary is missing'}]
    n = review._verify_finding_labels(findings, threads)
    assert n == 0
    assert findings[0]['label'] == 'A1'


def test_verify_finding_labels_dedupes_inrun():
    # Two findings sharing one label in a single run (a parser slip) get distinct labels.
    findings = [
        {'label': 'A1', 'location': 'a.py:1', 'desc': 'one'},
        {'label': 'A1', 'location': 'a.py:2', 'desc': 'two'},
    ]
    n = review._verify_finding_labels(findings, {})
    assert n == 1
    assert [f['label'] for f in findings] == ['A1', 'B1']


def test_prior_findings_by_reviewer_groups_by_number():
    # Threads are grouped by the reviewer number in the label; resolved threads and unlabeled
    # (human) comments are skipped, and replies are attached to their finding.
    payload = json.dumps({'data': {'repository': {'pullRequest': {'reviewThreads': {
        'nodes': [
            {'id': 'T1', 'isResolved': False, 'comments': {'nodes': [
                {'isMinimized': False, 'path': 'a.py', 'line': 3,
                 'body': '**[A1] MAJOR**: keep this'},
                {'isMinimized': False, 'path': 'a.py', 'line': 3,
                 'body': 'user: actually fixed it'},
            ]}},
            {'id': 'T2', 'isResolved': False, 'comments': {'nodes': [
                {'isMinimized': False, 'path': 'b.py', 'line': 5,
                 'body': '**[B2] MINOR**: yours too'},
            ]}},
            {'id': 'T3', 'isResolved': True, 'comments': {'nodes': [
                {'isMinimized': False, 'path': 'c.py', 'line': 1,
                 'body': '**[C2] MINOR**: already resolved'},
            ]}},
            {'id': 'T4', 'isResolved': False, 'comments': {'nodes': [
                {'isMinimized': False, 'path': 'd.py', 'line': 1,
                 'body': 'a human comment, no label'},
            ]}},
        ]}}}}})

    async def fake_gh(*a, **k):
        if a and a[0] == 'repo':
            return (0, '{"nameWithOwner": "acme/widget"}', '')
        return (0, payload, '')

    with patch.object(review, '_gh', new=fake_gh):
        out = asyncio.run(review._prior_findings_by_reviewer(123))

    assert set(out) == {1, 2}
    assert [f['label'] for f in out[1]] == ['A1']
    assert out[1][0]['replies'] == ['user: actually fixed it']
    assert out[1][0]['location'] == 'a.py:3'
    assert [f['label'] for f in out[2]] == ['B2']


def test_reviewer_prior_block_format():
    prior = [
        {'label': 'A1', 'severity': 'MAJOR', 'location': 'a.py:3', 'desc': 'keep this',
         'replies': ['user: actually fixed it']},
        {'label': 'B1', 'severity': 'MINOR', 'location': '', 'desc': 'maybe not', 'replies': []},
    ]
    block = review._reviewer_prior_block(1, prior)
    assert '# Your prior review findings' in block
    assert '[A1] MAJOR a.py:3 - keep this' in block
    assert 'user: actually fixed it' in block
    assert '[B1] MINOR - maybe not' in block
    assert 'EXACT label' in block


def test_prior_conversations_block_lists_settled():
    # Every prior conversation not re-raised this pass is listed (with its replies + status) for
    # the consolidator to drop; re-raised ones are not, and an empty input yields no block.
    convs = [
        {'label': 'A1', 'location': 'a.py:3', 'desc': 'missing type hint',
         'replies': ['Rejected. out of scope.'], 'is_resolved': True},
        {'label': 'B1', 'location': 'b.py:9', 'desc': 'off-by-one',
         'replies': [], 'is_resolved': False},
    ]
    block = review._prior_conversations_block(convs, {'A1'})
    assert 'do NOT re-open' in block
    assert '[B1] b.py:9 (open) - off-by-one' in block
    assert '[A1]' not in block  # re-raised, so not in the list
    assert review._prior_conversations_block(convs, {'A1', 'B1'}) == ''
    assert review._prior_conversations_block([], {'A1'}) == ''


def test_run_personas_appends_prior_block():
    # The per-reviewer prior block is appended only to the reviewer its number is keyed under.
    captured = []

    async def fake_run_with_tools(mapper, request, ctx=None, debug=False,
                                  max_rounds=tools.MAX_TOOL_ROUNDS):
        m = re.search(r'You are review #(\d+)', mapper.system)
        captured.append((int(m.group(1)) if m else None, request))
        return 'NO FINDINGS'

    prior = {1: '\n# Your prior review findings on this PR\n- [A1] MAJOR a.py:3 - keep this'}
    with patch.object(tools, 'run_with_tools', new=fake_run_with_tools), \
         patch.object(personas, 'get_mapper',
                      new=lambda *a, **k: types.SimpleNamespace(n_results=1, system=a[0])):
        asyncio.run(personas.run_personas(
            [('Sage', 'body', 1), ('Eli', 'body', 2)], 'base msg', 'm', 'review',
            tool_ctx=tools.ToolContext(phase='review', workdir='.', notes=[]),
            prior_block_by_number=prior))
    by_num = dict(captured)
    assert prior[1] in by_num[1]
    assert 'Your prior review findings' not in by_num[2]
