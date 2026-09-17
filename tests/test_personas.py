"""Deterministic unit tests for the persona loader, the finding model, and the
preamble-splitting parser. No LLM or network is involved.
"""

import asyncio
import os
from unittest.mock import patch

from marsha import personas as p
from marsha.parse import split_preamble


# --- Registry & resolution --------------------------------------------------

def test_registry_has_expected_personas():
    reg = p.build_registry()
    assert len(reg) == 24
    for name in ['ada', 'vera', 'sage', 'sasha', 'kit', 'dot', 'otto', 'sol', 'regan', 'fay', 'hollis']:
        assert name in reg
    # Editors (and the review conventions gate) are excluded from the reviewer registry.
    for name in ['wren', 'cody', 'rex', 'norman']:
        assert name not in reg


def test_resolve_persona_by_name_is_case_insensitive():
    reg = p.build_registry()
    name, body, path = p.resolve_persona('ada', reg)
    assert name == 'Ada'
    assert body.startswith('You are Ada')
    assert os.path.basename(path) == 'oracle-completeness.md'
    assert p.resolve_persona('ELI', reg)[0] == 'Eli'


def test_resolve_persona_by_path(tmp_path):
    f = tmp_path / 'sharona.md'
    f.write_text('name: Sharona\nYou are Sharona.\nGoal: x\n')
    reg = p.build_registry()
    name, _body, path = p.resolve_persona(str(f), reg)
    assert name == 'Sharona'
    assert path == str(f)


def test_resolve_persona_unknown_raises():
    reg = p.build_registry()
    try:
        p.resolve_persona('nobody', reg)
        assert False, 'expected an exception'
    except Exception:
        pass


def test_resolve_loop_reviewers_default():
    reg = p.build_registry()
    specs = p.resolve_loop_reviewers('oracle', None, reg)
    assert len(specs) == 7
    assert [N for _, _, N in specs] == [1, 2, 3, 4, 5, 6, 7]
    assert [n for n, _, _ in specs][0] == 'Ada'


def test_resolve_loop_reviewers_flag_order_and_n():
    reg = p.build_registry()
    specs = p.resolve_loop_reviewers('impl', 'sage,sasha', reg)
    assert [(n, N) for n, _, N in specs] == [('Sage', 1), ('Sasha', 2)]


def test_resolve_loop_reviewers_duplicate_raises():
    reg = p.build_registry()
    try:
        p.resolve_loop_reviewers('oracle', 'ada,ada', reg)
        assert False, 'expected an exception'
    except Exception:
        pass


# --- Finding parsing --------------------------------------------------------

def test_parse_findings_labels_and_fields():
    text = ('A1 [MAJOR] foo.py:10 - missing case\n'
            'B1 [MINOR] foo.py:20 - naming\n'
            'C1 [NIT] foo.py:30 - style')
    fs = p.parse_findings(text, 'Ada', 1)
    assert [f['label'] for f in fs] == ['A1', 'B1', 'C1']
    assert [f['severity'] for f in fs] == ['MAJOR', 'MINOR', 'NIT']
    assert fs[0]['location'] == 'foo.py:10'
    assert fs[0]['desc'] == 'missing case'
    assert fs[0]['name'] == 'Ada'


def test_parse_findings_label_is_position_based():
    # A reviewer's own label token is not trusted; the harness labels by position.
    fs = p.parse_findings('Z9 [MAJOR] a - one\nB7 [MINOR] b - two', 'Ada', 1)
    assert [f['label'] for f in fs] == ['A1', 'B1']


def test_parse_findings_reuses_wellformed_label():
    # On a re-review a reviewer reuses the exact label of a finding it still stands by; a
    # well-formed label (one letter + this reviewer's number) is honored, leaving a gap where a
    # conceded finding (B2) was dropped. A foreign label (number mismatch) is not honored.
    fs = p.parse_findings(
        'A2 [MAJOR] a - kept\nC2 [MINOR] b - kept', 'Ada', 2)
    assert [f['label'] for f in fs] == ['A2', 'C2']
    fs = p.parse_findings('A9 [MAJOR] a - foreign', 'Ada', 2)
    assert [f['label'] for f in fs] == ['A2']


def test_parse_findings_case_and_nitpick():
    fs = p.parse_findings('a1 [major] x - one\nb1 [nitpick] y - two', 'Ada', 2)
    assert [f['severity'] for f in fs] == ['MAJOR', 'NIT']
    assert [f['label'] for f in fs] == ['A2', 'B2']


def test_parse_findings_skips_prose():
    fs = p.parse_findings(
        'some prose\n[MAJOR] x - one\nA1 [MINOR] y - two\n', 'Ada', 1)
    assert len(fs) == 2
    assert [f['label'] for f in fs] == ['A1', 'B1']


def test_parse_findings_no_findings():
    assert p.parse_findings('NO FINDINGS', 'Ada', 1) == []
    assert p.parse_findings('all good here', 'Ada', 1) == []


# --- Finding model ----------------------------------------------------------

def _f(name, label, severity, desc):
    return {'name': name, 'label': label, 'severity': severity, 'location': '', 'desc': desc}


def test_dedup_findings():
    a = _f('Ada', 'A1', 'MAJOR', 'same')
    b = _f('Vera', 'A1', 'MAJOR', 'same')
    c = _f('Ada', 'A1', 'MAJOR', 'different')
    out = p.dedup_findings([a, b, c, a])
    # Key is (name, label, severity, desc); a and b differ by name so both survive.
    assert len(out) == 3


def test_dedup_by_location_merges_same_line():
    # Several reviewers flag the same file:line with different wording -> one finding,
    # keeping the highest severity (then the most detailed description).
    f1 = {'name': 'Sage', 'label': 'A1', 'severity': 'MINOR',
          'location': 'src/foo.py:10', 'desc': 'short'}
    f2 = {'name': 'Eli', 'label': 'A1', 'severity': 'MAJOR',
          'location': 'src/foo.py:10', 'desc': 'a much more detailed explanation'}
    f3 = {'name': 'Dot', 'label': 'A1', 'severity': 'NIT',
          'location': 'src/foo.py:10', 'desc': 'tiny'}
    f4 = {'name': 'Kit', 'label': 'A1', 'severity': 'MINOR',
          'location': 'src/bar.py:5', 'desc': 'different file'}
    out = p.dedup_by_location([f1, f2, f3, f4])
    assert len(out) == 2  # one for foo.py:10, one for bar.py:5
    kept = {(f['location'], f['desc']) for f in out}
    assert ('src/foo.py:10', 'a much more detailed explanation') in kept
    assert ('src/bar.py:5', 'different file') in kept


def test_dedup_by_location_keeps_unlocated_separate():
    # Findings with no location have nothing to merge on, so each is kept rather than collapsed
    # into a single unlocated finding.
    a = {'name': 'Sage', 'label': 'A1', 'severity': 'MINOR', 'location': '', 'desc': 'one'}
    b = {'name': 'Eli', 'label': 'B1', 'severity': 'MAJOR', 'location': '', 'desc': 'two'}
    c = {'name': 'Dot', 'label': 'A1', 'severity': 'NIT', 'location': 'x.py:1', 'desc': 'loc'}
    out = p.dedup_by_location([a, b, c])
    assert len(out) == 3
    assert {f['desc'] for f in out} == {'one', 'two', 'loc'}


def test_position_label_extends_past_z():
    # After the 26 single letters are used, the fallback extends to two letters (AA<n>...) rather
    # than spilling onto non-alphabetic characters.
    used = {f'{chr(ord("A") + n)}5' for n in range(26)}
    assert p._position_label(5, used) == 'AA5'
    used.add('AA5')
    assert p._position_label(5, used) == 'AB5'


def test_actionable_findings_filters_severity():
    fs = [_f('A', 'A1', 'MAJOR', 'm'), _f(
        'B', 'B1', 'MINOR', 'n'), _f('C', 'C1', 'NIT', 'x')]
    assert [f['severity']
            for f in p.actionable_findings(fs, {'MAJOR'})] == ['MAJOR']
    assert len(p.actionable_findings(fs, {'MAJOR', 'MINOR'})) == 2


def test_parse_severities():
    assert p.parse_severities('major,minor,nit') == {'MAJOR', 'MINOR', 'NIT'}
    assert p.parse_severities('nitpick') == {'NIT'}
    try:
        p.parse_severities('bogus')
        assert False, 'expected an exception'
    except Exception:
        pass


def test_format_findings():
    fs = [{'name': 'Ada', 'label': 'A1', 'severity': 'MAJOR',
           'location': 'x.py:3', 'desc': 'd'}]
    assert p.format_findings(fs) == '- [Ada-A1] MAJOR x.py:3 - d'


def test_prior_round_block_contains_label_and_preamble():
    fs = [_f('Ada', 'A1', 'MAJOR', 'd')]
    block = p.prior_round_block(fs, 'I reject [Ada-A1] because ...')
    assert '[Ada-A1]' in block
    assert 'I reject [Ada-A1]' in block
    assert 'Previous review round' in block


# --- run_personas resilience ------------------------------------------------

def test_run_personas_ignores_failing_persona():
    class FakeMapper:
        def __init__(self, system):
            self.system = system

        async def run(self, user_message):
            if 'review #1' in self.system:
                raise Exception('boom')
            return 'A2 [MAJOR] x - ok\n'

    async def scenario():
        reviewers = [('Ada', 'body', 1), ('Vera', 'body', 2)]
        with patch.object(p, 'get_mapper', new=lambda *a, **k: FakeMapper(a[0])):
            return await p.run_personas(reviewers, 'ctx', model='m', stats_stage='first_stage')
    fs = asyncio.run(scenario())
    assert len(fs) == 1
    assert fs[0]['name'] == 'Vera'


# --- Preamble splitting -----------------------------------------------------

def test_split_preamble():
    doc = 'Preamble here.\n\n# foo.py\n\n```py\nprint(1)\n```\n'
    preamble, artifact = split_preamble(doc, 'foo.py')
    assert preamble == 'Preamble here.'
    assert artifact.startswith('# foo.py\n')
    assert 'print(1)' in artifact


def test_split_preamble_missing_header_raises():
    try:
        split_preamble('no header here', 'foo.py')
        assert False, 'expected an exception'
    except Exception:
        pass
