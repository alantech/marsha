import importlib.util
import json
import os
import sys
from typing import Any

from unittest.mock import patch

import pytest

import marsha.review as review

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO_ROOT, '.watchman_eval.py')
FIXTURES_PATH = os.path.join(REPO_ROOT, 'examples', 'watchman', 'fixtures.json')


def _load_eval() -> Any:
    # The harness is a top-level tool (like .time.py), so load it by file path.
    spec = importlib.util.spec_from_file_location('watchman_eval', EVAL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixtures_file_invariants() -> None:
    # The shipped examples are 30 honest (the record supports the clearance) and 30 cheating
    # (it does not), each a well-formed single CLEARED thread with a consistent label.
    fixtures = json.load(open(FIXTURES_PATH))
    assert len(fixtures) == 60
    honest = [f for f in fixtures if f['kind'] == 'honest']
    cheat = [f for f in fixtures if f['kind'] == 'cheat']
    assert len(honest) == 30 and len(cheat) == 30
    for f in fixtures:
        assert f['expected'] == ('validated' if f['kind'] == 'honest'
                                 else 'not-validated')
        t = f['thread']
        assert t['label'] and t['thread_id'] and t['location'] and t['desc']
        assert isinstance(t['replies'], list)
        assert all(isinstance(e, list) and len(e) == 2 for e in f['evidence'])


def _fixtures() -> list[dict[str, Any]]:
    # A tiny controlled set (unique thread ids) so the counting logic can be checked in isolation.
    return [
        {'kind': 'honest', 'expected': 'validated',
         'thread': {'label': 'A1', 'thread_id': 'PRRT_A1', 'location': 'a.py:1',
                    'desc': 'x', 'replies': []}, 'evidence': []},
        {'kind': 'cheat', 'expected': 'not-validated',
         'thread': {'label': 'B2', 'thread_id': 'PRRT_B2', 'location': 'b.py:2',
                    'desc': 'y', 'replies': []}, 'evidence': []},
    ]


def _mock_watchman(reply: str) -> Any:
    # A fixed-reply stand-in for the real (LLM) watchman.
    async def fake(candidates: Any, evidence: Any, model: Any, base_name: Any,
                   base_ref: Any, cwd: Any, debug: bool = False, **k: Any) -> Any:
        return {candidates[0]['thread_id']: reply}
    return fake


def test_run_eval_counts_correct() -> None:
    # A watchman that returns each fixture's expected label scores every fixture.
    eval_mod = _load_eval()
    expected_by_id = {'PRRT_A1': 'validated', 'PRRT_B2': 'not-validated'}

    async def perfect(candidates: Any, evidence: Any, model: Any, base_name: Any,
                      base_ref: Any, cwd: Any, **k: Any) -> Any:
        return {candidates[0]['thread_id']: expected_by_id[candidates[0]['thread_id']]}

    with patch.object(review, '_watchman_validate', new=perfect):
        correct, total, wrong = eval_mod.run_eval(_fixtures())
    assert (correct, total, wrong) == (2, 2, [])


def test_run_eval_counts_partial() -> None:
    # A watchman that always validates gets only the honest fixture right.
    eval_mod = _load_eval()
    with patch.object(review, '_watchman_validate', new=_mock_watchman('validated')):
        correct, total, _wrong = eval_mod.run_eval(_fixtures())
    assert (correct, total) == (1, 2)
    # And one that never validates gets only the cheating fixture right.
    with patch.object(review, '_watchman_validate',
                      new=_mock_watchman('not-validated')):
        correct, total, _wrong = eval_mod.run_eval(_fixtures())
    assert (correct, total) == (1, 2)


def test_run_eval_fails_closed_on_no_validation() -> None:
    # A watchman that validates nothing (the fail-closed outcome of a failed call) scores no
    # fixture correctly: every decision is "missing" and thus wrong.
    eval_mod = _load_eval()

    async def empty(candidates: Any, evidence: Any, model: Any, base_name: Any,
                    base_ref: Any, cwd: Any, **k: Any) -> Any:
        return {}

    with patch.object(review, '_watchman_validate', new=empty):
        correct, total, wrong = eval_mod.run_eval(_fixtures())
    assert (correct, total) == (0, 2)
    assert len(wrong) == 2


def test_run_eval_no_verdict_is_not_a_correct_negative() -> None:
    # A watchman that produces no recognized verdict (malformed/empty output) yields 'no-verdict',
    # which is neither 'validated' nor a correct 'not-validated': a cheating fixture it mishandles
    # is counted wrong, not right, so a broken watchman cannot bank correct-negatives.
    eval_mod = _load_eval()
    with patch.object(review, '_watchman_validate', new=_mock_watchman('no-verdict')):
        correct, total, _wrong = eval_mod.run_eval(_fixtures())
    assert (correct, total) == (0, 2)


def test_eval_rejects_out_of_range_threshold() -> None:
    # A negative, zero, or >100 threshold would let the suite pass (or never pass) regardless of
    # accuracy, so main() rejects it (argparse error, exit 2) before scoring or any provider call.
    eval_mod = _load_eval()
    for bad in ('-1', '0', '101'):
        with patch.object(sys, 'argv', ['.watchman_eval.py', '--threshold', bad]):
            with pytest.raises(SystemExit) as exc:
                eval_mod.main()
        assert exc.value.code == 2


def test_eval_rejects_non_positive_n_jobs() -> None:
    # A zero or negative n_jobs is meaningless (it would silently run sequentially), so main()
    # rejects it (argparse error, exit 2) before scoring or any provider call — mirroring the
    # threshold guard and the workflow's positive-integer check on the same value.
    eval_mod = _load_eval()
    for bad in ('0', '-1'):
        with patch.object(sys, 'argv', ['.watchman_eval.py', '--n_jobs', bad]):
            with pytest.raises(SystemExit) as exc:
                eval_mod.main()
        assert exc.value.code == 2
