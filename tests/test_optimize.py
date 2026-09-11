"""Deterministic tests for the --optimize per-phase review loops.

The LLM is mocked at the seams where each loop calls the reviewer personas
(`run_personas`) and the implementor editor (`_run_editor`), so these tests
verify control flow and the safety guardrail without any network access or LLM
non-determinism. The implementation-loop guardrail still runs the *real* test
suite (an offline, dependency-free oracle) to prove that a regressing
"improvement" is reverted and a passing one is kept.
"""

import asyncio
import tempfile
import types
from unittest.mock import AsyncMock, patch

from marsha import llm
from marsha.meta import MarshaMeta
from marsha.utils import read_file, write_file


def make_meta(filename='example'):
    # A minimal, fully-populated meta. The LLM is mocked, so only .filename and the
    # attributes the message builders touch are exercised.
    meta = MarshaMeta(f'{filename}.mrsh')
    meta.filename = filename
    meta.functions = []
    meta.void_funcs = []
    meta.types = None
    return meta


def make_args(level=1, **kw):
    base = dict(optimize=level, test_personas=None, impl_personas=None,
                fix_personas=None, optimize_severity='major,minor,nit')
    base.update(kw)
    return types.SimpleNamespace(**base)


def finding(name='Ada', label='A1', severity='MAJOR', desc='d', location='x.py:1'):
    return {'name': name, 'label': label, 'severity': severity,
            'location': location, 'desc': desc}


def impl_md(filename, code):
    return f'# {filename}.py\n\n```py\n{code}\n```\n'


# --- Oracle phase: optimize_test_suite -------------------------------------

def test_oracle_noop_at_level_zero():
    with patch.object(llm, 'run_personas', new=AsyncMock()) as mock:
        out = asyncio.run(llm.optimize_test_suite(
            make_meta(), 'ORACLE', make_args(0), False))
    assert out == 'ORACLE'
    mock.assert_not_called()


def test_oracle_converges_with_no_findings():
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[])), \
            patch.object(llm, '_run_editor', new=AsyncMock()) as editor:
        out = asyncio.run(llm.optimize_test_suite(
            make_meta(), 'ORACLE', make_args(3), False))
    assert out == 'ORACLE'
    editor.assert_not_called()


def test_oracle_applies_editor_update():
    new_oracle = '# example_test.py\n\n```py\nv2\n```\n'
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor',
                         new=AsyncMock(return_value=(new_oracle, 'preamble'))) as editor:
        out = asyncio.run(llm.optimize_test_suite(
            make_meta(), 'ORACLE_v1', make_args(1), False))
    assert out == new_oracle
    assert editor.call_count == 1


def test_oracle_stops_on_editor_failure():
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor', new=AsyncMock(return_value=(None, ''))):
        out = asyncio.run(llm.optimize_test_suite(
            make_meta(), 'ORACLE', make_args(3), False))
    assert out == 'ORACLE'


def test_oracle_converges_when_editor_returns_unchanged():
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor',
                         new=AsyncMock(return_value=('ORACLE', 'preamble'))):
        out = asyncio.run(llm.optimize_test_suite(
            make_meta(), 'ORACLE', make_args(3), False))
    assert out == 'ORACLE'


# --- Implementation phase: optimize_implementation guardrail ---------------

GOOD = 'def add(a, b):\n    return a + b\n'
BETTER = 'def add(a, b):\n    """Return the sum of a and b."""\n    return a + b\n'
BROKEN = 'def add(a, b):\n    return a + b + 1\n'
ORACLE = (
    'import unittest\nfrom example import add\n\n'
    'class TestAdd(unittest.TestCase):\n'
    '    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n'
    '\nif __name__ == "__main__":\n    unittest.main()\n'
)


def _impl_env(good_code=GOOD):
    d = tempfile.mkdtemp(prefix='marsha_opt_')
    write_file(f'{d}/example.py', good_code)
    write_file(f'{d}/example_test.py', ORACLE)
    return d, [f'{d}/example.py', f'{d}/example_test.py']


def test_impl_guardrail_keeps_passing_improvement():
    d, files = _impl_env()
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor',
                         new=AsyncMock(return_value=(impl_md('example', BETTER), 'p'))):
        asyncio.run(llm.optimize_implementation(
            make_args(1), make_meta(), files, False))
    assert read_file(files[0]).strip() == BETTER.strip()


def test_impl_guardrail_reverts_regression():
    d, files = _impl_env()
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor',
                         new=AsyncMock(return_value=(impl_md('example', BROKEN), 'p'))):
        asyncio.run(llm.optimize_implementation(
            make_args(1), make_meta(), files, False))
    # The regressing "improvement" must be rolled back to the last-good code.
    assert read_file(files[0]) == GOOD


def test_impl_noop_at_level_zero():
    d, files = _impl_env()
    with patch.object(llm, 'run_personas', new=AsyncMock()) as mock:
        asyncio.run(llm.optimize_implementation(
            make_args(0), make_meta(), files, False))
    mock.assert_not_called()
    assert read_file(files[0]) == GOOD


def test_impl_converges_with_no_findings():
    d, files = _impl_env()
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[])), \
            patch.object(llm, '_run_editor', new=AsyncMock()) as editor:
        asyncio.run(llm.optimize_implementation(
            make_args(3), make_meta(), files, False))
    editor.assert_not_called()
    assert read_file(files[0]) == GOOD


def test_impl_generation_normalizes_single_result_for_n1():
    # run() returns a bare string for a single result; gpt_implementation must treat it as one doc.
    async def scenario():
        class FakeMapper:
            async def run(self, req):
                return '# example.py\n\n```py\ndef f():\n    return 1\n```\n'
        with patch.object(llm, 'get_mapper', new=lambda *a, **k: FakeMapper()):
            return await llm.gpt_implementation(make_meta(), 'ORACLE', n_results=1, debug=False)
    mds = asyncio.run(scenario())
    assert isinstance(mds, list)
    assert len(mds) == 1


# --- Correction phase: validate_test_correction ----------------------------

def test_correction_converges_with_no_findings():
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[])), \
            patch.object(llm, '_run_editor', new=AsyncMock()) as editor:
        out = asyncio.run(llm.validate_test_correction(
            make_meta(), 'code', 'orig', 'CORRECTED', 'reason', make_args(2), False))
    assert out == 'CORRECTED'
    editor.assert_not_called()


def test_correction_applies_editor_revision():
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor',
                         new=AsyncMock(return_value=('REVISED', 'p'))):
        out = asyncio.run(llm.validate_test_correction(
            make_meta(), 'code', 'orig', 'CORRECTED', 'reason', make_args(1), False))
    assert out == 'REVISED'


def test_correction_keeps_on_editor_failure():
    with patch.object(llm, 'run_personas', new=AsyncMock(return_value=[finding()])), \
            patch.object(llm, '_run_editor', new=AsyncMock(return_value=(None, ''))):
        out = asyncio.run(llm.validate_test_correction(
            make_meta(), 'code', 'orig', 'CORRECTED', 'reason', make_args(2), False))
    assert out == 'CORRECTED'
