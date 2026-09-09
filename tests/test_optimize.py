"""Deterministic tests for the --optimize per-phase review loops.

The LLM is mocked at the seam where each loop calls it, so these tests verify
control flow and the safety guardrail without any network access or LLM
non-determinism. The one exception is the implementation-loop guardrail, which
runs the *real* test suite (an offline, dependency-free oracle) to prove that a
regressing "improvement" is reverted and a passing one is kept.
"""

import asyncio
import tempfile
import types
from unittest.mock import AsyncMock, patch

from marsha import llm
from marsha.meta import MarshaMeta
from marsha.utils import read_file, write_file


def make_meta(filename='example'):
    # A minimal, fully-populated meta. The LLM is mocked, so only .filename and
    # the attributes format_marsha_for_llm touches are exercised.
    meta = MarshaMeta(f'{filename}.mrsh')
    meta.filename = filename
    meta.functions = []
    meta.void_funcs = []
    meta.types = None
    return meta


def impl_md(filename, code):
    return f'# {filename}.py\n\n```py\n{code}\n```\n'


# --- Oracle phase: optimize_test_suite -------------------------------------

def test_oracle_loop_converges_when_unchanged():
    async def scenario():
        with patch.object(llm, 'gpt_optimize_test_suite',
                          new=AsyncMock(return_value='ORACLE')):
            return await llm.optimize_test_suite(make_meta(), 'ORACLE', level=3, debug=False)
    assert asyncio.run(scenario()) == 'ORACLE'


def test_oracle_loop_applies_updates_up_to_level():
    async def scenario():
        mock = AsyncMock(side_effect=['ORACLE_v2', 'ORACLE_v3'])
        with patch.object(llm, 'gpt_optimize_test_suite', new=mock):
            return await llm.optimize_test_suite(make_meta(), 'ORACLE_v1', level=2, debug=False)
    assert asyncio.run(scenario()) == 'ORACLE_v3'


def test_oracle_loop_stops_on_invalid_review():
    async def scenario():
        mock = AsyncMock(return_value=None)
        with patch.object(llm, 'gpt_optimize_test_suite', new=mock):
            return await llm.optimize_test_suite(make_meta(), 'ORACLE', level=3, debug=False)
    assert asyncio.run(scenario()) == 'ORACLE'


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
    args = types.SimpleNamespace(optimize=1)

    async def scenario():
        with patch.object(llm, 'gpt_optimize_implementation',
                          new=AsyncMock(return_value=impl_md('example', BETTER))):
            await llm.optimize_implementation(args, make_meta(), files, debug=False)
    asyncio.run(scenario())
    assert read_file(files[0]).strip() == BETTER.strip()


def test_impl_guardrail_reverts_regression():
    d, files = _impl_env()
    args = types.SimpleNamespace(optimize=1)

    async def scenario():
        with patch.object(llm, 'gpt_optimize_implementation',
                          new=AsyncMock(return_value=impl_md('example', BROKEN))):
            await llm.optimize_implementation(args, make_meta(), files, debug=False)
    asyncio.run(scenario())
    # The regressing "improvement" must be rolled back to the last-good code.
    assert read_file(files[0]) == GOOD


def test_impl_loop_is_noop_at_level_zero():
    d, files = _impl_env()
    args = types.SimpleNamespace(optimize=0)
    mock = AsyncMock(return_value=impl_md('example', BETTER))

    async def scenario():
        with patch.object(llm, 'gpt_optimize_implementation', new=mock):
            await llm.optimize_implementation(args, make_meta(), files, debug=False)
    asyncio.run(scenario())
    mock.assert_not_called()
    assert read_file(files[0]) == GOOD


def test_impl_loop_stops_on_invalid_review():
    d, files = _impl_env()
    args = types.SimpleNamespace(optimize=2)
    mock = AsyncMock(return_value=None)

    async def scenario():
        with patch.object(llm, 'gpt_optimize_implementation', new=mock):
            await llm.optimize_implementation(args, make_meta(), files, debug=False)
    asyncio.run(scenario())
    assert mock.call_count == 1
    assert read_file(files[0]) == GOOD


# --- Correction phase: validate_test_correction ----------------------------

def test_correction_loop_converges_when_valid():
    async def scenario():
        with patch.object(llm, 'gpt_validate_test_correction',
                          new=AsyncMock(return_value='CORRECTED')):
            return await llm.validate_test_correction(
                make_meta(), 'code', 'orig', 'CORRECTED', 'reason', level=2, debug=False)
    assert asyncio.run(scenario()) == 'CORRECTED'


def test_correction_loop_applies_revisions_up_to_level():
    async def scenario():
        mock = AsyncMock(side_effect=['REVISED_2', 'REVISED_3'])
        with patch.object(llm, 'gpt_validate_test_correction', new=mock):
            return await llm.validate_test_correction(
                make_meta(), 'code', 'orig', 'REVISED_1', 'reason', level=2, debug=False)
    assert asyncio.run(scenario()) == 'REVISED_3'


def test_correction_loop_keeps_correction_on_invalid_review():
    async def scenario():
        mock = AsyncMock(return_value=None)
        with patch.object(llm, 'gpt_validate_test_correction', new=mock):
            return await llm.validate_test_correction(
                make_meta(), 'code', 'orig', 'CORRECTED', 'reason', level=2, debug=False)
    assert asyncio.run(scenario()) == 'CORRECTED'
