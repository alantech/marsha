"""Deterministic tests for the low-level subprocess helper in marsha.utils."""

import asyncio

from marsha.utils import run_subprocess


def test_run_subprocess_timeout_kills_and_reaps() -> None:
    # On timeout the child is killed AND reaped (`await wait()`), so it is not left as a zombie
    # with open pipes; the timeout still surfaces as the documented error.
    async def scenario() -> tuple[str, int | None]:
        proc = await asyncio.create_subprocess_exec(
            'sleep', '30', stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            await run_subprocess(proc, timeout=0.5)
            return ('ok', None)
        except Exception:
            return ('timeout', proc.returncode)

    kind, returncode = asyncio.run(scenario())
    assert kind == 'timeout'
    assert returncode is not None  # reaped: returncode is set, not a lingering zombie


def test_run_subprocess_returns_streams() -> None:
    # The happy path returns decoded stdout and stderr.
    async def scenario() -> tuple[str, str]:
        proc = await asyncio.create_subprocess_exec(
            'printf', 'hello', stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        return await run_subprocess(proc, timeout=10)

    out, err = asyncio.run(scenario())
    assert out == 'hello'
    assert err == ''
