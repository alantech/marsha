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


def test_run_subprocess_bounded_output() -> None:
    # With max_bytes, output within the limit is returned in full...
    async def scenario() -> tuple[str, str]:
        proc = await asyncio.create_subprocess_exec(
            'printf', 'hello', stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        return await run_subprocess(proc, timeout=10, max_bytes=100)

    out, err = asyncio.run(scenario())
    assert out == 'hello'
    assert err == ''


def test_run_subprocess_overflow_fails_and_reaps() -> None:
    # ...and output over the limit is refused (before it is buffered) with the child killed
    # and reaped, so an oversized remote response cannot exhaust memory.
    async def scenario() -> tuple[str, int | None]:
        proc = await asyncio.create_subprocess_exec(
            'head', '-c', '200', '/dev/zero', stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            await run_subprocess(proc, timeout=10, max_bytes=100)
            return ('ok', None)
        except Exception:
            return ('overflow', proc.returncode)

    kind, returncode = asyncio.run(scenario())
    assert kind == 'overflow'
    assert returncode is not None  # reaped: returncode is set, not a lingering zombie
