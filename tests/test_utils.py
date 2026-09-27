"""Deterministic tests for the low-level subprocess helper in marsha.utils."""

import asyncio
import os
from typing import Any

from marsha.utils import run_subprocess, write_file_no_follow


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


def test_run_subprocess_cancellation_kills_the_child() -> None:
    # Cancelling the read (not just timing it out) must also kill the child: a cancelled
    # refine/review task must not leave the CLI subprocess running.
    async def scenario() -> int | None:
        proc = await asyncio.create_subprocess_exec(
            'sleep', '30', stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        task = asyncio.ensure_future(
            run_subprocess(proc, timeout=30, max_bytes=1000))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Give the transport a moment to reap the killed child.
        for _ in range(50):
            if proc.returncode is not None:
                break
            await asyncio.sleep(0.1)
        return proc.returncode

    returncode = asyncio.run(scenario())
    assert returncode is not None  # the child was killed, not left running


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


def test_run_subprocess_bounded_drains_stderr_concurrently() -> None:
    # stderr is drained while stdout is read: a child that fills the stderr pipe (well over
    # the 64KB buffer) while still writing stdout must not deadlock the bounded reader.
    async def scenario() -> tuple[str, str]:
        proc = await asyncio.create_subprocess_exec(
            'sh', '-c', 'head -c 200000 /dev/zero 1>&2; printf ok',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        return await run_subprocess(proc, timeout=15, max_bytes=300000)

    out, err = asyncio.run(scenario())
    assert out == 'ok'
    assert len(err) == 200000


def test_run_subprocess_bounded_stderr_overflow_fails() -> None:
    # stderr is bounded too (the cap covers the combined output): a child that writes over
    # the limit to stderr, with little stdout, fails with the overflow error and is reaped
    # instead of buffering unbounded memory.
    async def scenario() -> tuple[str, int | None]:
        proc = await asyncio.create_subprocess_exec(
            'sh', '-c', 'head -c 200000 /dev/zero 1>&2; printf ok',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            await run_subprocess(proc, timeout=15, max_bytes=10000)
            return ('ok', None)
        except Exception:
            return ('overflow', proc.returncode)

    kind, returncode = asyncio.run(scenario())
    assert kind == 'overflow'
    assert returncode is not None  # reaped: returncode is set, not a lingering zombie


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


def test_write_file_no_follow_refuses_symlink(tmp_path: Any) -> None:
    # O_NOFOLLOW: writing through a path that has been swapped for a symlink must fail rather
    # than overwrite the link's target (the load-time symlink check is not atomic with the
    # write). Regular paths are written as usual.
    target = tmp_path / 'target.txt'
    target.write_text('target content')
    link = tmp_path / 'link.txt'
    os.symlink(target, link)
    try:
        write_file_no_follow(str(link), 'rewrite')
        raise AssertionError('a symlinked path must not be written through')
    except OSError:
        pass
    assert target.read_text() == 'target content'  # the target is untouched
    regular = tmp_path / 'regular.txt'
    write_file_no_follow(str(regular), 'hello')
    assert regular.read_text() == 'hello'
