from __future__ import annotations

import asyncio
from asyncio.subprocess import Process
from collections.abc import Callable
import os
import shutil
from typing import cast


# A JSON-serializable value: the shape of data that crosses an untyped boundary — a
# json.loads/json.load result, a GitHub GraphQL payload, an LLM's JSON output, a /models
# discovery response. Tighter than Any: it rules out non-JSON values (tuples, objects,
# callables) while still allowing the arbitrary nesting that real JSON has.
type JSON = bool | str | int | float | list[JSON] | dict[str, JSON] | None


def prettify_time_delta(delta: float, max_depth: int = 2) -> str:
    rnd: 'Callable[[float], int]' = round if max_depth == 1 else int
    if not max_depth:
        return ''
    if delta < 1:
        return f'''{format(delta * 1000, '3g')}ms'''
    elif delta < 60:
        sec = rnd(delta)
        subdelta = delta - sec
        return f'''{format(sec, '2g')}sec {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()
    elif delta < 3600:
        mn = rnd(delta / 60)
        subdelta = delta - mn * 60
        return f'''{format(mn, '2g')}min {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()
    elif delta < 86400:
        hr = rnd(delta / 3600)
        subdelta = delta - hr * 3600
        return f'''{format(hr, '2g')}hr {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()
    else:
        day = rnd(delta / 86400)
        subdelta = delta - day * 86400
        return f'''{format(day, '2g')}days {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()


def read_file(filename: str, mode: str = 'r') -> str | bytes:
    # Pin the text encoding so reads/writes are deterministic across platforms; binary mode
    # takes no encoding argument.
    if 'b' in mode:
        with open(filename, mode) as f:
            return cast(bytes, f.read())
    with open(filename, mode, encoding='utf-8') as f:
        return cast(str, f.read())


def write_file(filename: str, content: str | bytes, mode: str = 'w') -> None:
    if 'b' in mode:
        with open(filename, mode) as f:
            f.write(content)
    else:
        with open(filename, mode, encoding='utf-8') as f:
            f.write(content)


def write_composed(files: dict[str, str], subdir: str | None = None) -> list[str]:
    # Write a composed on-disk layout ({path: content}) and return the written paths.
    paths: list[str] = []
    for name, content in files.items():
        path = f'{subdir}/{name}' if subdir is not None else name
        if subdir is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        write_file(path, content)
        paths.append(path)
    return paths


def copy_file(src: str, dest: str) -> None:
    shutil.copyfile(src, dest)


def copy_tree(src: str, dest: str) -> None:
    shutil.copytree(src, dest)


def get_filename_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


async def run_subprocess(stream: Process, timeout: float = 60.0,
                         input: bytes | None = None,
                         max_bytes: int | None = None) -> tuple[str, str]:
    if max_bytes is None or input is not None:
        read = stream.communicate(input)
    else:
        # Bounded: stream stdout in chunks and fail at max_bytes, so an oversized output is
        # refused before it is buffered into memory (a caller's size guard must get the
        # chance to run on the size, not after the whole payload is in RAM).
        chunks: list[bytes] = []
        total = 0

        async def _bounded() -> tuple[bytes, bytes]:
            nonlocal total
            assert stream.stdout is not None and stream.stderr is not None
            # Drain stderr concurrently: reading it only after stdout would deadlock on a
            # child that fills the stderr pipe while keeping stdout open (it would block on
            # the full pipe and never close stdout).
            err_task = asyncio.ensure_future(stream.stderr.read())
            try:
                while True:
                    chunk = await stream.stdout.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        err_task.cancel()
                        raise Exception(
                            f'command output exceeds the {max_bytes}-byte limit')
                    chunks.append(chunk)
                err = await err_task
            except BaseException:
                # Any failure (overflow, outer cancellation): if the stderr drain is still
                # pending, cancel and reap it so it does not linger as an unhandled task.
                if not err_task.done():
                    err_task.cancel()
                    try:
                        await err_task
                    except BaseException:
                        pass
                raise
            await stream.wait()
            return b''.join(chunks), err

        read = _bounded()
    try:
        stdout, stderr = await asyncio.wait_for(read, timeout)
    except asyncio.exceptions.TimeoutError as e:
        try:
            stream.kill()
        except OSError:
            # Ignore 'no such process' error
            pass
        # Reap the killed child and close its pipes so it does not linger as a zombie (or leak
        # its transports) until garbage collection; ignore a follow-up failure if it is
        # already gone.
        try:
            await stream.wait()
        except Exception:
            pass
        # Chain the original TimeoutError so callers can tell a timeout apart from other errors.
        raise Exception('run_subprocess timeout...') from e
    except Exception:
        # An overflow (or any read failure): the child is no longer wanted — kill and reap it
        # the same way before the error propagates.
        try:
            stream.kill()
        except OSError:
            pass
        try:
            await stream.wait()
        except Exception:
            pass
        raise
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))
