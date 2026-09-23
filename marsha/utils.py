from __future__ import annotations

import asyncio
from asyncio.subprocess import Process
from collections.abc import Callable
import os
import shutil
from typing import cast


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
                         input: bytes | None = None) -> tuple[str, str]:
    try:
        stdout, stderr = await asyncio.wait_for(stream.communicate(input), timeout)
    except asyncio.exceptions.TimeoutError as e:
        try:
            stream.kill()
        except OSError:
            # Ignore 'no such process' error
            pass
        # Reap the killed child and close its pipes so it does not linger as a zombie (or leak
        # its transports) until garbage collection; ignore a follow-up failure if it is already
        # gone.
        try:
            await stream.wait()
        except Exception:
            pass
        # Chain the original TimeoutError so callers can tell a timeout apart from other errors.
        raise Exception('run_subprocess timeout...') from e
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))
