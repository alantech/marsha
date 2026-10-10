from __future__ import annotations

import asyncio
from asyncio.subprocess import Process
from collections.abc import Callable
import os
import shutil
import sys
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


def write_file_no_follow(filename: str, content: str) -> None:
    # Write `content` to `filename`, refusing to follow a symlink at the final path component.
    # For writes to a user-named path whose symlink-ness was checked earlier in the run: the
    # check and the write are not atomic, so a path swapped for a symlink in between must not
    # be followed — the write would overwrite the symlink's target instead. On Unix the open
    # itself carries O_NOFOLLOW, so the refusal is atomic. Where os.open has no no-follow
    # flag (Windows), a symlink present at check time is refused up front, and the write
    # itself then goes through an atomic no-follow open, so a swap that lands in between
    # is refused by the open rather than followed.
    if hasattr(os, 'O_NOFOLLOW'):
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        fd = os.open(filename, flags, 0o644)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return
    if os.path.islink(filename):
        raise OSError(f'refusing to write through a symlink: {filename}')
    _win32_write_no_follow(filename, content)


def _win32_write_no_follow(filename: str, content: str) -> None:
    # The no-follow write for a platform whose os.open cannot carry the guarantee:
    # CreateFileW with FILE_FLAG_OPEN_REPARSE_POINT opens a symlink at the final path
    # component as the reparse point itself — without following it — so a path swapped
    # for a symlink after the caller's check is opened (and refused) as the reparse
    # point it is: one atomic open, no check-then-open window. A regular path ignores
    # the flag and opens as usual (created or truncated, as O_CREAT|O_TRUNC would).
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        generic_write = 0x40000000
        file_share_all = 0x7
        create_always = 2
        file_attribute_normal = 0x80
        file_flag_open_reparse_point = 0x200000
        file_attribute_reparse_point = 0x400

        class _file_information(ctypes.Structure):
            _fields_ = [('dw_file_attributes', wintypes.DWORD),
                        ('ft_creation_time', wintypes.FILETIME),
                        ('ft_last_access_time', wintypes.FILETIME),
                        ('ft_last_write_time', wintypes.FILETIME),
                        ('dw_volume_serial_number', wintypes.DWORD),
                        ('n_file_size_high', wintypes.DWORD),
                        ('n_file_size_low', wintypes.DWORD),
                        ('n_number_of_links', wintypes.DWORD),
                        ('n_file_index_high', wintypes.DWORD),
                        ('n_file_index_low', wintypes.DWORD)]

        kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_file_information)]
        kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel32.WriteFile.argtypes = [
            wintypes.HANDLE, ctypes.c_char_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        kernel32.WriteFile.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.CreateFileW(
            filename, generic_write, file_share_all, None, create_always,
            file_attribute_normal | file_flag_open_reparse_point, None)
        if handle is None or handle >= 0xFFFFFFFF:  # INVALID_HANDLE_VALUE
            raise OSError(
                f'could not open {filename} for writing (CreateFileW)')
        try:
            info = _file_information()
            if not kernel32.GetFileInformationByHandle(
                    handle, ctypes.byref(info)):
                raise OSError(f'could not inspect {filename} '
                              '(GetFileInformationByHandle)')
            if info.dw_file_attributes & file_attribute_reparse_point:
                raise OSError(
                    f'refusing to write through a symlink: {filename}')
            written = wintypes.DWORD(0)
            data = content.encode('utf-8')
            if not kernel32.WriteFile(handle, data, len(data),
                                      ctypes.byref(written), None):
                raise OSError(f'could not write {filename} (WriteFile)')
        finally:
            kernel32.CloseHandle(handle)
        return
    # Not really Windows (the no-O_NOFOLLOW fallback off a non-Windows platform):
    # the plain open is all that is available; the caller already refused a symlink
    # present at its check.
    fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
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
        # Bounded: stream both pipes in chunks and fail at max_bytes of combined output, so an
        # oversized response is refused before it is buffered into memory (a caller's size
        # guard must get the chance to run on the size, not after the whole payload is in RAM).
        chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        total = 0

        async def _bounded() -> tuple[bytes, bytes]:
            assert stream.stdout is not None

            # A caller may merge stderr into stdout (stderr=STDOUT, as exec and validation do);
            # then stream.stderr is None and only stdout carries output, so err_chunks stays empty.
            # Otherwise both pipes exist and are drained concurrently: a child that fills one pipe
            # while keeping the other open would otherwise block on the full pipe and never close
            # the other, deadlocking a read that only ever looked at one of them.
            async def _drain(reader: asyncio.StreamReader, sink: list[bytes]) -> None:
                nonlocal total
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise Exception(
                            f'command output exceeds the {max_bytes}-byte limit')
                    sink.append(chunk)

            out_task = asyncio.ensure_future(_drain(stream.stdout, chunks))
            err_task = (asyncio.ensure_future(_drain(stream.stderr, err_chunks))
                        if stream.stderr is not None else None)
            try:
                if err_task is not None:
                    await asyncio.gather(out_task, err_task)
                else:
                    await out_task
            except BaseException:
                # Any failure (overflow, outer cancellation): cancel and reap the drain that
                # is still running so it does not linger as an unhandled task.
                for task in (out_task, err_task):
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await task
                        except BaseException:
                            pass
                raise
            await stream.wait()
            return b''.join(chunks), b''.join(err_chunks)

        read = _bounded()
    try:
        stdout, stderr = await asyncio.wait_for(read, timeout)
    except asyncio.exceptions.TimeoutError as e:
        await _kill_and_reap(stream)
        # Chain the original TimeoutError so callers can tell a timeout apart from other errors.
        raise Exception('run_subprocess timeout...') from e
    except asyncio.exceptions.CancelledError:
        # A cancellation of the calling task is a failure of the read too: without the same
        # cleanup the child would be left running.
        await _kill_and_reap(stream)
        raise
    except Exception:
        # An overflow (or any read failure): the child is no longer wanted — kill and reap it
        # the same way before the error propagates.
        await _kill_and_reap(stream)
        raise
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))


async def _kill_and_reap(stream: Process) -> None:
    """Kill a child that is no longer wanted and reap it so it does not linger as a zombie
    (or leak its transports); a follow-up failure (already gone) is ignored. The kill is
    synchronous, so it runs even when a cancellation makes the reap await fail — in that
    case the transport's own exit handling reaps the already-dead child."""
    try:
        stream.kill()
    except OSError:
        # Ignore 'no such process' error
        pass
    try:
        await stream.wait()
    except BaseException:
        pass
