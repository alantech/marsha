import asyncio
from asyncio.subprocess import Process
import os
import shutil


def prettify_time_delta(delta, max_depth=2):
    rnd = round if max_depth == 1 else int
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


def read_file(filename: str, mode: str = 'r'):
    with open(filename, mode) as f:
        content = f.read()
    return content


def write_file(filename: str, content: str, mode: str = 'w'):
    with open(filename, mode) as f:
        f.write(content)


def write_composed(files: dict, subdir: str = None) -> list[str]:
    # Write a composed on-disk layout ({path: content}) and return the written paths.
    paths = []
    for name, content in files.items():
        path = f'{subdir}/{name}' if subdir is not None else name
        if subdir is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        write_file(path, content)
        paths.append(path)
    return paths


def copy_file(src: str, dest: str):
    shutil.copyfile(src, dest)


def copy_tree(src: str, dest: str):
    shutil.copytree(src, dest)


def get_filename_from_path(path: str):
    return os.path.splitext(os.path.basename(path))[0]


async def run_subprocess(stream: Process, timeout: float = 60.0) -> tuple[str, str]:
    stdout = ''
    stderr = ''
    try:
        stdout, stderr = await asyncio.wait_for(stream.communicate(), timeout)
    except asyncio.exceptions.TimeoutError:
        try:
            stream.kill()
        except OSError:
            # Ignore 'no such process' error
            pass
        raise Exception('run_subprocess timeout...')
    except Exception as e:
        raise e
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))
