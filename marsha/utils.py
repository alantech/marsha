from inspect import getsourcefile
import autopep8
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


def autoformat_files(files: list[str]):
    for file in files:
        before = read_file(file)
        after = autopep8.fix_code(before)
        write_file(file, after)


def copy_file(src: str, dest: str):
    shutil.copyfile(src, dest)


def copy_tree(src: str, dest: str):
    shutil.copytree(src, dest)


def get_filename_from_path(path: str):
    return os.path.splitext(os.path.basename(path))[0]


def add_helper(filename: str):
    helper = os.path.join(os.path.dirname(
        os.path.abspath(getsourcefile(lambda: 0))), 'helper.py')
    with open(filename, 'a') as o, open(helper, 'r') as i:
        o.write(i.read())
