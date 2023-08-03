from inspect import getsourcefile
import autopep8
import os
import shutil


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
