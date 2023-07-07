import autopep8
import os
import shutil


def read_file(filename: str, mode: str = 'r'):
    f = open(filename, mode)
    content = f.read()
    f.close()
    return content


def write_file(filename: str, content: str, mode: str = 'w'):
    f = open(filename, mode)
    f.write(content)
    f.close()


def autoformat_files(files: list[str]):
    for file in files:
        before = read_file(file)
        after = autopep8.fix_code(before)
        write_file(file, after)


def copy_file(src: str, dest: str):
    shutil.copyfile(src, dest)


def delete_dir_and_content(filename: str):
    dir = os.path.dirname(filename)
    if os.path.isdir(dir):
        shutil.rmtree(dir)


def get_filename_from_path(path: str):
    return os.path.splitext(os.path.basename(path))[0]
