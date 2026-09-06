#!/usr/bin/env python
"""Write the marsha launcher script for the current platform.

Usage: make_launcher.py <bin_dir> <venv_dir>
"""

import os
import stat
import subprocess
import sys

PLACEHOLDER = '__MARSHA_VENV__'


def build_launcher(venv_dir):
    if os.name == 'nt':
        template, filename = 'marsha.bat.in', 'marsha.bat'
    else:
        template, filename = 'marsha.sh.in', 'marsha'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), template), 'r') as f:
        text = f.read()
    text = text.replace(PLACEHOLDER, venv_dir)
    if os.name == 'nt':
        text = text.replace('\r\n', '\n').replace('\n', '\r\n')
    return filename, text


def path_note(bin_dir):
    entries = [
        e.strip().rstrip('/\\').lower()
        for e in os.environ.get('PATH', '').split(os.pathsep)
    ]
    if bin_dir.rstrip('/\\').lower() not in entries:
        print(f'note: {bin_dir} is not in your PATH')


def main():
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} <bin_dir> <venv_dir>', file=sys.stderr)
        sys.exit(2)
    bin_dir, venv_dir = sys.argv[1], sys.argv[2]
    filename, text = build_launcher(venv_dir)
    os.makedirs(bin_dir, exist_ok=True)
    path = os.path.join(bin_dir, filename)
    with open(path, 'w', newline='') as f:
        f.write(text)
    if os.name != 'nt':
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run([path, '--help'], stdout=subprocess.DEVNULL, check=True)
    print(f'installed marsha to {path} (venv: {venv_dir})')
    path_note(bin_dir)


if __name__ == '__main__':
    main()
