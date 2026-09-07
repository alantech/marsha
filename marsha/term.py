import re
import sys

from rich.console import Console
from rich.markdown import Markdown

_stderr_console = None


def _console():
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(file=sys.stderr, soft_wrap=True)
    return _stderr_console


def strip_markdown(text):
    text = re.sub(r'(?m)^\s*>\s?', '', text)
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'(?m)^\s*[-*+]\s+', '', text)
    text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)
    return text.strip()


def print_diagnostic(kind, text):
    """Print a warning or error to stderr, rendering markdown when attached to a terminal"""
    console = _console()
    if console.is_terminal:
        color = 'yellow' if kind == 'warning' else 'red'
        console.print(f'[bold {color}]{kind}:[/]', Markdown(text))
    else:
        print(f'{kind}: {strip_markdown(text)}', file=sys.stderr)
