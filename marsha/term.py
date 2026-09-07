import sys

from rich.console import Console
from rich.markdown import Markdown

_stderr_console = None


def _console():
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(file=sys.stderr, soft_wrap=True)
    return _stderr_console


def print_diagnostic(kind, text):
    """Print a warning or error to stderr; rich renders the markdown and omits color when not attached to a terminal"""
    color = 'yellow' if kind == 'warning' else 'red'
    _console().print(f'[bold {color}]{kind}:[/]', Markdown(text))
