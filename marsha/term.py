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
    """Print a warning or error to stderr. Rich renders the markdown (with color only when attached
    to a terminal); the trailing whitespace its list renderer leaves on each line is trimmed."""
    color = 'yellow' if kind == 'warning' else 'red'
    console = _console()
    with console.capture() as capture:
        console.print(f'[bold {color}]{kind}:[/]', Markdown(text))
    for line in capture.get().splitlines():
        print(line.rstrip(), file=sys.stderr)
