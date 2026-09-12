"""Target-language backends and their registry.

The core keeps all orchestration; it delegates the language-specific leaves to the
backend bound for the run. `select()` binds one from the CLI's --target (by id or
alias); `current()` is what the core calls at each leaf. Only `python` is wired in
this change; the registry + dispatch are ready for `rust` (#183) and later targets.
"""

from marsha.backends.base import LanguageBackend
from marsha.backends.python import PythonBackend

__all__ = ['LanguageBackend', 'PythonBackend', 'DEFAULT_TARGET', 'available',
           'resolve_target', 'select', 'current']

DEFAULT_TARGET = 'python'

_registry = {}
_aliases = {}
_current = None


def register(backend):
    if backend.id in _registry:
        raise Exception(f'Duplicate language backend id: {backend.id}')
    _registry[backend.id] = backend
    for alias in backend.aliases:
        if alias in _aliases:
            raise Exception(f'Duplicate language backend alias: {alias}')
        _aliases[alias] = backend.id


def available():
    return sorted(_registry)


def resolve_target(name):
    # Resolve a target by id or alias (case-insensitive); raise listing what is available.
    key = (name or '').strip().lower()
    if key in _registry:
        return _registry[key]
    if key in _aliases:
        return _registry[_aliases[key]]
    raise Exception(
        f'Unknown target language: {name} (available: {", ".join(available())})')


def select(name):
    # Bind the target backend for this run (called once from the CLI). Returns the backend.
    global _current
    backend = resolve_target(name)
    _current = backend
    return backend


def current():
    # The backend bound for this run; defaults to the default target before any select().
    global _current
    if _current is None:
        _current = resolve_target(DEFAULT_TARGET)
    return _current


register(PythonBackend())
