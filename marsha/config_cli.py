"""The `marsha config` subcommand.

The user config file (see marsha.config) used to be hand-edited only, with the supported
keys known only from reading the source. This command makes it discoverable and editable:
`list` shows both the values currently set and every supported key (with its default and a
description), while `set`/`unset` add and remove keys. Everything here is plain local file
I/O — no LLM client is created and no toolchain is required.
"""

from __future__ import annotations

import sys
from typing import Any, cast

from marsha.config import (
    CONFIG_KEYS, PROVIDERS, get_config_path, load_config_file, save_config_file)


def run_config(args: Any) -> int:
    # args is the argparse Namespace from the `config` subparser; base.run() has already
    # rejected a missing action, so config_action is one of list/set/add/unset/remove.
    if args.config_action in ('set', 'add'):
        return _set(args.key, args.value)
    if args.config_action in ('unset', 'remove'):
        return _unset(args.key)
    return _list(args.show_secrets)


def _list(show_secrets: bool) -> int:
    config = _load_plain()
    print(f'Config file: {get_config_path()}')
    print()
    print('Current settings:')
    if config:
        for name, value in config.items():
            print(f'  {name} = {_display_value(name, value, show_secrets)}')
    else:
        print('  (none — the file is missing or empty, so all defaults apply)')
    print()
    print('Supported settings (`marsha config set/unset <key> ...`):')
    width = max(len(key.name) for key in CONFIG_KEYS)
    for key in CONFIG_KEYS:
        print(f'  {key.name.ljust(width)}  {key.description}')
        print(f'  {" ".ljust(width)}  Default: {key.default}')
    known = {key.name for key in CONFIG_KEYS}
    unknown = [name for name in config if name not in known]
    if unknown:
        # Hand-edited files drift (a typo'd key is silently ignored by the resolvers), so
        # surface anything we don't recognize.
        print()
        print('Warning: unrecognized keys in the config file '
              '(they are ignored; typos?):', file=sys.stderr)
        for name in unknown:
            print(f'  {name}', file=sys.stderr)
    return 0


def _set(name: str, value: str) -> int:
    if name not in {key.name for key in CONFIG_KEYS}:
        return _unknown_key(name)
    if name == 'provider' and value not in PROVIDERS:
        # Mirror resolve_provider's wording so the error reads the same here and at runtime.
        print(f'Unknown LLM provider: {value} '
              f'(expected one of: {", ".join(PROVIDERS)})', file=sys.stderr)
        return 2
    config = _load_plain()
    config[name] = value
    save_config_file(config)
    print(f'Set {name} = {_display_value(name, value, False)} '
          f'in {get_config_path()}')
    return 0


def _unset(name: str) -> int:
    if name not in {key.name for key in CONFIG_KEYS}:
        return _unknown_key(name)
    config = _load_plain()
    if name not in config:
        print(f'{name} is not set in {get_config_path()}')
        return 0
    del config[name]
    save_config_file(config)
    print(f'Removed {name} from {get_config_path()}')
    return 0


def _load_plain() -> dict[str, str]:
    # load_config_file() validates the file and its values are documented to be strings; the
    # cast to a plain dict keeps any unknown keys intact for round-tripping (set/unset only
    # touch the named key) and sidesteps TypedDict's literal-key indexing in strict mode. A
    # corrupt file raises from load_config_file before we ever write, so it can't be
    # clobbered silently.
    return cast(dict[str, str], dict(load_config_file()))


def _display_value(name: str, value: str, show_secrets: bool) -> str:
    # Secrets (API keys) are masked unless --show-secrets was passed: config output lands in
    # screenshots and scrollback often enough that plaintext is a bad default. The last four
    # characters are kept so two keys can still be told apart.
    secret = any(key.secret for key in CONFIG_KEYS if key.name == name)
    if show_secrets or not secret:
        return value
    if len(value) <= 4:
        return '****'
    return f'****{value[-4:]}'


def _unknown_key(name: str) -> int:
    print(f'Unknown config key: {name}', file=sys.stderr)
    print(f'Supported keys: {", ".join(key.name for key in CONFIG_KEYS)}',
          file=sys.stderr)
    return 2
