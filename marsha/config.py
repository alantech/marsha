from __future__ import annotations

import json
import os
import platform
from typing import NamedTuple, TypedDict, cast

APP_NAME = 'marsha'
CONFIG_FILENAME = 'config.json'
DEFAULT_API_BASE = 'https://api.openai.com/v1'
DEFAULT_PROVIDER = 'openai'
DEFAULT_MODEL = 'gpt-6-luna'
DEFAULT_STRONG_MODEL = 'gpt-5.6-terra'
ANTHROPIC_DEFAULT_MODEL = 'claude-sonnet-5'
ANTHROPIC_DEFAULT_STRONG_MODEL = 'claude-opus-5'
PROVIDERS = ('openai', 'anthropic')


# The shape of the optional user config file (config.json). Every key is optional; values are
# strings (a provider/model name, a base URL, or an API key).
class ConfigFile(TypedDict, total=False):
    provider: str
    api_base: str
    api_key: str
    claude_api_key: str
    model: str
    model_strong: str


# Documentation for one supported config file key: what it is for, what applies when it is
# unset, and whether the value is a secret (masked in `marsha config` output).
class ConfigKey(NamedTuple):
    name: str
    description: str
    default: str
    secret: bool = False


# The supported config file keys, in display order. This is the single source of truth for
# `marsha config list` and for the key validation in `set`/`unset`; keep it in sync with the
# resolvers below and the README's config-file section.
CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey(
        'provider',
        'LLM provider to use: openai (any OpenAI-compatible API) or anthropic (Claude). '
        'Overridden by --provider.',
        DEFAULT_PROVIDER),
    ConfigKey(
        'api_base',
        'Base URL of an OpenAI-compatible API, e.g. a local llama.cpp server. Overridden by '
        'the OPENAI_BASE_URL environment variable and --api-base (openai provider only).',
        DEFAULT_API_BASE),
    ConfigKey(
        'api_key',
        'API key for the openai provider. Overridden by the OPENAI_SECRET_KEY / OPENAI_API_KEY '
        'environment variables.',
        '(none)',
        secret=True),
    ConfigKey(
        'claude_api_key',
        'API key for the anthropic provider. Overridden by the CLAUDE_API_KEY / '
        'ANTHROPIC_API_KEY environment variables.',
        '(none)',
        secret=True),
    ConfigKey(
        'model',
        'Model to use for code generation. Overridden by --model.',
        f'{DEFAULT_MODEL} ({ANTHROPIC_DEFAULT_MODEL} with provider=anthropic)'),
    ConfigKey(
        'model_strong',
        'Strong model: used for the test-fixing stage and as the escalation target when a '
        'prompt exceeds the standard model\'s context.',
        f'{DEFAULT_STRONG_MODEL} ({ANTHROPIC_DEFAULT_STRONG_MODEL} with provider=anthropic)'),
)


_cli_model: str | None = None
_cli_strong_model: str | None = None
_cli_provider: str | None = None
_cli_api_base: str | None = None


def get_config_dir() -> str:
    system = platform.system()
    if system == 'Windows':
        base = os.environ.get('LOCALAPPDATA')
        if base is None:
            base = os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
        return os.path.join(base, APP_NAME)
    if system == 'Darwin':
        return os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', APP_NAME)
    base = os.environ.get('XDG_CONFIG_HOME')
    if base is None:
        base = os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(base, APP_NAME)


def get_config_path() -> str:
    return os.path.join(get_config_dir(), CONFIG_FILENAME)


def load_config_file() -> ConfigFile:
    config: ConfigFile = {}
    path = get_config_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except (OSError, ValueError) as e:
            raise Exception(f'Failed to read config file at {path}: {e}')
        if not isinstance(loaded, dict):
            raise Exception(
                f'Invalid config file at {path}: expected a JSON object')
        config = cast(ConfigFile, loaded)
    return config


def save_config_file(config: dict[str, str]) -> None:
    # Write the config file, creating the config directory on first use. The file can hold
    # API keys, so permissions are tightened to owner-only (0600; a no-op where the OS does
    # not support them).
    path = get_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
        f.write('\n')
    os.chmod(path, 0o600)


def set_cli_model(model: str | None) -> None:
    global _cli_model
    _cli_model = model


def set_cli_strong_model(model: str | None) -> None:
    global _cli_strong_model
    _cli_strong_model = model


def set_cli_provider(provider: str) -> None:
    global _cli_provider
    _cli_provider = provider


def set_cli_api_base(base: str) -> None:
    global _cli_api_base
    _cli_api_base = base


def resolve_provider() -> str:
    if _cli_provider:
        provider = _cli_provider
    else:
        provider = load_config_file().get('provider') or DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        raise Exception(
            f'Unknown LLM provider: {provider} (expected one of: {", ".join(PROVIDERS)})')
    return provider


def resolve_api_base(cli_value: str | None = None) -> str:
    if cli_value:
        return cli_value
    if _cli_api_base:
        return _cli_api_base
    env_value = os.getenv('OPENAI_BASE_URL')
    if env_value:
        return env_value
    file_value = load_config_file().get('api_base')
    if file_value:
        return file_value
    return DEFAULT_API_BASE


def is_local_backend() -> bool:
    # True when the OpenAI provider is pointed at a non-default base (a local or
    # OpenAI-compatible server such as llama.cpp). Such servers serialize requests and are
    # slow, so callers use this to run reviewers one at a time and relax the client timeout.
    # Real OpenAI and Anthropic handle concurrent requests fine.
    return resolve_provider() == 'openai' and resolve_api_base() != DEFAULT_API_BASE


def resolve_api_key(provider: str | None = None) -> str | None:
    provider = provider or resolve_provider()
    if provider == 'anthropic':
        env_value = os.getenv('CLAUDE_API_KEY') or os.getenv(
            'ANTHROPIC_API_KEY')
        if env_value:
            return env_value
        return load_config_file().get('claude_api_key')
    env_value = os.getenv('OPENAI_SECRET_KEY') or os.getenv('OPENAI_API_KEY')
    if env_value:
        return env_value
    return load_config_file().get('api_key')


def resolve_model() -> str:
    if _cli_model:
        return _cli_model
    file_value = load_config_file().get('model')
    if file_value:
        return file_value
    if resolve_provider() == 'anthropic':
        return ANTHROPIC_DEFAULT_MODEL
    return DEFAULT_MODEL


def resolve_strong_model() -> str:
    if _cli_strong_model:
        return _cli_strong_model
    file_value = load_config_file().get('model_strong')
    if file_value:
        return file_value
    if resolve_provider() == 'anthropic':
        return ANTHROPIC_DEFAULT_STRONG_MODEL
    return DEFAULT_STRONG_MODEL


def model_is_pinned() -> bool:
    # True when the standard model was explicitly chosen (--model or the config file) rather than
    # falling back to the default. Pinned models are never remapped by model auto-matching.
    return _cli_model is not None or bool(load_config_file().get('model'))


def strong_model_is_pinned() -> bool:
    # As model_is_pinned, for the strong model (the model_strong config key).
    return _cli_strong_model is not None or bool(
        load_config_file().get('model_strong'))
