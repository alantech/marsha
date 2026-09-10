import json
import os
import platform

APP_NAME = 'marsha'
CONFIG_FILENAME = 'config.json'
DEFAULT_API_BASE = 'https://api.openai.com/v1'
DEFAULT_PROVIDER = 'openai'
DEFAULT_MODEL = 'gpt-5-mini'
DEFAULT_STRONG_MODEL = 'gpt-5'
ANTHROPIC_DEFAULT_MODEL = 'claude-sonnet-5'
ANTHROPIC_DEFAULT_STRONG_MODEL = 'claude-opus-5'
PROVIDERS = ('openai', 'anthropic')

_cli_model = None
_cli_strong_model = None
_cli_provider = None
_cli_api_base = None


def get_config_dir():
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


def get_config_path():
    return os.path.join(get_config_dir(), CONFIG_FILENAME)


def load_config_file():
    config = {}
    path = get_config_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, ValueError) as e:
            raise Exception(f'Failed to read config file at {path}: {e}')
        if not isinstance(config, dict):
            raise Exception(
                f'Invalid config file at {path}: expected a JSON object')
    return config


def set_cli_model(model):
    global _cli_model
    _cli_model = model


def set_cli_strong_model(model):
    global _cli_strong_model
    _cli_strong_model = model


def set_cli_provider(provider):
    global _cli_provider
    _cli_provider = provider


def set_cli_api_base(base):
    global _cli_api_base
    _cli_api_base = base


def resolve_provider():
    if _cli_provider:
        provider = _cli_provider
    else:
        provider = load_config_file().get('provider') or DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        raise Exception(
            f'Unknown LLM provider: {provider} (expected one of: {", ".join(PROVIDERS)})')
    return provider


def resolve_api_base(cli_value=None):
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


def is_local_backend():
    # True when the OpenAI provider is pointed at a non-default base (a local or
    # OpenAI-compatible server such as llama.cpp). Such servers serialize requests and are
    # slow, so callers use this to run reviewers one at a time and relax the client timeout.
    # Real OpenAI and Anthropic handle concurrent requests fine.
    return resolve_provider() == 'openai' and resolve_api_base() != DEFAULT_API_BASE


def resolve_api_key(provider=None):
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


def resolve_model():
    if _cli_model:
        return _cli_model
    file_value = load_config_file().get('model')
    if file_value:
        return file_value
    if resolve_provider() == 'anthropic':
        return ANTHROPIC_DEFAULT_MODEL
    return DEFAULT_MODEL


def resolve_strong_model():
    if _cli_strong_model:
        return _cli_strong_model
    file_value = load_config_file().get('model_strong')
    if file_value:
        return file_value
    if resolve_provider() == 'anthropic':
        return ANTHROPIC_DEFAULT_STRONG_MODEL
    return DEFAULT_STRONG_MODEL


def apply_available_models(available):
    """Given the models actually served by an (OpenAI-compatible, e.g. local) backend, remap the
    standard and strong models to an available one when the configured model isn't served. A local
    server runs whatever is loaded and ignores the requested model name, so this makes marsha log
    and send the model that will actually be used. Returns a list of human-readable notes for each
    remap (empty if nothing changed)."""
    if not available:
        return []
    notes = []

    def _remap(name, resolver, setter):
        current = resolver()
        if current not in available:
            chosen = available[0]
            setter(chosen)
            notes.append(
                f'{name} {current!r} is not served by the backend; using {chosen!r}')

    _remap('model', resolve_model, set_cli_model)
    _remap('strong model', resolve_strong_model, set_cli_strong_model)
    return notes
