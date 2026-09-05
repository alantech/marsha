import json
import os
import platform

APP_NAME = 'marsha'
CONFIG_FILENAME = 'config.json'
DEFAULT_API_BASE = 'https://api.openai.com/v1'
DEFAULT_MODEL = 'gpt-5-mini'
DEFAULT_STRONG_MODEL = 'gpt-5'

_cli_model = None


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


def resolve_api_base(cli_value=None):
    if cli_value:
        return cli_value
    env_value = os.getenv('OPENAI_BASE_URL')
    if env_value:
        return env_value
    file_value = load_config_file().get('api_base')
    if file_value:
        return file_value
    return DEFAULT_API_BASE


def resolve_api_key():
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
    return DEFAULT_MODEL


def resolve_strong_model():
    file_value = load_config_file().get('model_strong')
    if file_value:
        return file_value
    return DEFAULT_STRONG_MODEL
