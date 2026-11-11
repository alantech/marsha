"""Deterministic tests for the `marsha config` subcommand.

These exercise `marsha config list` (current values, the supported-keys table, secret
masking, unrecognized-key warnings), `set`/`add` (key and provider validation, file and
directory creation, round-tripping of unknown keys), and `unset`/`remove`. Everything runs
through base.run() against a throwaway config dir ($XDG_CONFIG_HOME pointed at tmp_path) —
no LLM or network I/O is involved.
"""

import json
import os
import stat
from typing import Any

import pytest

import marsha.base as base


@pytest.fixture
def config_path(tmp_path: Any, monkeypatch: Any) -> str:
    # Point the config dir at a throwaway location so tests never touch the real one.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    return os.path.join(str(tmp_path), 'marsha', 'config.json')


def _write_config(config_path: str, config: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f)


def _read_config(config_path: str) -> dict[str, str]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return dict(json.load(f))


def test_run_config_no_action_prints_help_and_returns_two(capsys: Any) -> None:
    assert base.run(['config']) == 2
    assert 'usage: marsha config' in capsys.readouterr().err


def test_list_with_no_config_file(config_path: Any, capsys: Any) -> None:
    assert base.run(['config', 'list']) == 0
    out = capsys.readouterr().out
    assert f'Config file: {config_path}' in out
    assert '(none' in out
    # The supported-keys table lists every key the config file accepts, with defaults.
    for key in ('provider', 'api_base', 'api_key', 'claude_api_key',
                'model', 'model_strong'):
        assert key in out
    assert 'Default:' in out


def test_list_shows_current_values_and_masks_secrets(
        config_path: Any, capsys: Any) -> None:
    _write_config(config_path, {
                  'model': 'gpt-5-mini', 'api_key': 'sk-supersecretvalue123'})
    assert base.run(['config', 'list']) == 0
    out = capsys.readouterr().out
    assert 'model = gpt-5-mini' in out
    assert 'api_key = ****e123' in out
    assert 'sk-supersecretvalue123' not in out


def test_list_show_secrets_reveals_keys(config_path: Any, capsys: Any) -> None:
    _write_config(config_path, {'api_key': 'sk-supersecretvalue123'})
    assert base.run(['config', 'list', '--show-secrets']) == 0
    assert 'api_key = sk-supersecretvalue123' in capsys.readouterr().out


def test_list_warns_on_unrecognized_keys(config_path: Any, capsys: Any) -> None:
    _write_config(config_path, {'modle': 'gpt-5', 'model': 'gpt-6'})
    assert base.run(['config', 'list']) == 0
    cap = capsys.readouterr()
    assert 'modle = gpt-5' in cap.out
    assert 'unrecognized' in cap.err
    assert 'modle' in cap.err
    # Recognized keys are not warned about.
    assert 'model' not in cap.err.split('modle')[-1]


def test_set_creates_file_and_directory(config_path: Any, capsys: Any) -> None:
    assert not os.path.exists(config_path)
    assert base.run(['config', 'set', 'model', 'gpt-5-mini']) == 0
    assert f'Set model = gpt-5-mini in {config_path}' in capsys.readouterr().out
    assert _read_config(config_path) == {'model': 'gpt-5-mini'}
    # The file can hold API keys, so it is written owner-only.
    assert stat.S_IMODE(os.stat(config_path).st_mode) == 0o600


def test_set_add_alias(config_path: Any) -> None:
    assert base.run(['config', 'add', 'model', 'gpt-5']) == 0
    assert _read_config(config_path) == {'model': 'gpt-5'}


def test_set_rejects_unknown_key(config_path: Any, capsys: Any) -> None:
    assert base.run(['config', 'set', 'bogus', 'x']) == 2
    cap = capsys.readouterr()
    assert 'Unknown config key: bogus' in cap.err
    assert 'Supported keys' in cap.err
    assert not os.path.exists(config_path)


def test_set_validates_provider_value(config_path: Any, capsys: Any) -> None:
    assert base.run(['config', 'set', 'provider', 'bogus']) == 2
    assert 'Unknown LLM provider: bogus' in capsys.readouterr().err
    assert base.run(['config', 'set', 'provider', 'anthropic']) == 0
    assert _read_config(config_path) == {'provider': 'anthropic'}


def test_set_preserves_other_and_unknown_keys(config_path: Any) -> None:
    _write_config(config_path, {'api_base': 'http://x/v1', 'modle': 'typo'})
    assert base.run(['config', 'set', 'model', 'gpt-5']) == 0
    assert _read_config(config_path) == {
        'api_base': 'http://x/v1', 'modle': 'typo', 'model': 'gpt-5'}


def test_set_masks_secret_in_confirmation(config_path: Any, capsys: Any) -> None:
    assert base.run(['config', 'set', 'api_key', 'sk-abcdefgh5678']) == 0
    out = capsys.readouterr().out
    assert 'Set api_key = ****5678' in out
    assert 'sk-abcdefgh5678' not in out


def test_set_on_corrupt_file_raises_instead_of_clobbering(config_path: Any) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write('not valid json{')
    with pytest.raises(Exception, match='Failed to read config file'):
        base.run(['config', 'set', 'model', 'gpt-5'])
    with open(config_path, 'r', encoding='utf-8') as f:
        assert f.read() == 'not valid json{'


def test_unset_removes_key(config_path: Any, capsys: Any) -> None:
    _write_config(config_path, {'model': 'gpt-5', 'api_base': 'http://x/v1'})
    assert base.run(['config', 'unset', 'model']) == 0
    assert f'Removed model from {config_path}' in capsys.readouterr().out
    assert _read_config(config_path) == {'api_base': 'http://x/v1'}


def test_unset_remove_alias(config_path: Any) -> None:
    _write_config(config_path, {'api_base': 'http://x/v1'})
    assert base.run(['config', 'remove', 'api_base']) == 0
    assert _read_config(config_path) == {}


def test_unset_absent_key_is_a_noop(config_path: Any, capsys: Any) -> None:
    assert base.run(['config', 'unset', 'model']) == 0
    assert 'model is not set' in capsys.readouterr().out


def test_unset_rejects_unknown_key(capsys: Any) -> None:
    assert base.run(['config', 'unset', 'bogus']) == 2
    assert 'Unknown config key: bogus' in capsys.readouterr().err
