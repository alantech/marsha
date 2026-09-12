"""Tests for role-driven model auto-matching (closest-match selection for
OpenAI-compatible backends, issue #202). No network: discovery results are passed
in directly and the config file is stubbed, so the ranker, the per-role remap and
the pinning rules are verified deterministically.
"""

from unittest.mock import patch

import pytest

import marsha.config as cfg
from marsha import model_match


def _m(name, ctx=None):
    return {'id': name, 'context': ctx}


@pytest.fixture(autouse=True)
def _isolate_config():
    # No CLI overrides and no config file, so the default (unpinned) models resolve.
    cfg.set_cli_model(None)
    cfg.set_cli_strong_model(None)
    with patch.object(cfg, 'load_config_file', return_value={}):
        yield
    cfg.set_cli_model(None)
    cfg.set_cli_strong_model(None)


# --- the per-role ranker ----------------------------------------------------

def _profile(**overrides):
    profile = dict(model_match.ROLE_PROFILES['model'])
    profile.update(overrides)
    return profile


def test_rank_standard_smallest_fitting():
    # The target follows the configured model: gpt-5-mini documents a 400k window, so the
    # standard role wants a model at least as capable.
    models = [_m('small', 131072), _m('mid', 400000), _m('big', 1048576)]
    assert model_match.rank_models(
        models, 'model', current='gpt-5-mini') == 'mid'


def test_rank_standard_target_follows_configured_model():
    models = [_m('small', 200000), _m('big', 400000)]
    # claude-sonnet-5 documents 200k, so the 200k model fits; gpt-5-mini documents 400k,
    # so only the 400k model does.
    assert model_match.rank_models(
        models, 'model', current='claude-sonnet-5') == 'small'
    assert model_match.rank_models(
        models, 'model', current='gpt-5-mini') == 'big'


def test_rank_standard_no_current_defaults_to_default_window():
    models = [_m('small', 200000), _m('big', 400000)]
    assert model_match.rank_models(models, 'model') == 'small'


def test_rank_standard_explicit_target_overrides():
    models = [_m('tiny', 8192), _m('mid', 131072), _m('big', 262144)]
    assert model_match.rank_models(
        models, 'model', profile=_profile(target_context=65536)) == 'mid'


def test_rank_standard_exact_target_fits():
    models = [_m('just-below', 65535), _m('exact', 65536)]
    assert model_match.rank_models(
        models, 'model', profile=_profile(target_context=65536)) == 'exact'


def test_rank_standard_best_effort_largest_when_none_fit():
    models = [_m('tiny', 8192), _m('mid', 32768)]
    assert model_match.rank_models(
        models, 'model', current='gpt-5-mini') == 'mid'


def test_rank_standard_unknown_context_sorts_last():
    models = [_m('mystery'), _m('mid', 400000)]
    assert model_match.rank_models(
        models, 'model', current='gpt-5-mini') == 'mid'


def test_rank_strong_largest():
    models = [_m('tiny', 8192), _m('mid', 131072), _m('big', 262144)]
    assert model_match.rank_models(models, 'model_strong') == 'big'


def test_rank_strong_unknown_context_sorts_last():
    models = [_m('mystery'), _m('big', 262144)]
    assert model_match.rank_models(models, 'model_strong') == 'big'


def test_rank_no_signal_falls_back_to_list_order():
    models = [_m('first'), _m('second')]
    assert model_match.rank_models(models, 'model') == 'first'
    assert model_match.rank_models(models, 'model_strong') == 'first'


def test_rank_price_breaks_context_ties():
    # Both clear the 400k target with the same context; the known-cheaper one wins.
    models = [_m('gpt-5', 400000), _m('gpt-5-mini', 400000)]
    assert model_match.rank_models(
        models, 'model', current='gpt-5-mini') == 'gpt-5-mini'
    assert model_match.rank_models(
        models, 'model_strong', current='gpt-5') == 'gpt-5-mini'


def test_rank_unknown_price_never_beats_known():
    models = [_m('mystery', 131072), _m('gpt-5-mini', 131072)]
    assert model_match.rank_models(
        models, 'model', profile=_profile(target_context=65536)) == 'gpt-5-mini'


def test_rank_price_weight_zero_ignores_price():
    models = [_m('gpt-5', 131072), _m('gpt-5-mini', 131072)]
    profile = _profile(target_context=65536, price_weight=0)
    assert model_match.rank_models(models, 'model', profile=profile) == 'gpt-5'


def test_rank_profile_override_drives_role():
    models = [_m('tiny', 8192), _m('big', 262144)]
    strong = dict(model_match.ROLE_PROFILES['model_strong'])
    assert model_match.rank_models(models, 'model', profile=strong) == 'big'


def test_rank_empty_or_unknown_role():
    assert model_match.rank_models([], 'model') is None
    assert model_match.rank_models([_m('a')], 'no_such_role') is None


# --- per-role remap against a multi-model backend ---------------------------

def test_apply_remaps_each_role_to_its_closest_match():
    # gpt-5-mini documents a 400k window, so the standard role wants >= 400k: 'medium' is
    # the smallest that fits; the strong role takes the largest.
    models = [_m('small', 131072), _m('medium', 400000), _m('large', 1048576)]
    notes = model_match.apply_available_models(models)
    assert cfg.resolve_model() == 'medium'
    assert cfg.resolve_strong_model() == 'large'
    assert len(notes) == 2
    assert 'medium' in notes[0] and 'large' in notes[1]


def test_apply_no_fitting_model_both_roles_take_largest():
    models = [_m('small', 8192), _m('medium', 131072)]
    model_match.apply_available_models(models)
    assert cfg.resolve_model() == 'medium'
    assert cfg.resolve_strong_model() == 'medium'


def test_apply_single_model_remapped_to_it():
    notes = model_match.apply_available_models([_m('_probe-model', 8192)])
    assert cfg.resolve_model() == '_probe-model'
    assert cfg.resolve_strong_model() == '_probe-model'
    assert len(notes) == 2
    assert all('_probe-model' in n for n in notes)


def test_apply_keeps_models_that_are_served():
    current = [cfg.resolve_model(), cfg.resolve_strong_model()]
    models = [_m(name, 131072) for name in current]
    notes = model_match.apply_available_models(models)
    assert cfg.resolve_model() == current[0]
    assert cfg.resolve_strong_model() == current[1]
    assert notes == []


def test_apply_keeps_served_standard_and_ranks_strong():
    current = cfg.resolve_model()
    models = [_m(current, 131072), _m('monster', 524288)]
    notes = model_match.apply_available_models(models)
    assert cfg.resolve_model() == current
    assert cfg.resolve_strong_model() == 'monster'
    assert len(notes) == 1
    assert 'monster' in notes[0]


def test_apply_empty_or_none_is_a_noop():
    before = [cfg.resolve_model(), cfg.resolve_strong_model()]
    assert model_match.apply_available_models([]) == []
    assert model_match.apply_available_models(None) == []
    assert [cfg.resolve_model(), cfg.resolve_strong_model()] == before


# --- pinning: an explicit choice is never remapped ---------------------------

def test_apply_pinned_model_is_kept_and_flagged():
    cfg.set_cli_model('my-pinned')
    notes = model_match.apply_available_models([_m('other', 8192)])
    assert cfg.resolve_model() == 'my-pinned'
    # The strong role is not pinned, so it is still remapped.
    assert cfg.resolve_strong_model() == 'other'
    assert len(notes) == 2
    assert 'pinned' in notes[0]


def test_apply_pinned_models_in_list_are_silent():
    cfg.set_cli_model('served')
    cfg.set_cli_strong_model('served')
    notes = model_match.apply_available_models([_m('served', 8192)])
    assert cfg.resolve_model() == 'served'
    assert cfg.resolve_strong_model() == 'served'
    assert notes == []


def test_apply_pinned_via_config_file_is_kept():
    with patch.object(cfg, 'load_config_file',
                      return_value={'model': 'file-pinned'}):
        notes = model_match.apply_available_models([_m('other', 8192)])
        assert cfg.resolve_model() == 'file-pinned'
        assert cfg.resolve_strong_model() == 'other'
        assert any('pinned' in n for n in notes)
