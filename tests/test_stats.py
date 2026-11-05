"""Tests for per-invocation token-cost accounting and spend limits."""
from __future__ import annotations

import pytest
from collections.abc import Generator

from marsha import stats


@pytest.fixture(autouse=True)
def reset_usage() -> Generator[None, None, None]:
    stats.begin_invocation(None)
    yield
    stats.begin_invocation(None)


def test_recorded_usage_trips_cap_at_request_boundary() -> None:
    stats.begin_invocation(0.00005)
    stats.record_usage('gpt-6-luna', 1024, 0)

    assert stats.global_cost() == pytest.approx(0.00009765625)
    with pytest.raises(stats.PriceLimitExceeded, match='reached or exceeded'):
        stats.check_price_limit()


def test_cap_prevents_followup_request_after_threshold() -> None:
    stats.begin_invocation(0.00005)
    stats.record_usage('gpt-6-luna', 1024, 0)

    with pytest.raises(stats.PriceLimitExceeded, match='before another model request'):
        stats.ensure_price_limit('gpt-6-luna')


def test_cap_fails_closed_for_models_without_pricing() -> None:
    stats.begin_invocation(1.0)

    with pytest.raises(stats.PriceLimitExceeded, match='no token pricing'):
        stats.ensure_price_limit('custom-unknown-model')


def test_no_cap_keeps_unknown_models_usable() -> None:
    stats.begin_invocation(None)

    stats.ensure_price_limit('custom-unknown-model')
    stats.ensure_usage_available()


def test_cap_fails_closed_when_provider_omits_usage() -> None:
    stats.begin_invocation(1.0)

    with pytest.raises(stats.PriceLimitExceeded, match='returned no token usage'):
        stats.ensure_usage_available()


def test_invocation_reset_clears_usage_and_cap() -> None:
    stats.begin_invocation(1.0)
    stats.record_usage('gpt-6-luna', 1024, 2)

    stats.begin_invocation(None)

    assert stats.global_cost() == 0
    assert stats.global_tokens() == (0, 0)
    stats.ensure_price_limit('custom-unknown-model')
