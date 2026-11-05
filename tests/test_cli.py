"""Deterministic tests for the subcommand CLI in marsha.base.

These exercise the argv normalization (the deprecated bare form mapping onto
`compile`), the `help` subcommand, and the `run()` dispatch — none of which
need an LLM or network I/O (the compile path's runtime setup and main are
mocked where a dispatch is asserted).
"""

import signal
from typing import Any

import pytest

import marsha.base as base


def test_normalize_argv_maps_bare_form_to_compile() -> None:
    assert base._normalize_argv(['x.mrsh', '-t', 'python']) == \
        (['compile', 'x.mrsh', '-t', 'python'], True)
    # Flags may precede the positional in the bare form; both route to compile.
    assert base._normalize_argv(['-t', 'python', 'x.mrsh']) == \
        (['compile', '-t', 'python', 'x.mrsh'], True)


def test_normalize_argv_leaves_subcommands_and_help_alone() -> None:
    assert base._normalize_argv(['compile', 'x.mrsh']) == ([
        'compile', 'x.mrsh'], False)
    assert base._normalize_argv(['help']) == (['help'], False)
    assert base._normalize_argv(['help', 'compile']) == ([
        'help', 'compile'], False)
    assert base._normalize_argv(['-h']) == (['-h'], False)
    assert base._normalize_argv(['--help']) == (['--help'], False)
    assert base._normalize_argv([]) == ([], False)


def test_print_help_overview_lists_subcommands(capsys: Any) -> None:
    base.print_help(None)
    out = capsys.readouterr().out
    assert 'compile' in out
    assert 'help' in out
    assert 'deprecated' in out


def test_print_help_compile_shows_compile_usage(capsys: Any) -> None:
    base.print_help('compile')
    out = capsys.readouterr().out
    assert 'usage: marsha compile' in out
    assert '--optimize' in out
    assert '--max-price USD' in out


def test_max_price_is_available_on_every_model_subcommand() -> None:
    argv = {
        'compile': ['compile', 'spec.mrsh', '--max-price', '1.25'],
        'review': ['review', '--max-price', '1.25'],
        'refine': ['refine', 'spec.mrsh', '--max-price', '1.25'],
        'diff': ['diff', 'spec.mrsh', '--max-price', '1.25'],
    }
    for command, command_argv in argv.items():
        args = base.parser.parse_args(command_argv)
        assert args.command == command
        assert args.max_price == 1.25


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'not-a-price'])
def test_max_price_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(SystemExit):
        base.parser.parse_args(['review', '--max-price', value])


def test_diff_default_max_tool_failure_is_ten() -> None:
    args = base.parser.parse_args(['diff', 'spec.mrsh'])
    assert args.max_tool_failure == 10


def test_print_help_unknown_topic_reports_and_lists(capsys: Any) -> None:
    base.print_help('bogus')
    cap = capsys.readouterr()
    assert 'Unknown command: bogus' in cap.err
    assert 'compile' in cap.out


def test_run_help_returns_zero(capsys: Any) -> None:
    assert base.run(['help']) == 0
    assert 'compile' in capsys.readouterr().out


def test_run_help_topic_returns_zero(capsys: Any) -> None:
    assert base.run(['help', 'compile']) == 0
    assert 'usage: marsha compile' in capsys.readouterr().out


def test_run_reports_price_limit_and_returns_failure(
        monkeypatch: Any, capsys: Any) -> None:
    from marsha.stats import PriceLimitExceeded

    monkeypatch.setattr(base, '_setup_runtime', lambda *_args, **_kwargs: None)

    async def hit_cap(_args: Any) -> int:
        raise PriceLimitExceeded('test cap reached')

    monkeypatch.setattr(base, 'run_review', hit_cap)
    assert base.run(['review', '--max-price', '1']) == 1
    output = capsys.readouterr().err
    assert 'test cap reached' in output
    assert 'Total cost:' in output


def test_run_no_command_prints_help_and_returns_two(capsys: Any) -> None:
    assert base.run([]) == 2
    assert 'compile' in capsys.readouterr().err


def test_run_dispatches_compile(monkeypatch: Any) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(base, '_setup_runtime',
                        lambda args: calls.append('setup'))

    async def fake_main(args: Any) -> None:
        calls.append(('main', args.source))

    monkeypatch.setattr(base, 'main', fake_main)
    assert base.run(['compile', 'x.mrsh']) == 0
    assert calls == ['setup', ('main', 'x.mrsh')]


def test_run_bare_alias_warns_and_still_compiles(monkeypatch: Any, capsys: Any) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(base, '_setup_runtime',
                        lambda args: calls.append('setup'))

    async def fake_main(args: Any) -> None:
        calls.append(('main', args.source))

    monkeypatch.setattr(base, 'main', fake_main)
    assert base.run(['x.mrsh', '-t', 'python']) == 0
    assert 'deprecated' in capsys.readouterr().err
    assert calls == ['setup', ('main', 'x.mrsh')]


def test_sigint_handler_raises_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        base._sigint_handler(signal.SIGINT, None)


def test_install_sigint_handler_sets_handler() -> None:
    old = signal.getsignal(signal.SIGINT)
    try:
        base.install_sigint_handler()
        assert signal.getsignal(signal.SIGINT) is base._sigint_handler
    finally:
        signal.signal(signal.SIGINT, old)
