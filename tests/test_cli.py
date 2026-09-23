"""Deterministic tests for the subcommand CLI in marsha.base.

These exercise the argv normalization (the deprecated bare form mapping onto
`compile`), the `help` subcommand, and the `run()` dispatch — none of which
need an LLM or network I/O (the compile path's runtime setup and main are
mocked where a dispatch is asserted).
"""

from typing import Any

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
