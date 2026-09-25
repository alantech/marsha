from __future__ import annotations

import sys
import time

# Process start, so progress lines show elapsed time (matches `ps etime`) for spotting
# where a long run has stalled.
t0: float = time.time()

# Trace verbosity for the stderr channel. 0 = off, 2 = summary (--trace), 3 = full
# transcripts (--trace-full). Level 1 is reserved for the stdout `-d` debug, which is a
# separate code path and does not use this logger.
TRACE_OFF = 0
TRACE_SUMMARY = 2
TRACE_FULL = 3

_level: int = TRACE_OFF

# The per-LLM-call progress heartbeat ("Chat query took …"). On by default: for the long
# non-interactive runs (compile, review) it is the only live signal that a stage is still
# working between its stage banners. The interactive refine chat turns it off so it does not
# clobber the transcript; --trace always shows it.
_progress: bool = True


def set_level(level: int) -> None:
    global _level
    _level = int(level) if level else TRACE_OFF


def set_progress(enabled: bool) -> None:
    global _progress
    _progress = bool(enabled)


def set_enabled(value: bool) -> None:
    # Back-compat for earlier callers; maps on/off to the summary level.
    set_level(TRACE_SUMMARY if value else TRACE_OFF)


def _timestamp() -> str:
    elapsed = int(time.time() - t0)
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'


def _emit(message: str) -> None:
    # Write a timestamped line to stderr and flush immediately. stderr is unbuffered and
    # survives the block-buffering that hides stdout when it is piped to a file, so this is the
    # reliable channel for watching a run in real time.
    print(f'marsha[+{_timestamp()}] {message}', file=sys.stderr, flush=True)


def log(message: str) -> None:
    # A trace/debug line. No-op unless a trace level is set (--trace / --trace-full).
    if _level < TRACE_SUMMARY:
        return
    _emit(message)


def progress(message: str) -> None:
    # A per-LLM-call progress line (see _progress). On by default for the non-interactive runs;
    # suppressed in interactive modes unless --trace is on.
    if not (_progress or _level >= TRACE_SUMMARY):
        return
    _emit(message)


def dump(title: str, content: object) -> None:
    # Write the full transcript of one side of an LLM exchange (a request prompt or a response)
    # to stderr, bracketed by markers. Only enabled at the full trace level, since these are
    # large. A list (multiple completions) is joined with a blank line.
    if _level < TRACE_FULL:
        return
    if isinstance(content, (list, tuple)):
        text = '\n\n'.join(str(c) for c in content)
    else:
        text = '' if content is None else str(content)
    print(f'marsha[+{_timestamp()}] === {title} ({len(text)} chars) ===',
          file=sys.stderr, flush=True)
    print(text, file=sys.stderr, flush=True)
    print(f'marsha[+{_timestamp()}] === end {title} ===',
          file=sys.stderr, flush=True)
