import argparse
import asyncio
import sys
import tempfile
import time
import traceback

from marsha import backends
from marsha.config import (resolve_model, resolve_provider, resolve_api_base, is_local_backend,
                           set_cli_api_base, set_cli_model, set_cli_provider)
from marsha.context import discover_models
from marsha.model_match import apply_available_models
from marsha import log
from marsha.llm import generate_code, review_and_fix
from marsha.llm_client import create_client, set_client
from marsha.meta import MarshaMeta
from marsha.review import run_review
from marsha.stats import stats
from marsha.utils import read_file, copy_file, copy_tree, prettify_time_delta, write_composed

# Parse the input arguments. Subcommands hang off `marsha <command>`; the
# deprecated bare form `marsha <source> [flags]` is rewritten to `compile`
# in run() (see _normalize_argv).
parser = argparse.ArgumentParser(
    prog='marsha',
    description='Marsha AI Compiler',
)
sub = parser.add_subparsers(
    dest='command', title='commands', metavar='COMMAND')
compile_parser = sub.add_parser(
    'compile',
    help='Compile a .mrsh definition into generated code and a test suite.')
compile_parser.add_argument('source')
compile_parser.add_argument('-t', '--target', default='python',
                            help='Target language for the generated code, by backend id or alias (default: python). Only `python` is wired today; the registry is ready for more.')
compile_parser.add_argument('--target-version',
                            help='Version of the target language the generated code should target, eg 3.12 for Python (where it becomes the project requires-python). Default: the interpreter running Marsha.')
compile_parser.add_argument('-d', '--debug', action='store_true',
                            help='Turn on debug logging')
compile_parser.add_argument('--trace', action='store_true',
                            help='Also write a live, timestamped progress trace to stderr (each phase and every LLM request, with its label and duration). Implies -d. Useful for watching a slow run in real time, e.g. against a local llama.cpp server.')
compile_parser.add_argument('--trace-full', action='store_true',
                            help='As --trace, but also dump the full input prompt and output of every LLM call to stderr. Implies --trace (and -d). Use for debugging exact prompts and responses.')
compile_parser.add_argument('-q', '--quick-and-dirty', action='store_true',
                            help='Code generation with no correction stages run')
compile_parser.add_argument('-a', '--attempts', type=int, default=1)
compile_parser.add_argument(
    '-n', '--n-parallel-executions', type=int, default=3)
compile_parser.add_argument('--exclude-main-helper', action='store_true',
                            help='Skips addition of helper code for running as a script')
compile_parser.add_argument('--exclude-sanity-check', action='store_true',
                            help='Skips an initial sanity check that the definition is self-consistent')
compile_parser.add_argument('--no-tools', action='store_true',
                            help='Disable the LLM tool interface (the fake terminal where the LLM can look up dependency APIs and the web with $ commands: search-dependencies, dependency-docs, web-search, view-web-page, calc, and — in the optimize/correction loops — installed-environment introspection). Enabled by default.')
compile_parser.add_argument('--no-warn', action='store_true',
                            help='Do not display warnings about ambiguous areas of the definition from the sanity check')
compile_parser.add_argument('--optimize', type=int, default=0,
                            help='Optimization level: number of per-phase LLM review iterations (test-suite coverage/fidelity, implementation quality, and test-correction validation). 0 (default) disables the optimization loops.')
compile_parser.add_argument('--test-personas',
                            help='Comma-separated reviewer personas for the test-suite (oracle) loop. Each entry is a built-in name (e.g. ada) or a path to a custom persona file (e.g. ./sharona.md). Default: all built-in oracle reviewers.')
compile_parser.add_argument('--impl-personas',
                            help='Comma-separated reviewer personas for the implementation loop. Each entry is a built-in name (e.g. sage) or a path to a custom persona file. Default: all built-in impl reviewers.')
compile_parser.add_argument('--fix-personas',
                            help='Comma-separated reviewer personas for the test-correction (oracle-fix) loop. Each entry is a built-in name (e.g. sol) or a path to a custom persona file. Default: all built-in correction reviewers.')
compile_parser.add_argument('--optimize-severity', default='major,minor,nit',
                            help='Comma-separated finding severities to act on during --optimize (major,minor,nit). Default: all three.')
compile_parser.add_argument('--context-window', type=int, default=None,
                            help='Override the context window (in tokens) used to size review/editor prompts. Auto-detected from the service when possible, else documented defaults. Set it if your backend mis-reports its window.')
compile_parser.add_argument('--context-cap', type=float, default=0.5,
                            help='Fraction of the context window a single prompt may occupy before its findings are compacted (default 0.5).')
compile_parser.add_argument('-s', '--stats', action='store_true',
                            help='Save stats and write them to a file')
compile_parser.add_argument('--api-base',
                            help='Base URL of an OpenAI-compatible API to use for LLM requests, e.g. a local llama.cpp server. Overrides the OPENAI_BASE_URL environment variable and the config file (openai provider only)')
compile_parser.add_argument('--model',
                            help='Model to use for code generation, overriding the model in the config file')
compile_parser.add_argument('--provider',
                            choices=['openai', 'anthropic'],
                            help='LLM provider to use: openai (default; any OpenAI-compatible API) or anthropic (Claude)')

help_parser = sub.add_parser(
    'help',
    help='Show this overview, or detailed help for a subcommand.')
help_parser.add_argument('topic', nargs='?', default=None,
                         help='Subcommand to describe (e.g. compile). '
                              'Omit for the overview.')
review_parser = sub.add_parser(
    'review',
    help='Review a branch diff against the default branch with the review personas.')
review_parser.add_argument('--pr', type=int, default=None,
                           help='GitHub PR number to review (needs the gh CLI). '
                                'Checks the PR out; the working tree must be clean.')
review_parser.add_argument('--linear', default=None,
                           help='Linear ticket name whose requirements seed the review '
                                '(needs the linear CLI).')
review_parser.add_argument('--post-review', action='store_true',
                           help='Post the findings to the PR as an inline review '
                                '(requires --pr).')
review_parser.add_argument('--review-rounds', type=int, default=1,
                           help='Conventions-feedback rounds after the initial review: '
                                'the conventions gate rebuts findings that violate a real '
                                'convention, and the panel revises with the rebuttal. '
                                '0 disables the gate. Default: 1.')
review_parser.add_argument('--personas', default=None,
                           help='Comma-separated review personas (a built-in name or a '
                                'path). Default: all built-in impl reviewers.')
review_parser.add_argument('--target', default='python',
                           help='Target language, for reviewer guidance (default: python). '
                                'Its toolchain need not be installed.')
review_parser.add_argument('--target-version', default=None,
                           help='Target language version (for guidance).')
review_parser.add_argument('-d', '--debug', action='store_true',
                           help='Turn on debug logging')
review_parser.add_argument('--trace', action='store_true',
                           help='Live progress trace to stderr. Implies -d.')
review_parser.add_argument('--trace-full', action='store_true',
                           help='As --trace, but also dump full prompts/responses.')
review_parser.add_argument('--api-base',
                           help='OpenAI-compatible API base URL for the LLM.')
review_parser.add_argument('--model',
                           help='Model to use for the review.')
review_parser.add_argument('--provider', choices=['openai', 'anthropic'],
                           help='LLM provider: openai (default) or anthropic (Claude).')


def _normalize_argv(argv):
    # The deprecated bare form `marsha <source> [flags]` maps onto
    # `marsha compile <source> [flags]`. `compile`/`help` are real
    # subcommands; a top-level -h/--help shows the subcommand overview.
    if not argv or argv[0] in ('compile', 'help', 'review', '-h', '--help'):
        return argv, False
    return ['compile'] + argv, True


def print_help(topic):
    overview = (
        'Marsha AI Compiler\n'
        '\n'
        'Usage: marsha COMMAND [ARGS]\n'
        '\n'
        'Commands:\n'
        '  compile   Compile a .mrsh definition into generated code and a\n'
        '            test suite. Example: marsha compile your.mrsh\n'
        '            Key flags: -t/--target, -a/--attempts, --optimize,\n'
        '            --model, --provider, --api-base, -d/--debug, --trace.\n'
        '\n'
        '  review    Review a branch diff against the default branch with the\n'
        '            review personas. Example: marsha review --pr 123\n'
        '            Key flags: --pr, --linear, --post-review, --personas.\n'
        '\n'
        '  help      Show this overview, or detailed help for a subcommand.\n'
        '            Example: marsha help compile\n'
        '\n'
        'Run `marsha help <command>` or `marsha <command> --help` for the full\n'
        'options. Note: `marsha <source.mrsh>` (no subcommand) is a deprecated\n'
        'alias for `marsha compile <source.mrsh>`.\n'
    )
    if topic is None:
        print(overview)
    elif topic in ('compile', 'help', 'review'):
        submap = {'compile': compile_parser, 'help': help_parser,
                  'review': review_parser}
        print(submap[topic].format_help())
    else:
        print(f'Unknown command: {topic}', file=sys.stderr)
        print(overview)


def run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv, is_legacy = _normalize_argv(argv)
    if is_legacy:
        print('Warning: bare `marsha <source>` is deprecated; use '
              '`marsha compile <source>` instead. See `marsha help`.',
              file=sys.stderr)
    args = parser.parse_args(argv)
    if args.command == 'help':
        print_help(args.topic)
        return 0
    if args.command == 'review':
        _setup_runtime(args, require_toolchain=False)
        return asyncio.run(run_review(args))
    if args.command != 'compile':
        parser.print_help(sys.stderr)
        return 2
    _setup_runtime(args)
    asyncio.run(main(args))
    return 0


def _setup_runtime(args, require_toolchain=True):
    # --trace routes a live, flushed progress trace to stderr so a run can
    # be watched in real time, even when stdout is piped to a file. --trace-full
    # adds full request/response transcripts on top of the summary.
    if args.trace_full:
        log.set_level(log.TRACE_FULL)
    elif args.trace:
        log.set_level(log.TRACE_SUMMARY)
    else:
        log.set_level(log.TRACE_OFF)
    # Set up the shared LLM client
    set_cli_model(args.model)
    set_cli_provider(args.provider)
    set_cli_api_base(args.api_base)
    client = create_client(args.api_base)
    set_client(client)
    # On a local/OpenAI-compatible server the requested model name is ignored
    # and whatever is loaded is served. Remap the standard/strong models to the
    # closest match served for each role (smallest-fitting vs. most capable, by
    # context size with price as a tiebreaker) so marsha logs and sends the
    # models that will really be used. Explicitly pinned models are left alone.
    if is_local_backend():
        for note in apply_available_models(discover_models(resolve_api_base())):
            print(f'Note: {note}')
    # Bind the target-language backend (--target) and verify its toolchain.
    target = backends.select(args.target)
    if require_toolchain and not target.toolchain_ok():
        raise Exception(f'{args.target} toolchain not found')
    # Resolve the target version (--target-version) against the backend's rules.
    if args.target_version is not None:
        target.target_version = target.resolve_target_version(
            args.target_version)
    if args.debug or args.trace or args.trace_full:
        print(f'Using target language: {target.id}')
        print(f'Using target version: {target.target_version}')
        print(f'Using LLM provider: {resolve_provider()}')
        print(f'Using LLM endpoint: {client.base_url}')
        print(f'Using LLM model: {resolve_model()}')


async def main(args):
    t1 = time.time()
    input_file = args.source
    # Name without extension
    meta = await MarshaMeta(input_file).populate()
    print(f'Compiling functions for {meta.filename}...')
    quick_and_dirty = args.quick_and_dirty
    # --trace (and --trace-full) imply -d, so they get both the stdout debug detail and the stderr trace.
    debug = args.debug or args.trace or args.trace_full
    should_write_stats = args.stats
    attempts = args.attempts
    n_results = args.n_parallel_executions
    if debug:
        print(f'Number of attempts: {attempts}')
        print(f'Number of parallel executions: {n_results}')
    backend = backends.current()
    while attempts:
        attempts = attempts - 1
        # First stage: generate code for functions and classes
        try:
            cands = await generate_code(args, meta, n_results, debug)
        except Exception:
            continue
        # Early exit if quick and dirty
        if quick_and_dirty:
            print('Writing generated code to files...')
            for impl, oracle in cands[:2]:
                write_composed(backend.compose(impl, oracle))
            attempts = attempts + 1
            break
        # Writing generated code to temporary files in preparation for next stages
        file_groups = list()
        tmp_directories = []
        for idx, (impl, oracle) in enumerate(cands):
            print('Writing generated code to temporary files...')
            tmpdir = tempfile.TemporaryDirectory(
                suffix=f'_-_{meta.filename}_{idx}')
            tmp_directories.append(tmpdir)
            file_groups = file_groups + \
                [write_composed(
                    backend.compose(impl, oracle), subdir=tmpdir.name)]
        if debug:
            for filename in [filename for file_group in file_groups for filename in file_group]:
                print(f'# {filename}\n{read_file(filename)}\n')
        # Create tasks to run in parallel using asyncio
        tasks = []
        for file_group in file_groups:
            tasks.append(asyncio.create_task(
                review_and_fix(args, meta, file_group, debug), name=file_group[0]))
        try:
            done_task_name = await run_parallel_tasks(tasks)
            print('Writing generated code to files...')
            group = [g for g in file_groups if g[0] == done_task_name][0]
            source_dest = backend.source_name(meta.filename)
            copy_file(done_task_name, source_dest)
            if not args.exclude_main_helper:
                backend.make_executable(source_dest)
            test_file = [f for f in group if f.endswith(
                backend.test_name(meta.filename))][0]
            copy_file(test_file, backend.test_name(meta.filename))
            manifests = [f for f in group if f.endswith(
                backend.manifest_name())]
            if len(manifests) > 0:
                copy_file(manifests[0], backend.manifest_name())
        except Exception as e:
            print('Failed to generate working code.')
            print(e)
            if debug:
                traceback.print_tb(e.__traceback__)
                # Copy the temporary directories to a new directory for debugging
                for tmpdir in tmp_directories:
                    tmpdir_suffix = tmpdir.name.split('_-_')[-1]
                    copy_tree(tmpdir.name, f'{tmpdir_suffix}_failed')
            print('Retrying...')
            continue
        finally:
            cleanup_tmp_directories(tmp_directories)
        # Done! Add one back to `attempts` to avoid accidentally erroring out on success
        attempts = attempts + 1
        break
    if attempts == 0:
        t2 = time.time()
        stats.aggregate(prettify_time_delta(t2 - t1), args.attempts)
        if should_write_stats:
            stats.to_file()
        raise Exception(
            f'Failed to generate working code for {meta.filename}. Total time elapsed: {prettify_time_delta(t2 - t1)}. Total cost: {round(stats.total_cost, 2)}.')
    t2 = time.time()
    stats.aggregate(prettify_time_delta(t2 - t1), args.attempts - attempts + 1)
    if should_write_stats:
        stats.to_file()
    print(
        f'{meta.filename} done! Total time elapsed: {prettify_time_delta(t2 - t1)}. Total cost: {round(stats.total_cost, 2)}.')


async def run_parallel_tasks(tasks: list) -> str:
    print('Running tasks in parallel...')
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    done_task = done.pop()
    if done_task.exception() is None:
        print('Task completed successfully. Cancelling pending tasks...')
        for task in pending if pending is not None else []:
            task.cancel()
        return done_task.get_name()
    elif len(pending) > 0:
        print('Task completed with error. Waiting for pending tasks to finish...')
        return await run_parallel_tasks(pending)
    else:
        print('All tasks failed. Raising exception...')
        if done_task is not None and done_task.exception() is not None:
            raise done_task.exception()
        raise Exception('All tasks failed.')


def cleanup_tmp_directories(tmp_directories: list):
    for tmp_directory in tmp_directories:
        try:
            tmp_directory.cleanup()
        except Exception:
            pass
