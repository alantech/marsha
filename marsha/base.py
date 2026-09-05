import argparse
import asyncio
import os
import tempfile
import time
import traceback

from marsha.config import resolve_model, set_cli_model
from marsha.llm import generate_python_code, review_and_fix
from marsha.llm_client import create_client, set_client
from marsha.meta import MarshaMeta
from marsha.parse import write_files_from_markdown
from marsha.stats import stats
from marsha.utils import read_file, copy_file, add_helper, copy_tree, prettify_time_delta

# Parse the input arguments
parser = argparse.ArgumentParser(
    prog='marsha',
    description='Marsha AI Compiler',
)
parser.add_argument('source')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Turn on debug logging')
parser.add_argument('-q', '--quick-and-dirty', action='store_true',
                    help='Code generation with no correction stages run')
parser.add_argument('-a', '--attempts', type=int, default=1)
parser.add_argument('-n', '--n-parallel-executions', type=int, default=3)
parser.add_argument('--exclude-main-helper', action='store_true',
                    help='Skips addition of helper code for running as a script')
parser.add_argument('--exclude-sanity-check', action='store_true',
                    help='Skips an initial sanity check that function defintions will reliably generate working code')
parser.add_argument('-s', '--stats', action='store_true',
                    help='Save stats and write them to a file')
parser.add_argument('--api-base',
                    help='Base URL of an OpenAI-compatible API to use for LLM requests, e.g. a local llama.cpp server. Overrides the OPENAI_BASE_URL environment variable and the config file')
parser.add_argument('--model',
                    help='Model to use for code generation, overriding the model in the config file')

args = parser.parse_args()

# Set up the shared LLM client
set_cli_model(args.model)
client = create_client(args.api_base)
set_client(client)
if args.debug:
    print(f'Using LLM endpoint: {client.base_url}')
    print(f'Using LLM model: {resolve_model()}')


async def main():
    t1 = time.time()
    input_file = args.source
    # Name without extension
    meta = await MarshaMeta(input_file).populate()
    print(f'Compiling functions for {meta.filename}...')
    quick_and_dirty = args.quick_and_dirty
    debug = args.debug
    should_write_stats = args.stats
    attempts = args.attempts
    n_results = args.n_parallel_executions
    if args.debug:
        print(f'Number of attempts: {attempts}')
        print(f'Number of parallel executions: {n_results}')
    while attempts:
        attempts = attempts - 1
        # First stage: generate code for functions and classes
        try:
            mds = await generate_python_code(args, meta, n_results, debug)
        except Exception:
            continue
        # Early exit if quick and dirty
        if quick_and_dirty:
            print('Writing generated code to files...')
            for md in mds[:2]:
                write_files_from_markdown(md)
            attempts = attempts + 1
            break
        # Writing generated code to temporary files in preparation for next stages
        file_groups = list()
        tmp_directories = []
        for idx, md in enumerate(mds):
            print('Writing generated code to temporary files...')
            tmpdir = tempfile.TemporaryDirectory(
                suffix=f'_-_{meta.filename}_{idx}')
            tmp_directories.append(tmpdir)
            file_groups = file_groups + \
                [write_files_from_markdown(
                    md, subdir=tmpdir.name)]
        if args.debug:
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
            filename = done_task_name
            copy_file(filename, f'{meta.filename}.py')
            if not args.exclude_main_helper:
                add_helper(f'{meta.filename}.py')
            test_filename = filename.replace('.py', '_test.py')
            copy_file(test_filename, f'{meta.filename}_test.py')
            directory = os.path.dirname(filename)
            requirements_filename = os.path.join(
                directory, 'requirements.txt')
            if os.path.exists(requirements_filename):
                copy_file(requirements_filename, 'requirements.txt')
        except Exception as e:
            print('Failed to generate working code.')
            print(e)
            if args.debug:
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
