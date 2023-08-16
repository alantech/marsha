import argparse
import asyncio
import os
import openai
import tempfile
import time
import traceback
import sys

from marsha.llm import gpt_can_func_python, gpt_improve_func, gpt_func_to_python, lint_and_fix_files, test_and_fix_files
from marsha.meta import MarshaMeta
from marsha.parse import write_files_from_markdown
from marsha.stats import stats
from marsha.utils import read_file, autoformat_files, copy_file, add_helper, copy_tree, prettify_time_delta

# Set up OpenAI
openai.organization = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_SECRET_KEY')

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

args = parser.parse_args()


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
            mds = await generate_python_code(meta, n_results, debug)
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
                review_and_fix(meta, file_group, debug), name=file_group[0]))
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


async def generate_python_code(meta: MarshaMeta, n_results: int, debug: bool) -> list[str]:
    t1 = time.time()
    print('Generating Python code...')
    mds = None
    try:
        if not args.exclude_sanity_check:
            if not await gpt_can_func_python(meta, n_results):
                await gpt_improve_func(meta)
                sys.exit(1)
        mds = await gpt_func_to_python(meta, n_results, debug=debug)
    except Exception as e:
        print('First stage failure')
        print(e)
        if debug:
            traceback.print_tb(e.__traceback__)
        print('Retrying...')
        raise e
    finally:
        t2 = time.time()
        stats.first_stage.total_time = prettify_time_delta(
            t2 - t1)
    return mds


async def review_and_fix(meta: MarshaMeta, files: list[str], debug: bool = False):
    t_ssi = time.time()
    print('Parsing generated code...')
    try:
        await lint_and_fix_files(meta.filename, files, debug=debug)
    except Exception as e:
        print('Second stage failure')
        print(e)
        raise e
    finally:
        t_ssii = time.time()
        stats.second_stage.total_time = prettify_time_delta(
            t_ssii - t_ssi)
    if args.debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
    t_tsi = time.time()
    print('Verifying and correcting generated code...')
    try:
        await test_and_fix_files(meta, files, debug=debug)
    except Exception as e:
        print('Third stage failure')
        print(e)
        raise e
    finally:
        t_tsii = time.time()
        stats.third_stage.total_time = prettify_time_delta(
            t_tsii - t_tsi)
    if args.debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
    print('Formatting code...')
    autoformat_files(files)
    if args.debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')


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
