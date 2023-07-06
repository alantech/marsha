import argparse
import asyncio
import os
import openai
import time
import traceback

from llm import gpt_func_to_python, lint_and_fix_files, test_and_fix_files, prettify_time_delta
from parse import extract_functions_and_types, extract_type_name, write_files_from_markdown, is_defined_from_file, extract_type_filename
from utils import read_file, write_file, autoformat_files, copy_file, delete_dir_and_content, get_filename_from_path, get_file_fullpath

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
parser.add_argument('-s', '--stats', action='store_true',
                    help='Save stats and write them to a file')

args = parser.parse_args()

# TODO: make this a class?
# Stats for the run of the compiler
stats = {
    'class_generation': {
        'total_time': 0,
        'total_calls': 0,
        'gpt-3.5-turbo': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
        'gpt-4': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
    },
    'first_stage': {
        'total_time': 0,
        'total_calls': 0,
        'gpt-3.5-turbo': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
        'gpt-4': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
    },
    'second_stage': {
        'total_time': 0,
        'total_calls': 0,
        'gpt-3.5-turbo': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
        'gpt-4': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
    },
    'third_stage': {
        'total_time': 0,
        'total_calls': 0,
        'gpt-3.5-turbo': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
        'gpt-4': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0,
        },
    },
    'total_time': 0,
    'total_calls': 0,
    'attempts': 0,
    'total_cost': 0,
}


async def main():
    t1 = time.time()
    input_file = args.source
    # Name without extension
    marsha_filename = get_filename_from_path(input_file)
    marsha_file_content = read_file(input_file)
    functions, types = extract_functions_and_types(marsha_file_content)
    types_defined = None
    # Pre-process types in case we need to open a file to get the type definition
    if len(types) > 0:
        types_defined = await process_types(types)
    print(f'Compiling functions for {marsha_filename}...')
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
            mds = await generate_python_code(
                marsha_filename, functions, types_defined, n_results, debug, stats)
        except Exception as e:
            continue
        # Early exit if quick and dirty
        if quick_and_dirty:
            print('Writing generated code to files...')
            for md in mds[:2]:
                write_files_from_markdown(md)
            attempts = attempts + 1
            break
        # Writing generated code to temporal files in preparation for next stages
        file_groups = list()
        for idx, md in enumerate(mds):
            print('Writing generated code to temporary files...')
            file_groups = file_groups + \
                [write_files_from_markdown(
                    md, subdir=f'{marsha_filename}_{idx}')]
        if args.debug:
            for filename in [filename for file_group in file_groups for filename in file_group]:
                print(f'# {filename}\n{read_file(filename)}\n')
        # Create tasks to run in parallel using asyncio
        tasks = []
        for file_group in file_groups:
            tasks.append(asyncio.create_task(
                review_and_fix(marsha_filename, file_group, functions, stats, debug), name=file_group[0]))
        task_names = [task.get_name() for task in tasks]
        try:
            done_task_name = await run_parallel_tasks(tasks)
            for name in task_names:
                if name != done_task_name:
                    delete_dir_and_content(name)
                else:
                    print('Writing generated code to files...')
                    filename = name
                    test_filename = filename.replace('.py', '_test.py')
                    copy_file(filename, f'{marsha_filename}.py')
                    copy_file(test_filename, f'{marsha_filename}_test.py')
                    delete_dir_and_content(filename)
        except Exception as e:
            print('Failed to generate working code.')
            print(e)
            if not args.debug:
                for name in task_names:
                    delete_dir_and_content(name)
            else:
                traceback.print_tb(e.__traceback__)
            print('Retrying...')
            continue
        # Done! Add one back to `attempts` to avoid accidentally erroring out on success
        attempts = attempts + 1
        break
    if attempts == 0:
        t2 = time.time()
        stats['total_time'] = prettify_time_delta(t2 - t1)
        stats['attempts'] = args.attempts
        stats['total_calls'] = stats['first_stage']['total_calls'] + \
            stats['second_stage']['total_calls'] + \
            stats['third_stage']['total_calls'] + \
            stats['class_generation']['total_calls']
        stats['total_cost'] = stats['first_stage']['gpt-3.5-turbo']['total_cost'] + stats['first_stage']['gpt-4']['total_cost'] + \
            stats['second_stage']['gpt-3.5-turbo']['total_cost'] + stats['second_stage']['gpt-4']['total_cost'] + \
            stats['third_stage']['gpt-3.5-turbo']['total_cost'] + stats['third_stage']['gpt-4']['total_cost'] + \
            stats['class_generation']['gpt-3.5-turbo']['total_cost'] + \
            stats['class_generation']['gpt-4']['total_cost']
        if should_write_stats:
            stats_to_file(stats)
        raise Exception(
            f'Failed to generate working code for {marsha_filename}. Total time elapsed: {prettify_time_delta(t2 - t1)}. Total cost: {round(stats["total_cost"], 2)}.')
    t2 = time.time()
    stats['total_time'] = prettify_time_delta(t2 - t1)
    stats['attempts'] = args.attempts - attempts + 1
    stats['total_calls'] = stats['first_stage']['total_calls'] + \
        stats['second_stage']['total_calls'] + \
        stats['third_stage']['total_calls'] + \
        stats['class_generation']['total_calls']
    stats['total_cost'] = stats['first_stage']['gpt-3.5-turbo']['total_cost'] + stats['first_stage']['gpt-4']['total_cost'] + \
        stats['second_stage']['gpt-3.5-turbo']['total_cost'] + stats['second_stage']['gpt-4']['total_cost'] + \
        stats['third_stage']['gpt-3.5-turbo']['total_cost'] + stats['third_stage']['gpt-4']['total_cost'] + \
        stats['class_generation']['gpt-3.5-turbo']['total_cost'] + \
        stats['class_generation']['gpt-4']['total_cost']
    if should_write_stats:
        stats_to_file(stats)
    print(
        f'{marsha_filename} done! Total time elapsed: {prettify_time_delta(t2 - t1)}. Total cost: {round(stats["total_cost"], 2)}.')


async def generate_python_code(marsha_filename: str, functions: list[str], types_defined: list[str], n_results: int, debug: bool, stats: dict) -> list[str]:
    t1 = time.time()
    print('Generating Python code...')
    mds = None
    try:
        mds = await gpt_func_to_python(marsha_filename, functions, types_defined, n_results, stats, debug=debug)
    except Exception as e:
        print('First stage failure')
        print(e)
        if debug:
            traceback.print_tb(e.__traceback__)
        print('Retrying...')
        raise e
    finally:
        t2 = time.time()
        stats['first_stage']['total_time'] = prettify_time_delta(
            t2 - t1)
    return mds


async def process_types(raw_types: list[str]) -> list[str]:
    types_defined = []
    for raw_type in raw_types:
        type_name = extract_type_name(raw_type)
        # If type is defined from a file, read the file
        if is_defined_from_file(raw_type):
            print('Reading type from file...')
            filename = extract_type_filename(raw_type)
            full_path = get_file_fullpath(filename)
            type_data = read_file(full_path)
            raw_type = f'''# type {type_name}
{type_data}
            '''
        types_defined.append(raw_type)
    return types_defined


async def review_and_fix(marsha_filename: str, files: list[str], functions: list[str], stats: dict, debug: bool = False):
    t_ssi = time.time()
    print('Parsing generated code...')
    try:
        await lint_and_fix_files(marsha_filename, files, stats, debug=debug)
    except Exception as e:
        print('Second stage failure')
        print(e)
        raise e
    finally:
        t_ssii = time.time()
        stats['second_stage']['total_time'] = prettify_time_delta(
            t_ssii - t_ssi)
    if args.debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
    t_tsi = time.time()
    print('Verifying and correcting generated code...')
    try:
        await test_and_fix_files(marsha_filename, functions, files, stats, debug=debug)
    except Exception as e:
        print('Third stage failure')
        print(e)
        raise e
    finally:
        t_tsii = time.time()
        stats['third_stage']['total_time'] = prettify_time_delta(
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


def stats_to_file(stats: dict):
    stats_md = f'''# Stats
{stats['class_generation']['total_time'] != 0 and f"""## Class generation
Total time: {stats['class_generation']['total_time']}
Total calls: {stats['class_generation']['total_calls']}

"""}
## First stage
Total time: {stats['first_stage']['total_time']}
Total calls: {stats['first_stage']['total_calls']}

## Second stage
Total time: {stats['second_stage']['total_time']}
Total calls: {stats['second_stage']['total_calls']}

## Third stage
Total time: {stats['third_stage']['total_time']}
Total calls: {stats['third_stage']['total_calls']}

## Total
Total time: {stats['total_time']}
Total calls: {stats['total_calls']}
Attempts: {stats['attempts']}
Total cost: {stats['total_cost']}

'''
    write_file('stats.md', stats_md)


# Entry point
asyncio.run(main())
