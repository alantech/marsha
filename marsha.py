import argparse
import asyncio
import os
import shutil
import time

import autopep8
import openai

from llm import gpt_func_to_python, lint_and_fix_files, test_and_fix_files, prettify_time_delta, gpt_type_to_python
from parse import extract_function_name, extract_functions_and_types, extract_type_name, write_files_from_markdown

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
parser.add_argument('-n', '--n-parallel-executions', type=int, default=1)

args = parser.parse_args()


def autoformat_files(files):
    for file in files:
        f = open(file, 'r')
        before = f.read()
        f.close()
        after = autopep8.fix_code(before)
        f = open(file, 'w')
        f.write(after)
        f.close()


def copy_file(src, dest):
    shutil.copyfile(src, dest)


def delete_dir_and_content(filename):
    dir = os.path.dirname(filename)
    if os.path.isdir(dir):
        shutil.rmtree(dir)


async def process_types(types: list[str]) -> dict:
    classes_defined = {}
    for type in types:
        type_name = extract_type_name(type)
        print(f'Compiling type {type_name}...')
        class_defined = await gpt_type_to_python(type)
        classes_defined[type_name] = class_defined
    return classes_defined


async def fix_files_func(files, func):
    print('Parsing generated code...')
    try:
        await lint_and_fix_files(files)
    except Exception as e:
        print('Second stage failure')
        print(e)
        raise e
    if args.debug:
        for file in files:
            print(f'# {file}\n')
            f = open(file, 'r')
            print(f.read())
            f.close()
            print()
    print('Verifying and correcting generated code...')
    try:
        await test_and_fix_files(func, files)
    except Exception as e:
        print('Third stage failure')
        print(e)
        raise e
    if args.debug:
        for file in files:
            print(f'# {file}\n')
            f = open(file, 'r')
            print(f.read())
            f.close()
            print()
    print('Formatting code...')
    autoformat_files(files)
    if args.debug:
        for file in files:
            print(f'# {file}\n')
            f = open(file, 'r')
            print(f.read())
            f.close()
            print()


async def run_parallel_tasks(tasks) -> str:
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


async def main():
    t1 = time.time()
    f = open(args.source, 'r')
    marsha_file = f.read()
    f.close()
    functions, types = extract_functions_and_types(marsha_file)
    classes_defined = None
    if len(types) > 0:
        classes_defined = await process_types(types)
        if args.debug:
            for key, value in classes_defined.items():
                print(f'# type {key}\n')
                print(value)
                print()
    for func in functions:
        func_name = extract_function_name(func)
        print(f'Compiling function {func_name}...')
        attempts = args.attempts
        n_results = args.n_parallel_executions
        if args.debug:
            print(f'Number of attempts: {attempts}')
            print(f'Number of parallel executions: {n_results}')
        while attempts:
            attempts = attempts - 1
            print('Generating Python code...')
            mds = None
            try:
                mds = await gpt_func_to_python(func, types=classes_defined, debug=args.debug, n_results=n_results)
            except Exception as e:
                print('First stage failure')
                print(e)
                print('Retrying')
                continue
            if args.quick_and_dirty:
                print('Writing generated code to files...')
                for md in mds[:2]:
                    write_files_from_markdown(md)
                break
            filenames = list()
            for idx, md in enumerate(mds):
                print('Writing generated code to temporal files...')
                filenames = filenames + \
                    write_files_from_markdown(md, subdir=f'{func_name}_{idx}')
            if args.debug:
                for filename in filenames:
                    print(f'# {filename}\n')
                    f = open(filename, 'r')
                    print(f.read())
                    f.close()
                    print()
            # Create a new list of files having the i and i+1 files together
            # This is because we want to run the linting and testing in parallel
            # but we need to make sure that the linting and testing is done on
            # the same files (func and test) together
            files_grouped = [filenames[i:i + 2]
                             for i in range(0, len(filenames), 2)]
            # Create tasks to run in parallel using asyncio
            tasks = []
            for file_group in files_grouped:
                tasks.append(asyncio.create_task(
                    fix_files_func(file_group, func), name=file_group[0]))
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
                        copy_file(filename, f'{func_name}.py')
                        copy_file(test_filename, f'{func_name}_test.py')
                        delete_dir_and_content(filename)
            except Exception as e:
                print('Failed to generate working code.')
                print(e)
                for name in task_names:
                    delete_dir_and_content(name)
                print('Retrying...')
                continue
            # Done! Add one back to `attempts` to avoid accidentally erroring out on success
            attempts = attempts + 1
            break
        if attempts == 0:
            t2 = time.time()
            raise Exception(
                f'Failed to generate working code for {func_name}. Total time elapsed: {prettify_time_delta(t2 - t1)}')
        t2 = time.time()
        print(f'{func_name} done! Total time elapsed: {prettify_time_delta(t2 - t1)}')


asyncio.run(main())
