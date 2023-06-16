import argparse
import asyncio
import os
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


async def process_types(types: list[str]) -> dict:
    classes_defined = {}
    for type in types:
        type_name = extract_type_name(type)
        print(f'Compiling type {type_name}...')
        class_defined = await gpt_type_to_python(type)
        classes_defined[type_name] = class_defined
    return classes_defined


async def main():
    t1 = time.time()
    f = open(args.source, 'r')
    marsha_file = f.read()
    f.close()
    functions, types = extract_functions_and_types(marsha_file)
    if len(types) > 0:
        classes_defined = await process_types(types)
    # TODO: use classes_defined in function generation if necessary
    for func in functions:
        func_name = extract_function_name(func)
        print(f'Compiling function {func_name}...')
        attempts = 3
        while attempts:
            attempts = attempts - 1
            print('Generating Python code...')
            md = ''
            try:
                md = await gpt_func_to_python(func)
            except Exception as e:
                print('First stage failure')
                print(e)
                print('Retrying')
                continue
            files = write_files_from_markdown(md)
            if args.debug:
                for file in files:
                    print(f'# {file}\n')
                    f = open(file, 'r')
                    print(f.read())
                    f.close()
                    print()
            if args.quick_and_dirty:
                break
            print('Parsing generated code...')
            try:
                await lint_and_fix_files(files)
            except Exception as e:
                print('Second stage failure')
                print(e)
                print('Retrying')
                continue
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
                print('Retrying')
                continue
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
            # Done! Add one back to `attempts` to avoid accidentally erroring out on success
            attempts = attempts + 1
            break
        if attempts == 0:
            t2 = time.time()
            raise Exception(f'Failed to generate working code for {func_name}. Total time elapsed: {prettify_time_delta(t2 - t1)}')
        t2 = time.time()
        print(f'{func_name} done! Total time elapsed: {prettify_time_delta(t2 - t1)}')


asyncio.run(main())
