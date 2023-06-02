import argparse
import asyncio
import os
import sys
import time

from mistletoe import Document, ast_renderer
import openai
from pylama.main import parse_options, check_paths, DEFAULT_FORMAT

# Set up OpenAI
openai.organization = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_SECRET_KEY')

# Parse the input arguments (currently only one, the file to compile
parser = argparse.ArgumentParser(
        prog='marsha',
        description='Marsha AI Compiler',
)
parser.add_argument('source')
args = parser.parse_args()

# PyInstaller creates a temp folder and stores path in _MEIPASS
base_path = '.'
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
# Load examples used with ChatGPT
f = open(os.path.join(base_path, 'examples/fibonacci/fibonacci.mrsh'), 'r')
fibonacci_mrsh = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/fibonacci/fibonacci.py'), 'r')
fibonacci_py = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/fibonacci/fibonacci_test.py'), 'r')
fibonacci_test = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/connection_lint/before.py'), 'r')
connection_lint_before = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/connection_lint/pylama.txt'), 'r')
connection_lint_pylama = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/connection_lint/after.py'), 'r')
connection_lint_after = f.read()
f.close()


async def retry_chat_completion(query, model='gpt-3.5-turbo', max_tries=3):
    t1 = time.time()
    query['model'] = model
    while True:
        try:
            out = await openai.ChatCompletion.acreate(**query)
            t2 = time.time()
            print(f'''Chat query took {(t2 - t1) * 1000}ms, started at {t1}, ms/chars = {(t2 - t1) * 1000 / out.get('usage', {}).get('total_tokens', 9001)}''')
            return out
        except openai.error.InvalidRequestError as e:
            if e.code == 'context_length_exceeded':
                query['model'] = 'gpt-4'  # Try to cover up this error by choosing the bigger, more expensive model
            max_tries = max_tries - 1
            if max_tries == 0:
                raise e
            time.sleep(3 / max_tries)
        except Exception as e:
            max_tries = max_tries - 1
            if max_tries == 0:
                raise e
            time.sleep(3 / max_tries)
        if max_tries == 0:
            raise Exception('Could not execute chat completion')


def extract_function_name(func):
    ast = ast_renderer.get_ast(Document(func))
    if ast['children'][0]['type'] != 'Heading':
        raise Exception('Invalid Marsha function')
    header = ast['children'][0]['children'][0]['content']
    return header.split('(')[0].split('func')[1].strip()


def validate_first_stage_markdown(md, func_name):
    ast = ast_renderer.get_ast(md)
    if len(ast['children']) != 4:
        return False
    if ast['children'][0]['type'] != 'Heading':
        return False
    if ast['children'][2]['type'] != 'Heading':
        return False
    if ast['children'][1]['type'] != 'CodeFence':
        return False
    if ast['children'][3]['type'] != 'CodeFence':
        return False
    if ast['children'][0]['children'][0]['content'].strip() != f'{func_name}.py':
        return False
    if ast['children'][2]['children'][0]['content'].strip() != f'{func_name}_test.py':
        return False
    return True


async def gpt_func_to_python(func, retries=3):
    res = await retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': 'You are a senior software engineer assigned to write a Python 3 function. The assignment is written in markdown format, with a markdown title consisting of a pseudocode function signature (name, arguments, return type) followed by a description of the function and then a bullet-point list of example cases for the function. You write up a simple file that imports libraries if necessary and contains the function, and a second file that includes unit tests at the end based on the provided test cases. The filenames should follow the pattern of [function name].py and [function name]_test.py',
        }, {
            'role': 'user',
            'content': fibonacci_mrsh
        }, {
            'role': 'assistant',
            'content': f'''# fibonnaci.py

```py
{fibonacci_py}
```

# fibonacci_test.py

```py
{fibonacci_test}
```'''
        }, {
            'role': 'user',
            'content': func
        }],
    })
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        # If it fails to parse, it will throw here
        doc = Document(res.choices[0].message.content)
        # Some validation that the generated file matches the expected format of:
        # # function_name.py
        # ```py
        # <insert code here>
        # ```
        # # function_name_test.py
        # ```py
        # <insert code here>
        # ```
        if not validate_first_stage_markdown(doc, extract_function_name(func)):
            raise Exception('Invalid output format')
        return doc
    except Exception as e:
        if retries > 0:
            return await gpt_func_to_python(func, retries - 1)
        else:
            raise Exception('Failed to generate code', func)


def write_files_from_markdown(md):
    ast = ast_renderer.get_ast(md)
    filenames = []
    filename = ''
    filedata = ''
    for section in ast['children']:
        if section['type'] == 'Heading':
            filename = section['children'][0]['content']
            filenames.append(filename)
        elif section['type'] == 'CodeFence':
            filedata = section['children'][0]['content']
            f = open(filename, 'w')
            f.write(filedata)
            f.close()
    return filenames


async def fix_file(filename, lint_text, retries=3):
    f = open(filename, 'r')
    code = f.read()
    f.close()
    res = await retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': 'You are a senior software engineer working on a Python 3 function. You are using the pylama linting tool to find obvious errors and then fixing them. It uses pyflakes and pycodestyle under the hood to provide its recommendations, and all of the lint errors require fixing.',
        }, {
            'role': 'user',
            'content': f'''# extract_connection_info.py

```py
{connection_lint_before}
```

# pylama results

```
{connection_lint_pylama}
```''',
        }, {
            'role': 'assistant',
            'content': f'''# extract_connection_info.py

```py
{connection_lint_after}
```'''
        }, {
            'role': 'user',
            'content': f'''# {filename}

```py
{code}
```

# pylama results

```
{lint_text}
```''',
        }],
    })
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        parsed = Document(res.choices[0].message.content)
        write_files_from_markdown(parsed)
    except:
        if retries > 0:
            return await fix_file(filename, lint_text, retries - 1)
        else:
            raise Exception('Failed to generate code', func)


async def lint_and_fix_files(files, max_depth=10):
    if max_depth == 0:
        return
    options = parse_options()
    # Disabling pydocstyle and mccabe for now
    # options.linters = ['mccabe', 'pycodestyle', 'pydocstyle', 'pyflakes']
    options.linters = ['pycodestyle', 'pyflakes']
    options.paths = [os.path.abspath(f'./{file}') for file in files]
    lints = check_paths([os.path.abspath(f'./{file}') for file in files], options=options, rootdir='.')
    lints = [lint for lint in lints if lint.number != 'E501']
    if len(lints) == 0:
        return
    jobs = []
    for file in files:
        file_lints = [e.format(DEFAULT_FORMAT) for e in lints if e.filename == file]
        if len(file_lints) > 0:
            lint_text = '\n'.join(file_lints)
            print(lint_text)
            jobs.append(fix_file(file, lint_text))
    await asyncio.gather(*jobs)
    await lint_and_fix_files(files, max_depth - 1)


async def main():
    print('Compiling...')
    f = open(args.source, 'r')
    func = f.read()
    f.close()
    files = write_files_from_markdown(await gpt_func_to_python(func))
    await lint_and_fix_files(files)
    print('Done!')


asyncio.run(main())
