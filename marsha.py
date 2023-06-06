import argparse
import asyncio
import os
import shutil
import subprocess
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
f = open(os.path.join(base_path, 'examples/test_correction/func.mrsh'), 'r')
func_correction = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/test_correction/before.py'), 'r')
before_correction = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/test_correction/before_test.py'), 'r')
before_test_correction = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/test_correction/test_results.txt'), 'r')
test_results_correction = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/test_correction/after.py'), 'r')
after_correction = f.read()
f.close()
f = open(os.path.join(base_path, 'examples/test_correction/after_test.py'), 'r')
after_test_correction = f.read()
f.close()

# Determine what name the user's `python` executable is (`python` or `python3`)
python = 'python' if shutil.which('python') is not None else 'python3'
if shutil.which(python) is None:
    raise Exception('Python not found')


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
            'content': f'''# Requirements for function {fibonacci_mrsh.split('# func')[1]}'''
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
            'content': f'''# Requirements for function {func.split('# func')[1]}'''
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
            raise Exception('Failed to generate code', lint_text)


async def lint_and_fix_files(files, max_depth=10):
    if max_depth == 0:
        return
    options = parse_options()

    # Disabling pydocstyle and mccabe as they only do style checks, no compile-like checks
    options.linters = ['pycodestyle', 'pyflakes']
    options.paths = [os.path.abspath(f'./{file}') for file in files]

    # We're using the linter as a way to catch coarse errors like missing imports. We don't actually
    # want the LLM to fix the linting issues, we'll just run the output through Python Black at the
    # end, so we have a significant number of warnings and "errors" from the linter we ignore
    options.ignore = {
        'E111',  # indentation is not multiple of 4
        'E117',  # over-indented
        'E201',  # whitespace after `(`
        'E202',  # whitespace before `)`
        'E203',  # whitespace before `,` `;` `:`
        'E211',  # whitespace before `(`'
        'E221',  # multiple spaces before operator
        'E222',  # multiple spaces after operator
        'E223',  # tab before operator
        'E224',  # tab after operator
        'E225',  # missing whitespace around operator
        'E227',  # missing whitespace around bitwise or shift operator
        'E228',  # missing whitespace around modulo operator
        'E231',  # missing whitespace after `,` `;` `:`
        'E251',  # unexpected spaces around keyword / parameter equals
        'E261',  # at least two spaces before inline comment
        'E262',  # inline comment should start with `# `
        'E265',  # block comment should start with `# `
        'E266',  # too many `#` for block comment
        'E271',  # multiple spaces after keyword
        'E272',  # multiple spaces before keyword
        'E273',  # tab before keyword
        'E274',  # tab after keyword
        'E275',  # space missing after keyword
        'E301',  # expected 1 blank line, found 0
        'E302',  # expected 2 blank lines, found 0
        'E303',  # too many blank lines
        'E304',  # blank line after function decorator
        'E305',  # expected 2 blank lines after function or class
        'E306',  # expected 1 blank line before nested definition
        'E401',  # multiple imports on one line
        'E501',  # line too long
        'E502',  # blackslash redundant between brackets
        'E701',  # multiple statements on one line (colon)
        'E702',  # multiple statements on one line (semicolon)
        'E703',  # statement ends with a semicolon
        'E731',  # do not assign a lambda expression, use a def
        'W191',  # indentation contains tabs
        'W291',  # trailing whitespace
        'W292',  # no newline at end of file
        'W293',  # blank line contains whitespace
        'W391',  # blank line at end of file
    }

    lints = check_paths([os.path.abspath(f'./{file}') for file in files], options=options, rootdir='.')

    if len(lints) == 0:
        return

    jobs = []
    for file in files:
        file_lints = [e.format(DEFAULT_FORMAT) for e in lints if e.filename == file]
        if len(file_lints) > 0:
            lint_text = '\n'.join(file_lints)
            jobs.append(fix_file(file, lint_text))
    await asyncio.gather(*jobs)

    await lint_and_fix_files(files, max_depth - 1)


async def test_and_fix_files(func, files, max_depth=10):
    if max_depth == 0:
        return
    # There should only be two files, the test file and the code file
    test_file = [file for file in files if file.endswith('_test.py')][0]
    code_file = [file for file in files if not file.endswith('_test.py')][0]

    test_stream = subprocess.Popen([python, test_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = test_stream.communicate()
    test_results = f'{stdout}{stderr}'

    # Recursively work on fixing the files while the test suite fails, return when complete
    if "FAILED" in test_results:
        f = open(test_file, 'r')
        test = f.read()
        f.close()
        f = open(code_file, 'r')
        code = f.read()
        f.close()
        res = await retry_chat_completion({
            'messages': [{
                'role': 'system',
                'content': 'You are a senior software engineer helping a junior engineer fix some code that is failing. You are given the documentation of the function they were assigned to write, followed by the function they wrote, the unit tests they wrote, and the unit test results. There is little time before this feature must be included, so you are simply correcting their code for them using the original documentation as the guide and fixing the mistakes in the code and unit tests as necessary.',
            }, {
                'role': 'user',
                'content': f'''# Requirements for function {func_correction.split('# func')[1]}

# extract_connection_info.py

```py
{before_correction}
```

# extract_connection_info_test.py

```py
{before_test_correction}
```

# Test Results

{test_results_correction}''',
            }, {
                'role': 'assistant',
                'content': f'''# extract_connection_info.py

```py
{after_correction}
```

# extract_connection_info_test.py

```py
{after_test_correction}
```''',
            }, {
                'role': 'user',
                'content': f'''# Requirements for function {func.split('# func')[1]}

# {code_file}

```py
{code}
```

# {test_file}

```py
{test}
```

# Test Results

{test_results}''',
            }],
        }, 'gpt-4')

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
            write_files_from_markdown(doc)
        except Exception:
            if max_depth == 0:
                raise Exception('Failed to fix code', func)

        # We figure out if this pass has succeeded by re-running the tests recursively, where it
        # ejects from the iteration if the tests pass
        return await test_and_fix_files(func, files, max_depth - 1)


async def main():
    t1 = time.time()
    print('Compiling...')
    f = open(args.source, 'r')
    func = f.read()
    f.close()
    files = write_files_from_markdown(await gpt_func_to_python(func))
    print('Parsing generated code...')
    await lint_and_fix_files(files)
    print('Verifying and correcting generated code...')
    await test_and_fix_files(func, files)
    t2 = time.time()
    print(f'Done! Total time elapsed: {t2 - t1}')


asyncio.run(main())
