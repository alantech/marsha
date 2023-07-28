import asyncio
from asyncio.subprocess import Process
import functools
import openai
import os
import platform
import shutil
import subprocess
import sys
import time

from pylama.main import parse_options, check_paths, DEFAULT_FORMAT

from marsha.parse import validate_first_stage_markdown, validate_second_stage_markdown, write_files_from_markdown, format_marsha_for_llm, extract_func_name
from marsha.stats import MarshaStats
from marsha.utils import read_file

# OpenAI pricing model.
# Format: (tokens, price). Price per 1024 tokens.
PRICING_MODEL = {
    'gpt35': {
        'in': [(4096, 0.0015), (16384, 0.002)],
        'out': [(4096, 0.002), (16384, 0.004)]
    },
    'gpt4': {
        'in': [(8192, 0.03), (32768, 0.06)],
        'out': [(8192, 0.06), (32768, 0.12)]
    }
}

# Get time at startup to make human legible "start times" in the logs
t0 = time.time()

# PyInstaller creates a temp folder and stores path in _MEIPASS
base_path = '.'
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS

# Determine what name the user's `python` executable is (`python` or `python3`)
python = 'python' if shutil.which('python') is not None else 'python3'
if shutil.which(python) is None:
    raise Exception('Python not found')


def prettify_time_delta(delta, max_depth=2):
    rnd = round if max_depth == 1 else int
    if not max_depth:
        return ''
    if delta < 1:
        return f'''{format(delta * 1000, '3g')}ms'''
    elif delta < 60:
        sec = rnd(delta)
        subdelta = delta - sec
        return f'''{format(sec, '2g')}sec {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()
    elif delta < 3600:
        mn = rnd(delta / 60)
        subdelta = delta - mn * 60
        return f'''{format(mn, '2g')}min {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()
    elif delta < 86400:
        hr = rnd(delta / 3600)
        subdelta = delta - hr * 3600
        return f'''{format(hr, '2g')}hr {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()
    else:
        day = rnd(delta / 86400)
        subdelta = delta - day * 86400
        return f'''{format(day, '2g')}days {prettify_time_delta(subdelta, max_depth - 1)}'''.rstrip()


async def retry_chat_completion(query, model='gpt-3.5-turbo', max_tries=3, n_results=1):
    t1 = time.time()
    query['model'] = model
    query['n'] = n_results
    while True:
        try:
            out = await openai.ChatCompletion.acreate(**query)
            t2 = time.time()
            print(
                f'''Chat query took {prettify_time_delta(t2 - t1)}, started at {prettify_time_delta(t1 - t0)}, ms/chars = {(t2 - t1) * 1000 / out.get('usage', {}).get('total_tokens', 9001)}''')
            return out
        except openai.error.InvalidRequestError as e:
            if e.code == 'context_length_exceeded':
                # Try to cover up this error by choosing the bigger, more expensive model
                query['model'] = 'gpt-4'
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


async def gpt_func_to_python(marsha_filename: str, functions: list[str], defined_types: list[str], void_funcs: list[str], n_results: int, stats: MarshaStats, retries: int = 3, debug: bool = False):
    marsha_for_code_llm = format_marsha_for_llm(
        marsha_filename, functions + void_funcs, defined_types)
    marsha_for_test_llm = format_marsha_for_llm(
        marsha_filename, functions, defined_types)
    if debug:
        print(f'''marsha_for_llm =
    ---- start ----
{marsha_for_code_llm}
    ---- end ----''')

    reses = await asyncio.gather(retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': f'''You are a senior software engineer assigned to write Python 3 functions. 
The assignment is written in markdown format.
The description of each function should be included as a docstring.
Add type hints if feasible.
The filename should exactly match the name `{marsha_filename}.py`.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Generate `requirements.txt` file with all needed dependencies, do not add fixed version to dependencies.
If need to convert `type` to Python classes, you will receive a markdown where the heading is the class name followed by several rows following a comma separated CSV format where the first row contains all class properties and the following rows contain examples of the values of those properties. Make sure to add the __str__, __repr__, and __eq__ methods to the class.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{marsha_filename}.py`.
The content of the first section must be a python code block with the generated code.
The second section header must be the filename `requirements.txt`.
The content of the second section must be a text code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {marsha_filename}.py

```py
<generated code>
```

# requirements.txt

```txt
<dependencies needed>
```

''',
        }, {
            'role': 'user',
            'content': f'''{marsha_for_code_llm}'''
        }],
    }, n_results=n_results), retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': f'''You are a senior software engineer assigned to write a unit test suite for Python 3 functions.
The assignment is written in markdown format.
The unit tests created should exactly match the example cases provided for each function.
You have to create a TestCase per function provided.
The filename should exactly match the name `{marsha_filename}_test.py`.
Unknown imports might come from the file where the function is defined, or from the standard library.
If you are working with files, make sure to mock the file system since the tests will be run in a sandboxed environment.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{marsha_filename}_test.py`.
The content of the first section must be a python code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {marsha_filename}_test.py

```py
<generated code>
```

''',
        }, {
            'role': 'user',
            'content': f'''{marsha_for_test_llm}'''
        }],
    }, n_results=n_results))
    gather_stats(stats, 'first_stage', reses)
    # The output should be a valid list of Markdown documents. Parse each one and return the list of parsed doc, on failure
    # do not add it to the list. If the list to return is empty try again (or fully error out, for now)
    try:
        mds = list()
        for i in range(n_results):
            doc = reses[0].choices[i].message.content + \
                '\n\n' + reses[1].choices[i].message.content
            # Some validation that the generated file matches the expected format of:
            # # function_name.py
            # ```py
            # <insert code here>
            # ```
            # # requirements.txt
            # ```text
            # <dependency>
            # ```
            # # function_name_test.py
            # ```py
            # <insert code here>
            # ```
            if validate_first_stage_markdown(doc, marsha_filename):
                mds.append(doc)
            else:
                if debug:
                    print(f'''[First stage] Invalid doc:
{doc}''')
        if len(mds) == 0:
            raise Exception('Invalid output format')
        return mds
    except Exception:
        if debug:
            print(
                f'Failed to parse doc. Retries left = {retries}. Retrying...')
        if retries > 0:
            return await gpt_func_to_python(marsha_filename, functions, defined_types, void_funcs, n_results, stats, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', marsha_filename)


async def fix_file(marsha_filename: str, filename: str, lint_text: str, stats: MarshaStats, retries: int = 3, debug: bool = False):
    code = read_file(filename)
    res = await retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': f'''You are a senior software engineer working with Python 3.
You are using the `pylama` linting tool to find obvious errors and then fixing them. The linting tool uses `pyflakes` and `pycodestyle` under the hood to provide the recommendations.
All of the lint errors require fixing.
You should only fix the lint errors and not change anything else.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{filename}`.
The content of the first section must be a python code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {filename}

```py
<fixed code>
```

''',
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
    gather_stats(stats, 'second_stage', [res])
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        doc = res.choices[0].message.content
        if not validate_second_stage_markdown(doc, filename):
            if debug:
                print(f'''[Second stage] Invalid doc:
{doc}''')
            raise Exception('Invalid output format')
        write_files_from_markdown(doc)
    except Exception:
        if retries > 0:
            return await fix_file(marsha_filename, filename, lint_text, stats, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', lint_text)


async def lint_and_fix_files(marsha_filename: str, files: list[str], stats: MarshaStats, max_depth: int = 4, debug: bool = False):
    if max_depth == 0:
        raise Exception('Failed to fix code', files)
    options = parse_options()

    # Disabling pydocstyle and mccabe as they only do style checks, no compile-like checks
    options.linters = ['pycodestyle', 'pyflakes']
    options.paths = [os.path.abspath(f'./{file}') for file in files]

    # options.select = {
    #     'E112',  # expected an indented block
    #     'E113',  # unexpected indentation
    #     'E901',  # SyntaxError or IndentationError
    #     'E902',  # IOError
    #     'E0602', # undefined variable
    #     'E1122',  # unexpected keyword argument in function call
    #     'W0401', # wildcard import; unable to detect undefined names
    # }

    # We're using the linter as a way to catch coarse errors like missing imports. We don't actually
    # want the LLM to fix the linting issues, we'll just run the output through Python Black at the
    # end, so we have a significant number of warnings and "errors" from the linter we ignore
    options.ignore = {
        'E111',  # indentation is not multiple of 4
        'E117',  # over-indented
        'E126',  # continuation line over-indented for hanging indent
        'E127',  # continuation line over-indented for visual indent
        'E128',  # continuation line under-indented for visual indent
        'E129',  # visually indented line with same indent as next logical line
        'E131',  # continuation line unaligned for hanging indent
        'E133',  # closing bracket is missing indentation
        'E201',  # whitespace after `(`
        'E202',  # whitespace before `)`
        'E203',  # whitespace before `,` `;` `:`
        'E211',  # whitespace before `(`'
        'E221',  # multiple spaces before operator
        'E222',  # multiple spaces after operator
        'E223',  # tab before operator
        'E224',  # tab after operator
        'E225',  # missing whitespace around operator
        'E226',  # missing whitespace around arithmetic operator
        'E227',  # missing whitespace around bitwise or shift operator
        'E228',  # missing whitespace around modulo operator
        'E231',  # missing whitespace after `,` `;` `:`
        'E241',  # multiple spaces after `,` `;` `:`
        'E242',  # tab after `,` `;` `:`
        'E251',  # unexpected spaces around keyword / parameter equals
        'E252',  # missing whitespace around parameter equals
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
        'E722',  # do not use bare except, specify exception instead
        'E731',  # do not assign a lambda expression, use a def
        'W191',  # indentation contains tabs
        'W291',  # trailing whitespace
        'W292',  # no newline at end of file
        'W293',  # blank line contains whitespace
        'W391',  # blank line at end of file
        # https://github.com/AtomLinter/linter-pylama/blob/master/bin/pylama/lint/pylama_pyflakes.py
        'W0404',  # module is reimported multiple times
        'W0410',  # future import(s) after other imports
        'W0611',  # unused import
        'W0612',  # unused variable
    }

    lints = check_paths(
        [os.path.abspath(f'./{file}') for file in files], options=options, rootdir='.')

    if len(lints) == 0:
        return

    jobs = []
    for file in files:
        file_lints = [e.format(DEFAULT_FORMAT)
                      for e in lints if e.filename == file]
        if len(file_lints) > 0:
            lint_text = '\n'.join(file_lints)
            jobs.append(fix_file(marsha_filename, file,
                        lint_text, stats, debug=debug))
    await asyncio.gather(*jobs)

    await lint_and_fix_files(marsha_filename, files, stats, max_depth - 1, debug)


async def run_subprocess(stream: Process, timeout: float = 60.0) -> tuple[str, str]:
    stdout = ''
    stderr = ''
    try:
        stdout, stderr = await asyncio.wait_for(stream.communicate(), timeout)
    except asyncio.exceptions.TimeoutError:
        try:
            stream.kill()
        except OSError:
            # Ignore 'no such process' error
            pass
        raise Exception('run_subprocess timeout...')
    except Exception as e:
        raise e
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))


async def test_and_fix_files(marsha_filename: str, functions: list[str], defined_types: list[str], void_functions: list[str], files: list[str], stats: MarshaStats, retries: int = 4, debug: bool = False):
    break_line = '\n'
    if retries == 0:
        raise Exception('Failed to fix code', marsha_filename)
    # There should only be two files, the test file and the code file
    test_file = [file for file in files if file.endswith(
        f'{marsha_filename}_test.py')][0]
    code_file = [file for file in files if file.endswith(
        f'{marsha_filename}.py')][0]
    req_files = [file for file in files if file.endswith('requirements.txt')]
    # Define virtual environment path
    code_file_abspath = os.path.abspath(code_file)
    code_file_dir = os.path.dirname(code_file_abspath)
    venv_path = f'{code_file_dir}/venv'
    # Install requirements if needed
    req_file = None
    if len(req_files) > 0:
        req_file = req_files[0]
        if not os.path.exists(venv_path):
            print('Creating virtual environment...')
            try:
                create_venv_stream = await asyncio.create_subprocess_exec(
                    python, '-m', 'venv', venv_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                await run_subprocess(create_venv_stream)
            except Exception as e:
                if debug:
                    print('Failed to create virtual environment', e)
        print('Installing requirements...')
        try:
            # define pip executable based on os
            pip_exe = f'{venv_path}/Scripts/pip.exe' if platform.system(
            ) == 'Windows' else f'{venv_path}/bin/pip'
            pip_stream = await asyncio.create_subprocess_exec(
                pip_exe, 'install', '--disable-pip-version-check', '--no-compile', '-r', req_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await run_subprocess(pip_stream, 120)
        except Exception as e:
            if debug:
                print('Failed to install requirements', e)

    # Run the test suite
    if not os.path.exists(venv_path):
        python_exe = python
    else:
        # define python executable based on os
        python_exe = f'{venv_path}/Scripts/python.exe' if platform.system(
        ) == 'Windows' else f'{venv_path}/bin/python'
    try:
        test_stream = await asyncio.create_subprocess_exec(
            python_exe, test_file, '-f', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await run_subprocess(test_stream)
        test_results = f'''{stdout}{stderr}'''
    except Exception as e:
        print('Failed to run test suite...', e)
        test_results = None

    # Recursively work on fixing the files while the test suite fails, return when complete
    if test_results is not None and ("FAILED" in test_results or "Traceback" in test_results):
        if debug:
            print('Test failed, trying to fix code')
            print(test_results)
        test = read_file(test_file)
        code = read_file(code_file)
        requirements = read_file(req_file) if req_file is not None else None
        void_function_names = list(
            map(lambda f: extract_func_name(f), void_functions))
        res = await retry_chat_completion({
            'messages': [{
                'role': 'system',
                'content': f'''You are a senior software engineer helping a junior engineer fix some code that is failing.
You are given the documentation of the functions they were assigned to write, followed by the functions they wrote, the unit tests they wrote, and the unit test results.
Focus on just fixing the mistakes in the code and unit tests as necessary, trying to do the less number of changes.
Do not write new unit tests, just fix the existing ones.
{f"Do not make any reference to the functions {', '.join(void_function_names)} in `{marsha_filename}_test.py`." if len(void_function_names) > 0 else ""}
Make sure to produce working code that passes the unit tests.
Make sure to follow PEP8 style guidelines.
Make sure to include all needed standard Python libraries imports.
Generate `requirements.txt` file with all needed dependencies, do not add fixed version to dependencies.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{marsha_filename}.py`.
The content of the first section must be a python code block with the generated code.
The second section header must be the filename `requirements.txt`.
The content of the second section must be a text code block with the generated code.
The third section header must be the filename `{marsha_filename}_test.py`.
The content of the third section must be a python code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {marsha_filename}.py

```py
<fixed code>
```

# requirements.txt

```txt
<dependencies needed>
```

# {marsha_filename}_test.py

```py
<fixed code>
```

''',
            }, {
                'role': 'user',
                'content': f'''{format_marsha_for_llm(marsha_filename, functions + void_functions, defined_types)}

{f"""## Do not test the following functions:

{break_line.join(map(lambda f: f"- {f}", void_function_names))}""" if len(void_function_names) > 0 else ""}

# {code_file}

```py
{code}
```

# requirements.txt

```txt
{requirements if requirements is not None else ''}
```

# {test_file}

```py
{test}
```

# Test Results

{test_results}''',
            }],
        }, 'gpt-4')
        gather_stats(stats, 'third_stage', [res])
        # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
        # try again (or fully error out, for now)
        try:
            doc = res.choices[0].message.content
            # Some validation that the generated file matches the expected format of:
            # # function_name.py
            # ```py
            # <insert code here>
            # ```
            # # requirements.txt
            # ```txt
            # <dependency>
            # ```
            # # function_name_test.py
            # ```py
            # <insert code here>
            # ```
            if not validate_first_stage_markdown(doc, marsha_filename):
                raise Exception('Invalid output format')
            subdir = '/'.join(code_file.split('/')[:-1])
            files = write_files_from_markdown(doc, subdir=subdir)
        except Exception:
            if retries == 0:
                raise Exception('Failed to fix code', marsha_filename)

        # We figure out if this pass has succeeded by re-running the tests recursively, where it
        # ejects from the iteration if the tests pass
        return await test_and_fix_files(marsha_filename, functions, defined_types, void_functions, files, stats, retries - 1, debug)
    elif test_results is None:  # If the test suite failed to run, we try again
        return await test_and_fix_files(marsha_filename, functions, defined_types, void_functions, files, stats, retries - 1, debug)


def gather_stats(stats: MarshaStats, stage: str, res: list):
    rsetattr(stats, f'{stage}.total_calls', rgetattr(
        stats, f'{stage}.total_calls') + len(res))
    for r in res:
        model = 'gpt4' if r.model.startswith('gpt-4') else 'gpt35'
        input_tokens = r.usage.prompt_tokens
        rsetattr(stats, f'{stage}.{model}.input_tokens', rgetattr(
            stats, f'{stage}.{model}.input_tokens') + input_tokens)
        pricing = PRICING_MODEL[model]
        # Calculate input cost based on context length
        if (input_tokens <= pricing['in'][0][0]):
            rsetattr(stats, f'{stage}.{model}.input_cost', rgetattr(
                stats, f'{stage}.{model}.input_cost') + input_tokens * pricing['in'][0][1] / 1024)
        elif (input_tokens <= pricing['in'][1][0]):
            rsetattr(stats, f'{stage}.{model}.input_cost', rgetattr(
                stats, f'{stage}.{model}.input_cost') + input_tokens * pricing['in'][1][1] / 1024)
        output_tokens = r.usage.completion_tokens
        rsetattr(stats, f'{stage}.{model}.output_tokens', rgetattr(
            stats, f'{stage}.{model}.output_tokens') + output_tokens)
        # Calculate output cost based on context length
        if (output_tokens <= pricing['out'][0][0]):
            rsetattr(stats, f'{stage}.{model}.output_cost', rgetattr(
                stats, f'{stage}.{model}.output_cost') + output_tokens * pricing['out'][0][1] / 1024)
        elif (output_tokens <= pricing['out'][1][0]):
            rsetattr(stats, f'{stage}.{model}.output_cost', rgetattr(
                stats, f'{stage}.{model}.output_cost') + output_tokens * pricing['out'][1][1] / 1024)
        # Calculate total cost
        rsetattr(stats, f'{stage}.{model}.total_cost', rgetattr(stats, f'{stage}.{model}.total_cost') +
                 rgetattr(stats, f'{stage}.{model}.input_cost') + rgetattr(stats, f'{stage}.{model}.output_cost'))


# TODO: Move to utils
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        print(obj, attr)
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
