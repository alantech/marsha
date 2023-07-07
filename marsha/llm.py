import asyncio
from asyncio.subprocess import Process
import openai
import os
import shutil
import subprocess
import sys
import time

from pylama.main import parse_options, check_paths, DEFAULT_FORMAT

from marsha.parse import validate_first_stage_markdown, validate_second_stage_markdown, write_files_from_markdown, format_marsha_for_llm
from marsha.utils import read_file

# OpenAI pricing model.
# Format: (tokens, price). Price per 1024 tokens.
PRICING_MODEL = {
    'gpt-3.5-turbo': {
        'in': [(4096, 0.0015), (16384, 0.002)],
        'out': [(4096, 0.002), (16384, 0.004)]
    },
    'gpt-4': {
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


async def gpt_func_to_python(marsha_filename: str, functions: list[str], defined_types: list[str], n_results: int, stats: dict, retries: int = 3, debug: bool = False):
    marsha_for_llm = format_marsha_for_llm(
        marsha_filename, functions, defined_types)
    if debug:
        print(f'''marsha_for_llm =
    ---- start ----
{marsha_for_llm}
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
If you need to use external libraries, make sure to include the dependencies in a `requirements.txt` file. If there are no dependencies, do not include the file.
If need to convert `type` to Python classes, you will receive a markdown where the heading is the class name followed by several rows following a comma separated CSV format where the first row contains all class properties and the following rows contain examples of the values of those properties. Make sure to add the __str__, __repr__, and __eq__ methods to the class.
Your response must match exactly the following markdown format and nothing else:

# {marsha_filename}.py

```py
<generated code>
```

# requirements.txt

```txt
<dependencies>
```

In your response, do not include any explanation, notes, or comments.
''',
        }, {
            'role': 'user',
            'content': f'''{marsha_for_llm}'''
        }],
    }, n_results=n_results), retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': f'''You are a senior software engineer assigned to write a unit test suite for Python 3 functions.
The assignment is written in markdown format.
The unit tests created should exactly match the example cases provided for each function.
You have to create a TestCase per function provided.
You have to mock every external API call or database connection.
The filename should exactly match the name `{marsha_filename}_test.py`.
Unknown imports might come from the file where the function is defined, or from the standard library.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Your response must match exactly the following markdown format and nothing else:

# {marsha_filename}_test.py

```py
<generated code>
```

In your response, do not include any explanation, notes, or comments.
''',
        }, {
            'role': 'user',
            'content': f'''{marsha_for_llm}'''
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
                    print(f'''Invalid doc = {doc}''')
        if len(mds) == 0:
            raise Exception('Invalid output format')
        return mds
    except Exception:
        if debug:
            print(
                f'Failed to parse doc. Retries left = {retries}. Retrying...')
        if retries > 0:
            return await gpt_func_to_python(marsha_filename, functions, defined_types, n_results, stats, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', marsha_filename)


async def fix_file(marsha_filename: str, filename: str, lint_text: str, stats: dict, retries: int = 3, debug: bool = False):
    code = read_file(filename)
    res = await retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': f'''You are a senior software engineer working with Python 3.
You are using the `pylama` linting tool to find obvious errors and then fixing them. The linting tool uses `pyflakes` and `pycodestyle` under the hood to provide the recommendations.
All of the lint errors require fixing.
You should only fix the lint errors and not change anything else.
Your response must match exactly the following markdown format and nothing else:

# {marsha_filename}.py

```py
<fixed code>
```

# {marsha_filename}_test.py

```py
<fixed code>
```

In your response, do not include any explanation, notes, or comments.
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
                print(f'''Invalid doc = {doc}''')
            raise Exception('Invalid output format')
        write_files_from_markdown(doc)
    except Exception:
        if retries > 0:
            return await fix_file(marsha_filename, filename, lint_text, stats, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', lint_text)


async def lint_and_fix_files(marsha_filename: str, files: list[str], stats: dict, max_depth: int = 4, debug: bool = False):
    if max_depth == 0:
        raise Exception('Failed to fix code', files)
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


async def run_subprocess(stream: Process) -> tuple[str, str]:
    stdout = ''
    stderr = ''
    try:
        stdout, stderr = await asyncio.wait_for(stream.communicate(), 60)
    except asyncio.exceptions.TimeoutError:
        try:
            stream.kill()
        except OSError:
            # Ignore 'no such process' error
            pass
        raise
    return (stdout.decode('utf-8'), stderr.decode('utf-8'))


async def test_and_fix_files(marsha_filename: str, functions: list[str], files: list[str], stats: dict, retries: int = 4, debug: bool = False):
    if retries == 0:
        raise Exception('Failed to fix code', marsha_filename)
    # There should only be two files, the test file and the code file
    test_file = [file for file in files if file.endswith(
        f'{marsha_filename}_test.py')][0]
    code_file = [file for file in files if file.endswith(
        f'{marsha_filename}.py')][0]
    req_files = [file for file in files if file.endswith('requirements.txt')]

    # Install requirements if needed
    venv_path = None
    if len(req_files) > 0:
        req_file = req_files[0]
        req_file_abspath = os.path.abspath(req_file)
        req_file_dir = os.path.dirname(req_file_abspath)
        venv_path = f'{req_file_dir}/venv'
        if not os.path.exists(venv_path):
            print('Creating virtual environment...')
            create_venv_stream = await asyncio.create_subprocess_exec(
                python, '-m', 'venv', venv_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await run_subprocess(create_venv_stream)
        print('Installing requirements...')
        pip_stream = await asyncio.create_subprocess_exec(
            f'{venv_path}/bin/pip', 'install', '-r', req_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        await run_subprocess(pip_stream)

    # Run the test suite
    python_exe = f'{venv_path}/bin/python' if len(
        req_files) > 0 and venv_path is not None else python
    test_stream = await asyncio.create_subprocess_exec(
        python_exe, test_file, '-f', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await run_subprocess(test_stream)
    test_results = f'''{stdout}{stderr}'''

    # Recursively work on fixing the files while the test suite fails, return when complete
    if "FAILED" in test_results or "Traceback" in test_results:
        if debug:
            print('Test failed, trying to fix code')
            print(test_results)
        test = read_file(test_file)
        code = read_file(code_file)
        res = await retry_chat_completion({
            'messages': [{
                'role': 'system',
                'content': f'''You are a senior software engineer helping a junior engineer fix some code that is failing.
You are given the documentation of the functions they were assigned to write, followed by the functions they wrote, the unit tests they wrote, and the unit test results. 
Focus on just fixing the mistakes in the code and unit tests as necessary, trying to do the less number of changes.
Make sure to produce working code that passes the unit tests.
Make sure to follow PEP8 style guidelines.
Make sure to include all needed standard Python libraries imports.
If you need to use external libraries, make sure to include the dependencies in a `requirements.txt` file. If there are no dependencies, do not include the file.
Your response must match exactly the following markdown format and nothing else:

# {marsha_filename}.py

```py
<fixed code>
```

# requirements.txt

```txt
<dependency>
```

# {marsha_filename}_test.py

```py
<fixed code>
```

In your response, do not include any explanation, notes, or comments.
''',
            }, {
                'role': 'user',
                'content': f'''{format_marsha_for_llm(marsha_filename, functions)}

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
        return await test_and_fix_files(marsha_filename, functions, files, stats, retries - 1, debug)


def gather_stats(stats: dict, stage: str, res: list):
    stats[stage]['total_calls'] += len(res)
    for r in res:
        model = 'gpt-4' if r.model.startswith('gpt-4') else 'gpt-3.5-turbo'
        input_tokens = r.usage.prompt_tokens
        stats[stage][model]['input_tokens'] += input_tokens
        pricing = PRICING_MODEL[model]
        # Calculate input cost based on context length
        if (input_tokens <= pricing['in'][0][0]):
            stats[stage][model]['input_cost'] += input_tokens * \
                pricing['in'][0][1] / 1024
        elif (input_tokens <= pricing['in'][1][0]):
            stats[stage][model]['input_cost'] += input_tokens * \
                pricing['in'][1][1] / 1024
        output_tokens = r.usage.completion_tokens
        stats[stage][model]['output_tokens'] += output_tokens
        # Calculate output cost based on context length
        if (output_tokens <= pricing['out'][0][0]):
            stats[stage][model]['output_cost'] += output_tokens * \
                pricing['out'][0][1] / 1024
        elif (output_tokens <= pricing['out'][1][0]):
            stats[stage][model]['output_cost'] += output_tokens * \
                pricing['out'][1][1] / 1024
        # Calculate total cost
        stats[stage][model]['total_cost'] += stats[stage][model]['input_cost'] + \
            stats[stage][model]['output_cost']
