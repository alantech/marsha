import asyncio
from asyncio.subprocess import Process
import os
import platform
import time
import traceback
import shutil
import subprocess
import sys

from pylama.main import parse_options, check_paths, DEFAULT_FORMAT

from marsha.meta import MarshaMeta
from marsha.parse import validate_first_stage_markdown, validate_second_stage_markdown, write_files_from_markdown, format_marsha_for_llm, extract_func_name
from marsha.stats import stats
from marsha.utils import read_file, autoformat_files, prettify_time_delta
from marsha.mappers.chatgpt import ChatGPTMapper

# PyInstaller creates a temp folder and stores path in _MEIPASS
base_path = '.'
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS

# Determine what name the user's `python` executable is (`python` or `python3`)
python = 'python' if shutil.which('python') is not None else 'python3'
if shutil.which(python) is None:
    raise Exception('Python not found')


async def gpt_can_func_python(meta: MarshaMeta, n_results: int):
    gpt_can_func = ChatGPTMapper('''You are a senior software engineer reviewing an assignment to write a Python 3 function.
The assignment is written in markdown format.
It should include sections on the function name, inputs, outputs, a description of what it should do, and some examples of how it should be used.
You are assessing if this document has enough context such that a junior software engineer with a couple of years of experience should be able to write the desired function and a test suite to verify it.
The description must be precise enough to determine what to do.
The examples must be complete enough to likely catch all edge cases.
If the description and examples are broad enough that different engineers could reasonably create very different functions that supposedly meet the requirements but do different things, that is another reason to reject this assignment.
Your answer is consumed by project management software, so only respond with Y for yes or N for no.
''', max_tokens=1, n_results=n_results, stats_stage='first_stage')
    marsha_for_code_llm = format_marsha_for_llm(meta)
    gpt_opinions = await gpt_can_func.run(marsha_for_code_llm)
    if any([True if opinion == 'N' else False for opinion in gpt_opinions]):
        return False
    return True


gpt_improve = ChatGPTMapper('''You are a senior software engineer reviewing an assignment to write a Python 3 function that a junior software engineer has written.
The assignment is written in markdown format.
It includes sections on the function name, inputs, outputs, a description of what it should do, and some examples of how it should be used.
You have already decided this document is not written well enough such that another engineer can reliably write a working function that meets expectations, nor a test suite to verify proper functionality.
The description must be precise enough to determine what to do.
The examples must be complete enough to likely catch all edge cases.
You are writing a few paragraphs gently explaining the deficiencies in the task definition they have written, not coming up with examples assuming what they might have wanted, since that isn't clear in the first place, just why what they have provided is not precise enough.
In your response do not refer to the person at all or tell them what mistakes "they" have made. This is a blameless culture. The mistakes simply are, and that they made them isn't a problem, just that they should learn from them.
Do not include a "hello" or a "regards", etc, as your response is being attached to a code review system.
''', stats_stage='first_stage')


async def gpt_improve_func(meta: MarshaMeta):
    marsha_for_code_llm = format_marsha_for_llm(meta)
    improvements = await gpt_improve.run(marsha_for_code_llm)
    print(improvements)


async def gpt_func_to_python(meta: MarshaMeta, n_results: int, retries: int = 3, debug: bool = False):
    marsha_for_code_llm = format_marsha_for_llm(meta)
    gpt_gen_code = ChatGPTMapper(f'''You are a senior software engineer assigned to write Python 3 functions.
The assignment is written in markdown format.
The description of each function should be included as a docstring.
Add type hints if feasible.
The filename should exactly match the name `{meta.filename}.py`.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Generate `requirements.txt` file with all needed dependencies, do not add fixed version to dependencies.
If need to convert `type` to Python classes, you will receive a markdown where the heading is the class name followed by several rows following a comma separated CSV format where the first row contains all class properties and the following rows contain examples of the values of those properties. Make sure to add the __str__, __repr__, and __eq__ methods to the class.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{meta.filename}.py`.
The content of the first section must be a python code block with the generated code.
The second section header must be the filename `requirements.txt`.
The content of the second section must be a text code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {meta.filename}.py

```py
<generated code>
```

# requirements.txt

```txt
<dependencies needed>
```

''', n_results=n_results, stats_stage='first_stage')
    marsha_for_test_llm = format_marsha_for_llm(meta)
    gpt_gen_test = ChatGPTMapper(f'''You are a senior software engineer assigned to write a unit test suite for Python 3 functions.
The assignment is written in markdown format.
The unit tests created should exactly match the example cases provided for each function.
You have to create a TestCase per function provided.
The filename should exactly match the name `{meta.filename}_test.py`.
Unknown imports might come from the file where the function is defined, or from the standard library.
If you are working with files, make sure to mock the file system since the tests will be run in a sandboxed environment.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{meta.filename}_test.py`.
The content of the first section must be a python code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {meta.filename}_test.py

```py
<generated code>
```

''', n_results=n_results, stats_stage='first_stage')
    if debug:
        print(f'''marsha_for_llm =
    ---- start ----
{marsha_for_code_llm}
    ---- end ----''')

    reses = await asyncio.gather(gpt_gen_code.run(marsha_for_code_llm), gpt_gen_test.run(marsha_for_test_llm))
    # The output should be a valid list of Markdown documents. Parse each one and return the list of parsed doc, on failure
    # do not add it to the list. If the list to return is empty try again (or fully error out, for now)
    try:
        mds = list()
        for i in range(n_results):
            # TODO: This unfairly reduces the success probability of the separate GPT calls, requiring both in the same run
            # to pass. It should instead try to use the same pass if possible, but otherwise use a different pairing so bad
            # dice rolls don't compound each other.
            doc = reses[0][i] + '\n\n' + reses[1][i]
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
            if validate_first_stage_markdown(doc, meta.filename):
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
            return await gpt_func_to_python(meta, n_results, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', meta.filename)


async def fix_file(marsha_filename: str, filename: str, lint_text: str, retries: int = 3, debug: bool = False):
    code = read_file(filename)
    gpt_fix = ChatGPTMapper(f'''You are a senior software engineer working with Python 3.
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

''', stats_stage='second_stage')
    fixed_code = await gpt_fix.run(f'''# {filename}

```py
{code}
```

# pylama results

```
{lint_text}
```''')
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        if not validate_second_stage_markdown(fixed_code, filename):
            if debug:
                print(f'''[Second stage] Invalid doc:
{fixed_code}''')
            raise Exception('Invalid output format')
        write_files_from_markdown(fixed_code)
    except Exception:
        if retries > 0:
            return await fix_file(marsha_filename, filename, lint_text, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', lint_text)


async def lint_and_fix_files(marsha_filename: str, files: list[str], max_depth: int = 4, debug: bool = False):
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
                        lint_text, debug=debug))
    await asyncio.gather(*jobs)

    await lint_and_fix_files(marsha_filename, files, max_depth - 1, debug)


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


async def test_and_fix_files(meta: MarshaMeta, files: list[str], retries: int = 4, debug: bool = False):
    break_line = '\n'
    if retries == 0:
        raise Exception('Failed to fix code', meta.filename)
    # There should only be two files, the test file and the code file
    test_file = [file for file in files if file.endswith(
        f'{meta.filename}_test.py')][0]
    code_file = [file for file in files if file.endswith(
        f'{meta.filename}.py')][0]
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
            map(lambda f: extract_func_name(f), meta.void_funcs))
        gpt_fix = ChatGPTMapper(f'''You are a senior software engineer helping a junior engineer fix some code that is failing.
You are given the documentation of the functions they were assigned to write, followed by the functions they wrote, the unit tests they wrote, and the unit test results.
Focus on just fixing the mistakes in the code and unit tests as necessary, trying to do the less number of changes.
Do not write new unit tests, just fix the existing ones.
{f"Do not make any reference to the functions {', '.join(void_function_names)} in `{meta.filename}_test.py`." if len(void_function_names) > 0 else ""}
Make sure to produce working code that passes the unit tests.
Make sure to follow PEP8 style guidelines.
Make sure to include all needed standard Python libraries imports.
Generate `requirements.txt` file with all needed dependencies, do not add fixed version to dependencies.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{meta.filename}.py`.
The content of the first section must be a python code block with the generated code.
The second section header must be the filename `requirements.txt`.
The content of the second section must be a text code block with the generated code.
The third section header must be the filename `{meta.filename}_test.py`.
The content of the third section must be a python code block with the generated code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {meta.filename}.py

```py
<fixed code>
```

# requirements.txt

```txt
<dependencies needed>
```

# {meta.filename}_test.py

```py
<fixed code>
```

''', model='gpt-4', stats_stage='third_stage')
        fixed_code = await gpt_fix.run(f'''{format_marsha_for_llm(meta)}

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

{test_results}''')
        # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
        # try again (or fully error out, for now)
        try:
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
            if not validate_first_stage_markdown(fixed_code, meta.filename):
                raise Exception('Invalid output format')
            subdir = '/'.join(code_file.split('/')[:-1])
            files = write_files_from_markdown(fixed_code, subdir=subdir)
        except Exception:
            if retries == 0:
                raise Exception('Failed to fix code', meta.filename)

        # We figure out if this pass has succeeded by re-running the tests recursively, where it
        # ejects from the iteration if the tests pass
        return await test_and_fix_files(meta, files, retries - 1, debug)
    elif test_results is None:  # If the test suite failed to run, we try again
        return await test_and_fix_files(meta, files, retries - 1, debug)


async def generate_python_code(args, meta: MarshaMeta, n_results: int, debug: bool) -> list[str]:
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


async def review_and_fix(args, meta: MarshaMeta, files: list[str], debug: bool = False):
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
