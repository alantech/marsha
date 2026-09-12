"""The Python language backend.

Reproduces the historical, Python-hardcoded behavior of the generation pipeline
exactly: prompts, venv + pip test execution, pycodestyle/pyflakes linting,
autopep8 formatting, the `.py` / `_test.py` / `pyproject.toml` artifact
contract, the runnable-`main` reflection helper, and `python`/`python3`
toolchain detection. See issue #204.

Dependency manifests are PEP 621 `pyproject.toml` files (issue #206), the same
style as Marsha's own project file: the LLM is asked for a `[project]` table
plus a setuptools build configuration, so a generated project can be installed
with `uv sync` or `pip install .`.
"""

import asyncio
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib

import autopep8
import pycodestyle
import pyflakes.api
from mistletoe import Document, ast_renderer

from marsha.backends.base import LanguageBackend
from marsha.meta import void_note
from marsha.utils import read_file, write_file, run_subprocess

# pyflakes findings that are style noise (the linter previously suppressed them); everything else
# (e.g. undefined names) is a real problem worth showing the LLM.
_PYFLAKES_NOISE = ('imported but unused',
                   'is assigned to but never used', 'is defined but unused')

# pycodestyle (style/syntax) + pyflakes (semantics), run directly: pylama was dropped because
# it imports the deprecated pkg_resources API. We lint to catch coarse errors like undefined
# names; we do not want the LLM fixing every style nit — the output goes through autopep8
# anyway — so a large set of cosmetic rules is ignored below.
_LINT_IGNORE = {
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


class _StyleReport(pycodestyle.BaseReport):
    """Collect pycodestyle findings as (line, col, code, message) tuples, honouring the ignore set."""

    def __init__(self, options):
        super().__init__(options)
        self.errors = []

    def error(self, line_number, offset, text, check):
        code = super().error(line_number, offset, text, check)
        if code:
            self.errors.append((line_number, offset + 1, code, text[5:]))
        return code

    def get_file_results(self):
        return len(self.errors)


class _FlakeReport:
    """Collect pyflakes findings (undefined names, syntax errors, ...) as preformatted strings."""

    def __init__(self):
        self.errors = []

    def flake(self, message):
        self.errors.append(str(message))

    def syntaxError(self, filename, msg, lineno, offset, text):
        if offset is not None:
            self.errors.append(f'{filename}:{lineno}:{max(offset, 1)}: {msg}')
        else:
            self.errors.append(f'{filename}:{lineno}: {msg}')

    def unexpectedError(self, filename, msg):
        self.errors.append(f'{filename}: {msg}')


class PythonBackend(LanguageBackend):
    id = 'python'
    aliases = ('py',)
    code_fence_lang = 'py'

    def __init__(self, target_version=None):
        self._toolchain = None
        self.target_version = self.resolve_target_version(target_version)

    def resolve_target_version(self, requested):
        # The minimum Python version the generated project declares (its pyproject.toml
        # requires-python). Defaults to the interpreter running Marsha: the generated code
        # is only ever verified against that interpreter, so it is the only floor we can
        # actually stand behind.
        if requested is None:
            return f'{sys.version_info.major}.{sys.version_info.minor}'
        if not re.fullmatch(r'\d+(\.\d+)?', requested):
            raise Exception(
                f'Invalid target version: {requested!r} (expected a Python version, e.g. 3.12)')
        return requested

    # --- naming / contract ---------------------------------------------------

    def source_name(self, fn):
        return f'{fn}.py'

    def test_name(self, fn):
        return f'{fn}_test.py'

    def manifest_name(self):
        return 'pyproject.toml'

    def artifact_contract(self, fn):
        return [
            ('source', self.source_name(fn)),
            ('manifest', self.manifest_name()),
            ('test', self.test_name(fn)),
        ]

    # --- generation / review prompts (full templates) -------------------------

    def spec_check_prompt(self):
        return '''You are a senior software engineer reviewing an assignment to write a Python 3 function.
The assignment is written in markdown format.
It should include sections on the function name, inputs, outputs, a description of what it should do, and some examples of how it should be used.

First, decide whether the document is compilable. Use this test: could at least one implementation exist that satisfies every part of the document (description, inputs, outputs, and all examples) at the same time? If such an implementation could exist, the document is compilable.
Underspecification is not a reason the document is not compilable: like unspecified behavior in C, whatever the document leaves open is for the implementer to decide reasonably. If the description allows several outcomes (several valid orderings, several equivalent error messages, several formats) and the examples show one of them, an implementation that follows the examples satisfies the document, so it is compilable.
One section adding more detail than another is not a contradiction: sections only conflict when they state opposing views on what the code should be doing.
The document is not compilable only when no implementation could satisfy it as written, eg the description says the function prints its result while the examples compare its return value to a string, two examples give different outputs for the same input, or an example is malformed or violates a stated requirement.

Second, list warnings for significant ambiguities. A warning is for an underspecified or ambiguous area that could result in differently-behaving code between independent generation runs, eg a missing exception type or message, missing edge cases, an ambiguous output precision or format, or non-deterministic behavior that would make the generated code flaky to test.
Be careful not to wear out the user with useless warnings: only warn when the ambiguity is significant enough that two reasonable implementers could plausibly produce different behavior. Do not warn about style, and do not ask for more examples or more precision in areas that are merely unspecified but unlikely to change the behavior.

Respond with a single JSON object and nothing else, in exactly this shape:
{"compilable": true, "warnings": ["...", "..."]}
When the document is not compilable, include a third key, an "errors" array with one or more entries:
{"compilable": false, "warnings": ["..."], "errors": ["...", "..."]}
Each warning and each error is a markdown-formatted string that cites the relevant portion of the document using inline quotes of the document's own words. Each error must quote the sections that conflict with each other and explain why no implementation could satisfy both.
Do not wrap the JSON object in code fences.
'''

    def oracle_prompt(self, meta):
        return f'''You are a senior software engineer assigned to write a unit test suite for Python 3 functions.
The assignment is written in markdown format.
The test suite is the *oracle* used to judge generated implementations, so it must be trustworthy.
The unit tests created should exactly match the example cases provided for each function.
You have to create a TestCase per function provided.
{void_note(meta)}
The filename should exactly match the name `{meta.filename}_test.py`.
Unknown imports might come from the file where the function is defined, or from the standard library.
If you are working with files, make sure to mock the file system since the tests will be run in a sandboxed environment.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
The tests must be faithful to the assignment:
- Every test must correspond to an example of expected behavior in the assignment, or to behavior its description explicitly states.
- Do not assert behavior the assignment does not state. Do not test implementation details, internal structure, or the exact wording of error messages or output formats unless the assignment pins them down.
- Do not invent edge cases, inputs, or expected outputs that are not grounded in the assignment.
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

'''

    def impl_prompt(self, meta):
        return f'''You are a senior software engineer assigned to write Python 3 functions.
The assignment is written in markdown format.
The description of each function should be included as a docstring.
Add type hints if feasible.
The filename should exactly match the name `{meta.filename}.py`.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Generate a `pyproject.toml` file that declares the project: a `[project]` table whose `name` is `{meta.filename}`, with a `version`, a `requires-python`, and a `dependencies` array listing every third-party dependency the code needs (do not add fixed versions to dependencies), plus the `[build-system]` and `[tool.setuptools]` sections shown below so the project can be installed with `uv sync` or `pip install .`.
If need to convert `type` to Python classes, you will receive a markdown where the heading is the class name followed by several rows following a comma separated CSV format where the first row contains all class properties and the following rows contain examples of the values of those properties. Make sure to add the __str__, __repr__, and __eq__ methods to the class.
A unit test suite has already been written from this same assignment and is provided to you. Your implementation must satisfy the assignment AND pass this test suite. If anything in the test suite ever appears to conflict with the assignment, the assignment is authoritative.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{meta.filename}.py`.
The content of the first section must be a python code block with the generated code.
The second section header must be the filename `pyproject.toml`.
The content of the second section must be a toml code block with the generated file.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {meta.filename}.py

```py
<generated code>
```

# pyproject.toml

```toml
[project]
name = "{meta.filename}"
version = "0.1.0"
requires-python = ">={self.target_version}"
dependencies = []

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["{meta.filename}"]
```

'''

    def diagnose_prompt(self):
        return '''You are a senior software engineer debugging a Python 3 project.
You are given the assignment (in markdown), the implementation, the unit test suite (the oracle), and the unit test results.
The test suite was derived from the assignment and is authoritative for what the code should do, except where a test is itself wrong.
Determine the root cause of the failure:
- "implementation": the code is at fault — it does not correctly implement the assignment (a bug, a missing edge case, wrong logic, a missing or wrong import, etc.).
- "test": a test is at fault — it asserts behavior the assignment does not actually require (it is over-strict, it contradicts the assignment, it tests an implementation detail, or it pins down an exact error-message wording or output format that the assignment leaves open).
Choose "test" only when the failing assertion is genuinely not required by the assignment; when in doubt, choose "implementation".
Respond with a single JSON object and nothing else, in exactly this shape:
{"fault": "implementation", "reason": "..."}
or
{"fault": "test", "reason": "..."}
Do not wrap the JSON object in code fences.
'''

    def fix_impl_prompt(self, meta):
        return f'''You are a senior software engineer fixing a Python 3 implementation that is failing its unit tests.
You are given the assignment, the implementation, the unit test suite (the oracle), and the test results.
The unit test suite is authoritative and must NOT be modified.
Fix only the implementation so that it correctly implements the assignment and passes the test suite, making the least changes necessary.
Make sure to produce working code that passes the unit tests.
Make sure to follow PEP8 style guidelines.
Make sure to include all needed standard Python libraries imports.
Generate a `pyproject.toml` file that declares the project: a `[project]` table whose `name` is `{meta.filename}`, with a `version`, a `requires-python`, and a `dependencies` array listing every third-party dependency the code needs (do not add fixed versions to dependencies), plus the `[build-system]` and `[tool.setuptools]` sections shown below so the project can be installed with `uv sync` or `pip install .`.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{meta.filename}.py`.
The content of the first section must be a python code block with the generated code.
The second section header must be the filename `pyproject.toml`.
The content of the second section must be a toml code block with the generated file.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {meta.filename}.py

```py
<fixed code>
```

# pyproject.toml

```toml
[project]
name = "{meta.filename}"
version = "0.1.0"
requires-python = ">={self.target_version}"
dependencies = []

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["{meta.filename}"]
```

'''

    def correct_test_prompt(self, meta):
        return f'''You are a senior software engineer correcting a faulty unit test.
You are given the assignment, the implementation, the unit test suite, and the test results.
A diagnosis has determined that a TEST (not the implementation) is at fault: it asserts behavior the assignment does not actually require — for example it is over-strict, it contradicts the assignment, it tests an implementation detail, or it pins down an exact error-message wording or output format that the assignment leaves open.
Correct ONLY the faulty test(s) so that the test suite faithfully tests the assignment. Every change you make must be justified by the assignment: reference the part of the assignment that makes the current test wrong.
You must NOT weaken a test that is actually correct: if a failing assertion is genuinely required by the assignment, leave that test unchanged.
You must NOT modify the implementation, and you must not write new tests beyond correcting the faulty ones.
Your response must not comment on what you changed.
Your response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.
Your response must be a markdown file.
The first section header must be the filename `{meta.filename}_test.py`.
The content of the first section must be a python code block with the corrected test code.
The file should end with the code block, nothing else should be added to the file.
The desired response must look like the following:

# {meta.filename}_test.py

```py
<corrected code>
```

'''

    def lint_fix_prompt(self, filename):
        return f'''You are a senior software engineer working with Python 3.
You are using a Python linter to find obvious errors and then fixing them. The linter uses `pyflakes` and `pycodestyle` under the hood to provide the recommendations.
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

'''

    # --- layout / validation --------------------------------------------------

    def _sections(self, md):
        # The (header, content) pairs of a sectioned markdown doc, in document order. Empty code
        # fences are skipped, mirroring write_files_from_markdown.
        ast = ast_renderer.get_ast(Document(md))
        sections = []
        filename = ''
        for section in ast['children']:
            if section['type'] == 'Heading':
                filename = section['children'][0]['content']
            elif section['type'] == 'CodeFence':
                filedata = section['children'][0]['content']
                if filedata is None or filedata == '':
                    continue
                sections.append((filename, filedata))
        return sections

    def compose(self, impl, oracle):
        # Implementation files first (in document order), then the oracle's files: the
        # source file, the optional manifest, and the test suite.
        files = {}
        for name, content in self._sections(impl) + self._sections(oracle):
            files[name] = content
        return files

    def _validate_single(self, md, filename):
        # A single-file document: one header (the filename) followed by one code fence.
        ast = ast_renderer.get_ast(Document(md))
        if len(ast['children']) != 2:
            return False
        if ast['children'][0]['type'] != 'Heading':
            return False
        if ast['children'][1]['type'] != 'CodeFence':
            return False
        if ast['children'][0]['children'][0]['content'].strip() != filename:
            return False
        return True

    @staticmethod
    def _valid_manifest(text):
        # A dependency manifest is a PEP 621 pyproject.toml: parseable TOML with a [project]
        # table that names the project.
        if not text:
            return False
        try:
            data = tomllib.loads(text)
        except Exception:
            return False
        project = data.get('project') if isinstance(data, dict) else None
        return (isinstance(project, dict)
                and isinstance(project.get('name'), str)
                and project['name'].strip() != '')

    @staticmethod
    def _manifest_deps(manifest):
        # The third-party dependencies a pyproject.toml declares (its [project] dependencies
        # array; empty when the project declares none), or None when the file is not a
        # readable PEP 621 manifest.
        try:
            with open(manifest, 'rb') as f:
                project = tomllib.load(f).get('project')
            deps = project.get('dependencies', []) if isinstance(project, dict) else None
        except Exception:
            return None
        if not isinstance(deps, list):
            return None
        return [dep for dep in deps if isinstance(dep, str)]

    def _validate_impl(self, md, marsha_filename):
        # An implementation-only document: the code file, optionally followed by pyproject.toml
        ast = ast_renderer.get_ast(Document(md))
        if len(ast['children']) != 2 and len(ast['children']) != 4:
            return False
        if ast['children'][0]['type'] != 'Heading':
            return False
        if ast['children'][1]['type'] != 'CodeFence':
            return False
        if ast['children'][0]['children'][0]['content'].strip() != f'{marsha_filename}.py':
            return False
        if len(ast['children']) == 4:
            if ast['children'][2]['type'] != 'Heading':
                return False
            if ast['children'][3]['type'] != 'CodeFence':
                return False
            if ast['children'][2]['children'][0]['content'].strip() != 'pyproject.toml':
                return False
            manifest = ast['children'][3]['children'][0]['content']
            if not self._valid_manifest(manifest):
                return False
        return True

    def validate_markdown(self, doc, kind, name):
        if kind == 'impl':
            return self._validate_impl(doc, name)
        if kind == 'oracle':
            return self._validate_single(doc, self.test_name(name))
        if kind == 'unit':
            return self._validate_single(doc, name)
        return False

    # --- toolchain leaves -------------------------------------------------------

    def format_files(self, files):
        for file in files:
            before = read_file(file)
            after = autopep8.fix_code(before)
            write_file(file, after)

    def lint_files(self, files):
        """Run pycodestyle (style/syntax) and pyflakes (semantics) over the given files and return a
        mapping of filename -> list of '<file>:<line>:<col>: <code> <message>' strings. Replaces the
        former pylama call, whose plugin discovery imports the deprecated pkg_resources API."""
        findings = {f: [] for f in files}
        style_options = pycodestyle.StyleGuide(
            quiet=True, ignore=list(_LINT_IGNORE)).options
        for file in files:
            path = os.path.abspath(f'./{file}')
            if not os.path.exists(path):
                continue
            style = _StyleReport(style_options)
            pycodestyle.Checker(path, options=style_options,
                                report=style).check_all()
            findings[file].extend(f'{file}:{line}:{col}: {code} {msg}'
                                  for line, col, code, msg in style.errors)
            with open(path, encoding='utf-8') as fh:
                source = fh.read()
            flakes = _FlakeReport()
            pyflakes.api.check(source, file, flakes)
            for entry in flakes.errors:
                if any(noise in entry for noise in _PYFLAKES_NOISE):
                    continue
                findings[file].append(entry)
        return findings

    async def run_tests(self, code_file, test_file, manifest, debug=False):
        # Set up the venv (if needed), install the dependencies the generated pyproject.toml
        # declares, and run the test suite. Install failures are folded into the returned
        # results so the diagnose/fix loop can see (and repair) a broken manifest.
        # Returns (passed, results): passed is None if the suite could not be run at all,
        # False if it ran but failed, and True if it passed.
        python = self.toolchain()
        code_file_dir = os.path.dirname(os.path.abspath(code_file))
        venv_path = f'{code_file_dir}/.venv'
        install_note = ''
        if manifest and os.path.exists(manifest):
            deps = self._manifest_deps(manifest)
            if deps is None:
                install_note = (f'Failed to read the dependencies from {manifest}: '
                                f'it is not a valid pyproject.toml\n')
                print('Failed to read dependencies from pyproject.toml')
            elif deps:
                if not os.path.exists(venv_path):
                    print('Creating virtual environment...')
                    try:
                        create_venv_stream = await asyncio.create_subprocess_exec(
                            python, '-m', 'venv', venv_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        await run_subprocess(create_venv_stream)
                    except Exception as e:
                        if debug:
                            print('Failed to create virtual environment', e)
                print('Installing dependencies...')
                try:
                    pip_exe = f'{venv_path}/Scripts/pip.exe' if platform.system(
                    ) == 'Windows' else f'{venv_path}/bin/pip'
                    pip_stream = await asyncio.create_subprocess_exec(
                        pip_exe, 'install', '--disable-pip-version-check', '--no-compile', *deps, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = await run_subprocess(pip_stream, 120)
                    if pip_stream.returncode != 0:
                        install_note = f'Failed to install the declared dependencies:\n{out}{err}\n'
                except Exception as e:
                    install_note = f'Failed to install the declared dependencies: {e}\n'
                    if debug:
                        print('Failed to install dependencies', e)
        if not os.path.exists(venv_path):
            python_exe = python
        else:
            python_exe = f'{venv_path}/Scripts/python.exe' if platform.system(
            ) == 'Windows' else f'{venv_path}/bin/python'
        try:
            test_stream = await asyncio.create_subprocess_exec(
                python_exe, test_file, '-f', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await run_subprocess(test_stream)
            results = f'''{install_note}{stdout}{stderr}'''
        except Exception as e:
            print('Failed to run test suite...', e)
            return (None, '')
        passed = ('FAILED' not in results) and ('Traceback' not in results)
        return (passed, results)

    def make_executable(self, path):
        # Append the static reflection `__main__` helper (no LLM involved).
        helper = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'helper.py')
        with open(path, 'a') as o, open(helper, 'r') as i:
            o.write(i.read())

    def persona_guidance(self):
        return ('This project targets Python 3 — use type hints to aid static analysis, '
                'follow PEP8, code is formatted with autopep8, and third-party '
                'dependencies are declared in the project `pyproject.toml` '
                '([project] dependencies array, no pinned versions).')

    def toolchain(self):
        # Determine what name the user's `python` executable is (`python` or `python3`)
        if self._toolchain is None:
            name = 'python' if shutil.which(
                'python') is not None else 'python3'
            if shutil.which(name) is None:
                raise Exception('Python not found')
            self._toolchain = name
        return self._toolchain

    def toolchain_ok(self):
        try:
            self.toolchain()
            return True
        except Exception:
            return False
