"""Regression tests for the Python language backend (issue #204).

The refactor must be behavior-preserving for Python: these tests pin PythonBackend's
naming, artifact contract, compose layout, markdown validators, and prompt templates
to the pre-refactor hardcoded values, and exercise the toolchain leaves (helper append,
toolchain detection, test execution) against the real Python toolchain.
"""

import asyncio
import os
import sys
from unittest.mock import patch

from marsha import backends
from marsha.backends import PythonBackend
from marsha.meta import MarshaMeta
from marsha.utils import read_file, write_file


def make_meta(filename='example', void_funcs=()):
    meta = MarshaMeta(f'{filename}.mrsh')
    meta.filename = filename
    meta.functions = []
    meta.void_funcs = list(void_funcs)
    meta.types = None
    return meta


# --- registry / --target resolution ------------------------------------------

def test_registry_resolves_python_by_id_and_alias():
    assert backends.resolve_target('python').id == 'python'
    assert backends.resolve_target('py').id == 'python'
    assert backends.resolve_target('PY').id == 'python'
    assert backends.current().id == 'python'


def test_registry_rejects_unknown_target():
    try:
        backends.resolve_target('rust')
        assert False, 'expected an exception'
    except Exception as e:
        assert 'python' in str(e)


# --- naming / contract ---------------------------------------------------------

def test_naming():
    b = PythonBackend()
    assert b.id == 'python'
    assert 'py' in b.aliases
    assert b.code_fence_lang == 'py'
    assert b.source_name('example') == 'example.py'
    assert b.test_name('example') == 'example_test.py'
    assert b.manifest_name() == 'pyproject.toml'


def test_artifact_contract():
    b = PythonBackend()
    assert b.artifact_contract('example') == [
        ('source', 'example.py'),
        ('manifest', 'pyproject.toml'),
        ('test', 'example_test.py'),
    ]


def test_code_block():
    assert PythonBackend().code_block('x') == '```py\nx\n```'


# --- target version (--target-version, issue #206) -----------------------------

def test_resolve_target_version_defaults_to_running_interpreter():
    b = PythonBackend()
    assert b.target_version == f'{sys.version_info.major}.{sys.version_info.minor}'


def test_resolve_target_version_explicit():
    assert PythonBackend('3.12').target_version == '3.12'
    assert PythonBackend('3').target_version == '3'


def test_resolve_target_version_rejects_invalid():
    for bad in ('latest', '3.x', '3..1', '', 'v3.12', '3.14.1', '3.12.1.2'):
        try:
            PythonBackend(bad)
            assert False, f'expected an exception for {bad!r}'
        except Exception as e:
            assert 'Invalid target version' in str(e)


def test_impl_prompt_renders_target_version():
    meta = make_meta()
    assert 'requires-python = ">=3.12"' in PythonBackend('3.12').impl_prompt(meta)
    assert 'requires-python = ">=3.13"' in PythonBackend('3.13').impl_prompt(meta)
    assert 'requires-python = ">=3.12"' in PythonBackend('3.12').fix_impl_prompt(meta)


def test_impl_editor_template_renders_target_version():
    from marsha import personas
    _, system = personas.load_editor('impl')
    out = system.format(filename='example', void_note='', target_version='3.13')
    assert 'requires-python = ">=3.13"' in out


# --- compose layout -------------------------------------------------------------

def test_compose_with_manifest():
    b = PythonBackend()
    impl = ('# example.py\n\n```py\ncode\n```\n\n'
            '# pyproject.toml\n\n```toml\n[project]\nname = "example"\ndependencies = ["numpy"]\n```\n')
    oracle = '# example_test.py\n\n```py\ntests\n```\n'
    # Code-fence content is taken verbatim (including the trailing newline), exactly as
    # write_files_from_markdown used to write it.
    composed = b.compose(impl, oracle)
    assert composed == {
        'example.py': 'code\n',
        'pyproject.toml': '[project]\nname = "example"\ndependencies = ["numpy"]\n',
        'example_test.py': 'tests\n',
    }
    # The source file comes first, then the manifest, then the oracle.
    assert list(composed) == ['example.py', 'pyproject.toml', 'example_test.py']


def test_compose_without_manifest():
    b = PythonBackend()
    impl = '# example.py\n\n```py\ncode\n```\n'
    oracle = '# example_test.py\n\n```py\ntests\n```\n'
    assert list(b.compose(impl, oracle)) == ['example.py', 'example_test.py']


def test_compose_skips_empty_fences():
    b = PythonBackend()
    impl = ('# example.py\n\n```py\ncode\n```\n\n'
            '# pyproject.toml\n\n```toml\n```\n')
    oracle = '# example_test.py\n\n```py\ntests\n```\n'
    assert 'pyproject.toml' not in b.compose(impl, oracle)


# --- validators -----------------------------------------------------------------

def test_validate_impl():
    b = PythonBackend()
    ok = '# example.py\n\n```py\nx\n```\n'
    ok_manifest = (ok + '\n# pyproject.toml\n\n```toml\n'
                   '[project]\nname = "example"\ndependencies = []\n```\n')
    assert b.validate_markdown(ok, 'impl', 'example')
    assert b.validate_markdown(ok_manifest, 'impl', 'example')
    assert not b.validate_markdown(ok, 'impl', 'other')
    assert not b.validate_markdown(
        ok + '\n# other.txt\n\n```txt\ny\n```\n', 'impl', 'example')
    assert not b.validate_markdown('# example.py\n', 'impl', 'example')


def test_validate_impl_manifest_must_be_pyproject():
    b = PythonBackend()
    code = '# example.py\n\n```py\nx\n```\n'

    def with_manifest(toml):
        return code + f'\n# pyproject.toml\n\n```toml\n{toml}```\n'

    # Valid TOML with a [project] table that names the project.
    assert b.validate_markdown(
        with_manifest('[project]\nname = "example"\nversion = "0.1.0"\n'), 'impl', 'example')
    # Not valid TOML.
    assert not b.validate_markdown(
        with_manifest('[project\nname = "example"\n'), 'impl', 'example')
    # Valid TOML but no [project] table.
    assert not b.validate_markdown(
        with_manifest('[tool.setuptools]\npy-modules = ["example"]\n'), 'impl', 'example')
    # [project] without a name.
    assert not b.validate_markdown(
        with_manifest('[project]\nversion = "0.1.0"\n'), 'impl', 'example')
    # [project] name that is not a string.
    assert not b.validate_markdown(
        with_manifest('[project]\nname = 42\n'), 'impl', 'example')


def test_validate_oracle():
    b = PythonBackend()
    ok = '# example_test.py\n\n```py\nx\n```\n'
    assert b.validate_markdown(ok, 'oracle', 'example')
    assert not b.validate_markdown(ok, 'oracle', 'other')
    assert not b.validate_markdown('# example.py\n\n```py\nx\n```\n', 'oracle', 'example')


def test_validate_unit_uses_verbatim_name():
    b = PythonBackend()
    full = '# /tmp/x/example.py\n\n```py\nx\n```\n'
    assert b.validate_markdown(full, 'unit', '/tmp/x/example.py')
    assert not b.validate_markdown(full, 'unit', 'example.py')
    assert not b.validate_markdown(
        full + '\n# more.py\n\n```py\ny\n```\n', 'unit', '/tmp/x/example.py')


def test_validate_unknown_kind_is_false():
    assert not PythonBackend().validate_markdown('# a\n\n```py\nx\n```\n', 'bogus', 'a')


# --- prompt templates (byte-exact anchors) ----------------------------------------

FN = "example"

LINT_FILENAME = "/tmp/pytest-x/example.py"


SPEC_CHECK = 'You are a senior software engineer reviewing an assignment to write a Python 3 function.\nThe assignment is written in markdown format.\nIt should include sections on the function name, inputs, outputs, a description of what it should do, and some examples of how it should be used.\n\nFirst, decide whether the document is compilable. Use this test: could at least one implementation exist that satisfies every part of the document (description, inputs, outputs, and all examples) at the same time? If such an implementation could exist, the document is compilable.\nUnderspecification is not a reason the document is not compilable: like unspecified behavior in C, whatever the document leaves open is for the implementer to decide reasonably. If the description allows several outcomes (several valid orderings, several equivalent error messages, several formats) and the examples show one of them, an implementation that follows the examples satisfies the document, so it is compilable.\nOne section adding more detail than another is not a contradiction: sections only conflict when they state opposing views on what the code should be doing.\nThe document is not compilable only when no implementation could satisfy it as written, eg the description says the function prints its result while the examples compare its return value to a string, two examples give different outputs for the same input, or an example is malformed or violates a stated requirement.\n\nSecond, list warnings for significant ambiguities. A warning is for an underspecified or ambiguous area that could result in differently-behaving code between independent generation runs, eg a missing exception type or message, missing edge cases, an ambiguous output precision or format, or non-deterministic behavior that would make the generated code flaky to test.\nBe careful not to wear out the user with useless warnings: only warn when the ambiguity is significant enough that two reasonable implementers could plausibly produce different behavior. Do not warn about style, and do not ask for more examples or more precision in areas that are merely unspecified but unlikely to change the behavior.\n\nRespond with a single JSON object and nothing else, in exactly this shape:\n{"compilable": true, "warnings": ["...", "..."]}\nWhen the document is not compilable, include a third key, an "errors" array with one or more entries:\n{"compilable": false, "warnings": ["..."], "errors": ["...", "..."]}\nEach warning and each error is a markdown-formatted string that cites the relevant portion of the document using inline quotes of the document\'s own words. Each error must quote the sections that conflict with each other and explain why no implementation could satisfy both.\nDo not wrap the JSON object in code fences.\n'


DIAGNOSE = 'You are a senior software engineer debugging a Python 3 project.\nYou are given the assignment (in markdown), the implementation, the unit test suite (the oracle), and the unit test results.\nThe test suite was derived from the assignment and is authoritative for what the code should do, except where a test is itself wrong.\nDetermine the root cause of the failure:\n- "implementation": the code is at fault — it does not correctly implement the assignment (a bug, a missing edge case, wrong logic, a missing or wrong import, etc.).\n- "test": a test is at fault — it asserts behavior the assignment does not actually require (it is over-strict, it contradicts the assignment, it tests an implementation detail, or it pins down an exact error-message wording or output format that the assignment leaves open).\nChoose "test" only when the failing assertion is genuinely not required by the assignment; when in doubt, choose "implementation".\nRespond with a single JSON object and nothing else, in exactly this shape:\n{"fault": "implementation", "reason": "..."}\nor\n{"fault": "test", "reason": "..."}\nDo not wrap the JSON object in code fences.\n'


ORACLE = 'You are a senior software engineer assigned to write a unit test suite for Python 3 functions.\nThe assignment is written in markdown format.\nThe test suite is the *oracle* used to judge generated implementations, so it must be trustworthy.\nThe unit tests created should exactly match the example cases provided for each function.\nYou have to create a TestCase per function provided.\n\nThe filename should exactly match the name `example_test.py`.\nUnknown imports might come from the file where the function is defined, or from the standard library.\nIf you are working with files, make sure to mock the file system since the tests will be run in a sandboxed environment.\nMake sure to follow PEP8 guidelines.\nMake sure to include all needed standard Python libraries imports.\nThe tests must be faithful to the assignment:\n- Every test must correspond to an example of expected behavior in the assignment, or to behavior its description explicitly states.\n- Do not assert behavior the assignment does not state. Do not test implementation details, internal structure, or the exact wording of error messages or output formats unless the assignment pins them down.\n- Do not invent edge cases, inputs, or expected outputs that are not grounded in the assignment.\nYour response must not comment on what you changed.\nYour response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.\nYour response must be a markdown file.\nThe first section header must be the filename `example_test.py`.\nThe content of the first section must be a python code block with the generated code.\nThe file should end with the code block, nothing else should be added to the file.\nThe desired response must look like the following:\n\n# example_test.py\n\n```py\n<generated code>\n```\n\n'


IMPL = 'You are a senior software engineer assigned to write Python 3 functions.\nThe assignment is written in markdown format.\nThe description of each function should be included as a docstring.\nAdd type hints if feasible.\nThe filename should exactly match the name `example.py`.\nMake sure to follow PEP8 guidelines.\nMake sure to include all needed standard Python libraries imports.\nGenerate a `pyproject.toml` file that declares the project: a `[project]` table whose `name` is `example`, with a `version`, a `requires-python`, and a `dependencies` array listing every third-party dependency the code needs (do not add fixed versions to dependencies), plus the `[build-system]` and `[tool.setuptools]` sections shown below so the project can be installed with `uv sync` or `pip install .`.\nIf need to convert `type` to Python classes, you will receive a markdown where the heading is the class name followed by several rows following a comma separated CSV format where the first row contains all class properties and the following rows contain examples of the values of those properties. Make sure to add the __str__, __repr__, and __eq__ methods to the class.\nA unit test suite has already been written from this same assignment and is provided to you. Your implementation must satisfy the assignment AND pass this test suite. If anything in the test suite ever appears to conflict with the assignment, the assignment is authoritative.\nYour response must not comment on what you changed.\nYour response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.\nYour response must be a markdown file.\nThe first section header must be the filename `example.py`.\nThe content of the first section must be a python code block with the generated code.\nThe second section header must be the filename `pyproject.toml`.\nThe content of the second section must be a toml code block with the generated file.\nThe file should end with the code block, nothing else should be added to the file.\nThe desired response must look like the following:\n\n# example.py\n\n```py\n<generated code>\n```\n\n# pyproject.toml\n\n```toml\n[project]\nname = "example"\nversion = "0.1.0"\nrequires-python = ">=3.12"\ndependencies = []\n\n[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n\n[tool.setuptools]\npy-modules = ["example"]\n```\n\n'


LINT_FIX = 'You are a senior software engineer working with Python 3.\nYou are using a Python linter to find obvious errors and then fixing them. The linter uses `pyflakes` and `pycodestyle` under the hood to provide the recommendations.\nAll of the lint errors require fixing.\nYou should only fix the lint errors and not change anything else.\nYour response must not comment on what you changed.\nYour response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.\nYour response must be a markdown file.\nThe first section header must be the filename `/tmp/pytest-x/example.py`.\nThe content of the first section must be a python code block with the generated code.\nThe file should end with the code block, nothing else should be added to the file.\nThe desired response must look like the following:\n\n# /tmp/pytest-x/example.py\n\n```py\n<fixed code>\n```\n\n'


IMPL_FIX = 'You are a senior software engineer fixing a Python 3 implementation that is failing its unit tests.\nYou are given the assignment, the implementation, the unit test suite (the oracle), and the test results.\nThe unit test suite is authoritative and must NOT be modified.\nFix only the implementation so that it correctly implements the assignment and passes the test suite, making the least changes necessary.\nMake sure to produce working code that passes the unit tests.\nMake sure to follow PEP8 style guidelines.\nMake sure to include all needed standard Python libraries imports.\nGenerate a `pyproject.toml` file that declares the project: a `[project]` table whose `name` is `example`, with a `version`, a `requires-python`, and a `dependencies` array listing every third-party dependency the code needs (do not add fixed versions to dependencies), plus the `[build-system]` and `[tool.setuptools]` sections shown below so the project can be installed with `uv sync` or `pip install .`.\nYour response must not comment on what you changed.\nYour response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.\nYour response must be a markdown file.\nThe first section header must be the filename `example.py`.\nThe content of the first section must be a python code block with the generated code.\nThe second section header must be the filename `pyproject.toml`.\nThe content of the second section must be a toml code block with the generated file.\nThe file should end with the code block, nothing else should be added to the file.\nThe desired response must look like the following:\n\n# example.py\n\n```py\n<fixed code>\n```\n\n# pyproject.toml\n\n```toml\n[project]\nname = "example"\nversion = "0.1.0"\nrequires-python = ">=3.12"\ndependencies = []\n\n[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n\n[tool.setuptools]\npy-modules = ["example"]\n```\n\n'


TEST_CORRECT = 'You are a senior software engineer correcting a faulty unit test.\nYou are given the assignment, the implementation, the unit test suite, and the test results.\nA diagnosis has determined that a TEST (not the implementation) is at fault: it asserts behavior the assignment does not actually require — for example it is over-strict, it contradicts the assignment, it tests an implementation detail, or it pins down an exact error-message wording or output format that the assignment leaves open.\nCorrect ONLY the faulty test(s) so that the test suite faithfully tests the assignment. Every change you make must be justified by the assignment: reference the part of the assignment that makes the current test wrong.\nYou must NOT weaken a test that is actually correct: if a failing assertion is genuinely required by the assignment, leave that test unchanged.\nYou must NOT modify the implementation, and you must not write new tests beyond correcting the faulty ones.\nYour response must not comment on what you changed.\nYour response must not add any additional comments, clarifications, notes, information, explanations, details, examples or thoughts.\nYour response must be a markdown file.\nThe first section header must be the filename `example_test.py`.\nThe content of the first section must be a python code block with the corrected test code.\nThe file should end with the code block, nothing else should be added to the file.\nThe desired response must look like the following:\n\n# example_test.py\n\n```py\n<corrected code>\n```\n\n'


def test_prompts_are_byte_exact():
    # Pin the target version explicitly so the prompt pins do not depend on the
    # interpreter running the test suite (the default is the running interpreter).
    b = PythonBackend('3.12')
    meta = make_meta()
    assert b.spec_check_prompt() == SPEC_CHECK
    assert b.diagnose_prompt() == DIAGNOSE
    assert b.oracle_prompt(meta) == ORACLE
    assert b.impl_prompt(meta) == IMPL
    assert b.fix_impl_prompt(meta) == IMPL_FIX
    assert b.correct_test_prompt(meta) == TEST_CORRECT
    assert b.lint_fix_prompt(LINT_FILENAME) == LINT_FIX


def test_oracle_prompt_includes_void_note():
    b = PythonBackend()
    meta = make_meta(void_funcs=['# func print_thing(x: str)'])
    assert ('Do not create any tests for the void functions: print_thing.'
            in b.oracle_prompt(meta))


# --- persona guidance -------------------------------------------------------------

def test_persona_guidance_carries_python_conventions():
    g = PythonBackend().persona_guidance()
    for phrase in ('Python 3', 'type hints', 'PEP8', 'autopep8', 'pyproject.toml'):
        assert phrase in g


def test_persona_guidance_injected_into_reviewer_prompt():
    from marsha import personas

    seen = {}

    class CapturingMapper:
        def __init__(self, system, **kwargs):
            seen['system'] = system

        async def run(self, req):
            return 'NO FINDINGS'

    async def go(guidance):
        with patch.object(personas, 'get_mapper',
                          new=lambda *a, **k: CapturingMapper(*a, **k)):
            await personas.run_personas(
                [('Ada', 'You are Ada.', 1)], 'msg', 'model', 'first_stage',
                loop='oracle', guidance=guidance)

    asyncio.run(go(PythonBackend().persona_guidance()))
    assert 'You are Ada.' in seen['system']
    assert 'Python 3' in seen['system']
    # Without guidance the reviewer body is used verbatim (pre-refactor behavior).
    asyncio.run(go(''))
    assert seen['system'].startswith('You are Ada.')
    assert 'Python 3' not in seen['system']


# --- toolchain leaves ---------------------------------------------------------------

def test_make_executable_appends_helper(tmp_path):
    b = PythonBackend()
    path = f'{tmp_path}/example.py'
    write_file(path, 'def f():\n    return 1\n')
    before = read_file(path)
    b.make_executable(path)
    after = read_file(path)
    assert after.startswith(before)
    assert "if __name__ == '__main__':" in after


def test_toolchain():
    b = PythonBackend()
    assert b.toolchain() in ('python', 'python3')
    assert b.toolchain_ok()


def test_run_tests_pass_and_fail(tmp_path):
    b = PythonBackend()
    code = f'{tmp_path}/example.py'
    write_file(code, 'def add(a, b):\n    return a + b\n')
    header = ('import unittest\nfrom example import add\n\n'
              'class TestAdd(unittest.TestCase):\n')

    def write_test(name, expected):
        body = (f'    def test_add(self):\n        self.assertEqual(add(1, 2), {expected})\n\n'
                'if __name__ == "__main__":\n    unittest.main()\n')
        write_file(f'{tmp_path}/{name}', header + body)

    write_test('ok_test.py', 3)
    write_test('bad_test.py', 4)
    passed, _ = asyncio.run(b.run_tests(code, f'{tmp_path}/ok_test.py', None, False))
    assert passed is True
    passed, results = asyncio.run(b.run_tests(code, f'{tmp_path}/bad_test.py', None, False))
    assert passed is False
    assert 'AssertionError' in results or 'FAILED' in results


# --- pyproject.toml manifests (issue #206) ---------------------------------------

def test_valid_manifest():
    b = PythonBackend()
    assert b._valid_manifest('[project]\nname = "example"\nversion = "0.1.0"\n')
    assert b._valid_manifest('[project]\nname = "example"\ndependencies = ["numpy"]\n')
    assert not b._valid_manifest('')
    assert not b._valid_manifest(None)
    assert not b._valid_manifest('[project\nname = "example"\n')
    assert not b._valid_manifest('[tool.setuptools]\npy-modules = ["example"]\n')
    assert not b._valid_manifest('[project]\nversion = "0.1.0"\n')
    assert not b._valid_manifest('[project]\nname = 42\n')


def test_manifest_deps(tmp_path):
    b = PythonBackend()
    p = f'{tmp_path}/pyproject.toml'
    write_file(p, '[project]\nname = "example"\ndependencies = [\n    "numpy",\n    "six",\n]\n')
    assert b._manifest_deps(p) == ['numpy', 'six']
    write_file(p, '[project]\nname = "example"\ndependencies = []\n')
    assert b._manifest_deps(p) == []
    write_file(p, '[project]\nname = "example"\n')
    assert b._manifest_deps(p) == []
    write_file(p, 'this is not [toml')
    assert b._manifest_deps(p) is None
    assert b._manifest_deps(f'{tmp_path}/missing.toml') is None


def test_run_tests_with_pyproject_manifest(tmp_path):
    # A manifest without dependencies runs the suite with the system interpreter: no venv,
    # no installs.
    b = PythonBackend()
    code = f'{tmp_path}/example.py'
    write_file(code, 'def add(a, b):\n    return a + b\n')
    manifest = f'{tmp_path}/pyproject.toml'
    write_file(manifest, '[project]\nname = "example"\ndependencies = []\n')
    test = f'{tmp_path}/example_test.py'
    write_file(test, ('import unittest\nfrom example import add\n\n'
                      'class TestAdd(unittest.TestCase):\n'
                      '    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n\n'
                      'if __name__ == "__main__":\n    unittest.main()\n'))
    passed, results = asyncio.run(b.run_tests(code, test, manifest, False))
    assert passed is True
    assert not os.path.exists(f'{tmp_path}/.venv')


def test_run_tests_surfaces_unreadable_manifest(tmp_path):
    # A manifest that is not a readable pyproject.toml is reported in the results so the
    # diagnose/fix loop can repair it, even when the suite itself passes.
    b = PythonBackend()
    code = f'{tmp_path}/example.py'
    write_file(code, 'def add(a, b):\n    return a + b\n')
    manifest = f'{tmp_path}/pyproject.toml'
    write_file(manifest, 'this is not [toml')
    test = f'{tmp_path}/example_test.py'
    write_file(test, ('import unittest\nfrom example import add\n\n'
                      'class TestAdd(unittest.TestCase):\n'
                      '    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n\n'
                      'if __name__ == "__main__":\n    unittest.main()\n'))
    passed, results = asyncio.run(b.run_tests(code, test, manifest, False))
    assert passed is True
    assert 'not a valid pyproject.toml' in results
