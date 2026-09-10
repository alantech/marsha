import asyncio
from asyncio.subprocess import Process
import json
import os
import platform
import re
import time
import traceback
import shutil
import subprocess
import sys

import pycodestyle
import pyflakes.api

from marsha.config import resolve_model, resolve_provider, resolve_strong_model
from marsha.context import budget_tokens, estimate_tokens, fits, resolve_context_window
from marsha.meta import MarshaMeta
from marsha.log import log
from marsha.parse import validate_first_stage_markdown, validate_second_stage_markdown, validate_impl_markdown, write_files_from_markdown, format_marsha_for_llm, extract_func_name, split_preamble
from marsha.personas import (
    build_registry, resolve_loop_reviewers, load_editor, run_personas,
    actionable_findings, format_findings, prior_round_block, parse_severities,
    parse_compacted_findings,
)
from marsha.stats import stats
from marsha.term import print_diagnostic
from marsha.utils import read_file, write_file, autoformat_files, prettify_time_delta
from marsha.llm_client import get_client
from marsha.mappers import get_mapper
from marsha.mappers.chatgpt import uses_completion_tokens

# Determine what name the user's `python` executable is (`python` or `python3`)
python = 'python' if shutil.which('python') is not None else 'python3'
if shutil.which(python) is None:
    raise Exception('Python not found')


SPEC_CHECK_PROMPT = '''You are a senior software engineer reviewing an assignment to write a Python 3 function.
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

DIAGNOSE_PROMPT = '''You are a senior software engineer debugging a Python 3 project.
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


def parse_spec_check(text):
    """Parse the structured spec check response; raise on anything malformed"""
    t = text.strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1] if '\n' in t else ''
        if t.rstrip().endswith('```'):
            t = t.rstrip()[:-3]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find('{'), t.rfind('}')
        if start == -1 or end <= start:
            raise Exception(
                f'No JSON object in spec check response: {text[:200]}')
        obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict) or not isinstance(obj.get('compilable'), bool):
        raise Exception(f'Invalid spec check response: {text[:200]}')
    warnings = obj.get('warnings', [])
    errors = obj.get('errors', [])
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise Exception(f'Invalid spec check response: {text[:200]}')
    if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
        raise Exception(f'Invalid spec check response: {text[:200]}')
    if not obj['compilable'] and len(errors) == 0:
        raise Exception(f'Not compilable without errors: {text[:200]}')
    return {'compilable': obj['compilable'], 'warnings': warnings, 'errors': errors}


def parse_diagnosis(text):
    """Parse the structured failure diagnosis; raise on anything malformed"""
    t = text.strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1] if '\n' in t else ''
        if t.rstrip().endswith('```'):
            t = t.rstrip()[:-3]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find('{'), t.rfind('}')
        if start == -1 or end <= start:
            raise Exception(
                f'No JSON object in diagnosis response: {text[:200]}')
        obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise Exception(f'Invalid diagnosis response: {text[:200]}')
    fault = obj.get('fault')
    if fault not in ('implementation', 'test'):
        raise Exception(f'Invalid diagnosis response: {text[:200]}')
    reason = obj.get('reason', '')
    if not isinstance(reason, str):
        raise Exception(f'Invalid diagnosis response: {text[:200]}')
    return {'fault': fault, 'reason': reason}


async def gpt_check_spec(meta: MarshaMeta, retries: int = 2):
    # Reasoning models need a larger budget for their chain of thought
    if resolve_provider() == 'openai' and uses_completion_tokens(resolve_model()):
        answer = {'max_tokens': 8192, 'reasoning_effort': 'minimal'}
    elif resolve_provider() == 'anthropic':
        answer = {'max_tokens': 4096}
    else:
        # Local OpenAI-compatible servers: leave the output budget to the server
        answer = {}
    gpt_check = get_mapper(SPEC_CHECK_PROMPT, n_results=1,
                           stats_stage='first_stage', label='spec-check', **answer)
    marsha_for_code_llm = format_marsha_for_llm(meta)
    try:
        return parse_spec_check(await gpt_check.run(marsha_for_code_llm))
    except Exception:
        if retries > 0:
            return await gpt_check_spec(meta, retries - 1)
        raise


async def gpt_test_suite(meta: MarshaMeta, retries: int = 3, debug: bool = False):
    # Generate the oracle (the test suite) first, anchored to the spec. This is the
    # authoritative artifact the implementation will be judged against.
    void_function_names = list(
        map(lambda f: extract_func_name(f), meta.void_funcs))
    void_note = ''
    if len(void_function_names) > 0:
        void_note = f'Do not create any tests for the void functions: {", ".join(void_function_names)}.'
    gpt_gen_test = get_mapper(f'''You are a senior software engineer assigned to write a unit test suite for Python 3 functions.
The assignment is written in markdown format.
The test suite is the *oracle* used to judge generated implementations, so it must be trustworthy.
The unit tests created should exactly match the example cases provided for each function.
You have to create a TestCase per function provided.
{void_note}
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

''', n_results=1, stats_stage='first_stage', label='oracle-gen')
    marsha_for_test_llm = format_marsha_for_llm(meta)
    if debug:
        print(f'''marsha_for_llm =
    ---- start ----
{marsha_for_test_llm}
    ---- end ----''')
    try:
        doc = await gpt_gen_test.run(marsha_for_test_llm)
        if not validate_second_stage_markdown(doc, f'{meta.filename}_test.py'):
            if debug:
                print(f'''[Oracle] Invalid doc:
{doc}''')
            raise Exception('Invalid output format')
        return doc
    except Exception:
        if debug:
            print(
                f'Failed to generate test suite. Retries left = {retries}. Retrying...')
        if retries > 0:
            return await gpt_test_suite(meta, retries - 1, debug)
        else:
            raise Exception('Failed to generate test suite', meta.filename)


def _void_note(meta: MarshaMeta):
    # Note telling the oracle-related prompts not to test the void functions.
    void_function_names = list(
        map(lambda f: extract_func_name(f), meta.void_funcs))
    if len(void_function_names) == 0:
        return ''
    return f'Do not create any tests for the void functions: {", ".join(void_function_names)}.'


async def _run_editor(loop, meta, user_message, model, stats_stage, debug=False, retries=2):
    # One implementor (editor) iteration: load the loop's editor prompt, run it, and split its
    # response into (preamble, artifact). Returns (artifact, preamble), or (None, '') on failure.
    _, system = load_editor(loop)
    system = system.format(filename=meta.filename, void_note=_void_note(meta))
    if loop == 'impl':
        first_header = f'{meta.filename}.py'

        def validate(artifact):
            return validate_impl_markdown(artifact, meta.filename)
    else:
        first_header = f'{meta.filename}_test.py'

        def validate(artifact):
            return validate_second_stage_markdown(artifact, first_header)
    for attempt in range(retries):
        try:
            mapper = get_mapper(system, n_results=1,
                                stats_stage=stats_stage, model=model, label=f'{loop}-editor')
            doc = await mapper.run(user_message)
            preamble, artifact = split_preamble(doc, first_header)
            if not validate(artifact):
                if debug:
                    print(f'[Editor {loop}] Invalid artifact:\n{doc}')
                raise Exception('Invalid output format')
            return artifact, preamble
        except Exception:
            if debug:
                print(f'[Editor {loop}] attempt {attempt + 1} failed')
            if attempt == retries - 1:
                return None, ''
    return None, ''


_COMPACT_PROMPT = '''You are consolidating a list of code-review findings so it fits within a smaller context budget.
You are given findings, one per line, each referenced by a [Name-Label]. Different reviewers (the names) may have raised the same point, and some findings may already be resolved or no longer needed.
Reduce the list by doing BOTH of the following:
1. Drop any finding that is already resolved, superseded, or no longer needs to be acted on.
2. When two or more findings make the same point, keep ONLY the single most detailed one and drop the rest. Preserve the exact [Name-Label] of the most detailed one.
You MUST preserve each kept finding's [Name-Label] exactly as it was given: do not rename, renumber, or invent labels. The reduced list may therefore skip some letter/number combinations.
Respond with ONLY the reduced list, one finding per line, in exactly the format you were given:
- [Name-Label] SEVERITY <location> - <description>
Output no prose, no numbering, no code fences, and no finding that was not in the input. If no findings should be kept, respond with exactly: NO FINDINGS
'''


def _trim_findings_to_budget(findings, fits_check):
    # Deterministic last-resort reduction: drop findings from lowest to highest severity until
    # fits_check passes (or nothing is left). Guarantees the editor never receives an over-budget
    # prompt unless even the highest-severity findings alone exceed it.
    order = {'NIT': 0, 'MINOR': 1, 'MAJOR': 2}
    current = list(findings)
    while not fits_check(current) and current:
        worst = min(current, key=lambda f: order.get(f['severity'], 1))
        current.remove(worst)
    return current


async def compact_findings(meta, findings, model, debug=False, retries=2, prior_context=''):
    # A dedicated LLM pass that shrinks the findings list to fit the context budget: drop
    # resolved/redundant findings and merge cross-reviewer duplicates (keeping the more detailed
    # [Name-Label]). Survivors keep their original labels so cross-references stay valid. Returns
    # the original list unchanged unless the LLM produced a strictly smaller, well-formed list.
    gpt = get_mapper(_COMPACT_PROMPT, n_results=1,
                     stats_stage='third_stage', model=model, label='compact')
    user = f'''{format_marsha_for_llm(meta)}
{prior_context}
# Review findings to reduce

{format_findings(findings)}'''
    for attempt in range(retries):
        try:
            text = await gpt.run(user)
            compacted = parse_compacted_findings(text)
            if 0 < len(compacted) < len(findings):
                return compacted
            return findings
        except Exception:
            if debug:
                print(f'[Compact] attempt {attempt + 1} failed')
            if attempt == retries - 1:
                return findings
    return findings


async def _budgeted_findings(meta, findings, build, model, args, debug=False):
    # Ensure the editor prompt build(findings) fits the context budget. If not, compact the
    # findings (LLM), then deterministically trim by severity, so the editor never receives an
    # over-budget prompt. If the budget cannot be determined (e.g. no client in a test harness),
    # skip budgeting and send the findings as-is. Returns the (possibly reduced) findings.
    override = getattr(args, 'context_window', None)
    cap = getattr(args, 'context_cap', 0.5)
    try:
        client = get_client()
        ctx = await resolve_context_window(model=model, client=client, override=override)
    except Exception:
        return findings

    def fits_check(current):
        return fits(build(current), ctx, cap)
    if fits_check(findings):
        log(f'editor prompt ~{estimate_tokens(build(findings))} tokens, within budget '
            f'{budget_tokens(ctx, cap)}; no compaction needed')
        return findings
    log(f'editor prompt ~{estimate_tokens(build(findings))} tokens exceeds budget '
        f'{budget_tokens(ctx, cap)}; compacting {len(findings)} findings')
    reduced = await compact_findings(meta, findings, model, debug=debug)
    if fits_check(reduced):
        return reduced
    log('compaction insufficient; trimming lowest-severity findings to fit')
    return _trim_findings_to_budget(reduced, fits_check)


def _oracle_review_message(meta, oracle_md):
    return f'''{format_marsha_for_llm(meta)}
{_void_note(meta)}

# The current test suite (oracle)

{oracle_md}'''


def _oracle_editor_message(meta, oracle_md, findings):
    return f'''{format_marsha_for_llm(meta)}

# The current test suite (oracle)

{oracle_md}

# Review findings to address

{format_findings(findings)}'''


async def optimize_test_suite(meta: MarshaMeta, oracle_md: str, args, debug: bool = False) -> str:
    # Per-phase inner loop for the oracle: run the reviewer personas, hand their findings to the
    # editor (Wren), and iterate until there are no actionable findings or the suite stabilizes.
    level = args.optimize
    if level <= 0:
        return oracle_md
    registry = build_registry()
    reviewers = resolve_loop_reviewers('oracle', args.test_personas, registry)
    model = resolve_model()
    severities = parse_severities(args.optimize_severity)
    prior_findings, prior_preamble = [], ''
    for i in range(level):
        log(f'oracle loop iteration {i + 1}/{level}: running reviewers')
        user_message = _oracle_review_message(meta, oracle_md)
        if i > 0:
            user_message += prior_round_block(prior_findings, prior_preamble)
        findings = await run_personas(reviewers, user_message, model, 'first_stage', debug, loop='oracle')
        actionable = actionable_findings(findings, severities)
        if not actionable:
            if debug:
                print(
                    f'[Optimize oracle] iteration {i + 1}: no actionable findings, converged')
            break
        actionable = await _budgeted_findings(
            meta, actionable,
            lambda fs: _oracle_editor_message(meta, oracle_md, fs),
            model, args, debug)
        artifact, preamble = await _run_editor(
            'oracle', meta, _oracle_editor_message(
                meta, oracle_md, actionable),
            model, 'first_stage', debug)
        if artifact is None:
            if debug:
                print(
                    f'[Optimize oracle] iteration {i + 1}: editor failed, stopping')
            break
        if artifact.strip() == oracle_md.strip():
            if debug:
                print(
                    f'[Optimize oracle] iteration {i + 1}: suite unchanged, converged')
            break
        if debug:
            print(f'[Optimize oracle] iteration {i + 1}: suite updated')
        oracle_md = artifact
        prior_findings, prior_preamble = actionable, preamble
    return oracle_md


async def gpt_implementation(meta: MarshaMeta, oracle_md: str, n_results: int, retries: int = 3, debug: bool = False):
    # Generate implementations against the (already generated) oracle. The implementation
    # must satisfy the spec AND pass the provided test suite; on any conflict the spec wins.
    marsha_for_code_llm = format_marsha_for_llm(meta)
    gpt_gen_code = get_mapper(f'''You are a senior software engineer assigned to write Python 3 functions.
The assignment is written in markdown format.
The description of each function should be included as a docstring.
Add type hints if feasible.
The filename should exactly match the name `{meta.filename}.py`.
Make sure to follow PEP8 guidelines.
Make sure to include all needed standard Python libraries imports.
Generate `requirements.txt` file with all needed dependencies, do not add fixed version to dependencies.
If need to convert `type` to Python classes, you will receive a markdown where the heading is the class name followed by several rows following a comma separated CSV format where the first row contains all class properties and the following rows contain examples of the values of those properties. Make sure to add the __str__, __repr__, and __eq__ methods to the class.
A unit test suite has already been written from this same assignment and is provided to you. Your implementation must satisfy the assignment AND pass this test suite. If anything in the test suite ever appears to conflict with the assignment, the assignment is authoritative.
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

''', n_results=n_results, stats_stage='first_stage', label='impl-gen')
    user_request = f'''{marsha_for_code_llm}

## The unit test suite your implementation must pass

{oracle_md}'''
    if debug:
        print(f'''marsha_for_llm =
    ---- start ----
{marsha_for_code_llm}
    ---- end ----''')
    reses = await gpt_gen_code.run(user_request)
    if isinstance(reses, str):
        # run() returns a bare string for a single result; normalize to a list so -n 1 works.
        reses = [reses]
    # The output should be a valid list of implementation Markdown documents (code + optional
    # requirements). Parse each one and keep the valid docs; if none are valid, retry.
    try:
        mds = list()
        for doc in reses:
            if validate_impl_markdown(doc, meta.filename):
                mds.append(doc)
            else:
                if debug:
                    print(f'''[Implementation] Invalid doc:
{doc}''')
        if len(mds) == 0:
            raise Exception('Invalid output format')
        return mds
    except Exception:
        if debug:
            print(
                f'Failed to generate implementation. Retries left = {retries}. Retrying...')
        if retries > 0:
            return await gpt_implementation(
                meta, oracle_md, n_results, retries - 1, debug)
        else:
            raise Exception('Failed to generate code', meta.filename)


async def run_test_suite(code_file: str, test_file: str, req_file: str, debug: bool = False):
    # Set up the venv (if needed), install requirements, and run the test suite.
    # Returns (passed, results): passed is None if the suite could not be run at all,
    # False if it ran but failed, and True if it passed.
    code_file_dir = os.path.dirname(os.path.abspath(code_file))
    venv_path = f'{code_file_dir}/venv'
    if req_file and os.path.exists(req_file):
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
            pip_exe = f'{venv_path}/Scripts/pip.exe' if platform.system(
            ) == 'Windows' else f'{venv_path}/bin/pip'
            pip_stream = await asyncio.create_subprocess_exec(
                pip_exe, 'install', '--disable-pip-version-check', '--no-compile', '-r', req_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await run_subprocess(pip_stream, 120)
        except Exception as e:
            if debug:
                print('Failed to install requirements', e)
    if not os.path.exists(venv_path):
        python_exe = python
    else:
        python_exe = f'{venv_path}/Scripts/python.exe' if platform.system(
        ) == 'Windows' else f'{venv_path}/bin/python'
    try:
        test_stream = await asyncio.create_subprocess_exec(
            python_exe, test_file, '-f', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await run_subprocess(test_stream)
        results = f'''{stdout}{stderr}'''
    except Exception as e:
        print('Failed to run test suite...', e)
        return (None, '')
    passed = ('FAILED' not in results) and ('Traceback' not in results)
    return (passed, results)


def _impl_review_message(meta, oracle, code):
    return f'''{format_marsha_for_llm(meta)}

# The unit test suite (oracle) the implementation must keep passing

{oracle}

# The current implementation

```py
{code}
```'''


def _impl_editor_message(meta, oracle, code, findings):
    return f'''{format_marsha_for_llm(meta)}

# The unit test suite (oracle) the implementation must keep passing

{oracle}

# The current implementation

```py
{code}
```

# Review findings to address

{format_findings(findings)}'''


async def optimize_implementation(args, meta: MarshaMeta, files: list[str], debug: bool = False):
    # Per-phase inner loop for the implementation, run after the candidate already passes the
    # oracle. Reviewer personas propose findings; the editor (Cody) applies them; the guardrail
    # re-runs the oracle and reverts any change that regresses, so the loop can never ship broken.
    level = args.optimize
    if level <= 0:
        return
    code_file = [file for file in files if file.endswith(
        f'{meta.filename}.py')][0]
    test_file = [file for file in files if file.endswith(
        f'{meta.filename}_test.py')][0]
    req_files = [file for file in files if file.endswith('requirements.txt')]
    req_file = req_files[0] if len(req_files) > 0 else None
    subdir = os.path.dirname(os.path.abspath(code_file))
    oracle = read_file(test_file)
    registry = build_registry()
    reviewers = resolve_loop_reviewers('impl', args.impl_personas, registry)
    model = resolve_strong_model()
    severities = parse_severities(args.optimize_severity)
    prior_findings, prior_preamble = [], ''
    for i in range(level):
        log(f'impl loop iteration {i + 1}/{level}: running reviewers')
        current_code = read_file(code_file)
        user_message = _impl_review_message(meta, oracle, current_code)
        if i > 0:
            user_message += prior_round_block(prior_findings, prior_preamble)
        findings = await run_personas(reviewers, user_message, model, 'third_stage', debug, loop='impl')
        actionable = actionable_findings(findings, severities)
        if not actionable:
            if debug:
                print(
                    f'[Optimize impl] iteration {i + 1}: no actionable findings, converged')
            break
        actionable = await _budgeted_findings(
            meta, actionable,
            lambda fs: _impl_editor_message(meta, oracle, current_code, fs),
            model, args, debug)
        artifact, preamble = await _run_editor(
            'impl', meta, _impl_editor_message(
                meta, oracle, current_code, actionable),
            model, 'third_stage', debug)
        if artifact is None:
            if debug:
                print(
                    f'[Optimize impl] iteration {i + 1}: editor failed, stopping')
            break
        backup_code = current_code
        backup_req = read_file(req_file) if (
            req_file and os.path.exists(req_file)) else None
        write_files_from_markdown(artifact, subdir=subdir)
        if read_file(code_file) == backup_code:
            if debug:
                print(
                    f'[Optimize impl] iteration {i + 1}: unchanged, converged')
            break
        passed, _ = await run_test_suite(code_file, test_file, req_file, debug)
        if passed:
            if debug:
                print(f'[Optimize impl] iteration {i + 1}: improvement kept')
            prior_findings, prior_preamble = actionable, preamble
        else:
            write_file(code_file, backup_code)
            if backup_req is not None:
                write_file(req_file, backup_req)
            if debug:
                print(
                    f'[Optimize impl] iteration {i + 1}: regressed, reverted')
    return


async def fix_file(marsha_filename: str, filename: str, lint_text: str, retries: int = 3, debug: bool = False):
    code = read_file(filename)
    gpt_fix = get_mapper(f'''You are a senior software engineer working with Python 3.
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

''', stats_stage='second_stage', label='lint-fix')
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


# pyflakes findings that are style noise (the linter previously suppressed them); everything else
# (e.g. undefined names) is a real problem worth showing the LLM.
_PYFLAKES_NOISE = ('imported but unused',
                   'is assigned to but never used', 'is defined but unused')


def _lint_files(files, ignore):
    """Run pycodestyle (style/syntax) and pyflakes (semantics) over the given files and return a
    mapping of filename -> list of '<file>:<line>:<col>: <code> <message>' strings. Replaces the
    former pylama call, whose plugin discovery imports the deprecated pkg_resources API."""
    findings = {f: [] for f in files}
    style_options = pycodestyle.StyleGuide(
        quiet=True, ignore=list(ignore)).options
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


async def lint_and_fix_files(marsha_filename: str, files: list[str], max_depth: int = 4, debug: bool = False):
    if max_depth == 0:
        raise Exception('Failed to fix code', files)
    # pycodestyle (style/syntax) + pyflakes (semantics), run directly: pylama was dropped because
    # it imports the deprecated pkg_resources API. We lint to catch coarse errors like undefined
    # names; we do not want the LLM fixing every style nit — the output goes through Black anyway —
    # so a large set of cosmetic rules is ignored below.
    _lint_ignore = {
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

    lints_by_file = _lint_files(files, _lint_ignore)

    if all(len(v) == 0 for v in lints_by_file.values()):
        return

    jobs = []
    for file in files:
        file_lints = lints_by_file.get(file, [])
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


async def diagnose_failure(meta: MarshaMeta, code: str, tests: str, results: str, retries: int = 2):
    # Read-only: decide whether the implementation or a test is at fault. Uses the standard
    # (cheap) model, since the safety property is enforced structurally, not by this call.
    gpt_diag = get_mapper(DIAGNOSE_PROMPT, n_results=1,
                          stats_stage='third_stage', label='diagnose')
    user_request = f'''{format_marsha_for_llm(meta)}

# {meta.filename}.py

```py
{code}
```

# {meta.filename}_test.py

```py
{tests}
```

# Test Results

{results}'''
    try:
        return parse_diagnosis(await gpt_diag.run(user_request))
    except Exception:
        if retries > 0:
            return await diagnose_failure(meta, code, tests, results, retries - 1)
        # If diagnosis keeps failing, fall back to the common case (fix the implementation)
        return {'fault': 'implementation', 'reason': 'diagnosis unavailable; defaulting to fixing the implementation'}


async def fix_implementation(meta: MarshaMeta, code: str, tests: str, results: str, reason: str, retries: int = 3, debug: bool = False):
    # Edit ONLY the implementation. The oracle is provided as fixed context and is never
    # part of the editable output, so this path structurally cannot touch the test suite.
    gpt_fix = get_mapper(f'''You are a senior software engineer fixing a Python 3 implementation that is failing its unit tests.
You are given the assignment, the implementation, the unit test suite (the oracle), and the test results.
The unit test suite is authoritative and must NOT be modified.
Fix only the implementation so that it correctly implements the assignment and passes the test suite, making the least changes necessary.
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

''', model=resolve_strong_model(), stats_stage='third_stage', label='impl-fix')
    user_request = f'''{format_marsha_for_llm(meta)}

# {meta.filename}.py

```py
{code}
```

# {meta.filename}_test.py

```py
{tests}
```

# Test Results

{results}

# Diagnosis

The implementation is at fault: {reason}'''
    fixed_code = await gpt_fix.run(user_request)
    try:
        if not validate_impl_markdown(fixed_code, meta.filename):
            if debug:
                print(f'''[Fix implementation] Invalid doc:
{fixed_code}''')
            raise Exception('Invalid output format')
        return fixed_code
    except Exception:
        if retries > 0:
            return await fix_implementation(meta, code, tests, results, reason, retries - 1, debug)
        else:
            raise Exception('Failed to fix implementation', meta.filename)


async def correct_test(meta: MarshaMeta, code: str, tests: str, results: str, reason: str, retries: int = 3, debug: bool = False):
    # The "punt back": the only path that may edit the oracle, and only spec-anchored. It must
    # justify every change by the assignment and must never weaken a test that is actually
    # correct (so a misrouted diagnosis degrades to a no-op rather than a bent test).
    gpt_fix = get_mapper(f'''You are a senior software engineer correcting a faulty unit test.
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

''', model=resolve_strong_model(), stats_stage='third_stage', label='test-correct')
    user_request = f'''{format_marsha_for_llm(meta)}

# {meta.filename}.py

```py
{code}
```

# {meta.filename}_test.py

```py
{tests}
```

# Test Results

{results}

# Diagnosis

A test is at fault: {reason}'''
    fixed_test = await gpt_fix.run(user_request)
    try:
        if not validate_second_stage_markdown(fixed_test, f'{meta.filename}_test.py'):
            if debug:
                print(f'''[Correct test] Invalid doc:
{fixed_test}''')
            raise Exception('Invalid output format')
        return fixed_test
    except Exception:
        if retries > 0:
            return await correct_test(meta, code, tests, results, reason, retries - 1, debug)
        else:
            raise Exception('Failed to correct test', meta.filename)


def _code_from_test_md(md: str) -> str:
    # Extract the python code from a single-file test markdown document
    m = re.search(r'```[^\n]*\n(.*?)\n```', md, re.DOTALL)
    return m.group(1) if m else md


def _correction_review_message(meta, code, orig_test, corrected_code, reason):
    return f'''{format_marsha_for_llm(meta)}

# {meta.filename}.py

```py
{code}
```

# Original test suite

```py
{orig_test}
```

# Corrected test suite

```py
{corrected_code}
```

# Diagnosis

A test is at fault: {reason}'''


def _correction_editor_message(meta, code, orig_test, corrected_code, reason, findings):
    return _correction_review_message(
        meta, code, orig_test, corrected_code, reason) + f'''

# Review findings to address

{format_findings(findings)}'''


async def validate_test_correction(meta: MarshaMeta, code: str, orig_test: str, corrected_md: str, reason: str, args, debug: bool = False) -> str:
    # Third per-phase loop: reviewer personas check the test correction's reasoning and
    # spec-alignment before it is written, so the only path that can bend the oracle is itself
    # spec-anchored and double-checked. The editor (Rex) applies the findings.
    level = args.optimize
    if level <= 0:
        return corrected_md
    registry = build_registry()
    reviewers = resolve_loop_reviewers(
        'correction', args.fix_personas, registry)
    model = resolve_strong_model()
    severities = parse_severities(args.optimize_severity)
    prior_findings, prior_preamble = [], ''
    for i in range(level):
        log(f'correction loop iteration {i + 1}/{level}: running reviewers')
        corrected_code = _code_from_test_md(corrected_md)
        user_message = _correction_review_message(
            meta, code, orig_test, corrected_code, reason)
        if i > 0:
            user_message += prior_round_block(prior_findings, prior_preamble)
        findings = await run_personas(reviewers, user_message, model, 'third_stage', debug, loop='correction')
        actionable = actionable_findings(findings, severities)
        if not actionable:
            if debug:
                print(
                    f'[Validate correction] iteration {i + 1}: no actionable findings, validated')
            break
        actionable = await _budgeted_findings(
            meta, actionable,
            lambda fs: _correction_editor_message(
                meta, code, orig_test, corrected_code, reason, fs),
            model, args, debug)
        artifact, preamble = await _run_editor(
            'correction', meta,
            _correction_editor_message(
                meta, code, orig_test, corrected_code, reason, actionable),
            model, 'third_stage', debug)
        if artifact is None:
            if debug:
                print(
                    f'[Validate correction] iteration {i + 1}: editor failed, keeping current correction')
            break
        if artifact.strip() == corrected_md.strip():
            if debug:
                print(
                    f'[Validate correction] iteration {i + 1}: correction validated, unchanged')
            break
        if debug:
            print(
                f'[Validate correction] iteration {i + 1}: correction revised')
        corrected_md = artifact
        prior_findings, prior_preamble = actionable, preamble
    return corrected_md


async def test_and_fix_files(meta: MarshaMeta, files: list[str], retries: int = 4, debug: bool = False, args=None):
    if retries == 0:
        raise Exception('Failed to fix code', meta.filename)
    # There should only be two files, the test file and the code file
    test_file = [file for file in files if file.endswith(
        f'{meta.filename}_test.py')][0]
    code_file = [file for file in files if file.endswith(
        f'{meta.filename}.py')][0]
    req_files = [file for file in files if file.endswith('requirements.txt')]
    req_file = req_files[0] if len(req_files) > 0 else None
    passed, test_results = await run_test_suite(code_file, test_file, req_file, debug)
    if passed is None:  # If the test suite failed to run, we try again
        log('test suite failed to run; retrying')
        return await test_and_fix_files(meta, files, retries - 1, debug, args)
    log(f'test suite: {"passed" if passed else "FAILED"} (retries left={retries})')
    if not passed:
        if debug:
            print('Test failed, diagnosing the root cause')
            print(test_results)
        test = read_file(test_file)
        code = read_file(code_file)
        # Diagnose whether the implementation or a test is at fault, then route to the matching
        # fix. Only one artifact is ever edited per pass: the implementation fix structurally
        # cannot touch the oracle, and the (spec-anchored, double-checked) test correction is the
        # only path that may.
        verdict = await diagnose_failure(meta, code, test, test_results)
        log(f'diagnosis: fault={verdict["fault"]} - {verdict["reason"]}')
        subdir = '/'.join(code_file.split('/')[:-1])
        if verdict['fault'] == 'test':
            if debug:
                print(f'Punting back to the test layer: {verdict["reason"]}')
            fixed = await correct_test(
                meta, code, test, test_results, verdict['reason'], debug=debug)
            if args is not None and args.optimize > 0:
                fixed = await validate_test_correction(
                    meta, code, test, fixed, verdict['reason'], args, debug=debug)
            write_files_from_markdown(fixed, subdir=subdir)
        else:
            if debug:
                print(f'Fixing the implementation: {verdict["reason"]}')
            fixed = await fix_implementation(
                meta, code, test, test_results, verdict['reason'], debug=debug)
            write_files_from_markdown(fixed, subdir=subdir)
        # Re-run the tests recursively; the recursion ejects when they pass. The file paths are
        # stable across passes (same directory), so pass the original file list down unchanged.
        return await test_and_fix_files(meta, files, retries - 1, debug, args)


async def generate_python_code(args, meta: MarshaMeta, n_results: int, debug: bool) -> list[str]:
    t1 = time.time()
    print('Generating Python code...')
    log(f'first stage: {meta.filename} (n={n_results})')
    mds = None
    try:
        if not args.exclude_sanity_check:
            log('spec sanity check')
            check = await gpt_check_spec(meta)
            if not args.no_warn:
                for warning in check['warnings']:
                    print_diagnostic('warning', warning)
            for error in check['errors']:
                print_diagnostic('error', error)
            if not check['compilable']:
                sys.exit(1)
        # Oracle-first: generate the spec-anchored test suite, then generate implementations
        # against it. Each candidate keeps the standard (code, requirements, test) shape, but the
        # test section is the shared oracle, so the implementation is written to a fixed oracle.
        oracle = await gpt_test_suite(meta, debug=debug)
        log(f'oracle test suite generated ({len(oracle)} chars)')
        if args.optimize > 0 and not args.quick_and_dirty:
            print('Verifying test suite coverage and fidelity...')
            log(f'oracle optimize loop: {args.optimize} iteration(s)')
            oracle = await optimize_test_suite(meta, oracle, args, debug)
        log(f'generating {n_results} implementation(s)')
        impls = await gpt_implementation(meta, oracle, n_results, debug=debug)
        mds = []
        for impl in impls:
            doc = impl + '\n\n' + oracle
            if validate_first_stage_markdown(doc, meta.filename):
                mds.append(doc)
        if len(mds) == 0:
            raise Exception('No valid implementation candidates')
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
    log(f'second stage: {meta.filename} (lint & fix)')
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
    if debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
    t_tsi = time.time()
    print('Verifying and correcting generated code...')
    log('third stage: verify & correct')
    try:
        await test_and_fix_files(meta, files, debug=debug, args=args)
    except Exception as e:
        print('Third stage failure')
        print(e)
        raise e
    finally:
        t_tsii = time.time()
        stats.third_stage.total_time = prettify_time_delta(
            t_tsii - t_tsi)
    if args.optimize > 0:
        print('Optimizing implementation...')
        log(f'impl optimize loop: {args.optimize} iteration(s)')
        await optimize_implementation(args, meta, files, debug)
    if debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
    print('Formatting code...')
    log('formatting')
    autoformat_files(files)
    if debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
