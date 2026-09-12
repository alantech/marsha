import asyncio
import json
import os
import re
import time
import traceback
import sys

from marsha import backends
from marsha.config import resolve_model, resolve_provider, resolve_strong_model
from marsha.context import budget_tokens, estimate_tokens, fits, resolve_context_window
from marsha.meta import MarshaMeta, void_note
from marsha.log import log
from marsha.parse import write_files_from_markdown, format_marsha_for_llm, split_preamble
from marsha.personas import (
    build_registry, resolve_loop_reviewers, load_editor, run_personas,
    actionable_findings, format_findings, prior_round_block, parse_severities,
    parse_compacted_findings,
)
from marsha.stats import stats
from marsha.term import print_diagnostic
from marsha.utils import read_file, write_file, prettify_time_delta
from marsha.llm_client import get_client
from marsha.mappers import get_mapper
from marsha.mappers.chatgpt import uses_completion_tokens


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
    gpt_check = get_mapper(backends.current().spec_check_prompt(), n_results=1,
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
    b = backends.current()
    gpt_gen_test = get_mapper(b.oracle_prompt(meta), n_results=1,
                              stats_stage='first_stage', label='oracle-gen')
    marsha_for_test_llm = format_marsha_for_llm(meta)
    if debug:
        print(f'''marsha_for_llm =
    ---- start ----
{marsha_for_test_llm}
    ---- end ----''')
    try:
        doc = await gpt_gen_test.run(marsha_for_test_llm)
        if not b.validate_markdown(doc, 'oracle', meta.filename):
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


async def _run_editor(loop, meta, user_message, model, stats_stage, debug=False, retries=2):
    # One implementor (editor) iteration: load the loop's editor prompt, run it, and split its
    # response into (preamble, artifact). Returns (artifact, preamble), or (None, '') on failure.
    b = backends.current()
    _, system = load_editor(loop)
    system = system.format(filename=meta.filename, void_note=void_note(meta))
    if loop == 'impl':
        first_header = b.source_name(meta.filename)

        def validate(artifact):
            return b.validate_markdown(artifact, 'impl', meta.filename)
    else:
        first_header = b.test_name(meta.filename)

        def validate(artifact):
            return b.validate_markdown(artifact, 'oracle', meta.filename)
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
{void_note(meta)}

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
        findings = await run_personas(reviewers, user_message, model, 'first_stage', debug,
                                      loop='oracle', guidance=backends.current().persona_guidance())
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
    b = backends.current()
    marsha_for_code_llm = format_marsha_for_llm(meta)
    gpt_gen_code = get_mapper(b.impl_prompt(meta), n_results=n_results,
                              stats_stage='first_stage', label='impl-gen')
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
            if b.validate_markdown(doc, 'impl', meta.filename):
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


def _impl_review_message(meta, oracle, code):
    b = backends.current()
    return f'''{format_marsha_for_llm(meta)}

# The unit test suite (oracle) the implementation must keep passing

{oracle}

# The current implementation

{b.code_block(code)}'''


def _impl_editor_message(meta, oracle, code, findings):
    b = backends.current()
    return f'''{format_marsha_for_llm(meta)}

# The unit test suite (oracle) the implementation must keep passing

{oracle}

# The current implementation

{b.code_block(code)}

# Review findings to address

{format_findings(findings)}'''


async def optimize_implementation(args, meta: MarshaMeta, files: list[str], debug: bool = False):
    # Per-phase inner loop for the implementation, run after the candidate already passes the
    # oracle. Reviewer personas propose findings; the editor (Cody) applies them; the guardrail
    # re-runs the oracle and reverts any change that regresses, so the loop can never ship broken.
    level = args.optimize
    if level <= 0:
        return
    b = backends.current()
    code_file = [file for file in files if file.endswith(
        b.source_name(meta.filename))][0]
    test_file = [file for file in files if file.endswith(
        b.test_name(meta.filename))][0]
    req_files = [file for file in files if file.endswith(b.manifest_name())]
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
        findings = await run_personas(reviewers, user_message, model, 'third_stage', debug,
                                      loop='impl', guidance=b.persona_guidance())
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
        passed, _ = await b.run_tests(code_file, test_file, req_file, debug)
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
    b = backends.current()
    code = read_file(filename)
    gpt_fix = get_mapper(b.lint_fix_prompt(filename),
                         stats_stage='second_stage', label='lint-fix')
    fixed_code = await gpt_fix.run(f'''# {filename}

{b.code_block(code)}

# lint results

```
{lint_text}
```''')
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        if not b.validate_markdown(fixed_code, 'unit', filename):
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
    # The backend lints (style + semantics) with the target toolchain; we lint to catch coarse
    # errors like undefined names, not every style nit — the output goes through the backend
    # formatter anyway.
    lints_by_file = backends.current().lint_files(files)

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


async def diagnose_failure(meta: MarshaMeta, code: str, tests: str, results: str, retries: int = 2):
    # Read-only: decide whether the implementation or a test is at fault. Uses the standard
    # (cheap) model, since the safety property is enforced structurally, not by this call.
    b = backends.current()
    gpt_diag = get_mapper(b.diagnose_prompt(), n_results=1,
                          stats_stage='third_stage', label='diagnose')
    user_request = f'''{format_marsha_for_llm(meta)}

# {b.source_name(meta.filename)}

{b.code_block(code)}

# {b.test_name(meta.filename)}

{b.code_block(tests)}

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
    b = backends.current()
    gpt_fix = get_mapper(b.fix_impl_prompt(meta), model=resolve_strong_model(),
                         stats_stage='third_stage', label='impl-fix')
    user_request = f'''{format_marsha_for_llm(meta)}

# {b.source_name(meta.filename)}

{b.code_block(code)}

# {b.test_name(meta.filename)}

{b.code_block(tests)}

# Test Results

{results}

# Diagnosis

The implementation is at fault: {reason}'''
    fixed_code = await gpt_fix.run(user_request)
    try:
        if not b.validate_markdown(fixed_code, 'impl', meta.filename):
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
    b = backends.current()
    gpt_fix = get_mapper(b.correct_test_prompt(meta), model=resolve_strong_model(),
                         stats_stage='third_stage', label='test-correct')
    user_request = f'''{format_marsha_for_llm(meta)}

# {b.source_name(meta.filename)}

{b.code_block(code)}

# {b.test_name(meta.filename)}

{b.code_block(tests)}

# Test Results

{results}

# Diagnosis

A test is at fault: {reason}'''
    fixed_test = await gpt_fix.run(user_request)
    try:
        if not b.validate_markdown(fixed_test, 'oracle', meta.filename):
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
    b = backends.current()
    return f'''{format_marsha_for_llm(meta)}

# {b.source_name(meta.filename)}

{b.code_block(code)}

# Original test suite

{b.code_block(orig_test)}

# Corrected test suite

{b.code_block(corrected_code)}

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
        findings = await run_personas(reviewers, user_message, model, 'third_stage', debug,
                                      loop='correction', guidance=backends.current().persona_guidance())
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
    b = backends.current()
    # There should only be two files, the test file and the code file
    test_file = [file for file in files if file.endswith(
        b.test_name(meta.filename))][0]
    code_file = [file for file in files if file.endswith(
        b.source_name(meta.filename))][0]
    req_files = [file for file in files if file.endswith(b.manifest_name())]
    req_file = req_files[0] if len(req_files) > 0 else None
    passed, test_results = await b.run_tests(code_file, test_file, req_file, debug)
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


async def generate_code(args, meta: MarshaMeta, n_results: int, debug: bool) -> list[tuple[str, str]]:
    t1 = time.time()
    b = backends.current()
    print(f'Generating {b.id} code...')
    log(f'first stage: {meta.filename} (n={n_results})')
    cands = None
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
        # against it. Each candidate is (impl, oracle); the oracle is the single canonical string
        # every impl is written against, and the backend composes them into the on-disk layout
        # at write time, so the impl path can never touch the oracle.
        oracle = await gpt_test_suite(meta, debug=debug)
        log(f'oracle test suite generated ({len(oracle)} chars)')
        if args.optimize > 0 and not args.quick_and_dirty:
            print('Verifying test suite coverage and fidelity...')
            log(f'oracle optimize loop: {args.optimize} iteration(s)')
            oracle = await optimize_test_suite(meta, oracle, args, debug)
        log(f'generating {n_results} implementation(s)')
        impls = await gpt_implementation(meta, oracle, n_results, debug=debug)
        cands = [(impl, oracle) for impl in impls]
        if len(cands) == 0:
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
    return cands


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
    backends.current().format_files(files)
    if debug:
        for file in files:
            print(f'# {file}\n{read_file(file)}\n')
