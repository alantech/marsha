"""The implementation-phase planning pipeline for `marsha diff` (issue #219).

The monolithic implementor is replaced by a pipeline of specialized personas:
an explorer (maps relevant files), a rule checker (extracts conventions), and
a planner (produces a sequential markdown plan). The plan is the keystone
artifact: the implementor works step-by-step against it, the plan checker
verifies coverage, and the review threads track per-finding resolution.

This module owns the plan data structures, the markdown plan parser, the
planner prompt, and the runner functions for the three planning personas.
The step-by-step implement loop and the review-thread state machine live in
`marsha.diff` (Phase 2 and Phase 3).
"""
from __future__ import annotations

import dataclasses
import os
import re
from typing import cast

from marsha import tools
from marsha.log import log
from marsha.mappers import get_mapper
from marsha.personas import load_persona, personas_dir


# --- plan data structures -----------------------------------------------


@dataclasses.dataclass
class PlanStep:
    """One step in the implementation plan."""
    id: int
    goal: str
    files: list[str]
    test_first: str  # 'test_path::test_name' or 'none'
    approach: str
    validate: str


@dataclasses.dataclass
class Plan:
    """The parsed implementation plan."""
    title: str
    scope_files: list[str]
    scope_context: str
    constraints: list[str]
    steps: list[PlanStep]


# --- markdown plan parser -----------------------------------------------

_PLAN_TITLE_RE = re.compile(r'^# Plan:\s*(.+)', re.MULTILINE)
_SCOPE_RE = re.compile(r'^## Scope\s*\n(.*?)(?=^## |\Z)',
                       re.MULTILINE | re.DOTALL)
_SCOPE_FILE_RE = re.compile(r'^-\s*(\S+)\s*\((.+)\)')
_CONTEXT_RE = re.compile(r'^Context:\s*(.+)', re.MULTILINE)
_CONSTRAINTS_RE = re.compile(r'^## Constraints\s*\n(.*?)(?=^## |\Z)',
                             re.MULTILINE | re.DOTALL)
_CONSTRAINT_RE = re.compile(r'^-\s*(.+)', re.MULTILINE)
_STEPS_RE = re.compile(r'^## Steps\s*\n(.*)$', re.MULTILINE | re.DOTALL)
_STEP_RE = re.compile(r'^### Step (\d+):\s*(.+)\n((?:.+\n)*)', re.MULTILINE)
_STEP_FILES_RE = re.compile(r'^Files:\s*(.+)', re.MULTILINE)
_STEP_TEST_RE = re.compile(r'^Test first:\s*(.+)', re.MULTILINE)
_STEP_APPROACH_RE = re.compile(r'^Approach:\s*(.+)', re.MULTILINE)
_STEP_VALIDATE_RE = re.compile(r'^Validate:\s*(.+)', re.MULTILINE)


def parse_plan(text: str) -> Plan:
    """Parse a markdown plan into a Plan data structure.

    The plan format is:

        # Plan: <title>

        ## Scope
        Relevant files:
        - path (role)
        - path (role)

        Context: <1-2 sentences>

        ## Constraints
        - rule
        - rule

        ## Steps

        ### Step 1: <goal>
        Files: file1, file2
        Test first: test::name
        Approach: <text>
        Validate: <command>

        ### Step 2: <goal>
        Files: file3
        Test first: none
        Approach: <text>
        Validate: <command>
    """
    title_m = _PLAN_TITLE_RE.search(text)
    title = title_m.group(1).strip() if title_m else '(untitled)'

    scope_files: list[str] = []
    scope_m = _SCOPE_RE.search(text)
    if scope_m:
        for line in scope_m.group(1).split('\n'):
            fm = _SCOPE_FILE_RE.match(line.strip())
            if fm:
                scope_files.append(f'{fm.group(1)} ({fm.group(2)})')

    context_m = _CONTEXT_RE.search(text)
    context = context_m.group(1).strip() if context_m else ''

    constraints: list[str] = []
    constraints_m = _CONSTRAINTS_RE.search(text)
    if constraints_m:
        for cm in _CONSTRAINT_RE.finditer(constraints_m.group(1)):
            constraints.append(cm.group(1).strip())

    steps: list[PlanStep] = []
    steps_m = _STEPS_RE.search(text)
    if steps_m:
        steps_text = steps_m.group(1)
        for sm in _STEP_RE.finditer(steps_text):
            body = sm.group(3)
            files_m = _STEP_FILES_RE.search(body)
            files = ([f.strip() for f in files_m.group(1).split(',')]
                     if files_m else [])
            test_m = _STEP_TEST_RE.search(body)
            test_first = test_m.group(1).strip() if test_m else 'none'
            approach_m = _STEP_APPROACH_RE.search(body)
            approach = approach_m.group(1).strip() if approach_m else ''
            validate_m = _STEP_VALIDATE_RE.search(body)
            validate = validate_m.group(1).strip() if validate_m else ''
            steps.append(PlanStep(
                id=int(sm.group(1)),
                goal=sm.group(2).strip(),
                files=files,
                test_first=test_first,
                approach=approach,
                validate=validate))

    return Plan(
        title=title,
        scope_files=scope_files,
        scope_context=context,
        constraints=constraints,
        steps=steps)


def format_step(step: PlanStep) -> str:
    """Render one plan step in the canonical markdown format."""
    lines = [f'### Step {step.id}: {step.goal}',
             f'Files: {", ".join(step.files)}',
             f'Test first: {step.test_first}',
             f'Approach: {step.approach}',
             f'Validate: {step.validate}']
    return '\n'.join(lines)


# --- planner prompt -----------------------------------------------------

PLANNER_PROMPT = '''\
You are the planner. You are given a design-locked specification, an explorer's \
map of the relevant files and subsystems, and a list of constraints (conventions \
and rules). Your charge is to produce a sequential implementation plan.

The plan is a markdown document in this exact format:

# Plan: <spec title>

## Scope
Relevant files:
- <path> (<role>)
...

Context: <1-2 sentences>

## Constraints
- <rule>
...

## Steps

### Step 1: <goal>
Files: <file1>, <file2>
Test first: <test path::test name>
Approach: <1-2 sentences>
Validate: <command>

### Step 2: <goal>
Files: <file>
Test first: none
Approach: <1-2 sentences>
Validate: <command>

Rules:
- Each step is one focused change: a single component, module, or logical unit.
- Bias towards TDD: when a step modifies a function or module that has (or will \
have) a test file, set "Test first" to the test to write. The step writes the \
test, confirms it fails, implements, confirms it passes. For structural steps \
(new config, build-system change, module skeleton), set "Test first" to "none".
- The "Validate" command is the specific test or command that confirms the step \
is done. It must be runnable in the repository.
- Steps are sequential: each step assumes the prior steps are complete.
- The plan must cover the full scope of the spec: every requirement in the spec \
has a corresponding step.
- Keep the number of steps minimal but complete: 3-10 steps for a typical feature.
- The "Approach" is 1-2 sentences: what to do and how, not a full implementation.
- The "Files" list is the specific files to create or modify in this step.
- Echo the explorer's scope files and the constraints list verbatim in the Scope \
and Constraints sections.

Produce only the markdown plan, with no preamble or commentary.
'''


# --- runner functions ---------------------------------------------------


async def run_explorer(spec_text: str, cwd: str, model: str,
                       debug: bool = False) -> str:
    """Run the explorer persona: it reads the spec, explores the codebase, and
    produces a file map + context. Returns the explorer's markdown output."""
    name, body = load_persona(
        os.path.join(personas_dir(), 'explorer-scope.md'))
    ctx = tools.ToolContext(phase='review', workdir=cwd)
    system = body + tools.tool_instructions(ctx)
    mapper = get_mapper(
        system, n_results=1, model=model, label=f'plan:{name}',
        reasoning_effort='high')
    request = ('Explore the repository to identify the files and subsystems '
               'relevant to this specification.\n\n'
               + tools.wrap_untrusted('spec', spec_text))
    result = await tools.run_with_tools(
        mapper, request, ctx, debug=debug, max_rounds=50)
    log(f'plan: explorer ({name}) completed')
    return cast(str, result)


async def run_rule_checker(cwd: str, model: str,
                           debug: bool = False) -> str:
    """Run the rule checker persona: it reads AGENTS.md / CLAUDE.md and infers
    rules from codebase patterns. Returns the constraints list."""
    name, body = load_persona(
        os.path.join(personas_dir(), 'rule-checker-conventions.md'))
    ctx = tools.ToolContext(phase='review', workdir=cwd)
    system = body + tools.tool_instructions(ctx)
    mapper = get_mapper(
        system, n_results=1, model=model, label=f'plan:{name}',
        reasoning_effort='high')
    request = ('Identify the conventions and rules this repository follows that '
               'are relevant to new code. Produce the constraints list in the '
               'required format.')
    result = await tools.run_with_tools(
        mapper, request, ctx, debug=debug, max_rounds=30)
    log(f'plan: rule checker ({name}) completed')
    return cast(str, result)


async def run_planner(spec_text: str, explorer_output: str, rules: str,
                      model: str, debug: bool = False) -> Plan:
    """Run the planner: it takes the spec, explorer output, and rules, and
    produces the markdown plan. Returns the parsed Plan."""
    mapper = get_mapper(
        PLANNER_PROMPT, n_results=1, model=model, label='plan:planner',
        reasoning_effort='high')
    request = (
        f'## Specification\n\n{spec_text}\n\n'
        f'## Explorer output (relevant files and context)\n\n{explorer_output}\n\n'
        f'## Constraints (conventions and rules)\n\n{rules}\n\n'
        'Produce the implementation plan in the required markdown format.')
    text = await mapper.run(request)
    plan = parse_plan(text or '')
    log(f'plan: planner produced {len(plan.steps)} steps')
    return plan


def format_plan(p: Plan) -> str:
    """Render a Plan as canonical markdown (the format the planner produces
    and the plan checker / step loop consume)."""
    lines = [f'# Plan: {p.title}', '']
    if p.scope_files:
        lines.append('## Scope')
        lines.append('Relevant files:')
        for f in p.scope_files:
            lines.append(f'- {f}')
        lines.append('')
    if p.scope_context:
        lines.append(f'Context: {p.scope_context}')
        lines.append('')
    if p.constraints:
        lines.append('## Constraints')
        for c in p.constraints:
            lines.append(f'- {c}')
        lines.append('')
    if p.steps:
        lines.append('## Steps')
        lines.append('')
        for s in p.steps:
            lines.append(format_step(s))
            lines.append('')
    return '\n'.join(lines)


_PLAN_GAP_RE = re.compile(r'^-\s*Step\s+(\d+):\s*GAP\s*-\s*(.+)', re.MULTILINE)


async def run_plan_checker(p: Plan, cwd: str, model: str,
                           debug: bool = False) -> tuple[bool, list[str]]:
    """Run the plan checker: it verifies the working tree covers the plan.
    Returns (satisfied, gaps) where gaps is a list of gap descriptions."""
    name, body = load_persona(
        os.path.join(personas_dir(), 'plan-checker-coverage.md'))
    ctx = tools.ToolContext(phase='review', workdir=cwd)
    system = body + tools.tool_instructions(ctx)
    mapper = get_mapper(
        system, n_results=1, model=model, label=f'plan:{name}',
        reasoning_effort='high')
    plan_text = format_plan(p)
    request = (
        f'Verify that the working tree covers the plan below. For each step, '
        f'check that the required files exist and the goal is addressed.\n\n'
        f'## Plan\n{plan_text}\n\n'
        'Use the git tool and list-tree / find-in-file to read the working '
        'tree. Produce the plan check in the required format.')
    result = await tools.run_with_tools(
        mapper, request, ctx, debug=debug, max_rounds=30)
    result = cast(str, result)
    if 'PLAN SATISFIED' in result:
        log('plan: plan checker satisfied')
        return True, []
    gaps = []
    for m in _PLAN_GAP_RE.finditer(result):
        gaps.append(f'Step {m.group(1)}: {m.group(2).strip()}')
    if not gaps:
        log('plan: plan checker output unrecognized (no PLAN SATISFIED, no gaps)')
        return False, [
            'Unrecognized plan checker output; cannot verify coverage.']
    log(f'plan: plan checker found {len(gaps)} gap(s)')
    return False, gaps


_PLAN_MISSING_RE = re.compile(r'^-\s*MISSING:\s*(.+)', re.MULTILINE)


async def run_plan_completeness_check(spec_text: str, p: Plan, model: str,
                                      debug: bool = False) -> tuple[bool, list[str]]:
    """Verify the plan covers every requirement in the spec. Returns
    (complete, missing) where missing is a list of uncovered requirements."""
    name, body = load_persona(
        os.path.join(personas_dir(), 'plan-completeness-review.md'))
    mapper = get_mapper(
        body, n_results=1, model=model, label=f'plan:{name}',
        reasoning_effort='high')
    plan_text = format_plan(p)
    request = (
        f'## Specification\n\n{spec_text}\n\n'
        f'## Plan\n\n{plan_text}\n\n'
        'Verify that the plan covers every requirement in the specification. '
        'Produce your output in the required format.')
    result = await mapper.run(request)
    result = cast(str, result or '')
    if 'PLAN COMPLETE' in result:
        log('plan: plan completeness check passed')
        return True, []
    missing = []
    for m in _PLAN_MISSING_RE.finditer(result):
        missing.append(m.group(1).strip())
    if not missing:
        log('plan: plan completeness output unrecognized')
        return False, ['Unrecognized completeness check output.']
    log(
        f'plan: completeness check found {
            len(missing)} missing requirement(s)')
    return False, missing
