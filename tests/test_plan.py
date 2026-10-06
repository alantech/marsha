"""Tests for the implementation-phase planning pipeline (marsha.plan).

The plan parser is exercised with realistic markdown fixtures. The runner
functions (run_explorer, run_rule_checker, run_planner) are not tested here —
they are LLM-backed and will be exercised in Phase 2 integration tests.
"""
from typing import Any

from marsha import plan


# --- plan parser ---------------------------------------------------------------

SAMPLE_PLAN = '''\
# Plan: Add Rust and TypeScript code generation

## Scope
Relevant files:
- marsha/backends/__init__.py (backend registry)
- marsha/backends/base.py (LanguageBackend interface)
- marsha/backends/python.py (reference implementation)
- tests/test_backend.py (existing test patterns)

Context: The backend registry maps language IDs to LanguageBackend instances.
Python is the only wired target today.

## Constraints
- New backends must implement every LanguageBackend method (source: base.py:28)
- Test files follow `<name>_test.py` (source: inferred from existing backends)
- No new top-level dependencies without updating pyproject.toml (source: AGENTS.md)

## Steps

### Step 1: Register the Rust backend skeleton
Files: marsha/backends/rust.py, marsha/backends/__init__.py
Test first: tests/test_backend.py::test_rust_backend_registered
Approach: Create RustBackend(LanguageBackend) with id='rust', stub methods
that raise NotImplementedError. Register in backends/__init__.py.
Validate: pytest tests/test_backend.py::test_rust_backend_registered -q

### Step 2: Implement Rust naming and artifact contract
Files: marsha/backends/rust.py
Test first: tests/test_backend.py::test_rust_source_name
Approach: source_name -> `{fn}.rs`, test_name -> `{fn}_test.rs`
Validate: pytest tests/test_backend.py::test_rust_source_name -q

### Step 3: Add TypeScript backend skeleton
Files: marsha/backends/typescript.py, marsha/backends/__init__.py
Test first: none
Approach: Create TypeScriptBackend(LanguageBackend) with id='typescript'.
Validate: pytest tests/test_backend.py -q
'''


def test_parse_plan_title() -> None:
    p = plan.parse_plan(SAMPLE_PLAN)
    assert p.title == 'Add Rust and TypeScript code generation'


def test_parse_plan_scope_files() -> None:
    p = plan.parse_plan(SAMPLE_PLAN)
    assert len(p.scope_files) == 4
    assert p.scope_files[0] == 'marsha/backends/__init__.py (backend registry)'
    assert p.scope_files[1] == 'marsha/backends/base.py (LanguageBackend interface)'


def test_parse_plan_scope_context() -> None:
    p = plan.parse_plan(SAMPLE_PLAN)
    assert 'backend registry' in p.scope_context


def test_parse_plan_constraints() -> None:
    p = plan.parse_plan(SAMPLE_PLAN)
    assert len(p.constraints) == 3
    assert 'LanguageBackend method' in p.constraints[0]
    assert 'AGENTS.md' in p.constraints[2]


def test_parse_plan_steps() -> None:
    p = plan.parse_plan(SAMPLE_PLAN)
    assert len(p.steps) == 3
    assert p.steps[0].id == 1
    assert p.steps[0].goal == 'Register the Rust backend skeleton'
    assert p.steps[0].files == ['marsha/backends/rust.py',
                                'marsha/backends/__init__.py']
    assert p.steps[0].test_first == 'tests/test_backend.py::test_rust_backend_registered'
    assert p.steps[1].test_first == 'tests/test_backend.py::test_rust_source_name'
    assert p.steps[2].test_first == 'none'


def test_parse_plan_step_validate() -> None:
    p = plan.parse_plan(SAMPLE_PLAN)
    assert p.steps[0].validate == 'pytest tests/test_backend.py::test_rust_backend_registered -q'
    assert p.steps[2].validate == 'pytest tests/test_backend.py -q'


def test_parse_plan_empty() -> None:
    p = plan.parse_plan('')
    assert p.title == '(untitled)'
    assert p.steps == []
    assert p.constraints == []
    assert p.scope_files == []


def test_parse_plan_no_steps() -> None:
    p = plan.parse_plan('# Plan: Test\n\n## Scope\n\n## Constraints\n- rule\n')
    assert p.title == 'Test'
    assert p.constraints == ['rule']
    assert p.steps == []


def test_format_step() -> None:
    step = plan.PlanStep(
        id=1, goal='Do X', files=['a.py', 'b.py'],
        test_first='tests/test_a.py::test_x',
        approach='Write the test, then implement.',
        validate='pytest tests/test_a.py::test_x -q')
    text = plan.format_step(step)
    assert '### Step 1: Do X' in text
    assert 'Files: a.py, b.py' in text
    assert 'Test first: tests/test_a.py::test_x' in text
    assert 'Validate: pytest tests/test_a.py::test_x -q' in text


def test_parse_plan_roundtrip() -> None:
    # A parsed plan re-formatted step-by-step should re-parse to the same steps.
    p = plan.parse_plan(SAMPLE_PLAN)
    reformatted = ('# Plan: ' + p.title + '\n\n## Steps\n\n'
                   + '\n\n'.join(plan.format_step(s) for s in p.steps))
    p2 = plan.parse_plan(reformatted)
    assert len(p2.steps) == len(p.steps)
    for s1, s2 in zip(p.steps, p2.steps):
        assert s1.id == s2.id
        assert s1.goal == s2.goal
        assert s1.files == s2.files
