"""Deterministic tests for the .mrsh format parser (marsha.meta) and its LLM rendering
(marsha.parse.format_marsha_for_llm).

The parser is the single source of truth for what a .mrsh file may look like: compile runs it
as its first stage, and refine's format gate runs it over every rewritten .mrsh before a lock
is accepted. The format is line-oriented: top-level "# " headings delimit function/type
sections; "##" and deeper headings are subsections inside the current section; a '#' inside
text (an issue reference, a comment) does not delimit; and a "# " line inside a fenced code
block is code, not a heading.
"""

import asyncio
import os
import tempfile
from typing import Any

from marsha import meta
from marsha.parse import format_marsha_for_llm


def _valid_flat_spec() -> str:
    return ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum. This description is long '
            'enough to clear the minimum length rule for a marsha function section.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')


def _valid_subsection_spec() -> str:
    return ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum. This description is long '
            'enough to clear the minimum length rule for a marsha function section.\n'
            '\n'
            '## Behavior\n\n'
            'The addition is commutative and handles negative integers.\n\n'
            '## Usage examples\n\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')


async def _meta_for(spec: str) -> meta.MarshaMeta:
    with tempfile.NamedTemporaryFile('w', suffix='.mrsh', delete=False) as f:
        f.write(spec)
        path = f.name
    try:
        return await meta.MarshaMeta(path).populate()
    finally:
        os.unlink(path)


# --- _top_level_sections --------------------------------------------------------


def test_top_level_sections_splits_on_top_level_headings_only() -> None:
    spec = ('preamble text\n'
            '# func add(a: int, b: int): int\n'
            'description\n'
            '## subsection\n'
            'subsection body\n'
            '# func sub(a: int): int\n'
            'another description')
    sections = meta._top_level_sections(spec)
    assert sections[0] == 'preamble text'
    # The "##" heading stays inside the first function section; only the next "# " line
    # starts a new one.
    assert '## subsection\nsubsection body' in sections[1]
    assert sections[1].startswith(' func add(a: int, b: int): int')
    assert sections[2].startswith(' func sub(a: int): int')
    assert len(sections) == 3


def test_top_level_sections_hash_in_text_does_not_delimit() -> None:
    spec = ('# func add(a: int, b: int): int\n'
            'description referencing issue #218 and a `# comment`\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')
    sections = meta._top_level_sections(spec)
    assert len(sections) == 2  # preamble + the one func section
    assert 'issue #218' in sections[1]


def test_top_level_sections_ignores_headings_in_code_fences() -> None:
    spec = ('# func add(a: int, b: int): int\n'
            'description\n'
            '```\n'
            '# not a heading\n'
            '# func fake(x: int): int\n'
            '```\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')
    sections = meta._top_level_sections(spec)
    assert len(sections) == 2
    assert '# func fake' in sections[1]  # fence content stays in the section


def test_top_level_sections_long_fence_contains_short_fence() -> None:
    # A four-backtick fence can contain a triple-backtick line (CommonMark: the closing fence
    # must be at least as long as the opening one): the inner line is content, and a "# "
    # code line inside the fence does not start a section.
    spec = ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum. Padding padding padding '
            'padding padding padding padding padding padding padding padding padding.\n'
            '````\n'
            '```\n'
            '# func fake(x: int): int\n'
            '```\n'
            '````\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')
    sections = meta._top_level_sections(spec)
    assert len(sections) == 2  # the inner "```" line did not close the "````" fence
    functions, _, _ = meta.extract_functions_and_types(spec)
    assert len(functions) == 1  # the fenced "# func fake" line is code, not a section


def test_top_level_sections_leading_spaces_still_delimit() -> None:
    # Up to three leading spaces still mark a heading (CommonMark); the old character split
    # treated any '#' as a boundary, so this keeps the heading shape recognizable.
    sections = meta._top_level_sections('  # func add(a: int): int\nbody')
    assert sections[1].startswith(' func add(a: int): int')


# --- extract_functions_and_types ------------------------------------------------


def test_extract_flat_spec_still_valid() -> None:
    functions, types, void_funcs = meta.extract_functions_and_types(_valid_flat_spec())
    assert len(functions) == 1 and not types and not void_funcs
    assert 'func add' in functions[0]


def test_extract_spec_with_subsections() -> None:
    functions, types, void_funcs = meta.extract_functions_and_types(
        _valid_subsection_spec())
    assert len(functions) == 1 and not types and not void_funcs
    # The section keeps its subsections (they are part of the function's content).
    assert '## Behavior' in functions[0]
    assert '## Usage examples' in functions[0]


def test_extract_hash_in_text_keeps_the_section_intact() -> None:
    spec = ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum. See issue #218 for the '
            'history of this decision; the old draft is linked from the ticket.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')
    functions, _, _ = meta.extract_functions_and_types(spec)
    assert len(functions) == 1
    assert 'issue #218' in functions[0]


def test_extract_hash_in_text_does_not_swallow_the_next_function() -> None:
    spec = _valid_flat_spec() + '\n# func sub(a: int): int\n' + (
        'Subtracts one from another and returns the difference. This description is long '
        'enough to clear the minimum length rule for a marsha function section.\n\n'
        '* sub(5, 3) -> 2\n'
        '* sub(1, 5) -> -4')
    functions, _, _ = meta.extract_functions_and_types(spec)
    assert len(functions) == 2


def test_extract_requires_description_as_first_block() -> None:
    # A subsection immediately after the function heading is not the required description
    # paragraph.
    spec = ('# func add(a: int, b: int): int\n'
            '## Behavior\n\n'
            'The addition is commutative. Padding to make the length rule pass for sure.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')
    try:
        meta.extract_functions_and_types(spec)
        raise AssertionError('a subsection-first section must not validate')
    except Exception as e:
        assert 'Invalid description' in str(e)


def test_extract_examples_list_must_be_the_final_block() -> None:
    spec = ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum. This description is long '
            'enough to clear the minimum length rule for a marsha function section.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0\n'
            '\n'
            'A trailing paragraph after the examples list.\n')
    try:
        meta.extract_functions_and_types(spec)
        raise AssertionError('a section whose last block is not the examples list must '
                             'not validate')
    except Exception as e:
        assert 'usage examples' in str(e)


def test_extract_bare_func_without_heading_is_rejected() -> None:
    # A function signature with no "#" heading (a free-form document) is not a function
    # section: it is rejected with a clear message, not silently ignored.
    spec = 'func add(a: int, b: int): int'
    try:
        meta.extract_functions_and_types(spec)
        raise AssertionError('a heading-less func line must not validate')
    except Exception as e:
        assert 'Missing description' in str(e)


def test_extract_no_functions_or_types() -> None:
    try:
        meta.extract_functions_and_types('# Purpose\nA free-form document.')
        raise AssertionError('a document without func/type sections must not parse')
    except Exception as e:
        assert 'No functions or types found' in str(e)


# --- format_marsha_for_llm ------------------------------------------------------


def test_format_marsha_for_llm_subsections() -> None:
    meta_obj = asyncio.run(_meta_for(_valid_subsection_spec()))
    out = format_marsha_for_llm(meta_obj)
    # The subsections render into the description, and the section's final list is the
    # examples.
    assert '## Behavior' in out
    assert 'The addition is commutative' in out
    assert '### Examples of expected behavior' in out
    assert out.count('* add(1, 2) -> 3') == 1


def test_format_marsha_for_llm_list_in_subsection_is_not_the_examples() -> None:
    # An earlier list (e.g. an option list under a subsection) is description content; only
    # the section's final list is the usage examples. (Markdown merges lists that are only
    # blank-line apart, so the examples list is kept a block of its own, separated by a
    # paragraph.)
    spec = ('# func add(a: int, b: int): int\n'
            'Adds two integers together and returns the sum. This description is long '
            'enough to clear the minimum length rule for a marsha function section.\n'
            '\n'
            '## Options\n\n'
            '* option one\n'
            '* option two\n'
            '\n'
            'The examples below drive the test suite.\n'
            '\n'
            '* add(1, 2) -> 3\n'
            '* add(-1, 1) -> 0')
    meta_obj = asyncio.run(_meta_for(spec))
    out = format_marsha_for_llm(meta_obj)
    examples = out.split('### Examples of expected behavior')[1]
    assert 'option one' not in examples  # the subsection list is description, not examples
    assert '* add(1, 2) -> 3' in examples
