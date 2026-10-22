"""Read-only repository context for ``marsha diff``.

This module maps relevant files and extracts the repository's existing conventions. It does not
produce or rewrite a feature plan: the locked specification remains the implementation contract.
"""
from __future__ import annotations

import os
from typing import cast

from marsha import tools
from marsha.log import log
from marsha.mappers import get_mapper
from marsha.personas import load_persona, personas_dir


async def run_explorer(spec_text: str, cwd: str, model: str | None,
                       debug: bool = False, focus: str = '',
                       no_network: bool = False) -> str:
    """Map the repository files and subsystems relevant to the locked spec."""
    name, body = load_persona(os.path.join(
        personas_dir(), 'explorer-scope.md'))
    ctx = tools.ToolContext(phase='review', workdir=cwd, no_network=no_network)
    mapper = get_mapper(
        body + tools.tool_instructions(ctx), n_results=1, model=model,
        label=f'repo-context:{name}', reasoning_effort='high')
    request = ('Explore the repository to identify the files and subsystems relevant to this '
               'specification. The spec is the source of truth; this map is orientation only.\n\n'
               + tools.wrap_untrusted('spec', spec_text))
    if focus:
        request += ('\n\nThe implementor asked for this additional repository context:\n'
                    + tools.wrap_untrusted('focus', focus))
    result = await tools.run_with_tools(mapper, request, ctx, debug=debug, max_rounds=50)
    log(f'repo context: explorer ({name}) completed')
    return cast(str, result)


async def run_rule_checker(cwd: str, model: str | None,
                           debug: bool = False, focus: str = '',
                           no_network: bool = False) -> str:
    """Extract source-tagged conventions; do not infer feature requirements."""
    name, body = load_persona(
        os.path.join(personas_dir(), 'rule-checker-conventions.md'))
    ctx = tools.ToolContext(phase='review', workdir=cwd, no_network=no_network)
    mapper = get_mapper(
        body + tools.tool_instructions(ctx), n_results=1, model=model,
        label=f'repo-context:{name}', reasoning_effort='high')
    request = ('Identify the repository conventions relevant to new code. Return a concise, '
               'source-tagged constraints list. These are implementation context, not additional '
               'product requirements.')
    if focus:
        request += ('\n\nThe implementor asked to re-check this convention question:\n'
                    + tools.wrap_untrusted('focus', focus))
    result = await tools.run_with_tools(mapper, request, ctx, debug=debug, max_rounds=30)
    log(f'repo context: rule checker ({name}) completed')
    return cast(str, result)
