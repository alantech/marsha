"""Review threads for the `marsha diff` implementation phase (issue #219).

After the step-by-step implement loop completes, the full review panel runs.
Each finding gets a thread: the implementor responds (implement or push back),
and pushbacks go to the planner for adjudication. Threads are marked resolved
when the fix is made or the pushback is accepted.

This module owns the thread data structures and the resolution loop. The
integration into `run_diff` (calling the review panel, running the implementor
and planner for each thread) lives in `marsha.diff`.
"""
from __future__ import annotations

import dataclasses
import re
from typing import cast

from marsha import plan as plan_mod
from marsha import tools
from marsha.findings import Finding
from marsha.log import debug_print, log
from marsha.mappers import get_mapper

# The implementor's response and the planner's adjudication both end with an
# ACTION line. This regex extracts it.
_ACTION_RE = re.compile(
    r'ACTION:\s*(implemented|pushed_back|accepted|rejected)', re.IGNORECASE)


def parse_action(text: str) -> str:
    """Extract the ACTION keyword from a thread response, or '' when absent."""
    m = _ACTION_RE.search(text or '')
    return m.group(1).lower() if m else ''


@dataclasses.dataclass
class ThreadResponse:
    """One response in a review thread."""
    role: str  # 'reviewer', 'implementor', 'planner'
    action: str  # 'implemented', 'pushed_back', 'accepted', 'rejected', ''
    text: str


@dataclasses.dataclass
class ReviewThread:
    """A review thread: a finding and the responses to it."""
    finding: Finding
    responses: list[ThreadResponse]
    resolved: bool = False

    def format_for_prompt(self) -> str:
        """Render the thread for a prompt: the finding and all responses."""
        lines = [
            f"[{self.finding['severity']}] "
            f"{self.finding.get('location', '')} - "
            f"{self.finding['desc']}"]
        if self.finding.get('support'):
            lines.append(self.finding['support'])
        for r in self.responses:
            if r.role == 'reviewer':
                continue
            lines.append(
                f'\n--- {r.role} ({r.action or "response"}) ---\n{r.text}')
        return '\n'.join(lines)


PLANNER_ADJUDICATION_PROMPT = '''\
You are the planner. The implementor pushed back on a review finding. Your \
charge is to decide if the pushback is valid: does the finding identify a real \
problem that the implementor should address, or is the pushback correct (the \
concern is not actionable, already handled, or based on a misreading)?

You judge the substance, not the tone. A pushback that cites a real code path \
that handles the concern is valid. A pushback that merely disagrees without \
evidence is not.

Respond with exactly one of:
- ACTION: accepted — the pushback is valid; the finding is not actionable.
- ACTION: rejected — the finding stands; the implementor must address it.

Briefly state your reasoning after the ACTION line.
'''


async def ask_implementor_response(
        thread: ReviewThread, impl_ctx: tools.ToolContext, model: str,
        max_failures: int, debug: bool = False) -> ThreadResponse:
    """Ask the implementor to respond to a review finding: implement the fix
    or push back. The implementor uses the tool loop (write-file, run) to make
    the change if it implements. Returns the ThreadResponse."""
    from marsha.diff import IMPL_SYSTEM_PROMPT
    finding_text = (
        f"[{thread.finding['severity']}] "
        f"{thread.finding.get('location', '')} - "
        f"{thread.finding['desc']}\n"
        f"{thread.finding.get('support', '')}")
    request = (
        f'A review of the working-tree changes found the following issue:\n\n'
        f'{finding_text}\n\n'
        f'Respond with exactly one of:\n'
        f'- ACTION: implemented — you made the fix. Briefly describe what you '
        f'changed.\n'
        f'- ACTION: pushed_back — you disagree with the finding. Briefly '
        f'explain why.\n\n'
        f'If you implement the fix, use the write-file and run tools to make '
        f'the change and verify it. If you push back, do not make any code '
        f'changes. End your response with the ACTION line and a brief '
        f'explanation.')
    system = IMPL_SYSTEM_PROMPT + tools.tool_instructions(impl_ctx)
    if impl_ctx.run_whitelist:
        cmd_lines = '\n'.join(f'  {r.display}' for r in impl_ctx.run_whitelist)
        system += (f'\nCommands you may run in this repository '
                   f'(whitelisted; anything else is refused):\n{cmd_lines}\n')
    mapper = get_mapper(
        system, n_results=1, model=model, label='threads:implementor',
        reasoning_effort='high')
    from marsha.diff import IMPL_MAX_TOOL_ROUNDS
    text = await tools.run_with_tools(
        mapper, request, impl_ctx, debug=debug,
        max_rounds=IMPL_MAX_TOOL_ROUNDS,
        max_consecutive_failures=max_failures)
    action = parse_action(cast(str, text))
    return ThreadResponse(
        role='implementor', action=action, text=cast(str, text).strip())


async def ask_planner_adjudication(
        thread: ReviewThread, p: plan_mod.Plan, spec_text: str, model: str,
        debug: bool = False) -> ThreadResponse:
    """Ask the planner to adjudicate a pushback: accepted (finding not
    actionable) or rejected (finding stands). Returns the ThreadResponse."""
    pushback = next(
        (r.text for r in reversed(thread.responses) if r.role == 'implementor'),
        '')
    plan_text = plan_mod.format_plan(p)
    request = (
        f'The implementor pushed back on a review finding. Decide if the '
        f'pushback is valid.\n\n'
        f'## Finding\n{thread.finding["desc"]}\n\n'
        f'## Implementor\'s pushback\n{pushback}\n\n'
        f'## Plan\n{plan_text}\n\n'
        f'## Specification\n{spec_text[:48000]}\n\n'
        'Respond with the ACTION line and a brief reasoning.')
    mapper = get_mapper(
        PLANNER_ADJUDICATION_PROMPT, n_results=1, model=model,
        label='threads:planner', reasoning_effort='high')
    text = await mapper.run(request)
    action = parse_action(text or '')
    return ThreadResponse(
        role='planner', action=action, text=(text or '').strip())


async def resolve_threads(
        findings: list[Finding], impl_ctx: tools.ToolContext,
        p: plan_mod.Plan, spec_text: str, model: str, max_failures: int,
        debug: bool = False, max_cycles: int = 3) -> tuple[list[ReviewThread], bool]:
    """Resolve review findings via threads. For each finding, the implementor
    responds (implement or push back). Pushbacks go to the planner for
    adjudication. Loops until all threads are resolved or max_cycles is hit.
    Returns (threads, all_resolved)."""
    threads = [ReviewThread(
        finding=f,
        responses=[ThreadResponse(
            role='reviewer', action='', text=f['support'] or f['desc'])],
    ) for f in findings]
    for cycle in range(max_cycles):
        unresolved = [t for t in threads if not t.resolved]
        if not unresolved:
            break
        if debug:
            debug_print(f'[threads] cycle {cycle + 1}/{max_cycles}: '
                        f'{len(unresolved)} unresolved thread(s)')
        for thread in unresolved:
            try:
                response = await ask_implementor_response(
                    thread, impl_ctx, model, max_failures, debug)
            except (tools.ToolFailureLimitExceeded, KeyboardInterrupt):
                response = ThreadResponse(
                    role='implementor', action='', text='(implementor failed)')
            thread.responses.append(response)
            if response.action == 'implemented':
                thread.resolved = True
                log(f'threads: finding at {thread.finding.get("location", "?")} '
                    f'resolved (implemented)')
            elif response.action == 'pushed_back':
                try:
                    adjudication = await ask_planner_adjudication(
                        thread, p, spec_text, model, debug)
                except Exception as e:
                    log(f'threads: planner adjudication failed: {e}')
                    adjudication = ThreadResponse(
                        role='planner', action='rejected',
                        text='(adjudication failed; finding stands)')
                thread.responses.append(adjudication)
                if adjudication.action == 'accepted':
                    thread.resolved = True
                    log(f'threads: finding at {thread.finding.get("location", "?")} '
                        f'resolved (pushback accepted)')
                else:
                    log(f'threads: pushback rejected for '
                        f'{thread.finding.get("location", "?")}; '
                        f'stays open for next cycle')
    all_resolved = all(t.resolved for t in threads)
    return threads, all_resolved
