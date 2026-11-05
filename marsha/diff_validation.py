"""Validation evidence and the per-cycle meta-review for ``marsha diff``."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, cast

from marsha import tools
from marsha.findings import Finding
from marsha.log import log
from marsha.mappers import get_mapper
from marsha.personas import load_persona, personas_dir


@dataclass(frozen=True)
class ValidationCheck:
    command: str
    status: str  # passed | failed | unavailable | skipped
    output: str = ''
    blocking: bool = True

    @property
    def passed(self) -> bool:
        return self.status == 'passed'

    def render(self) -> str:
        details = self.output.strip()
        advisory = ' (ADVISORY)' if not self.blocking else ''
        return (f'### {self.command} — {self.status.upper()}{advisory}\n'
                + (details if details else '(no output)'))


class ValidationMetaFailure(Exception):
    """The meta-review did not return a complete, parseable finding ledger."""


@dataclass
class ValidationMetaResult:
    open_prior: list[Finding]
    coverage_gaps: list[Finding]
    summary: str


_META_PROMPT = '''\
You are a lightweight finding-ledger reconciler. Independent reviewers have already inspected the
working tree and tests; do not repeat their code review or perform a new specification-coverage
audit. You have no tools. Reconcile only the supplied prior findings, current-panel findings, and
validation outcomes.

For every supplied finding ID, return exactly one assessment. Current-panel findings must remain
open. Keep a prior finding open if a current finding describes the same unresolved issue; otherwise
mark it resolved because the complete current review panel did not report it. Validation findings
are current findings and must remain open until a later panel cycle no longer reports the failure.
Do not invent IDs or coverage gaps; return an empty coverage_gaps array.

Respond immediately with one JSON object and no prose:
{
  "assessments": [
    {"id": "Reviewer-Label", "status": "open" | "resolved", "reason": "..."}
  ],
  "coverage_gaps": [],
  "summary": "brief validation-aware disposition"
}
Include exactly one assessment for each prior or current finding ID, using the exact IDs supplied.
'''


def finding_id(finding: Finding) -> str:
    return f"{finding.get('name', '')}-{finding.get('label', '')}"


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValidationMetaFailure('meta-review returned no JSON object')
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ValidationMetaFailure(
            f'meta-review returned invalid JSON: {error}') from error
    if not isinstance(data, dict):
        raise ValidationMetaFailure('meta-review JSON must be an object')
    return data


async def run_validation_meta_review(
        prior_findings: list[Finding], current_findings: list[Finding],
        checks: list[ValidationCheck], model: str) -> ValidationMetaResult:
    """Reconcile a compact finding ledger without re-reviewing the code.

    The independent review panel owns code inspection and coverage. This stage only resolves prior
    IDs that the complete current panel no longer reports. The caller preserves all findings and
    continues repair if the response is malformed or unavailable.
    """
    known: dict[str, Finding] = {}
    for finding in prior_findings + current_findings:
        known[finding_id(finding)] = finding
    current_ids = {finding_id(f) for f in current_findings}
    prior_text = tools.truncate(json.dumps([
        {'id': finding_id(f), 'location': f.get('location', ''),
         'description': f.get('desc', '')}
        for f in prior_findings], ensure_ascii=False), limit=12_000)
    current_text = tools.truncate(json.dumps([
        {'id': finding_id(f), 'location': f.get('location', ''),
         'description': f.get('desc', '')}
        for f in current_findings], ensure_ascii=False), limit=12_000)
    validation_text = '\n\n'.join(check.render()
                                  for check in checks) or '(not run)'
    validation_text = tools.truncate(validation_text, limit=8_000)
    request = (
        f'## Prior open findings\n{prior_text}\n\n'
        f'## Current panel findings\n{current_text}\n\n'
        f'## Validation outcomes\n{validation_text}\n')
    _name, body = load_persona(
        os.path.join(personas_dir(), 'validation-meta-reviewer.md'))
    mapper = get_mapper(
        body + '\n\n' + _META_PROMPT,
        n_results=1, model=model,
        label='diff:validation-meta-reviewer', reasoning_effort='low')
    raw = cast(str, await mapper.run(request))
    data = _extract_json(raw)
    assessments = data.get('assessments')
    gaps = data.get('coverage_gaps')
    summary = data.get('summary')
    if not isinstance(assessments, list) or not isinstance(gaps, list) \
            or not isinstance(summary, str):
        raise ValidationMetaFailure(
            'meta-review JSON is missing required fields')

    by_id: dict[str, str] = {}
    for item in assessments:
        if not isinstance(item, dict):
            raise ValidationMetaFailure(
                'meta-review assessment must be an object')
        key, status = item.get('id'), item.get('status')
        if not isinstance(key, str) or status not in ('open', 'resolved') or key in by_id:
            raise ValidationMetaFailure(
                'meta-review assessment has an invalid ID or status')
        by_id[key] = status
    expected = set(known)
    missing, extra = expected - set(by_id), set(by_id) - expected
    if missing:
        raise ValidationMetaFailure(
            f'meta-review assessment IDs mismatch (missing={sorted(missing)})')
    if extra:
        log('diff: validation meta-review returned unknown assessment IDs; '
            f'ignoring {sorted(extra)}')
    by_id = {key: status for key, status in by_id.items() if key in expected}
    for key in current_ids:
        if by_id[key] != 'open':
            log(f'diff: validation meta-review tried to close current finding {key}; '
                'keeping it open for this cycle')
            by_id[key] = 'open'

    prior_ids = {finding_id(f) for f in prior_findings} - current_ids
    open_prior: list[Finding] = []
    seen_prior: set[str] = set()
    for finding in prior_findings:
        key = finding_id(finding)
        if key in prior_ids and key not in seen_prior and by_id[key] == 'open':
            open_prior.append(known[key])
            seen_prior.add(key)
    if gaps:
        log('diff: ignoring meta-review coverage gaps; code coverage is owned by the review panel')
    coverage_gaps: list[Finding] = []
    return ValidationMetaResult(open_prior, coverage_gaps, summary.strip())
