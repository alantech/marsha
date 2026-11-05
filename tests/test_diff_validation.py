"""Tests for validation evidence and the per-cycle meta-review."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from marsha import diff_validation
from marsha.diff_validation import (
    ValidationCheck,
    ValidationMetaFailure,
    run_validation_meta_review,
)
from marsha.findings import Finding


def _finding(name: str, label: str, desc: str) -> Finding:
    return {
        'name': name,
        'label': label,
        'severity': 'MAJOR',
        'location': 'src/module.py:12',
        'desc': desc,
        'support': 'The current code path does not perform the required check.',
    }


def _mapper(response: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(run=AsyncMock(return_value=json.dumps(response)))


def test_meta_review_gets_findings_and_validation_and_tracks_statuses() -> None:
    prior = [_finding('Sage', 'A1', 'old issue')]
    current = [_finding('Eli', 'B2', 'new issue')]
    response = {
        'assessments': [
            {'id': 'Sage-A1', 'status': 'resolved', 'reason': 'The check is now present.'},
            {'id': 'Eli-B2', 'status': 'open', 'reason': 'The new issue remains.'},
        ],
        'coverage_gaps': [],
        'summary': 'One prior item is fixed; current findings remain open.',
    }
    mapper = _mapper(response)
    with patch.object(diff_validation, 'get_mapper', return_value=mapper):
        result = asyncio.run(run_validation_meta_review(
            prior, current, [ValidationCheck('pytest -q', 'passed', '1 passed')],
            'test-model'))

    assert result.open_prior == []
    assert result.coverage_gaps == []
    assert mapper.run.await_args is not None
    prompt = mapper.run.await_args.args[0]
    assert 'Sage-A1' in prompt and 'Eli-B2' in prompt
    assert 'pytest -q — PASSED' in prompt
    assert 'Available commands:' not in prompt


def test_meta_review_ignores_coverage_gaps_from_ledger_model() -> None:
    response = {
        'assessments': [],
        'coverage_gaps': [{
            'requirement': 'An unreviewed requirement.',
            'description': 'This should be handled by the code review panel.',
            'location': 'src/service.py:10',
        }],
        'summary': 'No ledger findings.',
    }
    mapper = _mapper(response)
    with patch.object(diff_validation, 'get_mapper', return_value=mapper):
        result = asyncio.run(run_validation_meta_review(
            [], [], [ValidationCheck('pytest -q', 'passed')], 'test-model'))
    assert result.coverage_gaps == []


def test_meta_review_cannot_close_current_cycle_findings() -> None:
    current = [_finding('Eli', 'A1', 'current issue')]
    response = {
        'assessments': [
            {'id': 'Eli-A1', 'status': 'resolved', 'reason': 'The test passed.'},
        ],
        'coverage_gaps': [],
        'summary': 'Done.',
    }
    with patch.object(diff_validation, 'get_mapper', return_value=_mapper(response)):
        result = asyncio.run(run_validation_meta_review(
            [], current, [ValidationCheck('pytest -q', 'passed')], 'test-model'))
    assert result.open_prior == []  # current findings are retained by the outer ledger


def test_meta_review_missing_assessment_fails_closed() -> None:
    prior = [_finding('Sage', 'A1', 'prior issue')]
    response = {'assessments': [], 'coverage_gaps': [], 'summary': 'No issues.'}
    with patch.object(diff_validation, 'get_mapper', return_value=_mapper(response)):
        with pytest.raises(ValidationMetaFailure, match='assessment IDs mismatch'):
            asyncio.run(run_validation_meta_review(
                prior, [], [ValidationCheck('pytest -q', 'passed')], 'test-model'))


def test_meta_review_ignores_unknown_assessment_ids() -> None:
    prior = [_finding('Cleo', 'D12', 'the generation prompt omits a contract')]
    response = {
        'assessments': [
            {'id': 'Cleo-D12', 'status': 'open', 'reason': 'Still present.'},
            {'id': 'C-D12', 'status': 'open', 'reason': 'Unknown extra ID.'},
        ],
        'coverage_gaps': [],
        'summary': 'The prior finding remains open.',
    }
    with patch.object(diff_validation, 'get_mapper', return_value=_mapper(response)):
        result = asyncio.run(run_validation_meta_review(
            prior, [], [ValidationCheck('pytest -q', 'passed')], 'test-model'))
    assert result.open_prior == prior


def test_meta_review_preserves_prior_finding_order() -> None:
    prior = [_finding('Sage', 'B1', 'second'), _finding('Sage', 'A1', 'first')]
    response = {
        'assessments': [
            {'id': 'Sage-B1', 'status': 'open', 'reason': 'Still present.'},
            {'id': 'Sage-A1', 'status': 'open', 'reason': 'Still present.'},
        ],
        'coverage_gaps': [],
        'summary': 'Both findings remain open.',
    }
    with patch.object(diff_validation, 'get_mapper', return_value=_mapper(response)):
        result = asyncio.run(run_validation_meta_review(
            prior, [], [ValidationCheck('pytest -q', 'passed')], 'test-model'))
    assert [finding['label'] for finding in result.open_prior] == ['B1', 'A1']
