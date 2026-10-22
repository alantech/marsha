"""Tests for provider mapper retry and usage-accounting behavior."""
from __future__ import annotations

import asyncio
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from marsha.mappers import chatgpt, claude
from marsha import stats


@pytest.fixture(autouse=True)
def reset_usage() -> Generator[None, None, None]:
    stats.begin_invocation(None)
    yield
    stats.begin_invocation(None)


def test_chatgpt_usage_is_attributed_to_context_fallback_model() -> None:
    class ContextOverflow(Exception):
        code = 'context_length_exceeded'

    models: list[str] = []

    async def create(**query: object) -> SimpleNamespace:
        models.append(str(query['model']))
        if len(models) == 1:
            raise ContextOverflow('prompt too long')
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=3, prompt_tokens=2, completion_tokens=1))

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with patch.object(chatgpt, 'get_client', return_value=client), \
            patch.object(chatgpt, 'is_local_backend', return_value=False), \
            patch.object(chatgpt, 'resolve_strong_model', return_value='gpt-5'), \
            patch('marsha.mappers.chatgpt.time.sleep'), \
            patch('marsha.mappers.chatgpt.openai.BadRequestError', ContextOverflow), \
            patch('marsha.mappers.chatgpt.record_usage') as record:
        asyncio.run(chatgpt.retry_chat_completion({}, model='gpt-5-mini'))

    assert models == ['gpt-5-mini', 'gpt-5']
    record.assert_called_once_with('gpt-5', 2, 1)


def test_claude_usage_is_attributed_to_context_fallback_model() -> None:
    class ContextOverflow(Exception):
        def __init__(self) -> None:
            self.error = SimpleNamespace(message='prompt too long')

    models: list[str] = []

    class Stream:
        async def __aenter__(self) -> Stream:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get_final_message(self) -> SimpleNamespace:
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=2, output_tokens=1))

    class Messages:
        def stream(self, **query: object) -> Stream:
            models.append(str(query['model']))
            if len(models) == 1:
                raise ContextOverflow()
            return Stream()

    client = SimpleNamespace(messages=Messages())
    with patch.object(claude, 'get_client', return_value=client), \
            patch.object(claude, 'resolve_strong_model', return_value='claude-opus-5'), \
            patch('marsha.mappers.claude.time.sleep'), \
            patch('marsha.mappers.claude.anthropic.BadRequestError', ContextOverflow), \
            patch('marsha.mappers.claude.record_usage') as record:
        asyncio.run(claude.retry_message_create({}, model='claude-sonnet-5'))

    assert models == ['claude-sonnet-5', 'claude-opus-5']
    record.assert_called_once_with('claude-opus-5', 2, 1)


def test_chatgpt_mapper_stops_after_request_crosses_price_cap() -> None:
    calls: list[str] = []

    async def create(**query: object) -> SimpleNamespace:
        calls.append(str(query['model']))
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=1024, prompt_tokens=1024,
                                  completion_tokens=0),
            choices=[SimpleNamespace(message=SimpleNamespace(content='answer'))])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    stats.begin_invocation(0.00005)
    mapper = chatgpt.ChatGPTMapper(system='system', model='gpt-6-luna')

    with patch.object(chatgpt, 'get_client', return_value=client):
        with pytest.raises(stats.PriceLimitExceeded, match='reached or exceeded'):
            asyncio.run(mapper.run('prompt'))

    assert calls == ['gpt-6-luna']
    assert stats.global_cost() > 0.00005


def test_unknown_model_is_rejected_before_api_request_with_cap() -> None:
    calls: list[bool] = []

    async def create(**_query: object) -> SimpleNamespace:
        calls.append(True)
        raise AssertionError('the request must not be sent')

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    stats.begin_invocation(1.0)
    mapper = chatgpt.ChatGPTMapper(system='system', model='custom-unknown-model')

    with patch.object(chatgpt, 'get_client', return_value=client):
        with pytest.raises(stats.PriceLimitExceeded, match='no token pricing'):
            asyncio.run(mapper.run('prompt'))

    assert calls == []


def test_capped_claude_multiple_results_are_serialized_at_limit() -> None:
    calls: list[str] = []

    class Stream:
        async def __aenter__(self) -> Stream:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get_final_message(self) -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1024, output_tokens=0),
                model='claude-sonnet-5',
                content=[SimpleNamespace(type='text', text='answer')])

    class Messages:
        def stream(self, **query: object) -> Stream:
            calls.append(str(query['model']))
            return Stream()

    client = SimpleNamespace(messages=Messages())
    stats.begin_invocation(0.001)
    mapper = claude.ClaudeMapper(
        system='system', model='claude-sonnet-5', n_results=3)

    with patch.object(claude, 'get_client', return_value=client):
        with pytest.raises(stats.PriceLimitExceeded, match='before another model request'):
            asyncio.run(mapper.run('prompt'))

    assert calls == ['claude-sonnet-5']
