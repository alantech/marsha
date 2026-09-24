from __future__ import annotations

import os

import anthropic
import openai

from marsha.config import resolve_api_base, resolve_api_key, resolve_provider, is_local_backend

# A local/OpenAI-compatible server (e.g. llama.cpp) serializes and can take a very long time
# per request (large prefills plus slow token generation). The OpenAI client's default 600s
# read timeout would kill legitimately in-flight requests, so a local backend gets a long,
# finite timeout instead.
LOCAL_BACKEND_TIMEOUT = 4 * 60 * 60.0

_client: openai.AsyncOpenAI | anthropic.AsyncAnthropic | None = None


def create_client(api_base: str | None = None) -> openai.AsyncOpenAI | anthropic.AsyncAnthropic:
    provider = resolve_provider()
    if provider == 'anthropic':
        api_key = resolve_api_key('anthropic')
        if api_key is None:
            raise Exception(
                'No Anthropic API key found. Set the CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable, or add claude_api_key to the config file')
        return anthropic.AsyncAnthropic(api_key=api_key)
    api_key = resolve_api_key('openai')
    if api_key is None:
        raise Exception(
            'No OpenAI API key found. Set the OPENAI_SECRET_KEY or OPENAI_API_KEY environment variable, or add api_key to the config file')
    if is_local_backend():
        return openai.AsyncOpenAI(
            base_url=resolve_api_base(api_base),
            api_key=api_key,
            organization=os.getenv('OPENAI_ORG'),
            timeout=LOCAL_BACKEND_TIMEOUT,
        )
    return openai.AsyncOpenAI(
        base_url=resolve_api_base(api_base),
        api_key=api_key,
        organization=os.getenv('OPENAI_ORG'),
    )


def get_client() -> openai.AsyncOpenAI | anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = create_client()
    return _client


def set_client(client: openai.AsyncOpenAI | anthropic.AsyncAnthropic) -> None:
    global _client
    _client = client
