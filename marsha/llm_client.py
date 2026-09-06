import os

import anthropic
import openai

from marsha.config import resolve_api_base, resolve_api_key, resolve_provider

_client = None


def create_client(api_base=None):
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
    return openai.AsyncOpenAI(
        base_url=resolve_api_base(api_base),
        api_key=api_key,
        organization=os.getenv('OPENAI_ORG'),
    )


def get_client():
    global _client
    if _client is None:
        _client = create_client()
    return _client


def set_client(client):
    global _client
    _client = client
