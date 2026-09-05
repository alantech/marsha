import os

import openai

from marsha.config import resolve_api_base, resolve_api_key

_client = None


def create_client(api_base=None):
    return openai.AsyncOpenAI(
        base_url=resolve_api_base(api_base),
        api_key=resolve_api_key(),
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
