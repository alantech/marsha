import os

import openai

_client = None


def create_client():
    return openai.AsyncOpenAI(
        api_key=os.getenv('OPENAI_SECRET_KEY') or os.getenv('OPENAI_API_KEY'),
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
