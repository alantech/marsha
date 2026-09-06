from marsha.config import resolve_provider
from marsha.mappers.chatgpt import ChatGPTMapper
from marsha.mappers.claude import ClaudeMapper


def get_mapper(system, **kwargs):
    """Return a mapper instance for the resolved LLM provider"""
    if resolve_provider() == 'anthropic':
        return ClaudeMapper(system, **kwargs)
    return ChatGPTMapper(system, **kwargs)
