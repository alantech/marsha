from __future__ import annotations

from typing import Any

from marsha.config import resolve_provider
from marsha.mappers.base import BaseMapper
from marsha.mappers.chatgpt import ChatGPTMapper
from marsha.mappers.claude import ClaudeMapper


def get_mapper(system: str, **kwargs: Any) -> ChatGPTMapper | ClaudeMapper:
    """Return a mapper instance for the resolved LLM provider"""
    if resolve_provider() == 'anthropic':
        return ClaudeMapper(system, **kwargs)
    return ChatGPTMapper(system, **kwargs)
