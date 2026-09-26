from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import anthropic
from anthropic.types import Message

from marsha.config import resolve_model, resolve_strong_model
from marsha.log import log, progress
from marsha.llm_client import get_client
from marsha.mappers.base import BaseMapper, ContextOverflowError
from marsha.stats import stats
from marsha.utils import prettify_time_delta

# Get time at startup to make human legible "start times" in the logs
t0: float = time.time()

# Anthropic requires max_tokens on every request; use this budget when the
# caller did not specify one
DEFAULT_MAX_TOKENS = 32768


class Usage():
    """OpenAI-shaped usage so stats.py can consume both providers"""

    prompt_tokens: int
    completion_tokens: int

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class NormalizedResponse():
    """OpenAI-shaped response so stats.py can consume both providers"""

    model: str
    usage: Usage

    def __init__(self, model: str, usage: Usage) -> None:
        self.model = model
        self.usage = usage


def normalize_response(res: Message) -> NormalizedResponse:
    return NormalizedResponse(
        res.model, Usage(res.usage.input_tokens, res.usage.output_tokens))


def response_text(res: Message) -> str:
    return ''.join(block.text for block in res.content if block.type == 'text')


async def retry_message_create(query: dict[str, Any], model: str | None = None,
                               max_tries: int = 3, label: str | None = None) -> Message:
    # get_client returns a provider-union; this mapper is only constructed for the Anthropic
    # provider (see mappers.get_mapper), so the concrete client is always an AsyncAnthropic here.
    client = cast(anthropic.AsyncAnthropic, get_client())
    if model is None:
        model = resolve_model()
    label = label or 'llm'
    t1 = time.time()
    query['model'] = model
    if 'max_tokens' not in query:
        query['max_tokens'] = DEFAULT_MAX_TOKENS
    log(f'-> {label}: request sent (model={model})')
    while True:
        try:
            # Streaming is required by the API for requests that may take long
            async with client.messages.stream(**query) as stream:
                out = await stream.get_final_message()
            t2 = time.time()
            total_tokens = out.usage.input_tokens + out.usage.output_tokens
            progress(f'Chat query took {prettify_time_delta(t2 - t1)}, '
                     f'started at {prettify_time_delta(t1 - t0)}, '
                     f'ms/chars = {(t2 - t1) * 1000 / total_tokens}')
            log(f'<= {label}: done in {prettify_time_delta(t2 - t1)} (model={model})')
            return out
        except anthropic.BadRequestError as e:
            message = str(getattr(getattr(e, 'error', None), 'message', ''))
            if 'too long' in message or 'context' in message.lower():
                # A context overflow is a prompt-size problem, not a transient one. Try the
                # larger strong model once; if that cannot fit either, surface it so the caller
                # can compact the prompt rather than retry a prompt that cannot fit.
                if query['model'] != resolve_strong_model():
                    query['model'] = resolve_strong_model()
                else:
                    raise ContextOverflowError(f'prompt exceeds context window: {message}')
            max_tries = max_tries - 1
            if max_tries == 0:
                raise e
            time.sleep(3 / max_tries)
        except Exception as e:
            max_tries = max_tries - 1
            if max_tries == 0:
                raise e
            time.sleep(3 / max_tries)
        if max_tries == 0:
            raise Exception('Could not execute message creation')


class ClaudeMapper(BaseMapper):
    """Anthropic (Claude)-based mapper class"""

    system: str
    model: str | None
    max_tokens: int | None
    reasoning_effort: str | None
    seed: int | None
    max_retries: int
    n_results: int
    stats_stage: str | None

    def __init__(self, system: str, model: str | None = None, max_tokens: int | None = None,
                 reasoning_effort: str | None = None, seed: int | None = None, max_retries: int = 3,
                 n_results: int = 1, stats_stage: str | None = None,
                 label: str | None = None) -> None:
        BaseMapper.__init__(self)
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        # Anthropic has no reasoning_effort; accepted for signature parity
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.max_retries = max_retries
        self.n_results = n_results
        self.stats_stage = stats_stage
        self.label = label

    async def transform(self, user_request: Any) -> Any:
        # A bare string is a single user message; a list of {'role', 'content'}
        # dicts is a whole prior conversation (the tool-use follow-up calls).
        if isinstance(user_request, str):
            user_request = [{'role': 'user', 'content': user_request}]
        query_obj: dict[str, Any] = {
            'system': self.system,
            'messages': list(user_request),
        }
        if self.max_tokens is not None:
            query_obj['max_tokens'] = self.max_tokens
        if self.seed is not None:
            # Best-effort sampling reproducibility (Anthropic honors seed).
            query_obj['seed'] = self.seed
        # Anthropic has no n parameter, so fan out one request per result
        if self.n_results > 1:
            reses = list(await asyncio.gather(*[
                retry_message_create(
                    dict(query_obj), self.model, self.max_retries, self.label)
                for _ in range(self.n_results)
            ]))
        else:
            reses = [await retry_message_create(query_obj, self.model, self.max_retries, self.label)]

        if self.stats_stage is not None:
            stats.stage_update(self.stats_stage, [
                           normalize_response(res) for res in reses])

        texts = [response_text(res) for res in reses]
        return texts if self.n_results > 1 else texts[0]
