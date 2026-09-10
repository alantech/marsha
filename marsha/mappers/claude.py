import asyncio
import time

import anthropic

from marsha.config import resolve_model, resolve_strong_model
from marsha.log import log
from marsha.llm_client import get_client
from marsha.mappers.base import BaseMapper, ContextOverflowError
from marsha.stats import stats
from marsha.utils import prettify_time_delta

# Get time at startup to make human legible "start times" in the logs
t0 = time.time()

# Anthropic requires max_tokens on every request; use this budget when the
# caller did not specify one
DEFAULT_MAX_TOKENS = 32768


class Usage():
    """OpenAI-shaped usage so stats.py can consume both providers"""

    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class NormalizedResponse():
    """OpenAI-shaped response so stats.py can consume both providers"""

    def __init__(self, model, usage):
        self.model = model
        self.usage = usage


def normalize_response(res):
    return NormalizedResponse(
        res.model, Usage(res.usage.input_tokens, res.usage.output_tokens))


def response_text(res):
    return ''.join(block.text for block in res.content if block.type == 'text')


async def retry_message_create(query, model=None, max_tries=3, label=None):
    client = get_client()
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
            print(
                f'''Chat query took {prettify_time_delta(t2 - t1)}, started at {prettify_time_delta(t1 - t0)}, ms/chars = {(t2 - t1) * 1000 / total_tokens}''')
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

    def __init__(self, system, model=None, max_tokens=None, reasoning_effort=None, max_retries=3, n_results=1, stats_stage=None, label=None):
        BaseMapper.__init__(self)
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        # Anthropic has no reasoning_effort; accepted for signature parity
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.n_results = n_results
        self.stats_stage = stats_stage
        self.label = label

    async def transform(self, user_request):
        query_obj = {
            'system': self.system,
            'messages': [{
                'role': 'user',
                'content': user_request,
            }],
        }
        if self.max_tokens is not None:
            query_obj['max_tokens'] = self.max_tokens
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
