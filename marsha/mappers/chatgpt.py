from __future__ import annotations

import time
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletion

from marsha.config import is_local_backend, resolve_model, resolve_strong_model
from marsha.log import log, progress
from marsha.llm_client import get_client
from marsha.mappers.base import BaseMapper, ContextOverflowError
from marsha.stats import stats
from marsha.utils import prettify_time_delta

# Get time at startup to make human legible "start times" in the logs
t0: float = time.time()


def uses_completion_tokens(model: str) -> bool:
    # Reasoning models reject max_tokens and require max_completion_tokens. GPT-5, GPT-5.6 and
    # GPT-6 are all reasoning models (gpt-5.6-*/gpt-6-* do not match the 'gpt-5' prefix, so the
    # gpt-6 family is listed explicitly).
    return (model.startswith('gpt-5') or model.startswith('gpt-6')
            or model.startswith('o'))


async def retry_chat_completion(query: dict[str, Any], model: str | None = None,
                                max_tries: int = 3, n_results: int = 1,
                                label: str | None = None) -> ChatCompletion:
    # get_client returns a provider-union; this mapper is only constructed for the OpenAI
    # provider (see mappers.get_mapper), so the concrete client is always an AsyncOpenAI here.
    client = cast(openai.AsyncOpenAI, get_client())
    if model is None:
        model = resolve_model()
    label = label or 'llm'
    t1 = time.time()
    query['model'] = model
    query['n'] = n_results
    if 'max_tokens' in query and uses_completion_tokens(model):
        query['max_completion_tokens'] = query.pop('max_tokens')
    log(f'-> {label}: request sent (model={model}, n={n_results})')
    while True:
        try:
            out = cast(ChatCompletion, await client.chat.completions.create(**query))
            t2 = time.time()
            total_tokens = out.usage.total_tokens if out.usage is not None else 9001
            progress(f'Chat query took {prettify_time_delta(t2 - t1)}, '
                     f'started at {prettify_time_delta(t1 - t0)}, '
                     f'ms/chars = {(t2 - t1) * 1000 / total_tokens}')
            log(f'<= {label}: done in {prettify_time_delta(t2 - t1)} (model={model})')
            return out
        except openai.BadRequestError as e:
            if getattr(e, 'code', None) == 'context_length_exceeded':
                # A context overflow is a prompt-size problem, not a transient one. On a real
                # OpenAI backend the larger strong model may fit it, so try that once; otherwise
                # (a local backend, or already on the strong model) surface it so the caller can
                # compact the prompt rather than retry a prompt that cannot fit.
                if not is_local_backend() and query['model'] != resolve_strong_model():
                    query['model'] = resolve_strong_model()
                else:
                    raise ContextOverflowError(f'prompt exceeds context window: {e}')
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
            raise Exception('Could not execute chat completion')


class ChatGPTMapper(BaseMapper):
    """ChatGPT-based mapper class"""

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
            'messages': [{
                'role': 'system',
                'content': self.system,
            }] + user_request,
        }
        if self.max_tokens is not None:
            query_obj['max_tokens'] = self.max_tokens
        if self.reasoning_effort is not None:
            query_obj['reasoning_effort'] = self.reasoning_effort
        if self.seed is not None:
            # Best-effort reproducibility. Some reasoning models (gpt-5-mini) ignore it; the
            # parameter is harmless where it is not honored.
            query_obj['seed'] = self.seed
        res = await retry_chat_completion(query_obj, self.model, self.max_retries, self.n_results, self.label)

        if self.stats_stage is not None:
            stats.stage_update(self.stats_stage, [res])

        return [choice.message.content for choice in res.choices] if self.n_results > 1 else res.choices[0].message.content
