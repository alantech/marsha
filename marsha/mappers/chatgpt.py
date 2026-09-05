import time

import openai

from marsha.llm_client import get_client
from marsha.mappers.base import BaseMapper
from marsha.stats import stats
from marsha.utils import prettify_time_delta

# Get time at startup to make human legible "start times" in the logs
t0 = time.time()


async def retry_chat_completion(query, model='gpt-3.5-turbo', max_tries=3, n_results=1):
    client = get_client()
    t1 = time.time()
    query['model'] = model
    query['n'] = n_results
    while True:
        try:
            out = await client.chat.completions.create(**query)
            t2 = time.time()
            total_tokens = out.usage.total_tokens if out.usage is not None else 9001
            print(
                f'''Chat query took {prettify_time_delta(t2 - t1)}, started at {prettify_time_delta(t1 - t0)}, ms/chars = {(t2 - t1) * 1000 / total_tokens}''')
            return out
        except openai.BadRequestError as e:
            if getattr(e, 'code', None) == 'context_length_exceeded':
                # Try to cover up this error by choosing the bigger, more expensive model
                query['model'] = 'gpt-4'
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

    def __init__(self, system, model='gpt-3.5-turbo', max_tokens=None, max_retries=3, n_results=1, stats_stage=None):
        BaseMapper.__init__(self)
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.n_results = n_results
        self.stats_stage = stats_stage

    async def transform(self, user_request):
        query_obj = {
            'messages': [{
                'role': 'system',
                'content': self.system,
            }, {
                'role': 'user',
                'content': user_request,
            }],
        }
        if self.max_tokens is not None:
            query_obj['max_tokens'] = self.max_tokens
        res = await retry_chat_completion(query_obj, self.model, self.max_retries, self.n_results)

        if self.stats_stage is not None:
            stats.stage_update(self.stats_stage, [res])

        return [choice.message.content for choice in res.choices] if self.n_results > 1 else res.choices[0].message.content
