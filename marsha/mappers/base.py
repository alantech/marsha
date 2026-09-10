from marsha.log import dump


class ContextOverflowError(Exception):
    """Raised when a request is rejected because it exceeds the model's context window. This is a
    property of the prompt size (not a transient error), so callers should shrink the prompt
    rather than retry it verbatim."""


class BaseMapper():
    """Semi-abstract base for 'mappers' in Marsha"""

    def __init__(self):
        self.check_retries = 3
        self.output = None
        self.label = 'llm'

    async def transform(self, i):
        raise Exception('Not implemented')

    async def check(self):
        # Define a check if you want, but not necessary
        return self.output

    async def run(self, i):
        # Every LLM call funnels through here, so this is the single place to capture the full
        # input/output transcript for the --trace-full level.
        label = getattr(self, 'label', None) or 'llm'
        dump(f'{label}: request', i)
        try:
            self.output = await self.transform(i)
        except Exception as e:
            # TODO: Log the error before re-raise?
            raise e
        dump(f'{label}: response', self.output)

        iters = self.check_retries
        while iters > 0:
            try:
                o = await self.check()
                return o
            except Exception:
                # Using the exception here as flow control
                iters = iters - 1

        raise Exception('Transformer failed to converge')
