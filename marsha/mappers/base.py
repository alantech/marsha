class BaseMapper():
    """Semi-abstract base for 'mappers' in Marsha"""

    def __init__(self):
        self.check_retries = 3
        self.output = None

    async def transform(self, i):
        raise Exception('Not implemented')

    async def check(self):
        # Define a check if you want, but not necessary
        return self.output

    async def run(self, i):
        try:
            self.output = await self.transform(i)
        except Exception as e:
            # TODO: Log the error before re-raise?
            raise e

        iters = self.check_retries
        while iters > 0:
            try:
                o = await self.check()
                return o
            except Exception:
                # Using the exception here as flow control
                iters = iters - 1

        raise Exception('Transformer failed to converge')
