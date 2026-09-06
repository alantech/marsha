from marsha.utils import write_file

# Price per 1024 tokens (input, output), matched by longest model name prefix
PRICING_MODEL = {
    'gpt-5-mini': (0.000244140625, 0.001953125),
    'gpt-5-nano': (0.000048828125, 0.000390625),
    'gpt-5': (0.001220703125, 0.009765625),
    'gpt-4': (0.03, 0.06),
    'gpt-3.5': (0.0015, 0.002),
    'claude-sonnet-5': (0.0029296875, 0.0146484375),
    'claude-opus-5': (0.0146484375, 0.0732421875),
}


def price_for(model):
    best = None
    for prefix in PRICING_MODEL:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return PRICING_MODEL[best] if best else (0.0, 0.0)


class ModelStats:
    def __init__(self, name, input_tokens, output_tokens, input_cost, output_cost, total_cost):
        self.name = name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.total_cost = total_cost


class StageStats:
    def __init__(self, name, total_time, total_calls):
        self.name = name
        self.total_time = total_time
        self.total_calls = total_calls
        self.models = {}

    def update(self, res: list):
        self.total_calls += len(res)
        for r in res:
            ms = self.models.get(r.model)
            if ms is None:
                ms = self.models[r.model] = ModelStats(r.model, 0, 0, 0, 0, 0)
            in_price, out_price = price_for(r.model)
            ms.input_tokens += r.usage.prompt_tokens
            ms.input_cost += r.usage.prompt_tokens * in_price / 1024
            ms.output_tokens += r.usage.completion_tokens
            ms.output_cost += r.usage.completion_tokens * out_price / 1024
            ms.total_cost = ms.input_cost + ms.output_cost


class MarshaStats:
    def __init__(self):
        self.total_time = 0
        self.total_calls = 0
        self.attempts = 0
        self.total_cost = 0
        self.first_stage = StageStats('first_stage', 0, 0)
        self.second_stage = StageStats('second_stage', 0, 0)
        self.third_stage = StageStats('third_stage', 0, 0)

    @property
    def stages(self):
        return [self.first_stage, self.second_stage, self.third_stage]

    def stage_update(self, stage: str, res: list):
        stage_stats = getattr(self, stage, None)
        if isinstance(stage_stats, StageStats):
            stage_stats.update(res)

    def aggregate(self, total_time, attempts):
        self.total_time = total_time
        self.attempts = attempts
        self.total_calls = sum(stage.total_calls for stage in self.stages)
        self.total_cost = sum(
            ms.total_cost for stage in self.stages for ms in stage.models.values())

    def to_file(self, filename: str = 'stats.md'):
        write_file(filename, content=self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        stage_titles = {
            'first_stage': 'First',
            'second_stage': 'Second',
            'third_stage': 'Third',
        }
        lines = ['# Stats', '']
        for stage in self.stages:
            total_cost = sum(ms.total_cost for ms in stage.models.values())
            lines.append(f'## {stage_titles[stage.name]} stage')
            lines.append(f'Total time: {stage.total_time}')
            lines.append(f'Total calls: {stage.total_calls}')
            lines.append(f'Total cost: {total_cost}')
            for ms in stage.models.values():
                lines.append(
                    f'  {ms.name}: {ms.input_tokens} input tokens, '
                    f'{ms.output_tokens} output tokens, cost {ms.total_cost}')
            lines.append('')
        lines.append('## Total')
        lines.append(f'Total time: {self.total_time}')
        lines.append(f'Total calls: {self.total_calls}')
        lines.append(f'Attempts: {self.attempts}')
        lines.append(f'Total cost: {self.total_cost}')
        return '\n'.join(lines)


stats = MarshaStats()
