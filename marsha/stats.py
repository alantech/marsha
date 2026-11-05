from __future__ import annotations

from typing import Any

from marsha.utils import write_file

# Price per 1024 tokens (input, output), matched by longest model name prefix
PRICING_MODEL: dict[str, tuple[float, float]] = {
    'gpt-6-luna': (0.00009765625, 0.00048828125),
    'gpt-6-sol': (0.001953125, 0.009765625),
    'gpt-5.6-terra': (0.001953125, 0.01171875),
    'gpt-5-mini': (0.000244140625, 0.001953125),
    'gpt-5-nano': (0.000048828125, 0.000390625),
    'gpt-5': (0.001220703125, 0.009765625),
    'gpt-4': (0.03, 0.06),
    'gpt-3.5': (0.0015, 0.002),
    'claude-haiku-4-5': (0.0009765625, 0.0048828125),
    'claude-sonnet-5': (0.0029296875, 0.0146484375),
    'claude-opus-5': (0.0146484375, 0.0732421875),
}


def price_for(model: str) -> tuple[float, float]:
    best: str | None = None
    for prefix in PRICING_MODEL:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return PRICING_MODEL[best] if best else (0.0, 0.0)


def price_known(model: str) -> bool:
    # True when the model matches a price-table entry (by prefix). For other models the price
    # is unknown, which consumers must treat as neutral (not as free).
    return any(model.startswith(prefix) for prefix in PRICING_MODEL)


class ModelStats:
    def __init__(self, name: str, input_tokens: int, output_tokens: int,
                 input_cost: float, output_cost: float, total_cost: float) -> None:
        self.name = name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.total_cost = total_cost


class StageStats:
    def __init__(self, name: str, total_time: int | str, total_calls: int) -> None:
        self.name = name
        self.total_time = total_time
        self.total_calls = total_calls
        self.models: dict[str, ModelStats] = {}

    def update(self, res: list[Any]) -> None:
        self.total_calls += len(res)
        for r in res:
            usage = r.usage
            if usage is None:
                continue
            ms = self.models.get(r.model)
            if ms is None:
                ms = self.models[r.model] = ModelStats(r.model, 0, 0, 0, 0, 0)
            in_price, out_price = price_for(r.model)
            ms.input_tokens += usage.prompt_tokens
            ms.input_cost += usage.prompt_tokens * in_price / 1024
            ms.output_tokens += usage.completion_tokens
            ms.output_cost += usage.completion_tokens * out_price / 1024
            ms.total_cost = ms.input_cost + ms.output_cost


class MarshaStats:
    total_time: int | str
    total_calls: int
    attempts: int
    total_cost: float
    first_stage: StageStats
    second_stage: StageStats
    third_stage: StageStats

    def __init__(self) -> None:
        self.total_time = 0
        self.total_calls = 0
        self.attempts = 0
        self.total_cost = 0
        self.first_stage = StageStats('first_stage', 0, 0)
        self.second_stage = StageStats('second_stage', 0, 0)
        self.third_stage = StageStats('third_stage', 0, 0)

    @property
    def stages(self) -> list[StageStats]:
        return [self.first_stage, self.second_stage, self.third_stage]

    def stage_update(self, stage: str, res: list[Any]) -> None:
        stage_stats = getattr(self, stage, None)
        if isinstance(stage_stats, StageStats):
            stage_stats.update(res)

    def aggregate(self, total_time: int | str, attempts: int) -> None:
        self.total_time = total_time
        self.attempts = attempts
        self.total_calls = sum(stage.total_calls for stage in self.stages)
        self.total_cost = sum(
            ms.total_cost for stage in self.stages for ms in stage.models.values())

    def to_file(self, filename: str = 'stats.md') -> None:
        write_file(filename, content=self.__str__())

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
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

# Global cost accumulator for --max-price and the cost summaries used by the headless commands.
# Updated by the mappers on every LLM call (including compile's staged stats).
_global_cost: float = 0.0
_global_in_tokens: int = 0
_global_out_tokens: int = 0
_max_price: float | None = None


class PriceLimitExceeded(BaseException):
    """Stop the whole command when its estimated model spend reaches the configured cap.

    This intentionally derives from BaseException so broad `except Exception` recovery paths
    cannot swallow a spend-limit stop and continue making model calls or commit partial work.
    """


def begin_invocation(max_price: float | None) -> None:
    """Reset per-command usage and install its optional estimated-spend limit."""
    global _global_cost, _global_in_tokens, _global_out_tokens, _max_price
    _global_cost = 0.0
    _global_in_tokens = 0
    _global_out_tokens = 0
    _max_price = max_price


def max_price_enabled() -> bool:
    return _max_price is not None


def ensure_price_limit(model: str) -> None:
    """Refuse a request when the cap is spent or its model cannot be priced."""
    if _max_price is None:
        return
    if _global_cost >= _max_price:
        raise PriceLimitExceeded(
            f'estimated model spend ${_global_cost:.6f} has reached --max-price '
            f'${_max_price:.6f}; stopping before another model request')
    if not price_known(model):
        raise PriceLimitExceeded(
            f'cannot enforce --max-price because no token pricing is configured '
            f'for model {model!r}')


def ensure_usage_available() -> None:
    """Fail closed if a capped request completed without provider token usage."""
    if _max_price is not None:
        raise PriceLimitExceeded(
            'cannot enforce --max-price because the provider returned no token usage')


def check_price_limit() -> None:
    """Stop after the completed model request that reaches or crosses the cap."""
    if _max_price is not None and _global_cost >= _max_price:
        raise PriceLimitExceeded(
            f'estimated model spend ${_global_cost:.6f} reached or exceeded '
            f'--max-price ${_max_price:.6f}; stopping before another model request')


def record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    in_price, out_price = price_for(model)
    global _global_cost, _global_in_tokens, _global_out_tokens
    _global_in_tokens += prompt_tokens
    _global_out_tokens += completion_tokens
    _global_cost += (prompt_tokens * in_price +
                     completion_tokens * out_price) / 1024


def global_cost() -> float:
    return _global_cost


def global_tokens() -> tuple[int, int]:
    return _global_in_tokens, _global_out_tokens


def format_cost() -> str:
    in_tok, out_tok = global_tokens()
    cost = global_cost()
    return (f'Total cost: ${round(cost, 2)} '
            f'({in_tok:,} input, {out_tok:,} output tokens)')
