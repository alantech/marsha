from utils import write_file


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
        self.gpt35 = ModelStats('gpt-3.5-turbo', 0, 0, 0, 0, 0)
        self.gpt4 = ModelStats('gpt-4', 0, 0, 0, 0, 0)


class MarshaStats:
    def __init__(self):
        self.total_time = 0
        self.total_calls = 0
        self.attempts = 0
        self.total_cost = 0
        self.first_stage = StageStats('first_stage', 0, 0)
        self.second_stage = StageStats('second_stage', 0, 0)
        self.third_stage = StageStats('third_stage', 0, 0)

    def aggregate(self, total_time, attempts):
        self.total_time = total_time
        self.attempts = attempts
        self.total_calls = self.first_stage.total_calls + \
            self.second_stage.total_calls + self.third_stage.total_calls
        self.total_cost = self.first_stage.gpt35.total_cost + self.first_stage.gpt4.total_cost + self.second_stage.gpt35.total_cost + \
            self.second_stage.gpt4.total_cost + \
            self.third_stage.gpt35.total_cost + self.third_stage.gpt4.total_cost

    def to_file(self, filename: str = 'stats.md'):
        write_file(filename, content=self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'''# Stats

## First stage
Total time: {self.first_stage.total_time}
Total calls: {self.first_stage.total_calls}
Total cost: {self.first_stage.gpt35.total_cost + self.first_stage.gpt4.total_cost}

## Second stage
Total time: {self.second_stage.total_time}
Total calls: {self.second_stage.total_calls}
Total cost: {self.second_stage.gpt35.total_cost + self.second_stage.gpt4.total_cost}

## Third stage
Total time: {self.third_stage.total_time}
Total calls: {self.third_stage.total_calls}
Total cost: {self.third_stage.gpt35.total_cost + self.third_stage.gpt4.total_cost}

## Total
Total time: {self.total_time}
Total calls: {self.total_calls}
Attempts: {self.attempts}
Total cost: {self.total_cost}
'''
