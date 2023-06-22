#!/usr/bin/env python

import argparse
import math
import os
import time

from llm import prettify_time_delta

# Parse the input arguments
parser = argparse.ArgumentParser(
    prog='.time.py',
    description='Time the execution of Marsha on the same source multiple times'
)
parser.add_argument('source')
parser.add_argument('attempts', type=int, default=3)
parser.add_argument('n_parallel_executions', type=int, default=1)
args = parser.parse_args()

exitcodes = []
times = []
for i in range(8):
    print(f'Run {i + 1} / 8')
    t_1 = time.time()
    print(
        f'Running ./dist/marsha {args.source} -a {args.attempts} -n {args.n_parallel_executions}')
    exitcode = os.system(
        f'./dist/marsha {args.source} -a {args.attempts} -n {args.n_parallel_executions}')
    t_2 = time.time()
    testtime = t_2 - t_1
    exitcodes.append(exitcode)
    times.append(testtime)

successes = [True if code == 0 else False for code in exitcodes]
totaltime = sum(times)
avgtime = totaltime / 8
square_errors = [(t - avgtime) ** 2 for t in times]
stddev = math.sqrt(sum(square_errors) / 8)

print("Test results")
print(f'{sum(successes)} / 8 runs successful')
print(
    f'Runtime of {prettify_time_delta(avgtime)} +/- {prettify_time_delta(stddev)}')
