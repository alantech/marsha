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
args = parser.parse_args()

exitcodes = []
times = []
for i in range(8):
    t_1 = time.time()
    exitcode = os.system(f'./dist/marsha {args.source}')
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
print(f'Runtime of {prettify_time_delta(avgtime)} +/- {prettify_time_delta(stddev)}')