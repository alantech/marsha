#!/usr/bin/env python

import argparse
import math
import os
import time

from llm import prettify_time_delta

from mistletoe import Document, ast_renderer


# Parse the input arguments
parser = argparse.ArgumentParser(
    prog='.time.py',
    description='Time the execution of Marsha on the same source multiple times'
)
parser.add_argument('source')
parser.add_argument('attempts', type=int, default=3)
parser.add_argument('n_parallel_executions', type=int, default=1)
parser.add_argument('stats', type=bool, default=False)
args = parser.parse_args()

exitcodes = []
times = []
calls = []
cost = []
total_runs = 16
for i in range(total_runs):
    print(f'Run {i + 1} / {total_runs}')
    t_1 = time.time()
    print(
        f'Running ./dist/marsha {args.source} -a {args.attempts} -n {args.n_parallel_executions} {args.stats and "-s"}')
    exitcode = os.system(
        f'./dist/marsha {args.source} -a {args.attempts} -n {args.n_parallel_executions} {args.stats and "-s"}')
    t_2 = time.time()
    testtime = t_2 - t_1
    exitcodes.append(exitcode)
    times.append(testtime)
    if args.stats:
        run_stats_file = open('stats.md', 'r')
        run_stats = run_stats_file.read()
        run_stats_file.close()
        try:
            ast = ast_renderer.get_ast(Document(run_stats))
            results_child = ast['children'].pop()
            calls.append(int(results_child[
                         'children'][2]['content'].split('Total calls: ').pop()))
            cost.append(float(results_child[
                'children'][6]['content'].split('Total cost: ').pop()))
        except Exception as e:
            print(f'Error: {e}')
            calls.append(0)
            cost.append(0)
        with open('agg_stats.md', 'a') as f:
            f.write(f'''# Run {i + 1} / {total_runs}
Exit code: {exitcode}
Time: {prettify_time_delta(testtime)}
Stats:

```md
{run_stats}
```

''')


successes = [True if code == 0 else False for code in exitcodes]
# Time calculations
totaltime = sum(times)
avgtime = totaltime / total_runs
square_errors = [(t - avgtime) ** 2 for t in times]
stddevtime = math.sqrt(sum(square_errors) / total_runs)
# Call calculations
totalcalls = sum(calls)
avgcalls = totalcalls / total_runs
square_errors = [(c - avgcalls) ** 2 for c in calls]
stddevcalls = math.sqrt(sum(square_errors) / total_runs)
# Cost calculations
totalcost = sum(cost)
avgcost = totalcost / total_runs
square_errors = [(c - avgcost) ** 2 for c in cost]
stddevcost = math.sqrt(sum(square_errors) / total_runs)

results = f'''
# Test results
{sum(successes)} / {total_runs} runs successful
Runtime of {prettify_time_delta(avgtime)} +/- {prettify_time_delta(stddevtime)}
GPT calls {avgcalls} +/- {stddevcalls}
Total cost {totalcost}
Average cost {avgcost} +/- {stddevcost}
'''
print(results)
res_file = open('results.md', 'w')
res_file.write(results)
res_file.close()

if args.stats:
    with open('agg_stats.md', 'r') as f:
        stats = f.read()
    print(stats)
