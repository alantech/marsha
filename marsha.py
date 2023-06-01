import argparse
import os
import time

import openai

openai.organization = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_SECRET_KEY')

parser = argparse.ArgumentParser(
        prog='marsha',
        description='Marsha AI Compiler',
)
parser.add_argument('source')
args = parser.parse_args()

def retry_chat_completion(query, model='gpt-3.5-turbo', max_tries=3):
    t1 = time.time()
    query['model'] = model
    while True:
        try:
            out = openai.ChatCompletion.create(**query)
            t2 = time.time()
            print(f'''Chat query took {(t2 - t1) * 1000}ms, started at {t1}, ms/chars = {(t2 - t1) * 1000 / out.get('usage', {}).get('total_tokens', 9001)}''')
            return out
        except openai.error.InvalidRequestError as e:
            if e.code == 'context_length_exceeded':
                query['model'] = 'gpt-4'  # Try to cover up this error by choosing the bigger, more expensive model
            max_tries = max_tries - 1
            if max_tries == 0:
                raise e
            time.sleep(3 / max_tries)
        if max_tries == 0:
            raise Exception('Could not execute chat completion')


def gpt_func_to_python(func):
    res = retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': 'You are a senior software engineer assigned to write a Python 3 function. The assignment is written in markdown format, with a markdown title consisting of a pseudocode function signature (name, arguments, return type) followed by a description of the function and then a bullet-point list of example cases for the function. You write up a simple file that imports libraries if necessary and contains the function, and a second file that includes unit tests at the end based on the provided test cases. The filenames should follow the pattern of [function name].py and [function name]_test.py',
        }, {
            'role': 'user',
            'content': '''# func fibonacci(integer): integer in the set of fibonacci numbers

This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)

* fibonacci(1) = 1
* fibonacci(2) = 1
* fibonacci(3) = 2
* fibonacci(0) throws an error''',
        }, {
            'role': 'assistant',
            'content': '''# fibonnaci.py

```py
def fibonacci(n):
  if n <= 0:
    raise Exception(\'The fibonacci sequence only exists in positive whole number space\')
  elif n == 1 or n == 2:
    return 1
  else:
    return fibonacci(n - 1) + fibonacci(n - 2)
```

# fibonacci_test.py

```py
import unittest
import fibonacci from fibonacci


class TestFibonacci(unittest.TestCase):

  def test_1(self):
    self.assertEqual(fibonacci(1), 1)

  def test_2(self):
    self.assertEqual(fibonacci(2), 1)

  def test_3(self):
    self.assertEqual(fibonacci(3), 2)

  def test_0(self):
    self.assertRaises(Exception, fibonacci, 0)


if __name__ == '__main__':
    unittest.main()
```'''
        }, {
            'role': 'user',
            'content': func
        }],
    })
    return res.choices[0].message.content


def main():
    f = open(args.source, 'r')
    print(gpt_func_to_python(f.read()))


main()
