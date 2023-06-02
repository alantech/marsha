import argparse
import asyncio
import os
import time

from mistletoe import Document, ast_renderer
import openai
from pylama.main import parse_options, check_paths, DEFAULT_FORMAT

openai.organization = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_SECRET_KEY')

parser = argparse.ArgumentParser(
        prog='marsha',
        description='Marsha AI Compiler',
)
parser.add_argument('source')
args = parser.parse_args()

async def retry_chat_completion(query, model='gpt-3.5-turbo', max_tries=3):
    t1 = time.time()
    query['model'] = model
    while True:
        try:
            out = await openai.ChatCompletion.acreate(**query)
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
        except:
            max_tries = max_tries - 1
            if max_tries == 0:
                raise e
            time.sleep(3 / max_tries)
        if max_tries == 0:
            raise Exception('Could not execute chat completion')


async def gpt_func_to_python(func, retries=3):
    res = await retry_chat_completion({
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
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        return Document(res.choices[0].message.content)
    except:
        if retries > 0:
            return await gpt_func_to_python(func, retries - 1)
        else:
            raise Exception('Failed to generate code', func)

def write_files_from_markdown(md):
    ast = ast_renderer.get_ast(md)
    filenames = []
    filename = ''
    filedata = ''
    for section in ast['children']:
        # TODO: Validate better that the markdown is in the form of Heading then CodeFence, etc
        if section['type'] == 'Heading':
            filename = section['children'][0]['content']
            filenames.append(filename)
        elif section['type'] == 'CodeFence':
            filedata = section['children'][0]['content']
            f = open(filename, 'w')
            f.write(filedata)
            f.close()
    return filenames


async def fix_file(filename, lint_text, retries=3):
    f = open(filename, 'r')
    code = f.read()
    f.close()
    res = await retry_chat_completion({
        'messages': [{
            'role': 'system',
            'content': 'You are a senior software engineer working on a Python 3 function. You are using the pylama linting tool to find obvious errors and then fixing them. It uses pyflakes, mccabe, and pycodestyle under the hood to provde its recommendations, and all of the lint errors require fixing.',
        }, {
            'role': 'user',
            'content': '''# database_info.py

```py
import re
import json

def extract_connection_info(database_url):
  result = {}

  if database_url == '':
    return result

  host_database = re.compile(r'^(\\w+):\\/\\/(\\w+):(\\w+)@([^\\/]+)\\/(\\w+)$')
  match = host_database.match(database_url)

  protocol = match.group(1)
  db_user = match.group(2)
  db_password = match.group(3)
  host = match.group(4)
  database = match.group(5)

  result['protocol'] = protocol
  result['dbUser'] = db_user
  result['dbPassword'] = db_password
  result['host'] = host
  result['database'] = database

  extra = re.compile(r'^\\w+:\\/\\/\\w+:\\w+@\\w+\\/\\w+\\?([\\w\\=&]+)$').search(database_url)
  extra_result = {}
  if extra:
    ssl_mode = re.compile(r'sslmode=([\\w-]+)').search(extra.group(1))
    if ssl_mode:
        extra_result['ssl'] = ssl_mode.group(1)

    result['extra'] = extra_result

  return json.dumps(result)
```

# pylama results

\`\`\`
database_info.py:4:1 E302 expected 2 blank lines, found 1 [pycodestyle]
database_info.py:5:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:7:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:10:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:11:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:13:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:14:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:15:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:16:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:17:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:19:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:20:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:21:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:22:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:23:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:25:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:26:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:27:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info.py:34:3 E111 indentation is not a multiple of 4 [pycodestyle]
```''',
        }, {
            'role': 'assistant',
            'content': '''# database_info.py

```py
import re
import json


def extract_connection_info(database_url):
    result = {}

    if database_url == '':
        return result

    host_database = re.compile(r'^(\\w+):\\/\\/(\\w+):(\\w+)@([^\\/]+)\\/(\\w+)$')
    match = host_database.match(database_url)

    protocol = match.group(1)
    db_user = match.group(2)
    db_password = match.group(3)
    host = match.group(4)
    database = match.group(5)

    result['protocol'] = protocol
    result['dbUser'] = db_user
    result['dbPassword'] = db_password
    result['host'] = host
    result['database'] = database

    extra = re.compile(r'^\\w+:\\/\\/\\w+:\\w+@\\w+\\/\\w+\\?([\\w\\=&]+)$').search(database_url)
    extra_result = {}
    if extra:
        ssl_mode = re.compile(r'sslmode=([\\w-]+)').search(extra.group(1))
        if ssl_mode:
            extra_result['ssl'] = ssl_mode.group(1)

        result['extra'] = extra_result

    return json.dumps(result)
```'''
        }, {
            'role': 'user',
            'content': f'''# {filename}

```py
{code}
```

# pylama results

```
{lint_text}
```''',
        }],
    })
    # The output should be a valid Markdown document. Parse it and return the parsed doc, on failure
    # try again (or fully error out, for now)
    try:
        parsed = Document(res.choices[0].message.content)
        write_files_from_markdown(parsed)
    except:
        if retries > 0:
            return await fix_file(filename, lint_text, retries - 1)
        else:
            raise Exception('Failed to generate code', func)


async def lint_and_fix_files(files, max_depth=10):
    if max_depth == 0:
        return
    options = parse_options()
    # Disabling pydocstyle for now
    # options.linters = ['mccabe', 'pycodestyle', 'pydocstyle', 'pyflakes']
    options.linters = ['mccabe', 'pycodestyle', 'pyflakes']
    options.paths = [os.path.abspath(f'./{file}') for file in files]
    lints = check_paths([os.path.abspath(f'./{file}') for file in files], options=options, rootdir='.')
    if len(lints) == 0:
        return
    jobs = []
    for file in files:
        file_lints = [e.format(DEFAULT_FORMAT) for e in lints if e.filename == file]
        if len(file_lints) > 0:
            lint_text = '\n'.join(file_lints)
            print(lint_text)
            jobs.append(fix_file(file, lint_text))
    await asyncio.gather(*jobs)
    await lint_and_fix_files(files, max_depth - 1)


async def main():
    print('Compiling...')
    f = open(args.source, 'r')
    func = f.read()
    f.close()
    files = write_files_from_markdown(await gpt_func_to_python(func))
    await lint_and_fix_files(files)
    print('Done!')


asyncio.run(main())
