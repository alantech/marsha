import { Configuration, OpenAIApi } from 'openai';

const configuration = new Configuration({
  organization: process.env.OPENAI_ORG,
  apiKey: process.env.OPENAI_SECRET_KEY,
});

const openai = new OpenAIApi(configuration);

async function retryChatCompletion(query, model='gpt-3.5-turbo', maxTries=3) {
  const t1 = performance.now();
  query.model = model;
  do {
    try {
      const out = await openai.createChatCompletion(query);
      const t2 = performance.now();
      console.log(`Chat Query took ${t2 - t1}ms, started at ${t1}, ms/chars = ${(t2 - t1) / (out.data.usage?.total_tokens ?? 9001)}`);
      return out;
    } catch (e) {
      if (e?.response?.data?.error?.code === 'context_length_exceeded') {
        query.model = 'gpt-4'; // Try to cover up this error by choosing the bigger, more expensive model
      }
      maxTries--;
      if (!maxTries) throw e;
      await new Promise(r => setTimeout(r, 3000 / maxTries));
    }
  } while(maxTries);
}

async function gptFixPythonLint(filename, content, lint, retries = 3) {
  const res = await retryChatCompletion({
    messages: [{
      role: 'system',
      content: 'You are a senior software engineer working on a Python 3 function. You are using the pylama linting tool to find obvious errors and then fixing them.',
    }, {
      role: 'user',
      content: `# database_info.py

\`\`\`py
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
\`\`\`

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
\`\`\``,
    }, {
      role: 'assistant',
      content: `# database_info.py

\`\`\`py
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
\`\`\``
    }, {
      role: 'user',
      content: `# ${filename}

\`\`\`py
${content}
\`\`\`

# pylama results

\`\`\`
${lint}
\`\`\``,
    }],
  });
  const fn = res.data.choices[0].message.content.split('```').find(r => /^ts\n/.test(r))?.replace(/^ts\n/, '');
  if (!!fn && !functionParses('test', fn) && retries) {
    return gptDeclaration(func, retries--);
  }
  return res.data.choices[0].message.content;
}

async function main() {
  console.log(await gptFixPythonLint(
    'database_info_test.py',
    `import unittest
from database_info import extract_connection_info

class TestExtractConnectionInfo(unittest.TestCase):

  def test_1(self):
    expected = { "protocol": "postgresql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0:5432", "database": "mydb" }
    self.assertEqual(extract_connection_info('postgresql://user:pass@0.0.0.0:5432/mydb'), json.dumps(expected))

  def test_2(self):
    expected = {"protocol": "postgresql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0:5432", "database": "mydb","extra": {"ssl": "require"}}
    self.assertEqual(extract_connection_info('postgresql://user:pass@0.0.0.0:5432/mydb?sslmode=require'), json.dumps(expected))

  def test_3(self):
    self.assertEqual(extract_connection_info(''), '{}')`,
    `database_info_test.py:4:1 E302 expected 2 blank lines, found 1 [pycodestyle]
database_info_test.py:6:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info_test.py:7:101 E501 line too long (127 > 100 characters) [pycodestyle]
database_info_test.py:7:17 E201 whitespace after '{' [pycodestyle]
database_info_test.py:7:126 E202 whitespace before '}' [pycodestyle]
database_info_test.py:8:101 E501 line too long (111 > 100 characters) [pycodestyle]
database_info_test.py:8:91 E0602 undefined name 'json' [pyflakes]
database_info_test.py:10:3 E111 indentation is not a multiple of 4 [pycodestyle]
database_info_test.py:11:101 E501 line too long (153 > 100 characters) [pycodestyle]
database_info_test.py:11:125 E231 missing whitespace after ',' [pycodestyle]
database_info_test.py:12:101 E501 line too long (127 > 100 characters) [pycodestyle]
database_info_test.py:12:107 E0602 undefined name 'json' [pyflakes]
database_info_test.py:14:3 E111 indentation is not a multiple of 4 [pycodestyle]`));
}

main();