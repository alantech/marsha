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

async function gptFuncToPython(func, retries = 3) {
  const res = await retryChatCompletion({
    messages: [{
      role: 'system',
      content: 'You are a senior software engineer assigned to write a Python 3 function. The assignment is written in markdown format, with a markdown title consisting of a pseudocode function signature (name, arguments, return type) followed by a description of the function and then a bullet-point list of example cases for the function. You write up a simple file that imports libraries if necessary and contains the function, and a second file that includes unit tests at the end based on the provided test cases. The filenames should follow the pattern of [function name].py and [function name]_test.py',
    }, {
      role: 'user',
      content: `# fibonacci(n: int): int

This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)

* fibonacci(1) = 1
* fibonacci(2) = 1
* fibonacci(3) = 2
* fibonacci(0) throws an error`,
    }, {
      role: 'assistant',
      content: `# fibonnaci.py

\`\`\`py
def fibonacci(n):
  if n <= 0:
    raise Exception('The fibonacci sequence only exists in positive whole number space')
  elif n == 1 or n == 2:
    return 1
  else:
    return fibonacci(n - 1) + fibonacci(n - 2)
\`\`\`

# fibonacci_test.py

\`\`\`py
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
\`\`\`
`
    }, {
      role: 'user',
      content: `${func}`
    }],
  });
  return res.data.choices[0].message.content;
}

async function main() {
  console.log(await gptFuncToPython(`# extract_connection_info(database url): JSON object with connection properties

It should extract from the database url all the connection properties in a JSON format.

* extract_connection_info('postgresql://user:pass@0.0.0.0:5432/mydb') = { "protocol": "postgresql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0:5432", "database": "mydb" }
* extract_connection_info('postgresql://user:pass@0.0.0.0:5432/mydb?sslmode=require') = { "protocol": "postgresql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0:5432", "database": "mydb", "extra": { "ssl": "require" } }
* extract_connection_info('') = {}`));
}

main();