"""Tests for the fake-terminal tool-use system (issue #197).

Network access is mocked: the web handlers are exercised against fixed
HTML/JSON fixtures, and the tool loop is driven by a scripted fake mapper,
so the tests are deterministic and offline.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from marsha import llm, tools
from marsha.meta import MarshaMeta


def make_meta(filename='example'):
    meta = MarshaMeta(f'{filename}.mrsh')
    meta.filename = filename
    meta.functions = []
    meta.void_funcs = []
    meta.types = None
    return meta


DOC = '# example_test.py\n\n```py\ndef test():\n    pass\n```\n'

VALID_ORACLE = '# example_test.py\n\n```py\ndef test():\n    pass\n```\n'

VALID_IMPL = '# example.py\n\n```py\ndef f():\n    return 1\n```\n'


# --- $ command extraction ----------------------------------------------------

def test_extract_no_commands():
    assert tools.extract_pending_commands(DOC) == []
    assert tools.extract_pending_commands('') == []
    assert tools.extract_pending_commands(None) == []
    assert tools.extract_pending_commands(42) == []
    # A $ line that is content inside a closed code fence is not a command.
    assert tools.extract_pending_commands('# x.sh\n\n```sh\necho $HOME\n```\n') == []


def test_extract_single_command():
    text = DOC + '\n$ web-search "pandas read_csv parameters"\n'
    pending = tools.extract_pending_commands(text)
    assert [p.name for p in pending] == ['web-search']
    # A quoted phrase is a single argument (the handler joins args with spaces).
    assert pending[0].args == ['pandas read_csv parameters']
    assert pending[0].line == '$ web-search "pandas read_csv parameters"'


def test_extract_multiple_commands():
    text = DOC + '\n$ web-search "a b"\n$ view-web-page "https://example.com/docs"\n'
    pending = tools.extract_pending_commands(text)
    assert [(p.name, p.args) for p in pending] == [
        ('web-search', ['a b']),
        ('view-web-page', ['https://example.com/docs']),
    ]


def test_extract_ignores_trailing_blank_lines():
    text = DOC + '\n$ web-search "a"\n\n\n'
    assert [p.name for p in tools.extract_pending_commands(text)] == ['web-search']


def test_extract_unclosed_fence_yields_no_commands():
    # The document's last fence is still open, so the trailing $ line is
    # (malformed) content, not a command: no commands, and the stage's
    # validation/retry takes over.
    text = '# example_test.py\n\n```py\ndef test():\n    pass\n$ web-search "a"\n'
    assert tools.extract_pending_commands(text) == []


def test_extract_malformed_command_lines():
    for bad in (DOC + '\n$\n',
                DOC + '\n$web-search "a"\n',
                DOC + '\n$ web-search "unterminated\n'):
        assert tools.extract_pending_commands(bad) == [], bad


def test_extract_url_argument_is_a_single_token():
    text = DOC + '\n$ view-web-page "https://docs.python.org/3/library/json.html?x=1&y=2"\n'
    pending = tools.extract_pending_commands(text)
    assert pending[0].args == ['https://docs.python.org/3/library/json.html?x=1&y=2']


def test_extract_bare_command_line_is_whole_document():
    text = '$ web-search "a"\n$ view-web-page "https://example.com"\n'
    assert [p.name for p in tools.extract_pending_commands(text)] == [
        'web-search', 'view-web-page']


# --- command execution ---------------------------------------------------------

def test_execute_unknown_command_lists_available():
    out = asyncio.run(tools.execute_command('frobnicate', ['x']))
    assert 'unknown command: frobnicate' in out
    assert '$ web-search' in out
    assert '$ view-web-page' in out


def test_execute_known_command_runs_handler():
    # The registry captured the handler at import time, so patch the registry
    # entry, not the module attribute.
    with patch.object(tools.COMMANDS['web-search'], 'handler',
                      new=AsyncMock(return_value='OK')) as h:
        out = asyncio.run(tools.execute_command('web-search', ['q']))
    assert out == 'OK'
    h.assert_awaited_once_with(['q'])


def test_execute_handler_exception_is_error_text():
    async def boom(args):
        raise Exception('kaput')
    with patch.object(tools, 'COMMANDS',
                      {'kapow': tools.ToolCommand('kapow', '$ kapow', 'd', boom)}):
        out = asyncio.run(tools.execute_command('kapow', []))
    assert out == 'error: command kapow failed: kaput'


# --- web-search ----------------------------------------------------------------

DDG_HTML = '''<html><body>
<div class="results">
  <div class="result">
    <h2 class="result__title"><a rel="nofollow" class="result__a"
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpandas.pydata.org%2Fdocs%2Freference%2Fapi%2Fpandas.read_csv.html&amp;rut=xyz">pandas.read_csv &mdash; pandas documentation</a></h2>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=whatever">Read a <b>CSV</b> file into a DataFrame. Accepts in-memory and file-based inputs.</a>
  </div>
  <div class="result">
    <h2 class="result__title"><a class="result__a" href="https://example.com/csv-guide">A guide to CSV</a></h2>
    <a class="result__snippet">A plain-text guide to <b>CSV</b> parsing.</a>
  </div>
</div>
</body></html>'''


def test_web_search_parses_ddg_html():
    async def fake_get(url, timeout=None):
        assert url.startswith('https://html.duckduckgo.com/html/?q=')
        return 200, 'text/html', DDG_HTML.encode()
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.web_search(['pandas', 'read_csv']))
    assert 'Search results for: pandas read_csv' in out
    # The /l/?uddg= redirect wrapper is unwrapped to the real destination.
    assert 'https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html' in out
    assert 'A guide to CSV' in out
    assert 'Read a CSV file into a DataFrame' in out
    assert 'view-web-page' in out


def test_web_search_falls_back_to_instant_api():
    body = json.dumps({
        'Heading': 'Pandas',
        'AbstractText': 'Pandas is a software library written for data manipulation.',
        'AbstractURL': 'https://en.wikipedia.org/wiki/Pandas_(library)',
        'RelatedTopics': [
            {'Text': 'The pandas DataFrame is a two-dimensional labeled data structure.',
             'FirstURL': 'https://pandas.pydata.org/docs/user_guide/dsintro.html',
             'Name': 'DataFrame'},
            {'Topics': [{'Text': 'NumPy provides the ndarray object.',
                         'FirstURL': 'https://numpy.org/doc/', 'Name': 'NumPy'}]},
        ],
    }).encode()

    async def fake_get(url, timeout=None):
        if 'html.duckduckgo.com' in url:
            raise Exception('blocked')
        assert url.startswith('https://api.duckduckgo.com/')
        return 200, 'application/json', body
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.web_search(['pandas']))
    assert 'Pandas is a software library' in out
    assert 'https://en.wikipedia.org/wiki/Pandas_(library)' in out
    assert 'https://numpy.org/doc/' in out


def test_web_search_no_results_is_error_text():
    async def fake_get(url, timeout=None):
        if 'html.duckduckgo.com' in url:
            return 200, 'text/html', b'<html><body><p>no results</p></body></html>'
        return 200, 'application/json', b'{}'
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.web_search(['zzz']))
    assert out.startswith('error: no results')


def test_web_search_requires_a_query():
    out = asyncio.run(tools.web_search([]))
    assert out.startswith('error:')
    assert 'web-search' in out


# --- view-web-page ---------------------------------------------------------------

PAGE_HTML = '''<!DOCTYPE html>
<html><head><title>Ignore me</title><style>body{color:red}</style>
<script>var x = "<p>not content</p>";</script></head>
<body>
<h1>Pandas read_csv</h1>
<p>Read a <b>CSV</b> file into a <a href="/x">DataFrame</a>. Supports &quot;many&quot; options.</p>
<ul><li>sep</li><li>header</li></ul>
</body></html>'''


def test_view_web_page_renders_html_to_text():
    async def fake_get(url, timeout=None):
        assert url == 'https://pandas.pydata.org/docs'
        return 200, 'text/html; charset=utf-8', PAGE_HTML.encode()
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.view_web_page(['https://pandas.pydata.org/docs']))
    assert out.startswith('Content of https://pandas.pydata.org/docs (HTTP 200):')
    assert 'Pandas read_csv' in out
    assert 'Read a CSV file into a DataFrame. Supports "many" options.' in out
    assert 'sep' in out and 'header' in out
    assert 'not content' not in out
    assert 'Ignore me' not in out
    assert '<' not in out


def test_view_web_page_passes_through_plain_text():
    async def fake_get(url, timeout=None):
        return 200, 'text/plain', b'plain text body'
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.view_web_page(['https://example.com/readme']))
    assert 'plain text body' in out


def test_view_web_page_truncates_long_pages():
    async def fake_get(url, timeout=None):
        return 200, 'text/plain', b'x' * (tools.PAGE_CHAR_LIMIT + 100)
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.view_web_page(['https://example.com/big.txt']))
    assert out.rstrip().endswith('[page truncated]')
    assert len(out) < tools.PAGE_CHAR_LIMIT + 200


def test_view_web_page_rejects_bad_urls():
    assert asyncio.run(tools.view_web_page([])).startswith('error:')
    assert asyncio.run(tools.view_web_page(['a', 'b'])).startswith('error:')
    assert asyncio.run(tools.view_web_page(['ftp://example.com/x'])).startswith('error:')
    assert asyncio.run(tools.view_web_page(['not a url'])).startswith('error:')


def test_view_web_page_fetch_failure_is_error_text():
    async def fake_get(url, timeout=None):
        raise Exception('Connection refused')
    with patch.object(tools, '_http_get', new=fake_get):
        out = asyncio.run(tools.view_web_page(['https://example.com/']))
    assert out.startswith('error: failed to fetch')


# --- the tool loop ----------------------------------------------------------------

class ScriptedMapper:
    n_results = 1

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def run(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def test_run_with_tools_executes_pending_command():
    doc_with_cmd = DOC + '\n$ web-search "pandas read_csv"\n'
    mapper = ScriptedMapper([doc_with_cmd, DOC])
    with patch.object(tools, 'execute_command', new=AsyncMock(return_value='SEARCH-RESULT')) as ex:
        out = asyncio.run(tools.run_with_tools(mapper, 'REQ'))
    assert out == DOC
    ex.assert_awaited_once_with('web-search', ['pandas read_csv'])
    assert mapper.calls[0] == [{'role': 'user', 'content': 'REQ'}]
    assert len(mapper.calls[1]) == 3
    assert mapper.calls[1][0] == {'role': 'user', 'content': 'REQ'}
    assert mapper.calls[1][1] == {'role': 'assistant', 'content': doc_with_cmd}
    assert mapper.calls[1][2]['role'] == 'user'
    assert 'SEARCH-RESULT' in mapper.calls[1][2]['content']
    assert '$ web-search "pandas read_csv"' in mapper.calls[1][2]['content']


def test_run_with_tools_single_call_without_commands():
    mapper = ScriptedMapper([DOC])
    with patch.object(tools, 'execute_command', new=AsyncMock()) as ex:
        out = asyncio.run(tools.run_with_tools(mapper, 'REQ'))
    assert out == DOC
    ex.assert_not_awaited()
    assert len(mapper.calls) == 1


def test_run_with_tools_budget_exhausted():
    cmd_doc = DOC + '\n$ web-search "x"\n'
    mapper = ScriptedMapper([cmd_doc] * 10)
    with patch.object(tools, 'execute_command', new=AsyncMock(return_value='r')):
        try:
            asyncio.run(tools.run_with_tools(mapper, 'REQ', max_rounds=2))
            assert False, 'expected ToolBudgetError'
        except tools.ToolBudgetError:
            pass
    assert len(mapper.calls) == 2


def test_run_with_tools_requires_single_result_mapper():
    class Multi(ScriptedMapper):
        n_results = 3
    try:
        asyncio.run(tools.run_with_tools(Multi([DOC]), 'REQ'))
        assert False, 'expected an exception'
    except Exception as e:
        assert 'n_results' in str(e)


# --- prompt wiring in the generation stages ----------------------------------------

class CapturingMapper:
    def __init__(self, system, **kwargs):
        self.system = system
        self.kwargs = kwargs
        self.requests = []
        self.responses = []

    async def run(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def test_oracle_stage_appends_tool_instructions_when_enabled():
    seen = {}

    def make(system, **kw):
        m = CapturingMapper(system, **kw)
        m.responses = [VALID_ORACLE]
        seen['mapper'] = m
        return m

    with patch.object(llm, 'get_mapper', new=make):
        doc = asyncio.run(llm.gpt_test_suite(make_meta(), True, debug=False))
    assert doc == VALID_ORACLE
    assert tools.tool_instructions() in seen['mapper'].system
    assert 'web-search' in seen['mapper'].system
    assert isinstance(seen['mapper'].requests[0], list)


def test_oracle_stage_omits_tool_instructions_when_disabled():
    seen = {}

    def make(system, **kw):
        m = CapturingMapper(system, **kw)
        m.responses = [VALID_ORACLE]
        seen['mapper'] = m
        return m

    with patch.object(llm, 'get_mapper', new=make):
        doc = asyncio.run(llm.gpt_test_suite(make_meta(), False, debug=False))
    assert doc == VALID_ORACLE
    assert 'web-search' not in seen['mapper'].system
    assert isinstance(seen['mapper'].requests[0], str)


def test_oracle_stage_follows_up_on_commands():
    seen = {}

    def make(system, **kw):
        m = CapturingMapper(system, **kw)
        m.responses = [VALID_ORACLE + '\n$ web-search "pandas read_csv"\n',
                       VALID_ORACLE]
        seen['mapper'] = m
        return m

    with patch.object(llm, 'get_mapper', new=make), \
            patch.object(tools, 'execute_command', new=AsyncMock(return_value='RESULT')):
        doc = asyncio.run(llm.gpt_test_suite(make_meta(), True, debug=False))
    assert doc == VALID_ORACLE
    assert len(seen['mapper'].requests) == 2


def test_impl_stage_tool_use_runs_one_conversation_per_candidate():
    seen = []

    def make(system, **kw):
        m = CapturingMapper(system, **kw)
        m.responses = [VALID_IMPL + '\n$ web-search "httpx post"\n', VALID_IMPL]
        seen.append(m)
        return m

    with patch.object(llm, 'get_mapper', new=make), \
            patch.object(tools, 'execute_command',
                         new=AsyncMock(return_value='RESULT')) as ex:
        mds = asyncio.run(llm.gpt_implementation(
            make_meta(), 'ORACLE', n_results=2, tool_use=True, debug=False))
    assert mds == [VALID_IMPL, VALID_IMPL]
    assert len(seen) == 2
    for m in seen:
        assert m.kwargs.get('n_results') == 1
        assert tools.tool_instructions() in m.system
        assert len(m.requests) == 2
    assert ex.await_count == 2


def test_impl_stage_without_tools_keeps_single_call_fanout():
    seen = []

    def make(system, **kw):
        m = CapturingMapper(system, **kw)
        m.responses = [VALID_IMPL]
        seen.append(m)
        return m

    with patch.object(llm, 'get_mapper', new=make):
        mds = asyncio.run(llm.gpt_implementation(
            make_meta(), 'ORACLE', n_results=3, tool_use=False, debug=False))
    assert mds == [VALID_IMPL]
    assert len(seen) == 1
    assert seen[0].kwargs.get('n_results') == 3
    assert 'web-search' not in seen[0].system
    assert isinstance(seen[0].requests[0], str)


# --- mapper conversation support ------------------------------------------------------

def test_chatgpt_mapper_accepts_conversation():
    import marsha.mappers.chatgpt as chatgpt
    seen = {}

    async def fake_retry(query, model=None, max_tries=3, n_results=1, label=None):
        seen['query'] = query
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='out'))])

    with patch.object(chatgpt, 'retry_chat_completion', new=fake_retry):
        out = asyncio.run(chatgpt.ChatGPTMapper('SYS').transform(
            [{'role': 'user', 'content': 'a'},
             {'role': 'assistant', 'content': 'b'},
             {'role': 'user', 'content': 'c'}]))
    assert out == 'out'
    msgs = seen['query']['messages']
    assert msgs[0] == {'role': 'system', 'content': 'SYS'}
    assert msgs[1:] == [{'role': 'user', 'content': 'a'},
                        {'role': 'assistant', 'content': 'b'},
                        {'role': 'user', 'content': 'c'}]


def test_chatgpt_mapper_string_request_unchanged():
    import marsha.mappers.chatgpt as chatgpt
    seen = {}

    async def fake_retry(query, model=None, max_tries=3, n_results=1, label=None):
        seen['query'] = query
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='out'))])

    with patch.object(chatgpt, 'retry_chat_completion', new=fake_retry):
        asyncio.run(chatgpt.ChatGPTMapper('SYS').transform('hello'))
    assert seen['query']['messages'] == [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'hello'}]


def test_claude_mapper_accepts_conversation():
    import marsha.mappers.claude as claude
    seen = {}

    async def fake_retry(query, model=None, max_tries=3, label=None):
        seen['query'] = query
        return SimpleNamespace(
            content=[SimpleNamespace(type='text', text='out')],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            model='m')

    with patch.object(claude, 'retry_message_create', new=fake_retry):
        out = asyncio.run(claude.ClaudeMapper('SYS').transform(
            [{'role': 'user', 'content': 'a'},
             {'role': 'assistant', 'content': 'b'},
             {'role': 'user', 'content': 'c'}]))
    assert out == 'out'
    assert seen['query']['system'] == 'SYS'
    assert seen['query']['messages'] == [{'role': 'user', 'content': 'a'},
                                         {'role': 'assistant', 'content': 'b'},
                                         {'role': 'user', 'content': 'c'}]
