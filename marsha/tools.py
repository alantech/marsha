"""A simple tool-use system for the code and test generation stages (issue #197).

Those stages can currently only get feedback by generating code/tests and
then running the test suite to see the results, which is not enough for
non-trivial programs with external dependencies: the LLM will not have full
API specifications memorized, and even if it does, they can be out of date
versus the actual current state of the dependency. This module gives the LLM
a simple fake terminal: it may end a response with one or more lines, each
beginning with `$`, to invoke a command from a small, easy-to-extend set
(today: `web-search` and `view-web-page`). The harness detects the trailing
command lines, executes them, and feeds the results back into a follow-up
LLM call — repeating until the model produces final output with no pending
command.

The MCP standard is deliberately out of scope: it is more heavyweight and
extensible than this needs. Adding a command is a single entry in COMMANDS.
"""

import asyncio
import dataclasses
import html
import json
import re
import shlex
import urllib.parse
import urllib.request

from marsha.log import log

# Safety cap on how many LLM rounds one generation may spend issuing commands
# before the stage falls back to its normal retry logic.
MAX_TOOL_ROUNDS = 5

# Bounds on the web output fed back into the conversation, so a single tool
# result cannot blow the context budget.
HTTP_TIMEOUT = 30
MAX_HTTP_BYTES = 1000_000
SEARCH_RESULT_COUNT = 10
SNIPPET_CHAR_LIMIT = 300
PAGE_CHAR_LIMIT = 12_000

USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


class ToolBudgetError(Exception):
    """The model was still issuing commands after the round budget ran out.
    Its last response is not a usable final document, so the stage should
    retry generation from scratch."""


@dataclasses.dataclass
class ToolCommand:
    """One command of the fake terminal: its name, how it is spelled in a
    `$` line, a one-line description, and the async handler that executes it
    (args -> output text). Handlers return errors as `error: ...` text so
    the model can see what went wrong and adapt."""
    name: str
    usage: str
    description: str
    handler: 'callable'


@dataclasses.dataclass
class PendingCommand:
    """A `$` command line detected at the end of an LLM response."""
    line: str
    name: str
    args: list


# --- web commands -------------------------------------------------------------


async def _http_get(url, timeout=HTTP_TIMEOUT):
    """GET a URL off the event loop and return (status, content_type, body).
    The body is capped at MAX_HTTP_BYTES so a runaway page cannot exhaust
    memory before the text limits are applied."""
    def get():
        req = urllib.request.Request(
            url, headers={
                'User-Agent': USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.8',
            })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get('Content-Type', ''), resp.read(MAX_HTTP_BYTES)
    return await asyncio.to_thread(get)


def _strip_tags(fragment):
    # Drop tags; insert a space only where two word characters would otherwise
    # run together, so `</a>.` stays `.` and `<b>CSV</b> file` keeps one space.
    def repl(m):
        before = fragment[:m.start()]
        after = fragment[m.end():]
        if before and after and before[-1].isalnum() and after[0].isalnum():
            return ' '
        return ''
    return re.sub(r'(?s)<[^>]+>', repl, fragment)


def html_to_text(doc):
    """Reduce an HTML document to readable plain text: scripts, styles, and
    other non-content blocks are dropped, block boundaries become newlines,
    and entities are decoded."""
    t = re.sub(r'(?is)<(script|style|noscript|svg|head|iframe|template)\b.*?</\1>', ' ', doc)
    t = re.sub(r'(?s)<!--.*?-->', ' ', t)
    t = re.sub(r'(?i)</(p|div|li|ul|ol|tr|td|th|h[1-6]|pre|blockquote|section|article|'
               r'header|footer|table|figure|figcaption|dl|dt|dd)>', '\n', t)
    t = re.sub(r'(?i)<br\s*/?>', '\n', t)
    t = _strip_tags(t)
    t = html.unescape(t)
    lines = []
    blank = False
    for line in t.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)
            blank = False
        elif not blank:
            lines.append('')
            blank = True
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def _decode_ddg_href(href):
    """Resolve a result link from DuckDuckGo's HTML endpoint: protocol-relative
    links are absolutized and the `/l/?uddg=<url>` redirect wrapper is
    unwrapped to the real destination."""
    href = html.unescape(href).strip()
    if href.startswith('//'):
        href = 'https:' + href
    if 'duckduckgo.com/l/' in href:
        uddg = urllib.parse.parse_qs(
            urllib.parse.urlparse(href).query).get('uddg')
        if uddg:
            return uddg[0]
    return href


def _parse_ddg_html(doc):
    """Parse the results of DuckDuckGo's HTML search endpoint into
    (title, url, snippet) triples. Returns [] on any markup mismatch so the
    caller can fall back to the instant-answer API."""
    snippets = [re.sub(r'\s+', ' ', html.unescape(_strip_tags(m))).strip() for m in re.findall(
        r'(?s)<(?:a|div|span)\b[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|div|span)>', doc)]
    results = []
    for i, m in enumerate(re.finditer(r'(?s)<a\b[^>]*class="result__a"[^>]*>(.*?)</a>', doc)):
        href = re.search(r'href="([^"]*)"', m.group(0))
        if href is None:
            continue
        url = _decode_ddg_href(href.group(1))
        title = re.sub(r'\s+', ' ', html.unescape(_strip_tags(m.group(1)))).strip()
        if not url or not title or not url.startswith('http'):
            continue
        snippet = snippets[i] if i < len(snippets) else ''
        results.append((title, url, snippet))
    return results


async def _ddg_instant(query):
    """Fallback search via the DuckDuckGo instant-answer JSON API. Coverage is
    narrower than the HTML endpoint (entity-centric) but the endpoint is
    stable; returns (title, url, snippet) triples."""
    url = ('https://api.duckduckgo.com/?q=' + urllib.parse.quote_plus(query)
           + '&format=json&no_html=1&skip_disambig=1')
    try:
        _, _, body = await _http_get(url)
        data = json.loads(body.decode('utf-8', 'replace'))
    except Exception:
        return []
    results = []
    abstract = (data.get('AbstractText') or '').strip()
    abstract_url = (data.get('AbstractURL') or '').strip()
    if abstract and abstract_url:
        results.append((data.get('Heading') or query, abstract_url, abstract))

    def walk(topics):
        for topic in topics:
            if isinstance(topic, dict):
                # A group entry nests its members under a 'Topics' key.
                sub = topic.get('Topics')
                if isinstance(sub, list):
                    walk(sub)
                    continue
                text = (topic.get('Text') or '').strip()
                first = (topic.get('FirstURL') or '').strip()
                name = (topic.get('Name') or '').strip()
                if text and first:
                    results.append((name or first, first, text))
            elif isinstance(topic, list):
                walk(topic)

    walk(data.get('RelatedTopics') or [])
    return results


async def web_search(args):
    """`web-search "search terms"` — search the web and return the top
    results as numbered title/URL/snippet lines."""
    query = ' '.join(args).strip()
    if not query:
        return 'error: web-search needs a query, e.g. $ web-search "pandas read_csv parameters"'
    results = []
    try:
        _, _, body = await _http_get(
            'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote_plus(query))
        results = _parse_ddg_html(body.decode('utf-8', 'replace'))
    except Exception:
        results = []
    if not results:
        results = await _ddg_instant(query)
    if not results:
        return f'error: no results for: {query}'
    lines = [f'Search results for: {query}']
    for i, (title, url, snippet) in enumerate(results[:SEARCH_RESULT_COUNT], 1):
        lines.append(f'{i}. {title}')
        lines.append(url)
        if snippet:
            lines.append(f'   {snippet[:SNIPPET_CHAR_LIMIT]}')
    lines.append('Use view-web-page on a result URL to read the page itself.')
    return '\n'.join(lines)


async def view_web_page(args):
    """`view-web-page "https://url"` — fetch a web page and return its text
    content (HTML reduced to plain text), truncated to PAGE_CHAR_LIMIT."""
    if len(args) != 1:
        return 'error: view-web-page takes one argument, the URL, e.g. $ view-web-page "https://docs.python.org/3/"'
    url = args[0].strip()
    if not re.match(r'^https?://\S+$', url):
        return f'error: not a valid http(s) URL: {url}'
    try:
        status, ctype, body = await _http_get(url)
    except Exception as e:
        return f'error: failed to fetch {url}: {e}'
    doc = body.decode('utf-8', 'replace')
    head = doc.lstrip()[:512].lower()
    if 'html' in ctype.lower() or head.startswith('<!doctype html') or '<html' in head:
        text = html_to_text(doc)
    else:
        text = re.sub(r'[ \t]+', ' ', doc).strip()
    if not text.strip():
        return f'error: {url} returned no readable text (HTTP {status})'
    if len(text) > PAGE_CHAR_LIMIT:
        text = text[:PAGE_CHAR_LIMIT] + '\n[page truncated]'
    return f'Content of {url} (HTTP {status}):\n\n{text}'


# --- the fake terminal ---------------------------------------------------------


COMMANDS = {
    'web-search': ToolCommand(
        'web-search', '$ web-search "search terms"',
        'search the web; returns the top results as numbered title, URL, and snippet lines',
        web_search),
    'view-web-page': ToolCommand(
        'view-web-page', '$ view-web-page "https://url"',
        'fetch a web page and return its text content (truncated)',
        view_web_page),
}


def tool_instructions():
    """The fake-terminal protocol, appended to a system prompt when tool use
    is enabled for a stage."""
    lines = [
        'You may need information that is not in the assignment — for example the exact API of a third-party library the code must use, which you may not have memorized or which may have changed. You have a simple fake terminal for looking it up: to issue a command, end your response with one or more lines, each beginning with `$` followed by the command name and its arguments.',
        'Available commands:',
    ]
    for cmd in COMMANDS.values():
        lines.append(f'- {cmd.usage} — {cmd.description}')
    lines.extend([
        'Each command you issue is executed, and its output is returned to you in a follow-up message, where you may issue further commands or continue your work.',
        'Issue commands only when you genuinely need information you do not already have; start with web-search and use view-web-page on the most promising result.',
        'Once you have everything you need, produce your final response exactly as specified above, with no trailing command lines.',
    ])
    return '\n'.join(lines) + '\n'


_COMMAND_RE = re.compile(r'^\$\s+([A-Za-z0-9][A-Za-z0-9_-]*)(?:\s+(.*))?$')


def extract_pending_commands(text):
    """The trailing block of `$` command lines at the very end of an LLM
    response, as PendingCommand items; [] when the response does not end with
    a command call. Only a run of lines at the very end (after the last
    non-blank line) counts, lines at or under an unclosed code fence never
    count, and a trailing `$` line that does not parse as a command yields
    [] so the malformed response falls through to the stage's normal
    validation and retry instead of a confusing tool error."""
    if not isinstance(text, str):
        return []
    lines = text.splitlines()
    last = len(lines) - 1
    while last >= 0 and not lines[last].strip():
        last -= 1
    first = last
    while first >= 0 and lines[first].lstrip().startswith('$'):
        first -= 1
    if first == last:
        return []
    # A command block must sit outside code fences: an odd number of fence
    # lines above it means the document's last fence is still open.
    fences = sum(1 for line in lines[:first + 1] if line.lstrip().startswith('```'))
    if fences % 2 == 1:
        return []
    pending = []
    for line in lines[first + 1:last + 1]:
        m = _COMMAND_RE.match(line.strip())
        if m is None:
            return []
        name, rest = m.group(1), m.group(2) or ''
        try:
            args = shlex.split(rest)
        except ValueError:
            return []
        pending.append(PendingCommand(line, name, args))
    return pending


async def execute_command(name, args):
    """Run one fake-terminal command and return its output text. Errors are
    returned as `error: ...` text so the model can see what went wrong and
    adapt, instead of the loop raising."""
    cmd = COMMANDS.get(name)
    if cmd is None:
        available = '; '.join(c.usage for c in COMMANDS.values())
        return f'error: unknown command: {name}. Available commands: {available}'
    try:
        return await cmd.handler(args)
    except Exception as e:
        return f'error: command {name} failed: {e}'


def format_tool_output(pending, results):
    """The follow-up user message: each issued command echoed with its output."""
    parts = ['Command output:']
    for cmd, result in zip(pending, results):
        parts.append(f'\n{cmd.line}\n{result}\n')
    parts.append(
        '\nIf you need more information, end your next response with further '
        'command lines. Otherwise produce your final response now, in the '
        'exact format required, with no trailing command lines.')
    return '\n'.join(parts)


async def run_with_tools(mapper, request, debug=False, max_rounds=MAX_TOOL_ROUNDS):
    """Drive one LLM exchange with the fake terminal: call the mapper, and if
    the response ends with `$` command lines, execute them and feed the
    output back in a follow-up call (the conversation is extended with the
    model's own response and the command output), repeating until a response
    arrives with no pending command. The mapper must be single-result
    (n_results=1). Returns the final response text; raises ToolBudgetError if
    the model is still issuing commands after max_rounds rounds."""
    if getattr(mapper, 'n_results', 1) != 1:
        raise Exception('run_with_tools requires a single-result mapper (n_results=1)')
    messages = [{'role': 'user', 'content': request}]
    for round_ in range(max_rounds):
        text = await mapper.run(messages)
        pending = extract_pending_commands(text)
        if not pending:
            return text
        if debug:
            print(f'[tools] round {round_ + 1}/{max_rounds}: executing '
                  f'{len(pending)} command(s): {", ".join(c.name for c in pending)}')
        log(f'tools round {round_ + 1}/{max_rounds}: {len(pending)} command(s): '
            + ', '.join(c.name for c in pending))
        results = await asyncio.gather(
            *[execute_command(c.name, c.args) for c in pending])
        messages = messages + [
            {'role': 'assistant', 'content': text},
            {'role': 'user', 'content': format_tool_output(pending, list(results))},
        ]
    raise ToolBudgetError(f'still issuing commands after {max_rounds} rounds')
