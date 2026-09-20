import asyncio
import dataclasses
import os
import re

from marsha import tools
from marsha.config import is_local_backend
from marsha.log import log
from marsha.mappers import get_mapper

# Per-loop reviewer filename prefix and the (fixed-per-loop) editor/implementor file.
# For review, the "editor" is the conventions gate (Norman): it rebuts convention-violating
# findings between reviewer rounds rather than editing code.
LOOP_PREFIX = {
    'oracle': 'oracle-',
    'impl': 'impl-',
    'correction': 'correction-',
    'review': 'review-',
}
EDITOR_FILE = {
    'oracle': '_oracle-editor.md',
    'impl': '_impl-editor.md',
    'correction': '_correction-editor.md',
    'review': '_review-conventions.md',
}

# Appended to every reviewer's system prompt (formatted with the reviewer's number N). It fixes
# the output contract so a user-authored persona only has to describe *what* to check.
FINDINGS_CONTRACT = '''

You are review #{review_number}. Label each finding with a letter (A, then B, then C, ... in the order you report it) immediately followed by your review number, {review_number}. So your first finding is labeled A{review_number}, your second B{review_number}, and so on.
When re-reviewing after a previous round of posted comments (shown in the context), a comment whose label ends in {review_number} is one of your own prior findings. Re-raise it (reusing its exact label) only if you still believe it is a real issue and want to push back; otherwise close it — do not re-raise it — when you verify with your tools that the code no longer has the issue, or when the user rejected it and you agree and will not push back. Give any genuinely new finding the next unused letter followed by {review_number}, skipping the labels of your prior findings so a new finding never reuses a prior label (which would collide with an old thread).
Report each finding as a one-line headline, followed by 1-2 short paragraphs of supporting information, in exactly this form:
<LETTER>{review_number} [MAJOR|MINOR|NIT] <location> - <one-line description>
<one to two short paragraphs of supporting information: the concrete evidence you verified with the git tool (cite the specific file and lines you read), why it is a problem, and the concrete impact or risk>
For example:
A{review_number} [MAJOR] some_file.py:10 - the spec requires X but it is not tested
The spec ("Behavior", lines 20-24) requires the result to be sorted, but line 10 returns the list unsorted. I confirmed with `git show HEAD:some_file.py` that no sort is applied before the return, and the oracle (tests/test_x.py:40) asserts sorted order, so an unsorted result fails the suite.
Separate findings with a blank line.
Severity meanings: MAJOR = violates the spec or oracle contract, or would let a wrong artifact pass; MINOR = degrades quality but not correctness; NIT = a cheap style fix.
Report ONLY findings about the CURRENT code that you have verified with the git tool. Do not report a concern an earlier review already raised and that has since been fixed, or that the user rejected: those are settled, and re-raising them is noise, not a finding.
A finding must be a concrete, actionable problem in the current code. Do NOT comment on or evaluate a prior fix, ask the user to confirm or verify anything, or report a mere observation that is not a problem you can point at in the code. None of those are findings.
Before reporting a robustness or error-handling concern (for example "this swallows an error", "this can hide a failure", or "this leaves resources open"), check with the git tool how the code is actually used; if every caller already handles the case the concern is about, it is not a finding.
Do NOT report a performance or micro-optimization suggestion (precomputing, hoisting, caching, batching, parallelizing, or a complexity claim) unless you can show it is in a hot loop or works on data large enough to measurably affect overall performance; a one-off call, a pairwise pass over a single-digit-sized collection, or any complexity or allocation change with no evidence of real impact is not a finding.
Do NOT base a finding on a project convention, style rule, or requirement unless you can point to it in a config file, the PR or issue, or the surrounding code; a rule you cannot find written in the repository is not a finding.
If you have no new, verified finding, respond with exactly: NO FINDINGS
Do not restate your own name. Do not add any prose outside a finding (the supporting paragraphs are part of that finding).
'''

_DIR = os.path.dirname(os.path.abspath(__file__))
_registry_cache = None


def personas_dir():
    return _DIR


def _is_path(entry):
    # A persona entry is a file path (not a built-in name) when it is absolute, home-relative
    # (~), dot-relative (./), or contains a path separator in POSIX or Windows style, so the
    # check is portable to Windows (backslashes and drive-letter paths) as well as POSIX.
    return (os.path.isabs(entry) or entry.startswith('~')
            or entry.startswith('.') or '/' in entry or '\\' in entry)


def load_persona(path):
    # A persona file's first line must be `name: <name>`; the rest is the system-prompt body.
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    if not lines or not re.match(r'^name:\s*\S+', lines[0]):
        raise Exception(
            f'Persona file must start with a "name: <name>" line: {path}')
    name = lines[0].split(':', 1)[1].strip()
    body = '\n'.join(lines[1:]).strip()
    if not body:
        raise Exception(f'Persona file has no body: {path}')
    return (name, body)


def build_registry():
    # Scan the shipped personas dir for reviewer files (skip `_`-prefixed editors and README).
    global _registry_cache
    if _registry_cache is None:
        registry = {}
        for filename in sorted(os.listdir(_DIR)):
            if not filename.endswith('.md'):
                continue
            if filename.startswith('_') or filename == 'README.md':
                continue
            path = os.path.join(_DIR, filename)
            name, _ = load_persona(path)
            key = name.lower()
            if key in registry:
                raise Exception(
                    f'Duplicate persona name "{name}" in the registry')
            registry[key] = path
        _registry_cache = registry
    return _registry_cache


def reset_registry():
    # Primarily for tests that swap out the personas directory.
    global _registry_cache
    _registry_cache = None


def resolve_persona(entry, registry):
    # A file path loads that persona file directly; otherwise the entry is a built-in name.
    entry = entry.strip()
    if not entry:
        raise Exception('Empty persona entry')
    if _is_path(entry):
        path = os.path.abspath(os.path.expanduser(entry))
        if not os.path.exists(path):
            raise Exception(f'Persona file not found: {path}')
        name, body = load_persona(path)
        return (name, body, path)
    key = entry.lower()
    if key not in registry:
        available = ', '.join(sorted(registry))
        raise Exception(f'Unknown persona "{entry}". Available: {available}')
    path = registry[key]
    name, body = load_persona(path)
    return (name, body, path)


def resolve_loop_reviewers(loop, flag_value, registry=None):
    # Return the ordered list of (name, body, N) reviewers for a loop. N is the 1-based position.
    if registry is None:
        registry = build_registry()
    if flag_value:
        specs = []
        seen = set()
        for entry in flag_value.split(','):
            if not entry.strip():
                continue
            name, body, _ = resolve_persona(entry, registry)
            if name.lower() in seen:
                raise Exception(f'Duplicate persona in loop {loop}: {name}')
            seen.add(name.lower())
            specs.append((name, body))
    else:
        prefix = LOOP_PREFIX[loop]
        specs = []
        for key in sorted(registry):
            path = registry[key]
            if not os.path.basename(path).startswith(prefix):
                continue
            name, body = load_persona(path)
            specs.append((name, body))
    return [(name, body, i + 1) for i, (name, body) in enumerate(specs)]


def load_editor(loop):
    # The loop's fixed editor/implementor prompt (not part of the reviewer registry).
    path = os.path.join(_DIR, EDITOR_FILE[loop])
    return load_persona(path)


def parse_severities(value):
    # Parse a severity list (e.g. --severity / --optimize-severity) into the set of uppercase
    # tiers to act on.
    allowed = {'major', 'minor', 'nit', 'nitpick'}
    out = set()
    for part in (value or '').split(','):
        p = part.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise Exception(
                f'Invalid severity tier: {p} (choose from major, minor, nit)')
        out.add('nit' if p == 'nitpick' else p)
    return {s.upper() for s in out}


def _split_location(rest):
    # Split "<location> - <description>" on the first separator; the location is optional.
    for sep in (' - ', ' -- '):
        if sep in rest:
            location, description = rest.split(sep, 1)
            return location.strip(), description.strip()
    if ':' in rest:
        location, description = rest.split(':', 1)
        return location.strip(), description.strip()
    return '', rest


def position_label(review_number, used):
    # The first position-based label (A<n>, B<n>, ...) not already used, so a fallback label can
    # never collide with a reused one. Beyond the 26 single letters (a degenerate 27th+ finding
    # from one reviewer) it extends to two letters (AA<n>, AB<n>, ...) so the label always stays
    # alphabetic and unique instead of spilling onto non-letter characters.
    for n in range(26):
        cand = f'{chr(ord("A") + n)}{review_number}'
        if cand not in used:
            return cand
    for hi in range(26):
        for lo in range(26):
            cand = f'{chr(ord("A") + hi)}{chr(ord("A") + lo)}{review_number}'
            if cand not in used:
                return cand
    raise ValueError(f'no available label for reviewer {review_number}')


def parse_findings(text, name, review_number, prior_labels=None):
    # Parse one reviewer's findings. The canonical label is <letter><review_number>. On a
    # re-review a reviewer reuses the exact label of a prior finding it still stands by, so a
    # well-formed written label (one or more letters followed by THIS reviewer's number) is
    # honored; anything else is labeled by position, skipping the reviewer's prior labels, so a
    # malformed or foreign label cannot break it and a new finding never collides with a prior
    # thread's label. One-or-more letters keeps a reused two-letter label (AA<n>, ...) from being
    # misread as a foreign label and renumbered.
    prior = {lbl.upper() for lbl in (prior_labels or [])}
    findings = []
    used = set()
    own_label = re.compile(rf'[A-Z]+{review_number}')
    headline = re.compile(
        r'^\s*([A-Za-z]+\d+)?\s*\[(MAJOR|MINOR|NIT|NITPICK)\]\s*(.*)$', re.IGNORECASE)
    current = None
    support = []

    def flush():
        if current is not None:
            # Supporting paragraphs follow the headline; drop any leading indent per line but
            # keep internal blank lines (which separate paragraphs), then trim the ends.
            support_text = '\n'.join(ln.lstrip() for ln in support).strip()
            current['support'] = support_text
            findings.append(current)

    for line in (text or '').split('\n'):
        m = headline.match(line)
        if m:
            flush()
            severity = m.group(2).upper()
            if severity == 'NITPICK':
                severity = 'NIT'
            location, description = _split_location(m.group(3).strip())
            written = (m.group(1) or '').upper()
            if own_label.fullmatch(written) and written not in used:
                label = written
            else:
                label = position_label(review_number, used | prior)
            used.add(label)
            current = {
                'name': name,
                'label': label,
                'severity': severity,
                'location': location,
                'desc': description,
                'support': '',
            }
            support = []
        elif line.strip() or current is not None:
            # A non-headline line is the current finding's supporting text; blank lines are kept
            # (they separate paragraphs) but a line before any finding is ignored.
            if current is not None:
                support.append(line)
    flush()
    return findings


# The leading "- " is optional: the model sometimes omits the list bullet, and treating a
# well-formed finding without it as absent would silently drop it (see parse_compacted_findings).
_COMPACTED_LINE = re.compile(
    r'^\s*(?:-\s*)?\[([^\]]+)\]\s*(MAJOR|MINOR|NIT|NITPICK)\b\s*(.*)$', re.IGNORECASE)


def parse_compacted_findings(text):
    # Parse the output of the findings-compaction job. Each surviving line keeps its original
    # [Name-Label] (never renumbered), so references from other reviewers stay valid; the
    # sequence simply has gaps where findings were dropped.
    findings = []
    for line in (text or '').split('\n'):
        m = _COMPACTED_LINE.match(line)
        if not m:
            continue
        ref = m.group(1).strip()
        severity = m.group(2).upper()
        if severity == 'NITPICK':
            severity = 'NIT'
        location, description = _split_location(m.group(3).strip())
        name, label = ref.split('-', 1) if '-' in ref else (ref, '')
        findings.append({
            'name': name,
            'label': label,
            'severity': severity,
            'location': location,
            'desc': description,
            'support': '',
        })
    return findings


def dedup_findings(findings):
    seen = set()
    out = []
    for f in findings:
        key = (f['name'], f['label'], f['severity'], f['desc'].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def actionable_findings(findings, severities):
    # Keep only the enabled severity tiers, then de-duplicate across reviewers.
    return dedup_findings([f for f in findings if f['severity'] in severities])


def _location_key(location):
    # (path, line) for near-dedup: the integer after the final ':' if present, else None.
    # Two findings at the same path:line are almost always the same concern, so they can be
    # merged deterministically without a model.
    location = (location or '').strip()
    if not location:
        return ('', None)
    m = re.match(r'^(.*?):(\d+)\s*$', location)
    if m:
        return (m.group(1).strip(), int(m.group(2)))
    return (location, None)


def dedup_by_location(findings):
    # Merge findings that point at the same file:line — a common failure mode where several
    # reviewers flag the same spot with slightly different wording. For each location keep the
    # highest-severity finding, breaking ties on the more detailed (longer) description.
    # Findings with no location are kept as-is (there is no location to merge them on).
    order = {'NIT': 0, 'MINOR': 1, 'MAJOR': 2}
    best = {}
    unlocated = []
    for f in findings:
        if not (f['location'] or '').strip():
            unlocated.append(f)
            continue
        key = _location_key(f['location'])
        rank = (order.get(f['severity'], 1), len(f['desc'].strip()))
        cur = best.get(key)
        if cur is None or rank > (order.get(cur['severity'], 1),
                                  len(cur['desc'].strip())):
            best[key] = f
    return list(best.values()) + unlocated


def format_findings(findings):
    lines = []
    for f in findings:
        location = f' {f["location"]}' if f['location'] else ''
        lines.append(
            f'- [{f["name"]}-{f["label"]}] {f["severity"]}{location} - {f["desc"]}')
    return '\n'.join(lines)


def prior_round_block(findings, preamble, label='implementor'):
    # The prior-cycle context shown to reviewers in round >= 2, so each can recognize its own
    # [Name-Label] in the push-back and see which point was rejected. `label` names the role that
    # produced the push-back ("implementor" in the optimize loops, "conventions review" in review).
    block = '\n# Previous review round\n'
    if findings:
        block += '\nFindings raised last round:\n'
        block += format_findings(findings) + '\n'
    if preamble:
        block += f'\nWhat the {label} said last round:\n'
        block += preamble.strip() + '\n'
    return block


async def run_personas(reviewers, user_message, model, stats_stage, debug=False, loop=None, guidance='', tool_ctx=None, max_tool_rounds=None, prior_block_by_number=None, prior_labels_by_number=None, reasoning_effort=None, seed=None):
    # Run every reviewer independently; return the flattened labeled findings. On a local
    # (serial) backend the reviewers run one at a time so each gets the whole server; otherwise
    # they run concurrently. `guidance` is the target-language backend's persona_guidance(): the
    # per-language conventions the (language-agnostic) reviewer bodies leave out. `tool_ctx`, when
    # set, enables the fake terminal for this reviewer (see marsha.tools); each reviewer gets a
    # fresh copy of its `notes` list so their scratchpads do not leak across reviewers.
    # `max_tool_rounds` bounds the tool loop (defaults to tools.MAX_TOOL_ROUNDS).
    # `prior_block_by_number`, when set, maps a reviewer number to a block of that reviewer's OWN
    # prior (labeled) findings + the user's replies, appended to that reviewer's message so it
    # reuses a label only for a concern it still stands by. `prior_labels_by_number` maps a
    # reviewer number to the set of its prior labels so parse_findings can keep a new (position-
    # based) finding from colliding with a prior thread's label. `reasoning_effort` and `seed`,
    # when set, are forwarded to each reviewer's mapper (the review path passes them to make a
    # single pass more reliable and sampling as reproducible as the provider allows).
    async def one(spec):
        name, body, review_number = spec
        system = body
        if guidance:
            system += f'\n\n{guidance}'
        system += FINDINGS_CONTRACT.format(review_number=review_number)
        # A fresh notes list per reviewer (the `notes` tool mutates it); the command set is the
        # same for all reviewers, so the instructions can be built from either copy.
        rctx = dataclasses.replace(tool_ctx, notes=list(
            tool_ctx.notes)) if tool_ctx is not None else None
        if rctx is not None:
            system += tools.tool_instructions(rctx)
        label = f'{loop}:{name}' if loop else name
        user = user_message
        if prior_block_by_number and review_number in prior_block_by_number:
            user += prior_block_by_number[review_number]
        prior_labels = (prior_labels_by_number or {}).get(review_number)
        try:
            mapper = get_mapper(
                system, n_results=1, stats_stage=stats_stage, model=model, label=label,
                reasoning_effort=reasoning_effort, seed=seed)
            if rctx is not None:
                text = await tools.run_with_tools(
                    mapper, user, rctx,
                    debug=debug, max_rounds=max_tool_rounds or tools.MAX_TOOL_ROUNDS)
            else:
                text = await mapper.run(user)
        except Exception as e:
            if debug:
                print(f'[Personas] {name} failed: {e}')
            log(f'personas: {label} failed: {e}')
            return []
        return parse_findings(text, name, review_number, prior_labels=prior_labels)
    if is_local_backend():
        results = [await one(s) for s in reviewers]
    else:
        # A generator (not a list) avoids the intermediate list of coroutines gather would build.
        results = await asyncio.gather(*(one(s) for s in reviewers))
    return [f for sub in results for f in sub]
