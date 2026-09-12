import asyncio
import os
import re

from marsha.config import is_local_backend
from marsha.log import log
from marsha.mappers import get_mapper

# Per-loop reviewer filename prefix and the (fixed-per-loop) editor/implementor file.
LOOP_PREFIX = {
    'oracle': 'oracle-',
    'impl': 'impl-',
    'correction': 'correction-',
}
EDITOR_FILE = {
    'oracle': '_oracle-editor.md',
    'impl': '_impl-editor.md',
    'correction': '_correction-editor.md',
}

# Appended to every reviewer's system prompt (formatted with the reviewer's number N). It fixes
# the output contract so a user-authored persona only has to describe *what* to check.
FINDINGS_CONTRACT = '''

You are review #{review_number}. Label each finding with a letter (A, then B, then C, ... in the order you report it) immediately followed by your review number, {review_number}. So your first finding is labeled A{review_number}, your second B{review_number}, and so on.
Report each finding on its own line, in exactly this form:
<LETTER>{review_number} [MAJOR|MINOR|NIT] <location> - <one-line description>
For example: A{review_number} [MAJOR] some_file.py:10 - the spec requires X but it is not tested
Severity meanings: MAJOR = violates the spec or oracle contract, or would let a wrong artifact pass; MINOR = degrades quality but not correctness; NIT = a cheap style fix.
If the artifact is fully clean from your perspective, respond with exactly: NO FINDINGS
Do not restate your own name. Do not add any prose outside the findings.
'''

_DIR = os.path.dirname(os.path.abspath(__file__))
_registry_cache = None


def personas_dir():
    return _DIR


def _is_path(entry):
    return entry.startswith('/') or entry.startswith('.') or entry.startswith('~')


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
    # A path (leading /, ., or ~) loads that file directly; otherwise it is a built-in name.
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
    # Parse the --optimize-severity list into the set of uppercase tiers to act on.
    allowed = {'major', 'minor', 'nit', 'nitpick'}
    out = set()
    for part in (value or '').split(','):
        p = part.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise Exception(
                f'Invalid --optimize-severity tier: {p} (choose from major, minor, nit)')
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


def parse_findings(text, name, review_number):
    # Parse one reviewer's findings. The canonical label is <letter><review_number>, where the
    # letter is the finding's 1-based position, so a malformed reviewer label cannot break it.
    findings = []
    for line in text.split('\n'):
        m = re.match(
            r'^\s*(?:[A-Za-z]+\d+\s+)?\[(MAJOR|MINOR|NIT|NITPICK)\]\s*(.*)$', line, re.IGNORECASE)
        if not m:
            continue
        severity = m.group(1).upper()
        if severity == 'NITPICK':
            severity = 'NIT'
        location, description = _split_location(m.group(2).strip())
        findings.append({
            'name': name,
            'label': f'{chr(ord("A") + len(findings))}{review_number}',
            'severity': severity,
            'location': location,
            'desc': description,
        })
    return findings


_COMPACTED_LINE = re.compile(
    r'^\s*-\s*\[([^\]]+)\]\s*(MAJOR|MINOR|NIT|NITPICK)\b\s*(.*)$', re.IGNORECASE)


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


def format_findings(findings):
    lines = []
    for f in findings:
        location = f' {f["location"]}' if f['location'] else ''
        lines.append(
            f'- [{f["name"]}-{f["label"]}] {f["severity"]}{location} - {f["desc"]}')
    return '\n'.join(lines)


def prior_round_block(findings, preamble):
    # The prior-cycle context shown to reviewers in round >= 2, so each can recognize its own
    # [Name-Label] in the implementor's push-back and see which point was rejected.
    block = '\n# Previous review round\n'
    if findings:
        block += '\nFindings raised last round:\n'
        block += format_findings(findings) + '\n'
    if preamble:
        block += '\nWhat the implementor said last round:\n'
        block += preamble.strip() + '\n'
    return block


async def run_personas(reviewers, user_message, model, stats_stage, debug=False, loop=None, guidance=''):
    # Run every reviewer independently; return the flattened labeled findings. On a local
    # (serial) backend the reviewers run one at a time so each gets the whole server; otherwise
    # they run concurrently. `guidance` is the target-language backend's persona_guidance(): the
    # per-language conventions the (language-agnostic) reviewer bodies leave out.
    async def one(spec):
        name, body, review_number = spec
        system = body
        if guidance:
            system += f'\n\n{guidance}'
        system += FINDINGS_CONTRACT.format(review_number=review_number)
        label = f'{loop}:{name}' if loop else name
        try:
            mapper = get_mapper(
                system, n_results=1, stats_stage=stats_stage, model=model, label=label)
            text = await mapper.run(user_message)
        except Exception as e:
            if debug:
                print(f'[Personas] {name} failed: {e}')
            log(f'personas: {label} failed: {e}')
            return []
        return parse_findings(text, name, review_number)
    if is_local_backend():
        results = [await one(s) for s in reviewers]
    else:
        results = await asyncio.gather(*[one(s) for s in reviewers])
    return [f for sub in results for f in sub]
