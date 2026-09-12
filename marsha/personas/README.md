# Marsha review personas

These files are loaded at runtime by the `--optimize` review loops. There are three kinds of file:

- **Reviewer personas** (`oracle-*.md`, `impl-*.md`, `correction-*.md`) — independent reviewers, one per aspect. A loop runs its set of reviewers in parallel; each emits `MAJOR` / `MINOR` / `NIT` findings.
- **Editor personas** (`_*.md`) — the *implementor* for each loop. Editors are fixed per loop (not user-selectable) and are excluded from the reviewer registry by their `_` prefix. They take the consolidated findings and return a reasoning preamble plus the revised artifact.
- This `README.md` — documentation; excluded from the registry.

## Reviewer persona format

The **first line must be** `name: <name>`. Everything after it is the reviewer's system prompt (role, goal, checklist). The harness appends a fixed findings contract to every reviewer, so a persona only describes *what* to check — never *how* to report.

```
name: Ada
You are Ada, a meticulous QA engineer auditing a unit test suite (the oracle).
Goal: every piece of testable behavior the assignment states is covered by at least one test.
Check:
- Each worked example the assignment provides.
- Flag any such stated behavior that has no covering test.
```

## Selecting reviewers

The built-in registry is discovered by scanning this directory for `*.md` (excluding `_`-prefixed editors and this README) and reading each `name:` line. You select reviewers per loop with:

- `--test-personas` — oracle (test-suite) loop
- `--impl-personas` — implementation loop
- `--fix-personas` — test-correction (oracle-fix) loop

Each flag takes a comma-separated list. An entry is either a **built-in name** (e.g. `eli,max`) or a **path** to a custom file (detected by a leading `/`, `.`, or `~`, e.g. `./sharona.md`). Mixing is allowed: `--impl-personas eli,max,./sharona.md`. Omit a flag to run that loop's default set (all built-ins for the loop, name-sorted). A custom file must also start with `name: <name>`.

## Language-specific conventions

Reviewer bodies are deliberately language-agnostic ("an assignment", "an implementation") so one shared set serves every target language. Per-language and per-project conventions (for the Python backend: type hints, PEP8, autopep8, `pyproject.toml` manifests) are supplied by the target backend's `persona_guidance()` and injected into every reviewer run — the backend is selected with `--target` (see the top-level README).

## Finding labels

Each finding is labeled `<letter><N>` — the letter is the finding's position (A, B, C, ...) and `N` is the reviewer's 1-based position in the resolved list. References use `[Name-Label]` (e.g. `[Ada-A1]`), so in a later round a reviewer can recognize its own label in the implementor's push-back.
