# 003 - Tool-Driven Codebase Review and Review-Only Personas

## Current Status

### Proposed

2026-09-16

### Accepted

TBD

#### Approvers

- (pending)

### Implementation

- [ ] Implemented: (PR pending) 2026-09-16
- [ ] Revoked/Superceded by: —

## Author(s)

- David Ellis <isv.damocles@gmail.com>

## Summary

`marsha review` (issue #211, PR #214) currently hands each reviewer persona a
**static blob** — the full unified diff plus any PR/Linear context — and each
reviewer does a single `mapper.run` with no ability to explore. On top of that,
the reviewer bodies are framed for *generated-code-vs-oracle* review
(`FINDINGS_CONTRACT`: "MAJOR = violates the spec or oracle contract"), which
does not map onto reviewing a branch of an existing codebase (there is no
oracle). The net result of dogfooding was more false positives than correct
findings: reviewers flag from diff hunks alone, with no surrounding code, no
codebase conventions, and no history, and over-flag.

This RFC makes the review **tool-driven** and adds the review-only personas from
issue #212. Reviewers are given a read-only `git` tool (so they can page through
the diff, read changed files, and inspect history) and a `notes` scratchpad
(so they can record the facts they want to carry into the final review, and that
survive context compaction). Instead of force-feeding the full diff, each
reviewer is given `git diff <base> --stat` as a starting map. A new
`review-conventions` **feedback gate** (the "editor" role in a multi-round loop,
mirroring the optimize loop) rebuts findings that violate the repo's *actual*
conventions (AGENTS.md/CLAUDE.md/lint configs), which directly kills the
type-hints / `asyncio.run` class of false positives. A `review-git-history`
reviewer grounds findings in intent/history before flagging. A final
consolidation pass de-duplicates across reviewers.

## Proposal

### 1. A read-only `git` tool (new `git` category)

A new fake-terminal command, `git <subcommand> [args…]`, available only in the
`review` phase (a new category `git`, and a new `'review'` entry in
`PHASE_CATEGORIES`). It runs in the repository working directory
(`ToolContext.workdir`) with a hard timeout and `GIT_TERMINAL_PROMPT=0`
(non-interactive, so no credential prompt can hang a run).

**Allowlist, not a blocklist.** Only unambiguously read-only subcommands are
permitted; anything else returns the message that the reviewer may not modify
the git tree. An allowlist is preferred over the blocklist originally sketched
(commit/push/pull/checkout/…) because it cannot accidentally permit a mutating
subcommand, present or future:

```
diff log show blame grep ls-files ls-tree cat-file cat rev-parse status
describe shortlog rev-list show-ref for-each-ref count-objects ls-remote remote
```

Flag-level guards: `--output` / `-o` / `--output-directory` (which make
`git diff` write a file) are rejected. The reviewer reads the branch-under-review
with `git show HEAD:<path>` / `git cat-file`, the base with `git show <base>:<path>`,
per-file changes with `git diff <base>...HEAD -- <path>`, searches with
`git grep`, and history with `git log` / `git blame` / `git log -- <path>`.

### 2. A per-reviewer `notes` tool (new `notes` category)

A second review-scoped command: `notes add <text>` and `notes show`. Notes are
stored **server-side on the reviewer's `ToolContext`** (a `notes` list, fresh
per reviewer via a per-reviewer clone in `run_personas`) — not just in the
conversation. The reviewer is instructed to record each candidate finding as it
finds it (`notes add '<file:line> - <what and why>'`). This is the reviewer's
curated "keep this" list, which is what survives a lossy context compaction.

### 3. Feed `--stat`, not the full diff

`run_review` builds the reviewer message from `git diff <base> --stat` (the map:
which files changed and by how much), the exact base ref, and any PR/Linear
context (still wrapped as untrusted reference data) — **not** the full unified
diff. Each reviewer pages through the diff and the surrounding code itself with
the `git` tool. The full diff is still computed, but only when `--post-review`
is set, for inline-comment placement (`diff_new_lines`).

### 4. Budget-gated compaction with notes survival

The reviewer's tool loop (`run_with_tools`) is gated by the existing context
budget machinery (`fits` / `estimate_tokens` / `resolve_context_window`).
Before each call, if the accumulated prompt would exceed the budget, the loop
**compacts** the conversation with an LLM summarize pass and **re-attaches the
reviewer's `notes` verbatim** so they survive. When no compaction happens, the
notes are already in the conversation, so nothing is attached — the notes are
attached *only when a compaction was necessary*.

Reused vs. new: the budget gate and the LLM-summarize *pattern* are reused; the
summarization prompt for a **conversation** is new (the existing
`compact_findings` prompt is findings-list-specific). The loop's existing
`MAX_TOOL_ROUNDS` cap is raised for review via a `max_tool_rounds` parameter on
`run_personas`. This also fixes the current silent-failure mode where an
overflowing tool loop raised `ContextOverflowError` and the reviewer returned
no findings.

### 5. `review-conventions` as a feedback gate (issue #212)

`review-conventions` is **not** a panel member. It is the **editor role** in a
multi-round loop, mirroring the optimize loop's
`reviewers → editor(preamble) → reviewers(prior_round_block)` cycle. Per round:

1. The **panel** (the 11 impl reviewers + the new `review-git-history`) produces
   findings, exploring with `git` and recording with `notes`.
2. The **conventions gate** reads the repo's real conventions
   (AGENTS.md, CLAUDE.md, lint configs, sample code — via the `git` tool) and
   returns a **rebuttal** that cites specific findings by `[Name-Label]`
   (e.g. `A2 (Sage): add type hints — but this repo uses none; drop it`), or
   `''` when there are none.
3. If there is a rebuttal and rounds remain, the panel re-runs with
   `prior_round_block(prior_findings, rebuttal)` so each reviewer drops or
   adjusts its rebutted findings.

The loop converges when the gate returns no rebuttal. A `--review-rounds` flag
(default **1** = panel → gate → one revision) bounds it. `prior_round_block`
gains a label parameter so its "implementor said" section reads
"conventions review said" in this context.

Because in Marsha's code-generation the "respond negatively to a comment" cycle
is implicit in the code-building layer (the editor may push back on a finding,
and the next reviewer round sees that push-back), the review loop makes that
cycle explicit: the conventions gate *is* the push-back, and the next panel
round sees the prior findings plus the rebuttal.

### 6. `review-git-history` panel persona (issue #212)

A panel reviewer that grounds findings in intent and history before flagging:
it reads recent commits (`git log`), the branch's purpose, and per-line
authorship (`git blame`), and only flags a change when it is inconsistent with
its stated intent, regresses prior behavior, or misses something the history
suggests was anticipated. This reduces the "flagged a bug that is actually
intentional" false positives.

### 7. Cross-reviewer de-duplication (issue #212)

After the loop, a single consolidation pass — reusing the `compact_findings`
pattern but **meta-less** (review has no `MarshaMeta`) — drops resolved/superseded
findings and merges cross-reviewer duplicates (keeping the most detailed
`[Name-Label]`), so the same point raised by several reviewers is reported once.

### 8. Wiring

- `ToolContext` gains `notes` and a `'review'` phase; `PHASE_CATEGORIES` gains
  `'review': {git, notes}`.
- `run_personas` gains `max_tool_rounds`, a per-reviewer notes clone, and the
  tool wiring for review.
- `run_review` builds the review `ToolContext` (`phase='review'`,
  `workdir=<repo>`, `notes=[]`) and the `--stat` message, and drives the
  multi-round loop + gate + consolidation.
- `base.py` adds `--review-rounds` and updates the `review` help text.

## Alternatives Considered

- **Blocklist for the `git` tool** (allow everything except commit/push/pull/…).
  Rejected: a blocklist can silently permit a mutating subcommand; an allowlist
  of read-only commands is the safe default for a model running commands against
  a real repository.
- **Force-feed the full diff** (current behavior). Rejected: it is the root of
  the noise — reviewers cannot ground findings in surrounding code, conventions,
  or history, and it blows the context budget on large changes.
- **`review-conventions` as a 13th panel member.** Rejected: a panel member only
  *adds* findings; the value here is *removing* convention-violating findings
  from the others' output, which requires the editor/gate role and the
  multi-round feedback cycle.
- **A separate "final review from notes" call.** Superseded: the notes are
  re-attached on compaction (and are otherwise already in the conversation), so
  the reviewer's natural terminating response is the final review — no extra
  call needed.

## Expected Semver Impact

Minor: new functionality for `marsha review` (tools, multi-round conventions
gate, new personas, `--review-rounds`). No change to existing `compile` behavior
beyond the general budget-gated compaction in the tool loop, which only activates
when a prompt would otherwise overflow.

## Affected Components

- `marsha/tools.py` — `git` + `notes` commands, `git`/`notes` categories,
  `'review'` phase, `ToolContext.notes`, budget-gated compaction in
  `run_with_tools` + conversation-summarize prompt.
- `marsha/review.py` — `--stat` message, review `ToolContext`, multi-round loop,
  `_conventions_gate`, meta-less consolidation.
- `marsha/personas/__init__.py` — `run_personas` (`max_tool_rounds`, per-reviewer
  notes clone), `prior_round_block` label param, meta-less consolidation helper.
- `marsha/personas/review-conventions.md`, `marsha/personas/review-git-history.md`
  — new personas.
- `marsha/base.py` — `--review-rounds` flag + help.
- `tests/` — git allowlist/block, notes, `--stat` message, compaction + notes,
  conventions gate, multi-round loop, consolidation (temp git repo, mocked
  mapper/gh/linear).
- `README.md` — updated review section.

## Expected Timeline

Ordered; each step leaves `marsha review` runnable (the loop degrades to a
single pass until the gate/rounds are wired):

1. `git` tool (read-only allowlist + guards + non-interactive + timeout).
2. `notes` tool (per-reviewer scratchpad + `ToolContext.notes`).
3. `run_with_tools` budget-gated compaction + notes re-attach + summarize prompt.
4. Review feeding from `--stat` (full diff only for `--post-review`).
5. Conventions gate (`review-conventions` + `_conventions_gate` +
   `prior_round_block` label + multi-round loop + `--review-rounds`).
6. `review-git-history` panel persona.
7. Final consolidation (dedup) pass.
8. `run_personas` / `base.py` wiring (`--review-rounds`, review `ToolContext`).
9. Tests (temp git repo; mocked mapper/gh/linear).
10. README + full test suite + lint (`autopep8 -d` + `flake8`).
