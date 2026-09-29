from __future__ import annotations

from typing import NotRequired, TypedDict


# A single code-review finding as it flows through the pipeline: personas parse/produce them,
# the review loop critiques/consolidates/dedups them, and the evidence gate verifies them.
# `name`/`label`/`severity`/`location`/`desc` are always present; `support` (a 1-2 paragraph
# justification, often empty), `evidence` (the reviewer's retrieved git output), and `sources`
# (the doc/web sources it retrieved) are NotRequired — all read only via .get(), never by direct
# index.
class Finding(TypedDict):
    name: str
    label: str
    severity: str
    location: str
    desc: str
    support: NotRequired[str]
    evidence: NotRequired[list[tuple[str, str]]]
    sources: NotRequired[list[tuple[str, str]]]
