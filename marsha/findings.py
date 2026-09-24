from __future__ import annotations

from typing import NotRequired, TypedDict


# A single code-review finding as it flows through the pipeline: personas parse/produce them,
# the review loop critiques/consolidates/dedups them, and the evidence gate verifies them.
# `name`/`label`/`severity`/`location`/`desc` are always present; `support` (a 1-2 paragraph
# justification, often empty) and `evidence` (the reviewer's retrieved git output, attached
# downstream) are NotRequired — both are read only via .get(), never by direct index.
class Finding(TypedDict):
    name: str
    label: str
    severity: str
    location: str
    desc: str
    support: NotRequired[str]
    evidence: NotRequired[list[tuple[str, str]]]
