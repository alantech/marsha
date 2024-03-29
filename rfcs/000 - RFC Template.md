# 000 - RFC Template

## Current Status

### Proposed

YYYY-MM-DD

### Accepted

YYYY-MM-DD

#### Approvers

- Full Name <email@alantechnologies.com>

### Implementation

- [ ] Implemented: [One or more PRs](https://github.com/alantech/marsha/some-pr-link-here) YYYY-MM-DD
- [ ] Revoked/Superceded by: [RFC ###](./000 - RFC Template.md) YYYY-MM-DD

## Author(s)

- Author A <a@alantechnologies.com>
- Author B <b@alantechnologies.com>

## Summary

A brief, one to two paragraph summary of the problem and proposed solution goes here. The name of the PR *must* include "RFC" in it to be searchable.

## Proposal

A more detailed description of the proposed changes, what they will solve, and why it should be done. Diagrams and code examples very welcome!

Reviewers should *not* bring up alternatives in this portion unless the template is not being followed by the RFC author (which the RFC author should note in the PR with a detailed reason why). Reviewers should also not let personal distaste for a solution be the driving factor behind criticism of a proposal, there should be some rationale behind a criticism, though you can still voice your distaste since that means there's probably *something* there that perhaps another reviewer could spot (but distate on its own should not block).

Most importantly, be civil on both proposals and reviews. `iasql` is meant to be an approachable tool for developers and if we want to make it better we need to be approachable to each other. Some parts of the language may have been mistakes, but they certainly weren't intentional and all parts were thought over by prior contributors. New proposals come from people who see something that doesn't sit well with them and they have put forth the energy to write a proposal and we should be thankful that they care and want to make it better.

Ideally everyone can come to a refined version of the RFC that satisfies all arguments and is better than what anyone person could have come up with, but if an RFC is divisive, the "winning" side should be gracious, and the "losing" side should hopefully accept that the proposal was contentious.

### Alternatives Considered

After proposing the solution, any and all alternatives should be listed along with reasons why they are rejected.

Authors should *not* reject alternatives just because they don't "like" them, there should be a more solid reason

Reviewers should *not* complain about a lack of detail in the alternative descriptions especially if that is their own preferred solution -- they should attempt to positively describe the solution and bring their own arguments and proof for it.

## Expected Semver Impact

A brief description of the expected impact on the Semantic versioning.

Would this be considered a patch (no user-facing changes, but internal architectural changes. Bug fixes, new modules)?

Would this be considered a minor update (new functionality with zero impact on existing functionality. API changes, new iasql functions?)?

Would this be considered a major update (breaking the behavior of existing code)?

RFCs that are a major update are more likely to be rejected or modified to become a minor or patch update, if possible. If not possible, major version RFCs are likely to be delayed and batched together with other major version RFC updates.

## Affected Components

A brief listing of what part(s) of the engine will be impacted should be written here.

## Expected Timeline

An RFC proposal should define the set of work that needs to be done, in what order, and with an expected level of effort and turnaround time necessary. *No* multi-stage work proposal should leave the engine in a non-functioning state.
