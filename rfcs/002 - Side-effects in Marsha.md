# 002 - Side-effects in Marsha RFC 

## Current Status

### Proposed

2023-07-31

### Accepted

YYYY-MM-DD

#### Approvers

- Luis de Pombo <luis@alantechnologies.com>
- Alejandro Guillen <alejandro@alantechnologies.com>

### Implementation

- [ ] Implemented: [One or more PRs](https://github.com/alantech/marsha/some-pr-link-here) YYYY-MM-DD
- [ ] Revoked/Superceded by: [RFC ###](./000 - RFC Template.md) YYYY-MM-DD

## Author(s)

- David Ellis <david@alantechnologies.com>

## Summary

Side-effects are a part of all programming languages, even ones that try to reduce it as much as possible like Haskell. There is some sort of "hidden" input or output for a function, either affecting the behavior of the function in hard-to-predict ways, or causing an affect on something else in hard-to-measure ways. Because the *correctness* of a Marsha-generated function depends on the test suite that is generated in parallel, side-effects are particularly problematic if the test suite cannot correctly cover and validate these hidden inputs and outputs, something that is affecting the reliability of some of our existing examples, particularly those that make requests on the web for their outputs.

Improving the situation with side-effects in Marsha not only will improve the reliability of current functions, but should make Marsha functions that are intended to manipulate databases or to generate visualizations feasible. (TODO: Do we need special syntax for these, or can we get away with just special syntax for side-effects in general).

## Proposal

No proposal yet, just a collection of alternatives to think about until we make a decision. We may have to prototype the various options to see what really works, though.

### Alternatives Considered

#### Optional side-effect declaration lines

Using a `## SRC: ...` to specify a side-effect source (like a website, date/time, rng, etc) or `## DEST: ...` destination (stdout, image, web API, etc). These would be used by the main code generation to have a "better" idea of what sources to pull from or what exactly to push side-effects to, while the testing side of things would have a stronger indication of what to mock for inputs, while *also* being able to pull an example of the side-effect source to include in the prompt for more accurate mocks (to avoid the `cnn.mrsh` situation). For side-effect outputs, meanwhile, the test suite can inject a kind of mock to confirm the side-effect path was called correctly (intercepting `print` to make sure the correct text was printed, for instance).

It could look something like:

```md
# func cnn(string of section to take headlines from): list of headlines
## SRC: https://cnn.com

This function scrapes the cnn.com website for headlines. The section it takes the headlines from is passed to it, with 'home' referring to the homepage at cnn.com and should be special cased, while 'us' refers to cnn.com/us, 'politics' refers to cnn.com/politics, and so on for every top-level category CNN has.

* cnn('home') = ["Florida's new standards for teaching Black history spark outrage", "His books sold over 300 million copies and were translated into 63 languages. Now, a museum is acknowledging his racism", "Player quits match in tears as tennis world slams opponent’s ‘absolutely disgusting’ actions"]
* cnn('us') = ['18-year-old Miami woman arrested after allegedly trying to hire a hitman to go after her 3-year-old son', 'Investigation into Gilgo Beach serial killings suspect expands to Nevada and South Carolina', 'Rescue crews continue search for 2 children swept away by Pennsylvania floodwater that killed their mother']
* cnn('world') = ["Police raids follow shocking video of sexual assault in India’s Manipur state amid ethnic violence", 'Ukrainian air defenses in Odesa outgunned as Russia targets global grain supply', 'Anger boils over as Kenya’s cost of living protests shake the nation']
```

Presumably the text in the current example explaining to the LLM that the headline class name has changed and what it now is would not be necessary because the mock would be copied from the reality of the website itself.

This particular syntax might be confusing to non-developers, though, since the difference between a function input and output and a function side-effect source and destination could be very unclear, especially if "side-effect" is not in the nomenclature, but it would be brief as a syntax.

#### Optional side-effect "arguments"

Treating the side-effect inputs and outputs as non-side-effect inputs and outputs that are automatically set (or ignored for the return type) might make the concept less confusing. Wrapping the side-effects in both in brackets at the end of the lists could help here:

```md
# func cnn(string of section to take headlines from, [https://cnn.com]): list of headlines

This function scrapes the cnn.com website for headlines. The section it takes the headlines from is passed to it, with 'home' referring to the homepage at cnn.com and should be special cased, while 'us' refers to cnn.com/us, 'politics' refers to cnn.com/politics, and so on for every top-level category CNN has.

* cnn('home') = ["Florida's new standards for teaching Black history spark outrage", "His books sold over 300 million copies and were translated into 63 languages. Now, a museum is acknowledging his racism", "Player quits match in tears as tennis world slams opponent’s ‘absolutely disgusting’ actions"]
* cnn('us') = ['18-year-old Miami woman arrested after allegedly trying to hire a hitman to go after her 3-year-old son', 'Investigation into Gilgo Beach serial killings suspect expands to Nevada and South Carolina', 'Rescue crews continue search for 2 children swept away by Pennsylvania floodwater that killed their mother']
* cnn('world') = ["Police raids follow shocking video of sexual assault in India’s Manipur state amid ethnic violence", 'Ukrainian air defenses in Odesa outgunned as Russia targets global grain supply', 'Anger boils over as Kenya’s cost of living protests shake the nation']
```

Here it's clear that the `[https://cnn.com]` is *different*, though exactly how is not clear on a first reading. The fact that this input argument doesn't show up in the input arguments list of the generated function but it does show up in the function body might help clear it up a bit for the user just from inspection, but it could also confuse people that it's a "default" value of some sort that you can replace with some other value when that is not the case.

#### Optional Side-effects description

A new block below the description but above the examples that has a human-readable paragraph of side-effects to pull from or push to. This is less strict, but may also be less effective at handling side-effects in a reliable manner.

```md
# func cnn(string of section to take headlines from): list of headlines

This function scrapes the cnn.com website for headlines. The section it takes the headlines from is passed to it, with 'home' referring to the homepage at cnn.com and should be special cased, while 'us' refers to cnn.com/us, 'politics' refers to cnn.com/politics, and so on for every top-level category CNN has.

## Side effects

This function reads live data from https://cnn.com to work.

* cnn('home') = ["Florida's new standards for teaching Black history spark outrage", "His books sold over 300 million copies and were translated into 63 languages. Now, a museum is acknowledging his racism", "Player quits match in tears as tennis world slams opponent’s ‘absolutely disgusting’ actions"]
* cnn('us') = ['18-year-old Miami woman arrested after allegedly trying to hire a hitman to go after her 3-year-old son', 'Investigation into Gilgo Beach serial killings suspect expands to Nevada and South Carolina', 'Rescue crews continue search for 2 children swept away by Pennsylvania floodwater that killed their mother']
* cnn('world') = ["Police raids follow shocking video of sexual assault in India’s Manipur state amid ethnic violence", 'Ukrainian air defenses in Odesa outgunned as Russia targets global grain supply', 'Anger boils over as Kenya’s cost of living protests shake the nation']
```

This blends input and output side effects together and makes it a more flexible notice the user provides. It is not much different from the existing `cnn.mrsh` *except* hopefully you don't have to explicitly call out what has changed. It would depend on extra LLM calls to determine what the side-effect sources and destinations are, and then do the automated work to get an example of each source to use.

#### Disallow mocks in the test suite

One possibility to fix this situation is to disable mocking in the generated test suite at all. In that case, the side-effect functions *will* be executed for real. This could resolve the side-effect input problem by using real-world data for the test, but it would make the reliability of the generation lower (why we added mock support in the first place), It also doesn't do anything for the side-effect outputs, which would remain completely untested.

#### Explicit mock rule declaration in examples

Only generate mocks for things that were called out to be mocked and/or intercepted for side-effects inputs and outputs, respectively. This would similarly need extra work to get the actual payload to embed in the mocks.

```md
# func cnn(string of section to take headlines from): list of headlines

This function scrapes the cnn.com website for headlines. The section it takes the headlines from is passed to it, with 'home' referring to the homepage at cnn.com and should be special cased, while 'us' refers to cnn.com/us, 'politics' refers to cnn.com/politics, and so on for every top-level category CNN has.

* cnn('home') using a mock of https://cnn.com = ["Florida's new standards for teaching Black history spark outrage", "His books sold over 300 million copies and were translated into 63 languages. Now, a museum is acknowledging his racism", "Player quits match in tears as tennis world slams opponent’s ‘absolutely disgusting’ actions"]
* cnn('us') using a mock of https://cnn.com/us = ['18-year-old Miami woman arrested after allegedly trying to hire a hitman to go after her 3-year-old son', 'Investigation into Gilgo Beach serial killings suspect expands to Nevada and South Carolina', 'Rescue crews continue search for 2 children swept away by Pennsylvania floodwater that killed their mother']
* cnn('world') using a mock of https://cnn.com/world = ["Police raids follow shocking video of sexual assault in India’s Manipur state amid ethnic violence", 'Ukrainian air defenses in Odesa outgunned as Russia targets global grain supply', 'Anger boils over as Kenya’s cost of living protests shake the nation']
```

This approach seems pretty "natural" with the examples, bolting on some special behavior if you follow a particular pattern in the example set, but that may also reduce the discoverability of the mocking mechanism, or implying other features are automatically possible by using similar phrasing for other things.

## Expected Semver Impact

All of the proposed changes are optional additions to the syntax, so it would be a minor update if post-1.0.0

## Affected Components

The parser and llm portions would be affected, but this would not require new configuration options.

## Expected Timeline

TBD
