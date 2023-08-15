# 002 - Compiler Refactor RFC

## Current Status

### Proposed

2023-08-12

### Accepted

2023-08-15

#### Approvers

- Luis de Pombo <luis@alantechnologies.com>
- Alejandro Guillen <alejandro@alantechnologies.com>

### Implementation

- [ ] Implemented: [One or more PRs](https://github.com/alantech/marsha/some-pr-link-here) YYYY-MM-DD
- [ ] Revoked/Superceded by: [RFC ###](./000 - RFC Template.md) YYYY-MM-DD

## Author(s)

- David Ellis <david@alantechnologies.com>

## Summary

Multiple concurrent efforts to move Marsha forward (testing side-effect functions, special syntax for databases, using Llama V2 or WizardCoder for local LLM support) have been stymied by requiring very branchy code to implement on our simple fixed 3-stage parse-transform-emit compiler.

We either need to be very sure where we want to take Marsha to continue a fixed pipeline, or we need to consider alternative compiler structures.

## Proposal

#### How LLMs are like compilers

Simply put: compilers take some input (configuration flags, code, etc), parse it into an AST that can then be manipulated to produce some output in another format. LLMs take some input (SYSTEM directives, user requests, etc) parse it into tokens that can then be manipulated to produce a new output.

#### How LLMs are *not* like compilers

Compilers, when given the same input to the same compiler code, produces identical output. It may be difficult to get it to produce exactly the assembly instructions you're looking for, but once you find that magical incantation, it will continue to work. LLMs are not deterministic in that way. They will produce different output for the same input(s). Further, some inputs may produce an output that works some times and doesn't work other times.

#### What Marsha currently brings to the table

Marsha has three stages of operation, but those three stages together accomplish a singular task: make it so output from the LLM works as reliably as the test suite you provide it, or block the output entirely, instead of providing misleading output that doesn't work. This operates on the principle that verifying the output is more reliable than generating the output. (Which have precedence in things like the Sieve of Eratosthenese.) When the first stage fails to parse, it immediately returns an error message informing the user of the parse error. But when a verification fails in the second or third stage, it executes a different LLM call in a loop to attempt to get the LLM to correct itself. (The first stage also has an LLM-based sanity check that the even syntactically correct requests are "reasonable", but it doesn't loop, so it's not very relevant to this discussion.) Once it gets through all three stages it has working code (and a working test suite) that we then bolt helper logic to make CLI and HTTP based usage easier.

When it's broken down like that, we notice two basic kinds of operations:

1. `input -> transform -> output or error`
2. `input -> transform -> check and maybe retry -> output or error`

The fact that a transform or check is powered by an LLM or "regular" code is just an implementation detail.

A cross-cutting concern is the actual features of the language implemented within the entire flow. The entire flow is, roughly:

1. input
2. parse AST or error
3. transform AST to more verbose markdown or error
4. sanity check with LLM or "error" (on error, run a second LLM to try to give hints why it failed the sanity check)
5. prompt LLM to generate all functions and classes based on output from (3) or error (only if LLM unreachable), generating 3 different files by default to improve reliability for a greater expense
6. prompt LLM to generate test suite for all functions based on output from (3) or error (can be done in parallel when using ChatGPT), generating 3 different files by default to improve reliability for a greater expense
7. take pairs of output from (5) and (6) and verify they are valid markdown with the expected sections only, returning the pair(s) that work, or erroring if none do
8. take each pair of outputs from (5) and (6) and run them through a Python linter to confirm they are valid python (most stylistic rules removed, just syntax, unused imports, and a few others), if it errors it sends all of this to an LLM to generate new versions of the file, repeating on error until a limit is reached, then exiting the flow for that particular pair, erroring if all pairs fail.
9. take each pair of outputs from (8) and runs the test suite, passing them and the test suite errors to an LLM if they fail, otherwise continuing. If it continues to fail for too long, the particular pair exits the flow, if all pairs exit the flow, it errors.
10. the first pair to finish (9) cancels the rest of the operations, then by default it appends the CLI and HTTP server helper logic to the code file and exits.

For now, *most* of the Marsha language functionality exists within (3), and it is a synchronous operation, but most of the solutions for database support would at minimum require (3) to become async and involve various DB and ML operations (likely: connect to the DB and generate a schema dump, then chunk that dump and index it with vectors, then use vector search to find the most relevant table(s), stored procedure(s), etc to provide to (5) and (6)) but it's possible these features could "bleed" into the validation steps of Marsha, and the codebase starts turning (even more) into a big ball of mud. (For instance, validating the generated SQL DB access code in (5) against the actual DB schema in (7) seems likely).

There are other dimensions we would like to extend Marsha on:

* Which LLM is used to do the work.
* Which target language Marsha compiles to.
* Dev tools (debugging, decompiling, etc) for Marsha.

The solution we come up with needs to keep the complexity of all of these different features from compounding on one another, ideally.

#### Proposal: Marsha as Extensions that append to lists of layers

Taking a very rough inspiration from LLMs themselves, we'll have a singular kind of "mapper" that is the foundational piece of the compiler, and they will be organized into layers where each layer executes in parallel and their outputs are fed into the next layer, etc. Each logical feature will be an extension that concatenates its logic into the appropriate layers, with the "meaning" said layers being a convention per extension "group".

Now working backwards, an Extension Group is a collection of extensions that all declare membership of a particular group name. Extensions won't be allowed to mix between groups to reduce the complexity involved, and because an Extension Group declares how many layers exist and their input and output formats (most layers will likely be lists of some class type, but the input for the first layer will be `None` and use the quasi-global configuration data to set itself up and the output of the last layer can only be `None` or `str`).

A new extension that wishes to insert a special layer between existing layers would have to fork the entire extension group project, which seems wasteful, but as long as the layer definitions are "cheap" (simple and easy to read), just forking an extension group under a new name shouldn't be too big of a burden.

The "mapper" logic in each layer will consist of just the #2 operation type listed above: `input -> transform -> check and maybe retry -> output or error`, as the third step becomes a simple no-op in other cases, it could have a default `f(a) = a`-type implementation and made an optional element of that class.

Yes, class. The use of the word "mapper" here could be confusing, so this part is definitely subject to change, but there'd be a `BaseMapper` class that must be extended and where you must implement the `transform` method, but the `check` method could use the default implementation. Both of these methods are `async` so they can do whatever you want within them.

And you could always Bring-Your-Own-LLM, but it would probably be best if the Marsha project itself maintained a collection of `*Mapper` classes that automatically make LLMs like ChatGPT, LLamaV2, WizardCoder, etc, and other useful tools like Markdown AST, Python venv projects, etc, easily defineable with as few lines of code as possible. Each mapper is given the entirety of the prior layer's output, which will usually be a list of outputs when there are multiple mapper in a layer, but would be without the list wrapper if there is only one mapper in that layer (so for instance, loading the `*.mrsh` file could be a mapper in its own layer all by itself that loads and parses the AST, and then feeds that AST root node to the next layer that does the code and test LLM operations in parallel for ChatGPT, while Llama V2 locally could make that a singular mapper so they run sequentially).

Even though the mappers run in parallel, we'll use asyncio's `gather` mechanism to make sure they stay in the "expected" order.

You could conceivably just have the Extension Group directly place the amppers in the "right" places, but that means there'd be no way for a user to add or remove "optional" or "recommended" extensions, so there'd be zero configurability for them (and therefore *all* experimentation with how Marsha should work would require forking the extension group every time). If instead there are "required", "recommended", and "optional" extensions within an extension group, the user could manually disable recommended extensions or manually enable optional extensions to configure the behavior. Then the Extension Group defines the order in which each extension appends its mappers to the different layers, and each Extension defines which layers each transformer belongs to. This should allow experimentation with new syntax that is incompatible with existing syntax by marking the incompatible syntax a recommended extension and the new syntax an optional extension, and a savvy user could swap them out if desired.

These concepts would be baked into `Extension` and `ExtensionGroup` classes that would force the desired organization onto the code, and they, along with the `*Mapper`s could be put together into a Python module (or just a singular `*.py` file) and then [dynamically imported by Marsha](https://docs.python.org/3/library/importlib.html#module-importlib) so they could live outside of the codebase.

If the user does not specify an extension group, it would be assumed to use the default extension group that will be baked into Marsha (or later, if/when we add decompilation/debugging/etc functionality, it would depend on the subcommand called, like `marsha compile ...`, `marsha decompile ...`, `marsha debug ...`, etc). Similarly, if no extension manipulation is requested by the user, the required and recommended extensions would be loaded (otherwise, the set of extensions to load will be modified based on the inclusion/exclusion lists).

This gives us a fairly flexible control over how Marsha works and how it can be developed into the future, while also giving us a fractal-like organization of the code in question, making it easier to understand and maintain while keeping the code reasonably DRY and efficient.

### Alternatives Considered

#### Marsha as fully undirected graph of mappers

Here, each mapper simply declares a named source for its input and a named output destination, with `START` and `END` being special nodes. This is very similar in spirit to [queue-flow](https://dfellis.github.io/queue-flow/) that I wrote so many years ago, and could even handle multiple nodes reaching `END` at different times by using introspection on the event loop to decide when to actually quit.

It was rejected because while you can write pretty succinct code that efficiently handles sync or async functions, the named queues make it difficult/impossible to have runtime-configurable extension configuration with it, and it doesn't help the logical grouping of syntactic elements that are spread across multiple mappers throughout the graph, so readability would only be marginally improved. (That last part was pretty general to queue-flow -- I tended to get "write-only" code out of it that was fast, efficient, and near impossible for other developers to read, because the connections between the named queues were often spread across files and hard to follow.)

#### Marsha as database and trigger-transformers

There's been work on turning compilers into specialized databases, particularly for type inference or langserver use-cases where partial compilation of known data and multiple passes makes a lot of sense, and being able to easily query for things "other parts" of the compiler have "figured out" in the meantime can allow improvements in function generation in secondary passes, or simply allow compilation to complete at all for said functions. It's a very different way of thinking about how compilation works and definitely has some merit, but seems very similar functionally to the queue-flow-like approach described above: each registered operation would have to register what data it needs to do it's work and then get triggered when the required data is there. This makes an operation that depends on multiple sources of different data easier to implement than the queue-flow approach (but is doable with that approach with a reducer-transformer function that both sources push to and it only pushes to it's output queue once all expected data sources have arrived), but it has the exact same problem that the actual computational flow is very difficult to reason about, though it is easier to update a singular file and recompile just it and only parts of other files that are impacted, for instance, instead of having to start the world over again, so the advantage for language servers is there.

But the increased maintenance burden *plus* the LLM latency problem (such that sub-100ms response times for IDE-langserver integration to be meaningful is impossible) means we don't want to try for a language server for quite a while, if ever. So this approach is also rejected. It also suffers from actual language extensions being exceedingly difficult to do in this appraoch, because you would have to modify each transformer's query to adjust which inputs it consumes, which means it's an abstraction that doesn't fit if that's desirable. (Usually, extensible parts of the language go into the standard library and the syntax itself is not really extensible, just the set of symbols you're working with. I do wonder if/how Raku implements a lang server?)

#### Layers only Extensions

In this version, each Extension Group is a list of Extensions, and these extensions are actually the transformers, and executed in the order defined, one input leading to another output, and that's it. Extension Groups are just shorthand for particular transformer orders. This is much simpler than the proposal, but has several drawbacks: first, "related" transformers for a given feature are not obvious (except perhaps by naming convention) so misconfiguration by leaving part of a feature out is easily possible. It is also possible to accidentally put two incompatible transformers next to each other when another order of the same transformers is fine, but there would be no warning to you until you tried and it failed. Finally, it's impossible to run some transformers in parallel even if they really don't depend on each other, so this approach would reduce our current compiler performance noticeably.

#### Keep Hacking

Do nothing to change the way we're doing things and just hack the features on at will. The complexity of the codebase would temporarily go up while we're figuring things out, but would presumably drop down some again in the future once certain behaviors are determined irrelevant or are merged into more generalized ones. We could presumably keep going on this for a bit, but competing features would have to live on long-lived feature branches while being figured out. Because the reliability of an LLM-based transform is lower than a traditional compiler transform, some language ideas may look good on paper but can't actually work with the LLMs of today, at least, possibly never, and determining that ahead of time is sometimes frustratingly impossible, so this approach is rejected as putting too high of a maintenance burden on us, as well as keeping external contribution lower. It also never allows Marsha-the-OSS-project to be considered a separate thing from Marsha-the-AI-language. This means things like the decompiler, debugger, etc, would likely need to be parallel projects (or at least the various Marsha-related things would be wholly-separate sub-modules within a parent module and little to do with each other architecturally).

Finally, this approach feels antithetical to the bootstrapping goal. The proposed solution could conceivably have the transformers, extensions, and extension groups generated by Marsha itself, and the "core" of Marsha (the various classes defined) could be converted piecemeal, too, until it's all Marsha code. Handwritten, tightly-coupled compiler logic doesn't seem as amenable to that sort of rewrite.

## Expected Semver Impact

The language itself would have zero use-facing chnages, so that would be a patch version change, but Marsha as a more generalized tool beyond Marsha the AI language might imply it's a major version bump. Hard to decide.

## Affected Components

Absolutely everything in the codebase, but we could probably do this piecemeal, first reoganizing the three stages into transformers, and then writing the extension and extension group classes and handling logic and dropping the fixed pipeline later.

## Expected Timeline

1. Create `BaseMapper`, `ChatGPTMapper`, `MarkdownASTMapper`, etc classes
2. Rewrite the Marsha stages with these classes.
3. Create the `Extension` and `ExtensionGroup` classes (and/or better name for `ExtensionGroup`?)
4. Rewrite the pipeline to use these classes with a hardwired instantiation. (Temporarily dropping `--quick-and-dirty` and potentially other flags)
5. Convert the hardwired instantiation into a default that is used if the user doesn't provide an `ExtensionGroup` and extension configuration options, but use the specified `ExtensionGroup` if defined.
6. Restore `--quick-and-dirty` (and potentially others) as an `ExtensionGroup`
7. Start implementing `Llama2Mapper`, `WizardCoderMapper`, etc for some experiments, DB-specific ones for others, etc.
