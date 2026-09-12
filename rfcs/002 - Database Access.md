# 002 - Database Access RFC

## Current Status

### Proposed

2023-08-02

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

LLMs are good at foundational things. Loops, branches, recursion, etc. These things haven't really changed since almost the beginning of computing, and it can use them well. Widely-used, stable libraries it also does well on. But specifics to the exact problem at hand need to be fully described to it or it has no hope of working. Making queries to a database and then using the query result to generate some output *can* work with Marsha as is -- if you're willing to write the SQL query for it. The moment you simply describe the query you want, it will fail more often than not because it doesn't know what your database schema is, and will just guess a schema that seems reasonable based on the requested query, and then fail.

We have already demonstrated a way to handle this correctly with [SQL Pal](https://github.com/alantech/sqlpal), so we could integrate much of the work there into the Marsha compiler to handle the schema problem better. But we likely need new syntax to support this. Querying existing databases with externally-defined schemas might be doable without a new syntax, but it would be reasonable for Marsha applications to define their own database for their needs, especially Marsha applications meant to be run as a web server with different REST endpoints to accomplish different tasks.

## Proposal

Just alternatives right now as we dig into the possibilities here. The solution ideally should:

1. Declare what database will be used by the application, whether a local sqlite file, a remote Postgres/MySQL/etc server, or perhaps something more esoteric like MongoDB, Redis, etc.
2. Declare the connection is read-only vs read-write, so it *never* tries to mutate a read-only database connection.
3. Allow the user to define the schema within Marsha if the database is *owned* by the Marsha application (one level above read-write, as it should also automatically migrate the database on startup ideally)
4. When run as a web server, it should use a singleton connection instead of creating a new connection per request, since many databases handle large connection volumes poorly. (Likely the CLI approach could use the same mechanism, though it wouldn't matter there)
5. Multiple databases should be allowed in the same marsha application (the syntax shouldn't be a singleton type) so there needs to be a clear way to associate schemas and queries to a particular database
6. When possible, get the answer from the database during compilation, instead of making the user explicitly spell it out in Marsha text.

### Alternatives Considered

#### Fully separate syntax

Making a wholly-separate syntax added into Marsha: `database`, `query`, and `schema` blocks. The example syntaxes below are vaguely based on the syntax explanations [Postgres uses](https://www.postgresql.org/docs/current/sql-select.html). `{something}` is a value the user needs to provide and `[ something ]` is an optional part of the syntax. Unbracketed/unbraced words are keywords. But outside of the more rigid syntactic areas the example data is just that -- example data.

```md
# database {name} {url or path} [ro | rw]
```

Declares the database to be used, names it, and specifies the URL or path to the sqlite file. Optionally specifies read-only or read-write, defaulting to read-only.

```md
# query {name}(args) using {dbname}

Query description

---
return column 1, return column 2, return column 3
example 1 value 1, example 1 value 2, example 1 value 3
example 2 value 1, example 2 value 2, example 2 value 3
example 3 value 1, example 3 value 2, example 3 value 3
```

Similar to a function, but specifying which database it is connected to with `using {dbname}` and the return type is a custom type defined at the end after a horizontal rule `---`.

```md
# schema {name} for {dbname} {example.csv}
## index [{name} columns] {column name 1, column name 2}
## join {column name} = {schema name}.{column name}
```

```md
# schema {name} for {dbname}
column 1, column 2, column 3
example 1, example 2, example 3
example 4, example 5, example 6
## index [{name} columns] {column name 1, column name 2}
## join {column name} = {schema name}.{column name}
```

```md
# schema {name} for {dbname} using type {type name}
## index [{name} columns] {column name 1, column name 2}
## join {column name} = {schema name}.{column name}
```

Schemas are like `type`s, but with index, join, and dbname directives included. There are three variants for the schema, the first two matching the variants for declaring a type and the third one for just deriving a schema from a type. Similarly there should be a way to derive a type from a schema:

```md
# type {name} from schema {dbname}.{type name}
```

Queries would be used by Marsha functions by the user referencing it in the description (and then the query definition would be included with the function generation, so it would be folded into the function. So different functions don't get slightly different versions of the query, there should be a pre-pass to generate the query that would be embedded into the functions so it is a SQL template given to the LLM)

#### Embedded syntax

Databases would still be declared top-level, and the `# schema` type would also exist, but queries would be in a new `## query` subsection for a function, instead. This would have to be after the description section but before the function examples, so the query would be defined *after* it's "use" in the description, which might be a bit weird, but if defined before the description we would need a `## description` sub-heading to make it clear again.

#### References syntax

Rather than coming up with a special syntax for databases or websites to mock (the [other RFC being worked on in parallel](https://github.com/alantech/marsha/pull/159)), having a references block that has a listing of references [in the Markdown style](https://www.markdownguide.org/basic-syntax/#reference-style-links) that can be referred to by number within the function blocks may provide a similar benefit, though much more reliant on the LLM to make sense of the provided data.

It would still require us to connect to the URL/database/etc, to gather information to provide to the LLM, but it would be a more modular "document generator" mechanism that could be plugin-based where parsing of the markdown in the description section of the function blocks would figure out which document(s) to generate and provide to the LLM during function generation.

This reduces the amount of new syntax dramatically, but eschews any database schema management within Marsha for at least the time being.

Some of the desired functionality, like a shared connection object to reduce connection pressure on the database, could be done with a Marsha function told to memoize generated connection objects and other Marsha functions told to acquire the connection from said function. This places the burden on the user to "do the right thing" but keeps language simplicity and flexibility high. Potentially the user burden can be reduced by developing a standard library for Marsha and/or a module system.

```md
# func get_users(): list of user objects

This function gets all user records from [the database][1] and returns them, or fails if unable to connect to the database.

* get_users() = [{"name": "John Doe", "age": 27, "enjoys_udon": true}, {"name": "Jane Doe", "age": 28, "enjoys_udon": false}]
* get_users() raises database access exception

# references

[1]: ./db.sqlite
```

In this approach, *all* of the logic SQL Pal has to guess what tables are relevant will be necessary (or we make it very expensive with large-context GPT4 calls).

#### Extensions Syntax

LLM abilities probably haven't plateaued yet, and I see value in both the Fully-separated syntax and the References syntax -- which is desired may depend on personal preference more than anything, but needing to choose one over the other may limit the future of Marsha in undesirable ways. At least while we're still figuring out what to leave to the LLM, what to guide the LLM with, and what to explicitly require the user to provide, it could be useful to have users specify which manipulations of the Markdown syntax they even want to have with special interpreter directives defined at the beginning of the file, kinda like [Raku](https://www.raku.org/) or [Racket](https://racket-lang.org/).

We could define a `# using {extension 1} [, {extension 2} ...]` header that *must* be the first if present, specifying what parsing extensions are to be used on the text after it. We could even potentially put *all* functionality behind different extensions to give us the flexibility to make breaking changes to things like functions without breaking any code still using the older function behavior, but that could be a wonky barrier to entry for people who are not programmers.

These extensions ought to register sets of `# something` blocks they manage, and last one wins, then there can be a set of "base" blocks but an extension can replace it. Breaking changes could be tested with such an extension and then if accepted, promoted to base while the current base behavior is demoted into a back-compat extension you can switch back to if desired.

But for something like the `# references` block to make any sense, it also needs to *modify* the behavior of the `# func` block to update it with the documents the function description references, if any. This implies that it may be a better idea for the extensions to decide to either replace or extend the existing behavior of `# something` blocks they have referenced. The current behavior that parses the `# func` blocks into more verbose markdown would then have that extended markdown intercepted by the references extension and modified with the references used by the function.

Similarly, the current type-insertion behavior for `# func` blocks that specify an explicit type could be turned into an extension interception behavior in that way, simplifying and segregating those different concerns, likely with near-zero latency impact (the function could not be parsed by the LLM until the dependent type is turned into a Python `class`, but that is true today. It would just insert an extra function call in the Markdown parse and re-generation, afaict.

This also opens up a couple of interesting features:

1. With a standardized interface like this, user-written extensions, including extensions written in Marsha to modify Marsha, become possible since Python is an interpreted language so dynamic loading of these extensions should be doable.
2. The "base" and set of extensions could be derived from the target language, with Python just being the default. We could have different bases for different target languages using the same general "framework".
3. Embedded code for the target language could just be another extension, for instance a `# def my_python_function(argname):` could be the beginning of manually-written Python to include in the output automatically.
4. Modules/standard lib/etc could just be another extension that defines a block that then introduces the block(s) actually desired.
5. The five "layers" of Marsha compilation: `.mrsh` parse, code and test generation, syntactic validation, test-and-iterate, and final formatting + helper code writing, could themselves be turned into extensions. This would imply an event-based structure where extensions listen to some events and emit to others, extension selection determines which event listeners are registered, and compiler flags determine which event gets initial data pushed and which event is treated as the "done" event.

That last concept has an interesting implication for parallelization and true multicore usage, and where the LLM calls themselves could just be another event consumer and emitter and switching from, say, OpenAI to llama.cpp + LLaMa 2, could be made trivial. It also would make `# using` extensions and compiler configuration equivalent: any compiler flag ought to have an equivalent `# using` directive. So you could decide to make using compiler flags instead of `# using` directives a lint error, or you could decide to have a Marsha-based environment that uses extensions you provide that the user does not need to be aware of.

## Expected Semver Impact

Minor update if post-1.0.0

## Affected Components

TODO

## Expected Timeline

TODO
