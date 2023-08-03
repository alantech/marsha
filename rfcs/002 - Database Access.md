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
example 3 value 1, example 3 value 2, example 3, value 3
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

#### Something else?

TODO

## Expected Semver Impact

Minor update if post-1.0.0

## Affected Components

TODO

## Expected Timeline

TODO
