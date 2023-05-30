# 001 - Marsha Syntax

## Current Status

### Proposed

2023-05-25

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

Marsha as a higher-level language is going to need a syntax. This syntax should be:
- minimal, "obvious", and discourage lax or incomplete information that could lead to unpredictable behavior
- be mechanically parseable for syntax highlighting and quick feedback on correctness issues to the user
- make it easy to define examples and set different tolerance levels that will fail to compile if not enough examples are provided to reduce the probability of generating faulty code and provide the foundation for a test harness/suite itself

For now, only function and data structure definitions are being considered. What other elements to create will depend on the initial target audience we intend to tackle, and since we're still debating that, we'll update this document once we're ready.

Following the meta RFC, the functions will be a mixture of declarative and constraint style function definitions, and will have the following five parts: the function name, input arguments, return type, description, and examples of its usage. These should be enough for the LLM to do a pretty solid job at writing the actual code and the tests cases for us.

## Proposal

To be determined

### Data Syntax Alternatives Considered

#### Pure CSV Syntax with Types Row

```md
# type SKU
brand_id, sku_id, color, type
integer, integer, string that is a valid color name, string that is name of article of clothing
20, 10040, 'red', 'shirt'
50, 10059, 'blue', 'shirt'
```

#### Pure CSV Syntax with Optional Types

```md
# type SKU
brand_id: integer, sku_id: integer, color: string that is a valid color name, type: string that is name of article of clothing
20, 10040, 'red', 'shirt'
50, 10059, 'blue', 'shirt'

sku_from_new_brands(SKU[sku_id=10040])
```

#### Numbered Constructor Syntax with Optional Types

```md
# type SKU(brand_id: integer, sku_id: integer, color: string that is a valid color name, type: string that is name of article of clothing)
1. SKU(20, 10040, 'red', 'shirt')
2. SKU(50, 10059, 'blue', 'shirt')
```

All the numbered syntax examples can be referenced in the function examples using `#`:

```md
# sku_from_new_brand(SKU[], Brand[]) = SKU[]
<description>
- sku_from_new_brand([SKU#1], ...) = SKU#1
```

#### Numbered CSV-like Syntax with Optional Types

```md
# type SKU
brand_id: integer, sku_id: integer, color: string that is a valid color name, type: string that is name of article of clothing
1. 20, 10040, 'red', 'shirt'
2. 50, 10059, 'blue', 'shirt'
```

#### Numbered, CSV-like Syntax with Optional Types

```md
# type SKU(brand_id: integer, sku_id: integer, color: string that is a valid color name, type: string that is name of article of clothing)
1. 20, 10040, 'red', 'shirt'
2. 50, 10059, 'blue', 'shirt'
```

#### Numbered, TS-like Syntax with Optional Types

```md
# type SKU(brand_id: integer, sku_id: integer, color: string that is a valid color name, type: string that is name of article of clothing)
1. SKU { 20, 10040, 'red', 'shirt' }
2. SKU {50, 10059, 'blue', 'shirt' }
```

### Function Syntax Alternatives Considered

#### Markdown-like Syntax

A markdown-based syntax has several advantages in that you can just render the markdown and get something easily readable in a [literate programming](https://en.wikipedia.org/wiki/Literate_programming) style. It also provides some hints about blocks and ordering without needing to explicitly close blocks with curly braces or indent blocks like Python.

There are a few possibilities here on how to structure it, more verbosely and English-like, and more compact and math-like. It should be trivial to transform one into the other so we *could* also decide to support both styles, but it's probably best to choose one, or some minor blend of the two for clarity.

##### Verbose-style

```md
# Function fibonacci

## Inputs

1. n is an integer

## Output

An integer within the fibonacci set

## Description

This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)

## Examples

* fibonacci(1) = 1
* fibonacci(2) = 1
* fibonacci(3) = 2
* fibonacci(0) throws an error
```

##### Math-style

```md
# fibonacci(n: int): int

This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)

* fibonacci(1) = 1
* fibonacci(2) = 1
* fibonacci(3) = 2
* fibonacci(0) throws an error
```

The math-like form compacts the first three sections together and doesn't need to specify that it's a function since it follows a function syntax. Further, the examples at the end don't need an explicit subsection in this form because it is unambiguous that the first part without anything is the description and the bullet-point list are the examples. The more verbose form uses an enumerated list for the input arguments so argument order is explicit. The math-like form following a Typescript-like type system probably makes the most sense, but could also go with C style `int fibonacci(int n)` if we wanted to. It may even be the case that the LLM will be "fine" with either of them, or spelling out `integer` or `32-bit integer` and not have a problem to translate that into the target language of choice. Being explicit here may not actually be necessary, but if/when we have more than just functions, that may force us to choose something precise here so we can distinguish function blocks from other blocks. The verbose version replaces `:` with `is a(n)` so `n: int` became `n is an integer`.

#### "Notepad"/"Word" syntax

Markdown syntax does still require some syntax to learn, even if it is minor. A "pure english text" approach is another alternative, with the only "syntax" being a colon.

```
Function: fibonacci
Inputs: n is an integer
Output: An integer within the fibonacci set
Description: This function calculates the nth fibonacci number, where n is provided to it and starts with 1.
fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)
Examples:
fibonacci(1) = 1
fibonacci(2) = 1
fibonacci(3) = 2
fibonacci(0) throws an error
End Function
```

This syntax feels like it needs an explicit "End Function" declaration. It *probably* could be optional, but it might be easier on humans reading it to be there. In this syntax `Function: `, `Inputs: `, `Output: `, `Description: `, `Examples: `, and `End Function` are keywords if and only if they are at the very beginning of the line. They must also be in the listed order so there is some syntax highlighting and error reporting possible. The `Inputs` and `Examples` sections expect each input or example to be on a separate line. You can immediately use the same line that the keyword is located on, or you can start on the next line, whichever is easier. This syntax can/should ignore blank lines and it can consider the remainder of the line after the keyword as a blank line to ignore, so you can also "format" the example for greater clarity:

```
Function: fibonacci

Inputs: n is an integer

Output: An integer within the fibonacci set

Description:

This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)

Examples:

fibonacci(1) = 1
fibonacci(2) = 1
fibonacci(3) = 2

fibonacci(0) throws an error

End Function
```

This "syntax" has the advantage of incredibly bare-bones plain text and could reasonably be written by someone used to using Microsoft Word, for instance, which may have some value. It can still be syntax highlighted with a custom highlighter, but is still legible without one.

#### XML-based syntax

Tons of people were able to make websites back in the late 90s. The HTML syntax was simple enough at the time that hand editing it was okay, and people at least tolerated the verbosity. XML is the generalization and simplification of that HTML syntax, and so it could similarly be used here.

```xml
<function name="fibonacci">
  <inputs>
    <input name="n" type="integer" />
  </inputs>
  <output type="an integer within the fibonacci set" />
  <description>
    This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

    fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)
  </description>
  <examples>
    <e>fibonacci(1) = 1</e>
    <e>fibonacci(2) = 1</e>
    <e>fibonacci(3) = 2</e>
    <e>fibonacci(0) throws an error</e>
  </examples>
</function>
```

This syntax is super-easy to parse, syntax highlight, people have in the past tolerated it, and there's no ambiguity issue as the language is extended to handle more than just functions, but it is the most verbose option possible, and you probably need an editor to help you write it correctly. There's also the ambiguity surrounding properties on tags versus nested tags. Eg, could `<function name="fibonacci">` also be written `<function><name>fibonacci</name>`? Why or why not?

#### JSON/YAML/TOML syntaxes

The XML-based syntax above shows that we could similarly format this with any other data interchange format. Not going to make examples for these right now because I think they're a bad idea (and I'm low on time before the next meeting), but definitely possible.

#### SQL-inspired syntax

One of the most successful languages for non-developers is SQL, and many of the "software engineer adjacent" roles that we think this language could lower the barrier to entry for often know SQL, so we could use that as the base of our language.

```
FUNCTION fibonacci (n int) RETURNS an integer within the fibonacci set
DESCRIPTION
  This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

  fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)
EXAMPLES (
  fibonacci(1) = 1,
  fibonacci(2) = 1,
  fibonacci(3) = 2,
  fibonacci(0) throws an error
);
```

Examples become comma-delimited within parentheses, and the keywords `FUNCTION`, `RETURNS`, `DESCRIPTION`, and `EXAMPLES` must be provided in order, and presumably are case-insensitive. The indentations are trimmed from the text and are ignored by the language. This does mean that if you need to use the word `EXAMPLES` inside of your `DESCRIPTION` block, there would need to be some special escape mechanism (similarly in the function name, input arguments, and return type blocks, but less likely to occur there). We could similarly follow the SQL standard here and require single quotes around these keywords when used as not-a-keyword, like `'examples'`, but it may not be immediately obvious to someone new when they read it why that is the case.

#### C-inspired syntax

Including it anyways, because it should totally work, though I don't know if this is the best of ideas


```c
function fibonacci(int n) {
  /**
   * This function calculates the nth fibonacci number, where n is provided to it and starts with 1.
   * 
   * fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)
   **/
  assert fibonacci(1) = 1;
  assert fibonacci(2) = 1;
  assert fibonacci(3) = 2;
  assert fibonacci(0) throws an error;
  return an integer within the fibonacci set;
}
```

This looks more "code-like" and has the keywords `function`, `assert`, and `return`, using a "flowerbox" style comment section for the description (but would probably also support non-flowerbox style and lines of one-liner `//` comments, too), and uses curly braces for block scope and semi-colons for statement separators. It would be more familiar for developers, but also anyone who has taken at least an intro to programming course, so that may still be fine?

## Expected Semver Impact

If we were at a 1.0.0+ version (somehow before anything exists) this would be a major version bump ;)

## Affected Components

Everything

## Expected Timeline

An RFC proposal should define the set of work that needs to be done, in what order, and with an expected level of effort and turnaround time necessary. *No* multi-stage work proposal should leave the engine in a non-functioning state.
