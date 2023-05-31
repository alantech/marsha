# Marsha AI Lang

Marsha is a higher-level programming language. Samples of it should look a lot like the requirements given to a software engineer in English. The syntax is compiled by an LLM into software that meets the requirements described, but implication details can vary greatly. As such the software generated is not meant to be deterministic, but the more details and examples provided the more likely the output will be deterministic.

The Marsha syntax should be:
- minimal, "obvious", but also discourage lax or incomplete information that could lead to unpredictable behavior
- be mechanically parseable for syntax highlighting and quick feedback on correctness issues to the user
- make it easy to define examples to reduce the probability of generating faulty code and allow generating tests that the application code can be tested against

For now, only function and data structure definitions have been defined which allow for data mangling scripts to be created. The syntax is subject to change as Marsha is currently in an alpha state. What other elements to create will depend on the initial target audience and use case.