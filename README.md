&nbsp;

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="logo.png">
    <img height="250" />
  </picture>
</p>

[![Discord Follow](https://dcbadge.vercel.app/api/server/p5BTaWAdjm?style=flat)](https://discord.gg/p5BTaWAdjm)
[![GitHub Repo Stars](https://img.shields.io/github/stars/alantech/marsha?style=social)](https://github.com/alantech/marsha)

Marsha is a functional, higher-level, English-based programming language that gets compiled into tested Python software by an LLM. This repository contains an LLM-based compiler that implicitly defines the Marsha syntax.

## Usage

The Marsha compiler can be invoked via a CLI or `pip` module that can also be invoked in Jupyter notebooks.

<> TODO Demo

<> TODO some instructions

## Syntax

The Marsha syntax is meant to be:
- minimal and "obvious", but also discourage lax or incomplete information that could lead to unpredictable behavior
- be mechanically parseable for syntax highlighting and quick feedback on correctness issues to the user
- make it easy to define examples to reduce the probability of generating faulty code and allow generating tests that the application code can be tested against

The Marsha syntax looks a lot like markdown and is a mixture of English and mathematical notation. It has its own file format `.mrsh` and is used to define function(s) by providing an input, output (if not void), and the requirements described. The syntax is subject to change as Marsha is currently in an alpha state. If you have a legitimate use case for Marsha, please let us know.

<> Subsections for different blocks

## Compiler

Marsha is compiled by an LLM into tested software that meets the requirements described, but implication details can vary greatly across runs much like if different developers implemented it for you. There is typically more than one way to write software that fulfills a set of requirements. However, the compiler is best-effort and sometimes it will fail to generate the described program. We aim for 80%+ accuracy on our [examples](./examples/test/). In general, the more detailed the description and the more examples are provided the more likely the output will work.

## Roadmap

- support for endpoints, visualizations and data storage
- syntax highlighting
- support for Llama CP, currently 
- being able to bootstrap the Marsha compiler with a Marsha program
- more target languages other than Python