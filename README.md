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

The Marsha compiler can be used to compile the syntax using a `pip` module via a terminal or Jupyter Notebook:

```sh
pip install git+https://github.com/alantech/marsha
python -m marsha data_mangling.mrsh
```

## Syntax

The Marsha syntax looks a lot like markdown and is a mixture of English and mathematical notation. It has its own file format `.mrsh` that houses function definition(s). The syntax is subject to change as Marsha is currently in an alpha state. If you have a legitimate use case for Marsha, please let us know.

### Data Types

Data types provide function type safety which helps improve the accuracy of the code generation. The data type format is almost identical to the CSV format.

```md
# type EmployeeSkills
name, skill
Bob,	math
Jake,	spreadsheets
Lisa,	coding
Sue,	spreadsheets
```

It is also possible for Marsha to infer the data type from CSV file

```md
# type EmployeesByDepartment employees_by_department.csv
```

### Functions

Functions are the bread and butter of Marsha and can easily define transformations between different data types. They are defined by a name, input, output (if not void), a description of the requirements, and a list of samples. The examples are imperative to be able to generate the test suite for the generated software which helps ensure accuracy. Function samples should look a lot like the requirements given to a software engineer in English, but using shorthand mathematical notation.

```md
# func get_employee_skills(list of EmployeesByDepartment, list of DepartmentSkills): list of EmployeeSkills

This function receives a list of EmployeesByDepartment and a list of DepartmentSkills. The function should be able to create a response of EmployeeSkills merging the 2 list by department. Use the pandas library.

* get_employee_skills() = throws an error
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting')]) = throws an error
* get_employee_skills([], []) = []
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting')], []) = []
* get_employee_skills([], [DepartmentSkills('Accounting', 'math')]) = []
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting')], [DepartmentSkills('Accounting', 'math')]) = [EmployeeSkills('Joe', 'math')]
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting'), EmployeesByDepartment('Jake', 'Engineering')], [DepartmentSkills('Accounting', 'math')]) = [EmployeeSkills('Joe', 'math')]
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting'), EmployeesByDepartment('Jake', 'Engineering')], [DepartmentSkills('Accounting', 'math'), DepartmentSkills('Engineering', 'coding')]) = [EmployeeSkills('Joe', 'math'), EmployeeSkills('Jake', 'coding')]
```

### Goals

The Marsha syntax is meant to be:
- minimal and "obvious", but also discourage lax or incomplete information that could lead to unpredictable behavior
- be mechanically parseable for syntax highlighting and quick feedback on correctness issues to the user
- make it easy to define examples to reduce the probability of generating faulty code and allow generating tests that the application code can be tested against

## Compiler

Marsha is compiled by an LLM into tested software that meets the requirements described, but implication details can vary greatly across runs much like if different developers implemented it for you. There is typically more than one way to write software that fulfills a set of requirements. However, the compiler is best-effort and sometimes it will fail to generate the described program. We aim for 80%+ accuracy on our [examples](./examples/test/). In general, the more detailed the description and the more examples are provided the more likely the output will work.

## Roadmap

- Improve average accuracy for our test bed above 90%
- Support for endpoints, visualizations, and data storage
- Syntax highlighting
- Support for different types of LLM
- Being able to bootstrap the Marsha compiler with a Marsha program
- More target languages other than Python