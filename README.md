# Marsha AI Language

[![Discord Follow](https://dcbadge.vercel.app/api/server/p5BTaWAdjm?style=flat)](https://discord.gg/p5BTaWAdjm)
[![GitHub Repo Stars](https://img.shields.io/github/stars/alantech/marsha?style=social)](https://github.com/alantech/marsha)

<p align="center"><b>Describe Logic ⴲ Provide Examples ⴲ Run Reliably</b></p>

```md
# func duckduckgo(search text): top three link names and URLs separated by newline

This function executes a search using duckduckgo.com and returns the top three links (excluding ads) in the following format:

First link name: https://first.link/path
Second link name: https://www.secondlink.com/path
Third link name: https://thirdlink.org/path

* duckduckgo('search engine') = '21 Great Search Engines You Can Use Instead Of Google: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwivx6SS1qeAAxU5lWoFHbuqA4sQFnoECBEQAQ&url=https%3A%2F%2Fwww.searchenginejournal.com%2Falternative-search-engines%2F271409%2F&usg=AOvVaw1MhHGUxrHf8AkmiU64AotH&opi=89978449\nThe Top 11 Search Engines, Ranked by Popularity: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwivx6SS1qeAAxU5lWoFHbuqA4sQFnoECA8QAQ&url=https%3A%2F%2Fblog.hubspot.com%2Fmarketing%2Ftop-search-engines&usg=AOvVaw30ykZ9Ftz51L4pQTaMsmpQ&opi=89978449\nSearch engine - Wikipedia: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwivx6SS1qeAAxU5lWoFHbuqA4sQFnoECCsQAQ&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSearch_engine&usg=AOvVaw2JG-HuD9odcoxnHHkUd3sl&opi=89978449'
* duckduckgo(3) raises an exception
```

```sh
$ marsha duckduckgo.mrsh
Compiling functions for duckduckgo...
Generating Python code...
Chat query took 17sec 404.074ms, started at 7.32827ms, ms/chars = 9.494857455106361
Chat query took 26sec 404.137ms, started at 2.94709ms, ms/chars = 11.080208717150894
Writing generated code to temporary files...
Writing generated code to temporary files...
Running tasks in parallel...
Parsing generated code...
Verifying and correcting generated code...
Creating virtual environment...
Parsing generated code...
Verifying and correcting generated code...
Creating virtual environment...
Installing requirements...
Installing requirements...
Chat query took 33sec 516.078ms, started at 29sec 944.337ms, ms/chars = 11.6699436746932
Installing requirements...
Chat query took 35sec 580.238ms, started at 30sec 165.022ms, ms/chars = 13.202314606845269
Installing requirements...
Chat query took 39sec 352.607ms, started at  1min  6sec, ms/chars = 18.58885568664724
Installing requirements...
Formatting code...
Task completed successfully. Cancelling pending tasks...
Writing generated code to files...
duckduckgo done! Total time elapsed:  1min 46sec. Total cost: 0.57.
$ python -m duckduckgo "Hello, World"

                  en.wikipedia.org/wiki/"Hello,_World!"_program
                  : https://en.wikipedia.org/wiki/%22Hello,_World!%22_program

                  www.hackerrank.com/blog/the-history-of-hello-world/
                  : https://www.hackerrank.com/blog/the-history-of-hello-world/

                  learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/tutorials/hello-world
                  : https://learn.microsoft.com/en-us/dotnet/csharp/tour-of-csharp/tutorials/hello-world
```

Marsha is an LLM-based programming language. Describe what you want done with a simple syntax, provide examples of usage, and the Marsha compiler will guide an LLM to produce tested Python software.

## Usage

The Marsha compiler can be used to compile the syntax using a `pip` module via a terminal or Jupyter Notebook:

```bash
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

Functions are the bread and butter of Marsha and can easily define transformations between different data types. There are three sections to a Marsha function: the declaration, the description, and the examples.

The declaration is a Markdown section prefixed with `func`, then followed by a name, parenthesis containing the input type(s), and finally a colon followed by the output type. The name must be a single word, but the types don't need to be classic software types, or even the explicit data types defined above. They can themselves be simple descriptions of what the type is meant to be. Eg,

```md
# func get_employee_skills(list of EmployeesByDepartment, list of DepartmentSkills): list of EmployeeSkills
```

The next section is the description of the function. Here you explain what the function should do. Being more explicit here will reduce variability in the generated output and improve reliability in behavior, but it's up to you just how explicit you will be and how much you leave to the LLM to figure out. This is similar to declarative languages like SQL and HTML where there are defaults for things you do not specify. Eg,

```md
This function receives a list of EmployeesByDepartment and a list of DepartmentSkills. The function should be able to create a response of EmployeeSkills merging the 2 list by department. Use the pandas library.
```

The final section is the example section. Here you provide examples of calling the function and what its output should be. Marsha uses this to provide more information to the LLM to generate the logic you want, but also uses it to generate a test suite to validate that what it has generated actually does what you want it to. This feedback loop makes Marsha more reliable than directly using the LLM itself. In some ways, this is similar to Constraint-based programming languages where you validate and verify the behavior of your function in the definition of the function itself, but it is also less stringent than those, allowing incomplete constraints where constraint-based languages will fail to compile in the face of that ambiguity. Eg,

```md
* get_employee_skills() = throws an error
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting')]) = throws an error
* get_employee_skills([], []) = []
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting')], []) = []
* get_employee_skills([], [DepartmentSkills('Accounting', 'math')]) = []
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting')], [DepartmentSkills('Accounting', 'math')]) = [EmployeeSkills('Joe', 'math')]
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting'), EmployeesByDepartment('Jake', 'Engineering')], [DepartmentSkills('Accounting', 'math')]) = [EmployeeSkills('Joe', 'math')]
* get_employee_skills([EmployeesByDepartment('Joe', 'Accounting'), EmployeesByDepartment('Jake', 'Engineering')], [DepartmentSkills('Accounting', 'math'), DepartmentSkills('Engineering', 'coding')]) = [EmployeeSkills('Joe', 'math'), EmployeeSkills('Jake', 'coding')]
```

Altogether this produces:

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
- Support for visualizations and data storage (geek mode: handle side-effect logic better in general)
- Syntax highlighting (vim, vscode, etc)
- Support for different types of LLM
- Bootstrap the Marsha compiler with a Marsha program
- More target languages other than Python
- A module system
- Edits to Marsha mutating existing Python code instead of regenerating
- "Decompiler" from source code into Marsha syntax
- "Debugger" meta mode to take existing Marsha definition and an example of an unexpected failure and recommend what to update with the Marsha definition.
- Optmization "levels" (spend more time on more iterations with the LLM improving performance, security, etc)
- Marsha GUI mode: visual editor baked into the compiler (eventually with the decompiler/debugger/etc features), and able to generate a GUI wrapper for generated code, enabling end-to-end non-terminal usage
- Better support for a mixed environment (Marsha functions can be used by Python, but how to get Marsha to use hand-written Python functions)
- Better "web scraping" behavior (LLM likes to assume the internet still looks like it did in November 2021, but HTML structure has often changed for the largest websites; automatically correcting that assumption would be nice)