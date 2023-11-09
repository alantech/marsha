# Simple REST servers

- Let's create a virtual environment

```bash
python -m venv venv
```

- Install `marsha` via pip

```bash
pip install git+https://github.com/alantech/marsha
```

- Set your OpenAI key

```bash
export OPENAI_SECRET_KEY=sk-...
```

- Define the marsha file. The various function names become `/func_name` endpoints that you can POST to and get a response body back. If the function do not have parameters then you can use GET for that `/func_name` endpoint.

Let's say the following file is saved in `todos.mrsh`:

```md
# type task
name, status
cooking, pending
dishes, completed
cleaning, pending

# func save_task(task name): task dict

This function receives a task name and creates a `task` object with it. The initial status for all tasks is `pending`. The value is saved in a global dictionary.

* save_task() = throws an error
* save_task('test') = task('test', 'pending')

# func get(dictionary with name property): task dict

This function gets the requested task name from the global dictionary and return the task object.

* get({'name': 'cooking'}) = task('cooking', 'pending')
* get({'name': 'dishes'}) = task('dishes', 'completed')
* get() = throws an error

# func add(dictionary with name property): task dict

This function calls the `save_task` function and take the task with the requested task name.

* add({'name': 'cooking'}) = task('cooking', 'pending')
* add({'name': 'dishes'}) = task('dishes', 'pending')
* add() = throws an error
```

- Execute marsha to generate working code

```bash
python -m marsha todos.mrsh
```

- Run generated script as web server in the desired port. If any of your functions takes multiple arguments, it **must** be called in JSON mode with the arguments each being an element of a top-level array.

```bash
python -m todos --serve 8088
```

- Use your web server. If you set the `Content-Type` header to `application/json` the input and output will be JSON, if not it will be plain text.

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"name": "dishes"}' localhost:8088/add
```

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"name": "dishes"}' localhost:8088/get
```
