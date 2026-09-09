n_jobs ?= 1
provider ?= openai
runs ?= 30
PREFIX ?= $(HOME)/.local

./venv:
	uv venv venv
	uv pip install --python ./venv/bin/python -r requirements.txt
	uv pip install --python ./venv/bin/python --upgrade .

.PHONY: clean
clean:
	git clean -ffdx -e .env

.PHONY: install
install: ./venv
	uv pip install --python ./venv/bin/python --upgrade .
	./venv/bin/python ./make_launcher.py "$(PREFIX)/bin" "$(CURDIR)/venv"

.PHONY: uninstall
uninstall:
	rm -f $(PREFIX)/bin/marsha

.PHONY: format
format:
	./venv/bin/autopep8 -i marsha/*.py marsha/personas/*.py

.PHONY: test
test: ./venv
	./venv/bin/python -m pytest tests/

.PHONY: time
time: ./venv .time.py
	uv pip install --python ./venv/bin/python --upgrade .; ./venv/bin/python ./.time.py $(test) $(attempts) $(n_parallel_executions) $(stats) --n_jobs $(n_jobs) --provider $(provider) --runs $(runs)
