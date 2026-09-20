n_jobs ?= 1
provider ?= openai
runs ?= 30
PREFIX ?= $(HOME)/.local

./venv:
	uv venv venv
	uv pip install --python ./venv/bin/python --upgrade .

.PHONY: clean
clean:
	git clean -ffdx -e .env

.PHONY: install
install: ./venv uninstall
	# Fresh install from a clean slate. `uv pip install --upgrade .` rebuilds the local wheel
	# in-tree, but setuptools' build_py copies the package into build/lib/ without pruning, so a
	# renamed or removed file lingers there and gets re-baked into the next wheel (and re-installed)
	# even after a site-packages wipe. Clear the in-tree build artifacts so the wheel is built from
	# the current source; `uninstall` (a prerequisite above) drops the prior install first.
	rm -rf build/ dist/ *.egg-info
	uv pip install --python ./venv/bin/python --upgrade .
	./venv/bin/python ./make_launcher.py "$(PREFIX)/bin" "$(CURDIR)/venv"

.PHONY: uninstall
uninstall:
	rm -rf "$(CURDIR)"/venv/lib*/python*/site-packages/marsha*
	rm -f "$(PREFIX)/bin/marsha"

.PHONY: format
format:
	./venv/bin/autopep8 -i marsha/*.py marsha/personas/*.py

.PHONY: test
test: ./venv
	./venv/bin/python -m pytest tests/

.PHONY: time
time: ./venv .time.py
	uv pip install --python ./venv/bin/python --upgrade .; ./venv/bin/python ./.time.py $(test) $(attempts) $(n_parallel_executions) $(stats) --n_jobs $(n_jobs) --provider $(provider) --runs $(runs)
