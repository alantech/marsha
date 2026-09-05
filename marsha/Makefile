n_jobs ?= 1

./dist/marsha: ./venv ./*.py ./mappers/*.py ./marsha.spec requirements.txt
	uv pip install --python ./venv/bin/python -r requirements.txt
	./venv/bin/pyinstaller __main__.py --name marsha --onefile --collect-all pyflakes --collect-all mccabe --collect-all pycodestyle --collect-all pydocstyle --collect-all pylama --add-data ../examples:./examples --add-data ./helper.py:./marsha

./venv:
	uv venv venv

.PHONY: clean
clean:
	git clean -ffdx -e .env

.PHONY: install
install: ./dist/marsha
	cp ./dist/marsha /usr/local/bin/marsha

.PHONY: format
format:
	./venv/bin/autopep8 -i *.py

.PHONY: time
time: ./dist/marsha .time.py
	uv pip install --python ./venv/bin/python --upgrade ..; ./venv/bin/python ./.time.py $(test) $(attempts) $(n_parallel_executions) $(stats) --n_jobs $(n_jobs)
