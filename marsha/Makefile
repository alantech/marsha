./dist/marsha: ./venv ./*.py ./marsha.spec
	. ./venv/bin/activate; pip install -r requirements.txt
	. ./venv/bin/activate; pyinstaller __main__.py --name marsha --onefile --collect-all pyflakes --collect-all mccabe --collect-all pycodestyle --collect-all pydocstyle --collect-all pylama --add-data ../examples:./examples

./venv:
	(command -v $(python) && $(python) -m venv venv) || (command -v python && python -m venv venv) || (command -v python3 && python3 -m venv venv)

.PHONY: clean
clean:
	git clean -ffdx -e .env

.PHONY: install
install: ./dist/marsha
	cp ./dist/marsha /usr/local/bin/marsha

.PHONY: format
format:
	. ./venv/bin/activate; autopep8 -i *.py

.PHONY: time
time: ./dist/marsha .time.py
	. ./venv/bin/activate; pip install --upgrade ..; ./.time.py $(test) $(attempts) $(n_parallel_executions) $(stats)

.PHONY: time-win
time: .\dist\marsha .time.py
	. .\venv\bin\activate; pip install --upgrade ..; .\.time.py $(test) $(attempts) $(n_parallel_executions) $(stats)
