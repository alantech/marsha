./dist/marsha: ./venv ./marsha.py
	. ./venv/bin/activate; pip install -r requirements.txt
	. ./venv/bin/activate; pyinstaller marsha.py --onefile

./venv:
	(command -v python && python -m venv venv) || (command -v python3 && python3 -m venv venv)

.PHONY: clean
clean:
	git clean -ffdx -e .env

.PHONY: install
install: ./dist/marsha
	cp ./dist/marsha /usr/local/bin/marsha