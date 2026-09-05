n_jobs ?= 1
PREFIX ?= $(HOME)/.local

./venv:
	uv venv venv
	uv pip install --python ./venv/bin/python -r requirements.txt
	uv pip install --python ./venv/bin/python --upgrade ..

.PHONY: clean
clean:
	git clean -ffdx -e .env

.PHONY: install
install: ./venv
	uv pip install --python ./venv/bin/python --upgrade ..
	mkdir -p $(PREFIX)/bin
	sed 's|__MARSHA_VENV__|$(CURDIR)/venv|' ./marsha.sh.in > $(PREFIX)/bin/marsha
	chmod 755 $(PREFIX)/bin/marsha
	$(PREFIX)/bin/marsha --help > /dev/null
	@echo "installed marsha to $(PREFIX)/bin/marsha (venv: $(CURDIR)/venv)"
	@case ":$$PATH:" in *":$(PREFIX)/bin:"*) ;; *) echo "note: $(PREFIX)/bin is not in your PATH" ;; esac

.PHONY: uninstall
uninstall:
	rm -f $(PREFIX)/bin/marsha

.PHONY: format
format:
	./venv/bin/autopep8 -i *.py

.PHONY: time
time: ./venv .time.py
	uv pip install --python ./venv/bin/python --upgrade ..; ./venv/bin/python ./.time.py $(test) $(attempts) $(n_parallel_executions) $(stats) --n_jobs $(n_jobs)
