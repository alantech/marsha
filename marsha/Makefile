./dist/marsha: ./venv ./*.py ./marsha.spec ./bin/llamacpp
	. ./venv/bin/activate; pip install -r requirements.txt
	. ./venv/bin/activate; pyinstaller __main__.py --name marsha --onefile --collect-all pyflakes --collect-all mccabe --collect-all pycodestyle --collect-all pydocstyle --collect-all pylama --add-data ../examples:./examples --add-data ./helper.py:./marsha --add-data ./bin/llamacpp:./marsha/bin

./venv:
	(command -v $(python) && $(python) -m venv venv) || (command -v python && python -m venv venv) || (command -v python3 && python3 -m venv venv)

NVCC := $(shell command -v nvcc 2> /dev/null)

ifndef NVCC
./bin/llamacpp:
	mkdir -p ./bin
	mkdir -p /tmp/__marsha_setup__
	cd /tmp/__marsha_setup__ && git clone https://github.com/ggerganov/llama.cpp.git
	cd /tmp/__marsha_setup__/llama.cpp && make
	cp /tmp/__marsha_setup__/llama.cpp/main ./bin/llamacpp
	rm -rf /tmp/__marsha_setup__
else
./bin/llamacpp:
	mkdir -p ./bin
	mkdir -p /tmp/__marsha_setup__
	cd /tmp/__marsha_setup__ && git clone https://github.com/ggerganov/llama.cpp.git
	cd /tmp/__marsha_setup__/llama.cpp && make LLAMA_CUBLAS=1
	cp /tmp/__marsha_setup__/llama.cpp/main ./bin/llamacpp
	rm -rf /tmp/__marsha_setup__
endif

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
