.PHONY: format lint test play

format:
	python -m py_compile $(shell git ls-files '*.py')

lint:
	python -m py_compile $(shell git ls-files '*.py')

test:
	pytest -q

play:
	python play.py --white model --black search --max-moves 20 --depth 3 --time 0.5

