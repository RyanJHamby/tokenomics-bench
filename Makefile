.PHONY: install test lint
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check . && ruff format --check src tests
