.PHONY: install test lint demo
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check . && ruff format --check src tests

demo:  ## synthetic end-to-end run on the mock server; output is NOT a result
	python -m tokbench.runner configs/demo_synthetic.yaml --server mock --gpu fake \
		--usd-per-hr 1.0 --out demo_out --quiet-server
	python -m tokbench.report demo_out --png demo_out/frontier.png --ttft-slo 0.5
