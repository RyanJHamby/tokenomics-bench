.PHONY: install test lint demo estimate
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

estimate:  ## GPU-hours and dollars for every real config; set PRICE=<usd/hr> from live rates
	@for f in b1_knee b2_prefix_cache b3_quantization b4_cuda_graphs b5_power_caps; do \
		printf "%-18s" $$f; python -m tokbench.runner configs/$$f.yaml --usd-per-hr $(PRICE) --out /dev/null --dry-run; \
	done
