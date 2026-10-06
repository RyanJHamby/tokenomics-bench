.PHONY: install test lint demo estimate configs
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

CONFIGS = b1_capacity_pilot b2_power_clock.template b3a_fp8_capacity b3a_extra_capacity b3b_quant_fixed_load b4_graph_modes b5_overload_recovery b6_moe.template b7_sglang_crosscheck

estimate:  ## GPU-hours and dollars per config; PRICE=<live usd/hr>; capacity is a stand-in here
	@for f in $(CONFIGS); do \
		printf "%-26s" $$f; python -m tokbench.runner configs/$$f.yaml --usd-per-hr $(PRICE) \
			--out /dev/null --dry-run --assume-capacity 10 | head -1; \
	done

configs:  ## regenerate configs/*.yaml from tokbench/configgen.py (never hand-edit them)
	python -m tokbench.configgen
