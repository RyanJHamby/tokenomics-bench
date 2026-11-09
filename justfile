set shell := ["bash", "-c"]

# synthetic end-to-end run on the mock server; output is NOT a result
demo:
	make demo

# create the venv's editable install (pip install -e ".[dev]")
install:
	make install

# mock-server and fake-GPU tests
test:
	make test

# lint with ruff
lint:
	make lint

# what CI runs: lint then test
check:
	make check

# GPU-hours and dollars per config; usage: just estimate 1.0 (live USD per hour)
estimate price:
	make estimate PRICE={{price}}
