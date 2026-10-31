---
id: bench-readiness-report
goal: "`python -m tokbench.readiness --price <usd/hr>` prints a no-GPU pre-rental report (per-block GPU-hours and dollars, padded total vs remaining budget, config drift, unpinned model revisions, exact vLLM pin) and exits 0 only when everything is ready."
acceptance:
  - python -m pytest tests/test_readiness.py --runxfail -q
  - python -m pytest tests/test_readiness.py -q
scope:
  - src/tokbench/readiness.py
  - tests/test_readiness.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines once the work is done (the second acceptance command fails while the strict marker remains)
  - do not touch docs/PREREG-v2.md, docs/PREREG.md, docs/PREREG-v2-amendments.md, hypotheses, margins or budget/budget.yaml
  - do not rent, contact, or call any provider or GPU; this is pure computation over files in the repo
  - no new dependencies (pyyaml is already used)
  - do not edit existing modules (budget.py, config.py, configgen.py, runner.py); import and reuse them
  - do not edit Makefile or docs in this spec
risk_tier: 2
max_runs: 3
autonomy: inherit
status: ready
issue: 2
---

## Why
RUNBOOK "Before renting" asks the owner to check live price, estimate, remaining budget and drift by hand across several
commands. One command should answer "is it safe to start the meter?" without a GPU.

## What to build (`src/tokbench/readiness.py`)
`report(price: float, root: Path = Path(".")) -> dict` and `main(argv) -> int` (`--price` required, `--root`, `--json`).
Keys the test requires: `price_usd_per_hr, cap_usd, safety_margin, spent_usd, remaining_usd, blocks[{config,gpu_hours,usd}],
total_usd, total_gpu_hours, padded_total_usd, fits_budget, capacity_is_standin (always True), config_drift,
unpinned_variants, vllm_pin, ready`.
- Blocks: every name in `tokbench.configgen.build()`. Estimate like `make estimate` / `runner --dry-run --assume-capacity 10`:
  `load_config`, `resolve_loads(cfg, dict.fromkeys(refs, 10.0))` (refs = `runner._capacity_refs(cfg)`), `plan(cfg)`,
  `estimate_cost(cfg, price, launches=plan(cfg))` in `tokbench.config`. The stand-in capacity is NOT a measurement; keep the flag.
- Budget: `budget.load_policy(root/"budget")`, `budget.read_ledger`, `budget.spent`. `fits_budget = total_usd*(1+margin) <= remaining`.
- Drift: committed `root/configs/<name>.yaml` text != `configgen.render(name, header, cfg)` -> list the name.
- Unpinned: a variant whose `server_args` lacks `--revision` -> `"<config>:<variant>"`.
- vLLM pin: `vllm==X` line of `root/requirements-gpu.txt` -> `"X"`; anything else (`>=`, absent) -> `None`.
- `ready = fits_budget and not drift and not unpinned and vllm_pin is not None`. `main` exits 0 iff ready; plain output has one
  line per block and ends with `READY` or `NOT READY`.
- Missing `root/configs` raises `FileNotFoundError`. Use the style of `tokbench/budget.py` (argparse, `raise SystemExit(main())`).
Line length 100 (ruff).

## Log
- 2026-10-31 ready, issue #2 (handshake was provisional; repo not yet onboarded with ap init; waiting for the owner's agent-ready label)
