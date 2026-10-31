---
id: bench-run-provenance
goal: "`python -m tokbench.provenance <run_dir>` writes a deterministic PROVENANCE.json (sha256 and size of every artifact, cell and launch summary, manifest facts, synthetic flag) and `--verify` reports modified, missing and unlisted files."
acceptance:
  - python -m pytest tests/test_provenance.py --runxfail -q
  - python -m pytest tests/test_provenance.py -q
scope:
  - src/tokbench/provenance.py
  - tests/test_provenance.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines once the work is done (the second acceptance command fails while the strict marker remains)
  - do not touch docs/PREREG*.md, hypotheses, margins, or the budget
  - do not edit runner.py, capture.py, report.py, or any existing module; this is a standalone post-hoc tool
  - no new dependencies; stdlib only (hashlib, json)
  - no GPU, network, signing, or release-upload logic; no timestamps in the output
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
README Roadmap lists "a per-run provenance index" as not yet built. Raw results are shipped as release assets, not git, so
a reader needs a way to prove the files they downloaded are the files that were recorded.

## What to build (`src/tokbench/provenance.py`)
Run directory layout (see `tests/test_figures.py` for how a mock run is made): `manifest.json`, cell files `<variant>__<load>__r<k>.json`
(schema_version 2, keys `variant`, `repeat`, `synthetic`, ...) beside `.req.jsonl.gz/.gpu.csv.gz/.metrics*` artifacts,
`events.jsonl`, `launches/<variant>__r<k>/`. Cell loading rule (skip `manifest.json` and names containing `capacity`) is in
`report.load_cells`; reuse the idea, do not edit that module.
- `build_index(run_dir) -> dict` with `schema: 1`, `files` ({posix relpath: {sha256, bytes}} for every file except
  `PROVENANCE.json`), `cells` ([{file, variant, repeat, synthetic, schema_version}] sorted by file), `synthetic` (all cells
  synthetic), `launches` (sorted names under `launches/`), `manifest` ({tokbench_git_sha, gpu} from manifest.json).
  No cells -> `ValueError`.
- `write_index(run_dir) -> Path` writes `run_dir/PROVENANCE.json` with `json.dumps(..., indent=2, sort_keys=True)`; byte-identical on rerun.
- `verify(run_dir) -> list[str]` of `modified: <rel>`, `missing: <rel>`, `unlisted: <rel>`; no index -> `FileNotFoundError`.
- `main(argv)`: `<run>` writes the index (exit 0); `--verify` prints problems one per line, exit 1 if any.
Line length 100 (ruff). The test makes a real mock run (~6 s), no GPU.

## Log
