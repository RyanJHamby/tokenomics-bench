---
id: bench-claim-guard
goal: "`python -m tokbench.claimcheck` fails if README.md or docs/*.md quote a measured-style number (tok/s, req/s, J/token, $/Mtok) unless the same line links a complete results/<run>/ capture; this repo passes today."
acceptance:
  - python -m pytest tests/test_claimcheck.py --runxfail -q
  - python -m pytest tests/test_claimcheck.py -q
scope:
  - src/tokbench/claimcheck.py
  - tests/test_claimcheck.py
non_goals:
  - do not change any assertion in the test file; only delete the `XFAIL` constant and the `@XFAIL` decorator lines once the work is done (the second acceptance command fails while the strict marker remains)
  - do not edit README.md or any file in docs/ (the repo is clean today; the point is to keep it so), and never touch docs/PREREG*.md
  - do not add numbers, results, or results/ folders
  - no new dependencies; stdlib only
  - do not wire it into CI, Makefile, or .github (the owner decides that separately)
risk_tier: 1
max_runs: 3
autonomy: inherit
status: ready
issue: 4
---

## Why
AGENTS.md rule 1 and the README promise: no number is quoted unless it comes from a captured `pow run` in `results/`
(folder holds `cmd.txt`, `env.txt`, `output.log`; see results/README.md). Nothing checks this mechanically today.

## What to build (`src/tokbench/claimcheck.py`)
- `Finding` dataclass with `path: str` (posix, relative to root), `line: int` (1-based), `text: str`.
- `scan_text(text, root=None, path="") -> list[Finding]`: a line is a finding if it contains a number directly followed by
  one of `tok/s`, `tokens/s`, `req/s`, `J/tok`, `J/token`, `/Mtok` (also `$0.42/Mtok` and `0.42 $/Mtok`). Deliberately NOT
  flagged: `ms`, `%`, `GPU-hours`, margins, SLO values (those are pre-registered design, not results). A flagged line is
  backed (no finding) only if the SAME line has a link to `results/<name>/` and `root/results/<name>/` contains all of
  `cmd.txt`, `env.txt`, `output.log`. With `root=None` nothing is backed.
- `check_repo(root) -> list[Finding]`: scans `root/README.md` and `root/docs/*.md` except `PREDICTIONS.md`, `journal.md` and
  `PREREG*.md` (labelled predictions, dated log, frozen pre-registration). Missing README.md raises `FileNotFoundError`.
- `main(argv) -> int`: `--root` (default `.`); prints `path:line: text` per finding; exit 1 if any else 0.
Match the style of `src/tokbench/budget.py`. Line length 100 (ruff). The README currently names "J/token" and "tokens/s" only
without numbers, so the real-repo test passes only if the regex requires a digit before the unit.

## Log
- 2026-10-31 ready, issue #4 (handshake was provisional; repo not yet onboarded with ap init; waiting for the owner's agent-ready label)
