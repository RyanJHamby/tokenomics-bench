# Runbook: real-hardware runs

Nothing here has been run on a GPU yet. Treat the first pass as a smoke test and journal
every surprise (`pow log`).

## Before renting
1. `make estimate PRICE=<live on-demand $/hr>`. Prices move daily; use the provider's live
   rate. As of the dry run, B1-B5 are about 7.8 GPU-hours plus about 1 for B7.
2. Pick a GPU with **FP8 support** (compute capability >= 8.9: L4, L40S, H100) and a bare
   pod where **root can run `nvidia-smi -pl`**. Serverless platforms usually block this, so
   B5 (power caps) needs a pod. Pick on-demand, not interruptible.
3. B5 caps in `configs/b5_power_caps.yaml` assume a 350 W limit (L40S). Rescale to
   100/85/70/55% of the rented GPU's `power.default_limit` if it differs, and note the
   change in the journal before running.
4. Have `HF_TOKEN` with access to meta-llama/Llama-3.1-8B-Instruct. Budget a cap with the
   provider so a hung run can't bill overnight.

## On the pod
```
git clone https://github.com/RyanJHamby/tokenomics-bench && cd tokenomics-bench
export HF_TOKEN=...
scripts/setup_pod.sh
scripts/preflight.sh            # must print PREFLIGHT PASSED
```
Preflight checks: pinned vLLM version, the flags used by the configs (including
`--no-enable-prefix-caching`, without which the B2 "off" arm could silently stay on),
writable power cap, model access, FP8 support. If a flag is missing, fix the config and
record the deviation; do not run around it.

## Order of operations
Run a smoke test first (about 10 minutes): `PRICE=... scripts/run_all.sh b4` is the
cheapest block. Inspect one cell JSON by hand: are `server_metrics` populated, is
`mean_power_w` plausible, is `throttle_seen` false, are there failed requests?

Then, per block, in dependency order:
```
PRICE=<live rate> scripts/run_all.sh b1        # knee; B7 depends on it
PRICE=<live rate> scripts/run_all.sh b2 b3 b4 b5
PRICE=<live rate> scripts/run_all.sh b7
```
Each step is wrapped in `pow run`, which saves output and hardware info under `results/`.
Raw per-cell JSON lands in `results/raw/<stamp>-<block>/`. Quality gates run before their
block's timings; a failed gate means that block's speedups must not be claimed.

## After each block
- `python -m tokbench.report results/raw/<dir> --png results/<dir>/frontier.png`
- Commit raw results in a signed commit before analysing them. `pow log` what happened,
  including failures and anything that contradicts the pre-registration.
- Stop the pod. A forgotten pod is the biggest cost risk.

## Deviations
The pre-registration is tagged `prereg-v1` before the first run. Any change after that goes
in the journal as a dated deviation; docs/PREREG.md is not edited silently.
