# Runbook: real-hardware runs (prereg-v2)

Nothing here has run on a GPU yet. The first pass is partly a smoke test; journal every
surprise (`pow log`), including failures. Hard spend cap: **$60 across all attempts**
(`budget/budget.yaml`).

## Before renting
1. Pick a pod with **root and `nvidia-smi -pl`/`-lgc` allowed** (many containers forbid both;
   the preflight probes this, but cheaper to ask first). Full-VM providers are likelier than
   serverless. On-demand, not interruptible. Primary GPU: L40S (FP8-capable, 350 W).
2. Check the **live** $/hr. `make estimate PRICE=<live rate>` prints GPU-hours and dollars per
   block (the B2 template shows 1 of 8 arms and B6 1 of 2: multiply). Planned total is about
   14.7 GPU-hours for the core blocks and **about 18.2 with the exploratory blocks** (shape loads,
   overload/recovery, MoE, SGLang), before retries. At ~$1/h that is ~$18; at ~$2.7/h ~$49, which
   leaves no retry cover, so on an H100-priced pod run the core blocks only. Compare to remaining budget: `python -m tokbench.budget status`.
3. **Set a spending limit on the provider side.** The in-repo ledger is a guard, not
   enforcement: it cannot see the provider's bill.
4. If the GPU is not an L40S, regenerate `docs/PREDICTIONS.md` for it
   (`python -m tokbench.model --hw <name> --write docs/PREDICTIONS.md`), verify the hardware
   figures against the vendor datasheet, and commit **before** the first measured run.
5. Have `HF_TOKEN` with Llama 3.1 access, and `pow` installed on the pod.

## On the pod (inside tmux)
```
git clone https://github.com/RyanJHamby/tokenomics-bench && cd tokenomics-bench
export HF_TOKEN=...
scripts/setup_pod.sh
scripts/preflight.sh        # must print PREFLIGHT PASSED
```
Preflight checks, among others: pinned vLLM, driver >= 580, NVML energy counter, that `-pl` and
`-lgc` are writable (before you pay for a cap sweep), no stale cap, accepted CUDA-graph modes,
every flag the configs use, vCPU count, and a checksummed GSM8K file. If a check fails, fix it or
record a deviation; do not work around it silently.

## What gets saved, and what protects the run
- **Per cell**: `<cell>.json` (summary), `.req.jsonl.gz` (every request), `.gpu.csv.gz` (the full
  sample series), `.metrics.jsonl.gz` + `.metrics.raw.txt.gz` (vLLM series; raw keeps labels and
  histogram buckets). **Per launch**: `launches/<variant>__r<k>/server.log`, `launch.json` (wall +
  monotonic anchors, canary result), `soak.req.jsonl.gz`, `idle.gpu.csv.gz`, and `crash/` on failure.
  **Per run**: `events.jsonl`, `manifest.json`, `_env/` (pip freeze, `nvidia-smi -q` with identifiers
  hashed, lscpu, cgroup, HF cache snapshot, allow-listed env, never secrets). Expect ~100-250 MB for
  the whole plan: **do not commit it to git**; ship it as release assets. Re-derivation tests prove the
  summary and energy can be recomputed from these files alone.
- **Watchdog** (on by default for real GPU runs): a detached process resets clocks and the DEFAULT power
  limit and kills orphaned server process groups if the runner is SIGKILLed, OOM-killed or hangs.
  Check `results/raw/<stamp>-<block>/watchdog.log` after any abnormal end.
- **Canary**: after `/health` every launch sends one tiny completion and requires exact
  `usage.prompt_tokens` and (vLLM) `/metrics`; a broken server fails here, not 10 minutes into a soak.
- **Off-pod sync**: in a third tmux pane, `SYNC_DEST=<dir> scripts/sync_loop.sh results` (or
  `SYNC_RELEASE=<tag>`, which needs a repo-scoped token on the pod: use a fine-grained one and revoke it).
  A terminated pod then loses at most a minute. The pod cannot sign commits: pull the tarballs to the laptop,
  extract, and make the signed commit there.
- **Preflight log**: run `scripts/preflight.sh 2>&1 | tee results/preflight/preflight.log`; it also
  writes `results/preflight/telemetry_probe.json` (which optional NVML fields, DCGM and RAPL this pod exposes).
- **After the first real launch (a 2-minute smoke cell), check by hand** that `server.log`, the `.req`,
  `.gpu`, `.metrics` files and `launch.json` (with a passing canary) are all non-empty before starting B1.

## Order of operations (dependency order; stop when money runs short)
First, a ~10 GPU-minute validation of the load generator against vLLM's own benchmark on the
same server: `PRICE=<live> scripts/crosscheck.sh`. A DISAGREE verdict means a client-side artefact
(timestamping, event-loop saturation, window definition); understand it before trusting any latency.
```
PRICE=<live> scripts/run_all.sh b1                     # pilot: capacity, batch-1, saturation, variance
python -m tokbench.pilot results/raw/<STAMP>-b1        # launches needed per margin
B1_DIR=results/raw/<STAMP>-b1 PRICE=<live> scripts/run_all.sh b2gen
git add configs/b2_power_clock.yaml && git commit -S -m "B2 arms from pilot"   # BEFORE b2 runs
PRICE=<live> scripts/run_all.sh b2 b3a b3b b4          # core, confirmatory
# exploratory, ONLY after the core blocks finish and only if budget remains (first to be cut):
PRICE=<live> scripts/run_all.sh b5                      # overload + recovery
B1_DIR=results/raw/<STAMP>-b1 scripts/run_all.sh b6gen  # then commit configs/b6_moe.yaml
PRICE=<live> scripts/run_all.sh b6                      # Qwen3-30B-A3B-FP8 MoE
scripts/setup_sglang.sh && PRICE=<live> scripts/run_all.sh b7   # SGLang (unverified flags)
```
After the pilot, compare `required_repeats` to the plan. If the budget cannot afford the repeats
a margin needs, **widen** that margin (documented in the journal) rather than shrinking repeats
below 3 or narrowing a margin later. B2 (the core finding) comes first for that reason.

## After all blocks: the pre-registered analysis (mechanical)
```
python -m tokbench.analyze --b1 results/raw/<S>-b1 --b2 results/raw/<S>-b2 --b3a-fp8 results/raw/<S>-b3a-fp8 \
  --b3b results/raw/<S>-b3b --b4 results/raw/<S>-b4 --gsm8k-dir results/gates/<S> --out results/analysis.json
```
It applies the frozen P1-P6 procedure, lists excluded cells, and checks the model's predictions. It
makes no choices the pre-registration did not already make; omit an input and that test reports
`insufficient_data` instead of guessing.

## After each block
- `python -m tokbench.report results/raw/<dir> --png results/<dir>/frontier.png` and
  `make figures RUNS="results/raw/<dir> ..." ANALYSIS=results/analysis.json` (figures read only saved data;
  a figure whose inputs are missing is skipped, never faked; mock/fake-GPU runs are watermarked)
- Commit raw results in a **signed commit before analysing them**; `pow log` what happened.
- Stop the pod. Record the invoice: `python -m tokbench.budget add --provider ... --gpu ...
  --usd-per-hr ... --hours ... --session <name>`. A forgotten pod is the biggest cost risk.
- If a run is interrupted, re-run with the same `STAMP`: finished cells are skipped.

## Deviations
`prereg-v2` is tagged before the first run. Any change afterwards goes in the journal as a dated
deviation; `PREREG-v2.md` is not edited silently. Excluded cells (see its validity rules) are
listed in the write-up, never dropped quietly.
