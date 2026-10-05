# tokenomics-bench: engineering journal

Dated notes on what I tried, what happened, and why I made each call, including
the dead ends. Entries are appended with `pow log` and `pow run`, so dates match
commit history. Raw outputs for any number quoted in the README live in
[`results/`](../results/).

## 2026-10-04 18:58 EDT (at 9ed6a63)

Chose thesis: cost/energy-per-token frontier under a p99 SLO (serving -> quantization -> GPU power cap). Local-first: build harness against a mock server, rent GPU only for final runs. Preregistered plan committed before any GPU spend.

## 2026-10-04 19:01 EDT (at 8ad866c)

Loadgen built and validated against a mock server with known latency. Open loop measures from scheduled arrival (coordinated omission). 5 tests pass locally; no GPU numbers yet. Next: telemetry (NVML + vLLM /metrics) and analysis.

## 2026-10-04 19:07 EDT (at 658d8af)

Telemetry + analysis done on synthetic data (14 tests). Welch p-value cross-checked against scipy once, hardcoded in test. Bumped python to >=3.11 for typing.Self. Still no real-GPU numbers.

## 2026-10-04 19:12 EDT (at aacdbc7)

Runner + report built; make demo (mock server, fake GPU) runs end to end in ~2.5 min and draws the frontier chart watermarked SYNTHETIC. Demo output goes to gitignored demo_out/, not results/, so it can't be mistaken for a measurement. TODO before GPU spend: quality gates (greedy equivalence, eval subset), real B1-B7 configs, freeze PREREG.

## 2026-10-04 20:22 EDT (at fa28d67)

Wrote B1-B5 configs + B7 template, quality gates, overload derivation. Dry-run estimate at an ASSUMED $1.50/hr: ~7.8 GPU-h for B1-B5 (higher than the 3-5h I first guessed, driven by 3 repeats and per-variant restarts). Real price must come from live rates. Bug caught: AWQ variant sent the base model name; fixed. Not yet done: GSM8K data file, vLLM version pin, PREREG freeze.

## 2026-10-04 20:27 EDT (at 5bab543)

Froze PREREG (tag prereg-v1) after amending gates: cache on/off judged against same-config noise floor (GPU greedy isn't bit-reproducible); quant claimed lossless only within 2 SE on GSM8K-200. Pinned vllm==0.30.0 (latest on PyPI today); my flag check was against the local fork, so preflight re-verifies on the pod. Scripts are syntax-checked only, never run against vLLM.

## 2026-10-04 22:08 EDT (at 6b52080)

Tier 0 fixes landed after four adversarial reviews (methodology, code-break, frontier scout, hardware/power). Verified myself: prompts tokenized ~2.5x longer than labeled; prompt i was identical across loads (prefix-cache contamination). Fixes: exact-token prompts via /v1/completions salted per load+repeat; strict request validity (truncation/error events/short = failure); goodput + ITL; steady-window throughput and energy in ONE window; NVML energy counter preferred; throttle mask fixed; sampler death is loud; server lifecycle (port check, SIGKILL escalation, GPU-memory wait, SIGTERM/SIGHUP); power cap restores to DEFAULT limit and validates min/max; resume + atomic writes; fail-fast on dead runs; selection logic (eligible cells, unique names, NaN-safe). 74 tests. Budget: hard cap $60 total across all attempts (user decision 2026-10-04); guard in runner, ledger in budget/. Still unverified: everything on real vLLM/NVML. Next: prereg-v2 (supersedes v1; v1 tag kept), scripts, frontier experiment design.

## 2026-10-04 22:31 EDT (at 341d503)

Pre-registration v2 frozen (tag prereg-v2); v1 kept at prereg-v1 with a SUPERSEDED banner. Changes after the 4 adversarial reviews: goodput + one steady window for latency/throughput/energy; time-based warmup, soak, idle baseline, shuffled load order; capacity by bisection to 8% (replaces QPS grids); paired-launch t-interval stats with margin verdicts (replaces Welch on 3 repeats); numeric hypotheses P1-P6; clock lock vs power cap as the sharp question; arms generated from measured pilot power and card limits; paired full-GSM8K quality gate; roofline predictions committed first (docs/PREDICTIONS.md, hardware specs UNVERIFIED). Planned ~13.7 GPU-h vs $60 cap. NOT verified: anything on real vLLM/NVML/provider; whether any provider allows -pl/-lgc; the hardware datasheet figures; vLLM 0.30.0 flag acceptance (preflight checks). I (Claude) rewrote README AI-disclosure; author must confirm its accuracy.

## 2026-10-05 08:27 EDT (at 301ebb9)

Pre-run amendment to prereg-v2 validity rules (tightening only, not outcome-dependent): a cell is invalid if the load generator's event-loop lag p99 exceeds 10 ms (client saturation), per the hardware reviewer's concern that a single asyncio client could distort TTFT/ITL. Added scripts/crosscheck.sh: tokbench vs vllm bench serve on the SAME server, tolerances fixed in code before any result (throughput 10%, p50 15%, p99 30%). Mock-only so far; real behaviour unverified. CI green for prereg-v2 push.
