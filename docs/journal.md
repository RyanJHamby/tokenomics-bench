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
