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
