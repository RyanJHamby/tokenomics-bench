# tokenomics-bench

**Status: pre-registration phase. No GPU results yet, so this README quotes no numbers.**

Where does an LLM server's cheapest, most energy-efficient configuration sit once
you hold it to a p99 latency SLO? This repo measures the serving-layer, model, and
GPU-power knobs of vLLM together, rather than one at a time, and reports the
frontier of tokens/s, p99 TTFT/TPOT, joules/token, and $/Mtok.

Hypothesis (to be confirmed or refuted by the data): throughput-optimal is not
cost-optimal or energy-optimal under a p99 SLO.

- Pre-registered plan: [`docs/PREREG.md`](docs/PREREG.md)
- Method and threats to validity: [`docs/METHOD.md`](docs/METHOD.md)
- Journal: [`docs/journal.md`](docs/journal.md)

## Findings

_None yet. Every number added here must trace to a `pow run` capture in
[`results/`](results/), with the hardware named._

## How this was built

Design, experiments, and analysis are mine. I used AI coding assistants for
parts of the implementation, tests, and documentation, and reviewed every change.
The [engineering journal](docs/journal.md) records decisions and dead ends as
they happened, and [`results/`](results/) holds the raw output behind every
number quoted here.
