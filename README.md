# tokenomics-bench

**Status: harness built and pre-registered; no GPU results yet, so this README quotes no
numbers.** The hardware measurements are the next step, under a hard $60 total budget.

On one GPU serving Llama-3.1-8B, what is the frontier of **goodput per dollar** and
**goodput per joule** under a p99 latency SLO, and which knob moves it: an SM **clock lock**,
a **power cap**, quantization, or CUDA-graph mode? The sharp question is whether a clock lock
reduces energy per token where a power cap does not (decode on a small model may never reach
the cap), and whether the energy-optimal clock differs from the cost-optimal one.

## What makes this more than a sweep
- **Pre-registered** ([`PREREG-v2`](docs/PREREG-v2.md)) before any GPU run, with numeric
  hypotheses, a paired-launch statistical procedure, and an "inconclusive is reported as
  inconclusive" rule. v1 was superseded after adversarial review; v1 is kept at its tag.
- **Predictions first** ([`PREDICTIONS`](docs/PREDICTIONS.md)): roofline-model intervals are
  committed before measurement; the write-up reports predicted vs measured, misses included.
- **Goodput, not throughput**: failures, truncations and still-running requests count as SLO
  misses. Latency by arrival, throughput/energy by completion, over one steady-state window.
- **Open-loop load**, exact-token prompts salted per load (no prefix-cache leakage), capacity
  found by bisection, launches as the unit of replication.
- **Energy from the NVML counter**, with cap/lock arms generated from measured draw and the
  card's real limits.
- **Failure-hardened harness**: resumable, fail-fast, signal-safe, 120+ tests, a spend ledger.

## Findings

_None yet. Every number added here must trace to a `pow run` capture in
[`results/`](results/), with the hardware named._

## Reproduce
`make test` (mock-server tests, no GPU), `make demo` (synthetic end-to-end, watermarked),
`make estimate PRICE=<usd/hr>`. Real runs: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## How this was built

The research question, the thesis, the $60 budget cap, and the scoping decisions are the
author's. The harness, tests, pre-registration drafts, and the four adversarial design reviews
were produced with an AI coding assistant (Claude Code), working in this repo under the
author's direction; commits carry a `Co-Authored-By` trailer. The AI-generated reviews are
claims to check, not authority: several of their findings were re-verified by hand (noted in
the [journal](docs/journal.md)) and others were not. Statements about results are made only
from captured runs in [`results/`](results/). The author's own line-by-line review status is
tracked in the journal, not asserted here.
