# Pre-registration

Committed before any GPU run. Changes after the first GPU run are recorded in
the journal as deviations, never edited silently.

## Question
Under a fixed p99 SLO, which vLLM configuration minimizes $/Mtok and J/token,
and does it differ from the throughput-maximizing configuration?

## Fixed
- Model: Llama-3.1-8B-Instruct (FP16 baseline), single GPU.
- vLLM: one pinned release and commit, recorded in the hardware manifest.
- Workloads: (a) fixed-length synthetic (e.g. 512 in / 128 out), (b) shared-prefix
  synthetic at 0/50/90% prefix share, (c) a ShareGPT-like length distribution.

## Metrics
Throughput (output tokens/s), TTFT and TPOT at p50 and p99, GPU energy
(integral of NVML power) divided by output tokens, $/Mtok from on-demand GPU price,
plus vLLM preemption count and KV-cache usage.

## SLO (initial; fixed before runs)
p99 TTFT <= 2 s and p99 TPOT <= 100 ms. A config's feasible capacity is the
highest load that meets both.

## Load generation
Closed-loop concurrency {1, 4, 16, 64} and open-loop Poisson arrivals. Open loop
is primary for latency claims (avoids coordinated omission).

## Design
>= 3 repeats per cell, first N requests discarded as warmup, config order
interleaved across repeats. Percentiles reported with bootstrap 95% CIs.

## Gates (before timing is trusted)
- Prefix cache on/off: greedy outputs identical on a fixed prompt set.
- Quantization: quality delta vs FP16 on a fixed eval subset, reported next to
  any speedup.

## Decision rule
A difference is called real only if it exceeds a 5% minimum effect AND passes a
Welch's t-test (p < 0.05) across repeats, the same rule as
`measured-speedup-harness`. Otherwise it is reported as "not distinguishable".

## Hypotheses
H1 The latency knee appears before throughput saturates.
H2 Prefix caching reduces p50 TTFT materially only when prefix share is high.
H3 FP8 raises throughput at a quality cost within noise on the chosen eval.
H4 CUDA graphs matter most at low batch size (TPOT).
H5 A moderate power cap lowers J/token with a smaller throughput loss on
   decode-bound load.
H6 The cheapest SLO-feasible config differs from the highest-throughput one.
