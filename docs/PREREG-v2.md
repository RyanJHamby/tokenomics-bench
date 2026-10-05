# Pre-registration v2

**Status:** committed and tagged `prereg-v2` before any GPU run. It **supersedes**
[`PREREG.md`](PREREG.md) (tag `prereg-v1`, kept intact). No measurement data existed when
either was written, so nothing here is post hoc.

**Why superseded.** Before spending anything, four independent adversarial reviews
(methodology/statistics, code-breaking, frontier techniques, hardware/power; see the
journal) found that v1 could not support its own headline. The material problems:
$/Mtok was a constant divided by throughput, so "throughput-optimal is not cost-optimal"
was close to an identity; open-loop "throughput" was offered load; 200-request cells made
p99 roughly the second-largest sample; a Welch test on 3 launches confirmed a true 10%
effect about 6% of the time; prompt lengths were mislabeled by ~2.5x; the prefix cache
leaked across loads; power caps may never bind during decode. v2 fixes these in design.

## Question
On one GPU serving Llama-3.1-8B-Instruct under a p99 latency SLO, what is the frontier of
**goodput per dollar** and **goodput per joule**, and which knob moves it: clock lock, power
cap, quantization, or CUDA-graph mode? The sharp question: does a clock lock reduce energy
per token where a power cap does not, and is the energy-optimal clock different from the
cost-optimal one?

## Fixed
- Model `meta-llama/Llama-3.1-8B-Instruct`; vLLM `0.30.0`; one GPU, one pod per attempt.
  Primary GPU: L40S. If a different GPU is rented, `docs/PREDICTIONS.md` is regenerated for it
  and committed **before** the first measured run, and the change is journalled.
- Workload: 512 prompt / 128 output tokens, exact token counts (`/v1/completions`, token-id
  prompts), `ignore_eos`, greedy. Random token ids are not natural text: results do not
  transfer to speculative/n-gram decoding or to real traffic length distributions.
- **Primary SLO:** p99 TTFT <= 1.0 s and p99 TPOT <= 50 ms. Sensitivity: feasibility is also
  recomputed at (0.5 s, 30 ms) and (2 s, 100 ms) from stored p99s; goodput is only available at
  the primary SLO.
- Prefix caching is explicitly OFF in every block (it defaults ON in vLLM V1).

## Metrics (all over one measurement window)
- **Goodput** = output tokens/s from requests that met both SLOs and completed in the window.
  A request that fails, truncates, or is still running at the drain deadline is a failure and an
  SLO miss; it is never dropped.
- **J/output token** = board energy over the window / output tokens completed in that window.
  Energy source: NVML cumulative energy counter (power-integral reported alongside). Board
  power only; host CPU and cooling excluded. Also J/request.
- **$/Mtok (good)** = rental $/hr / (goodput x 3600) x 1e6, using the live on-demand price
  recorded per run. Electricity is ~3% of rent at stated prices and is reported but not added.
- **SLO capacity** of a variant: highest open-loop rate with zero failures and p99 TTFT and
  p99 TPOT within the SLO, found by doubling then bisection to **8%** resolution plus a
  confirmation probe.

## Measurement protocol
Per launch: start server -> 60 s saturating soak (thermal and compile warm-up) -> 10 s idle
power baseline -> loads in a **seeded random order**, each `warmup 30 s` + `measure 60 s`
(capacity probes) or `180 s` (comparisons) + up to 60 s drain. Latency/SLO by arrival time in
the window; throughput, goodput and energy by completion time in the same window. Prompts are
salted per (repeat, load) so each load starts cold. Variant launch order is shuffled per
repeat. A **launch is the unit of replication**; cells within a launch are not independent.

## Statistical procedure (frozen)
- Estimand: per-launch log ratio `log(m_A / m_B)` of a metric, A vs B, same load, paired by
  repeat index. Interval: t interval on the mean (not a bootstrap; n is small).
- Verdict against a pre-registered relative margin: `superior_*` (CI beyond the margin),
  `equivalent` (CI inside +/- margin), else `inconclusive`. **Inconclusive is reported as
  inconclusive, never as "no difference".**
- Holm correction over the **primary family** (P1-P6 below) only.
- Repeats: default 3 launches. After the pilot, `required_repeats(sd, margin)` with
  `sd = sqrt(2) x` the pilot's per-launch SD of the log metric gives the launches a margin needs.
  If the budget cannot afford it, the margin is **widened** to what the budget supports and
  stated; it is never narrowed after seeing results.
- Per-cell p99 TTFT carries a distribution-free CI (order statistics); cells with wide CIs
  are flagged.
- Quality: full 1319-item GSM8K, same items for baseline and candidate, exact McNemar plus
  paired CI, margin +/- 2 pp, `max_tokens` 1024.

## Primary hypotheses (confirmatory)
| # | Claim | Metric, comparison | Margin |
|---|---|---|---|
| P1 | A clock lock at ~70% of max SM clock cuts energy per token | J/token, lock70 vs baseline, at saturation and at 0.7 x capacity | superior_lower, 10% |
| P2 | ...without a goodput cost | goodput, lock70 vs baseline, same loads | equivalent, 5% |
| P3 | A power cap at 80% of saturated p95 draw does not bind during decode | J/token and goodput, cap80 vs baseline | equivalent, 5% |
| P4 | FP8 weights raise SLO capacity | capacity, fp8 vs fp16 (B3a vs B1) | superior_higher, 20% |
| P5 | ...with non-inferior quality | GSM8K, fp8 vs fp16 | non_inferior, 2 pp |
| P6 | CUDA-graph benefit is larger at c=1 than at c=64 | TPOT ratio none/full_and_piecewise at c=1 vs c=64 (paired difference of log ratios) | CI excludes 0 |

**Derived finding (reported, not a hypothesis test):** among the lock arms at 0.7 x capacity,
the J/token-minimizing arm versus the goodput-per-dollar-maximizing arm. The claim "they
differ" holds only if the J/token-optimal arm is >= 10% lower in J/token and >= 5% lower in
goodput than the goodput-optimal arm (both `superior`).

**Model check (reported with misses):** each interval in `docs/PREDICTIONS.md` vs the
measured value. The model is not refit.

## Exploratory (no claims of significance)
Prefix caching, speculative decoding, bursty arrivals, TP=2, FP8 KV cache beyond capacity
accounting, AWQ, ITL tails, per-arm temperature. Anything not in the table above is
exploratory and labelled so.

## Validity rules (cells excluded and rerun, listed, never hidden)
Excluded if: `workload_ok` false (server saw a different prompt length than configured);
`sampler_ok` false; hardware-slowdown or thermal throttle bits seen; any failure other than
`incomplete`; server restarted mid-launch. A power cap or lock the driver refuses is recorded
as a skipped arm. Temperature, SM clock and throttle OR are stored per cell.

## Budget and stopping
Hard cap **$60** across every attempt, tracked in `budget/ledger.jsonl`; the runner refuses
work whose padded estimate exceeds the remainder. Block order: B1 pilot -> B2 (core) -> B3a ->
B3b+quality -> B4. If money runs short, later blocks are dropped; primary comparisons are never
run with fewer than 3 launches. All spend is logged, including failed attempts.

## Threats to validity (acknowledged up front)
One GPU model and size, one model, one workload shape; synthetic token ids; board power only;
rental $/hr is a snapshot; AWQ is a different checkpoint (model + precision change); a pod's
silicon and cooling vary, so a result is one machine's; no multi-GPU, long-context, MoE or
disaggregation claim is made. Hardware figures in the model are unverified and may need fixing
before publication. Whether a given provider permits `-pl`/`-lgc` is unknown until the probe.
