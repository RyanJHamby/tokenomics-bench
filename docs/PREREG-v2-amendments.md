# Amendments to pre-registration v2 (tag `prereg-v2.1`)

Made **before any GPU run**; no measurement data existed. Each tightens or corrects the plan
and none depends on an outcome. [`PREREG-v2.md`](PREREG-v2.md) (tag `prereg-v2`) is left
unedited; where the two differ, this file governs. Rationale and dates are in the journal.

| # | Where v2 says | Amended to | Why |
|---|---|---|---|
| A1 | Derived finding evaluated "at 0.7 x capacity" | Evaluated at **saturation (c=64)** | At 0.7 x capacity every arm meets the SLO, so goodput equals offered load and cannot rank arms by cost. |
| A2 | P6 metric "TPOT ratio" | **TPOT p50** per launch | p50 is stable at c=1 and c=64; p99 at these sample sizes is noisy. |
| A3 | "Holm correction over the primary family (P1-P6)" | Holm over the **superiority** tests (P1a, P1b, P4, P6). P2 and P3 (equivalence) are intersection-union TOST at alpha .05 each and are not multiplicity-adjusted. | Holm adjusts tests of "no difference"; for equivalence claims that is the wrong null. |
| A4 | B3a: capacity of quantized variants | FP8 capacity runs **3 launches** (primary, P4). fp8-kv8 and AWQ-INT4 run 1 launch and are **exploratory**. | A paired P4 test needs >= 3 launches; this also enforces v2's own "never fewer than 3 launches" rule. |
| A5 | Validity rules | Add: a cell is invalid if the load generator's event-loop **lag p99 exceeds 10 ms** (client saturation). | A saturated client mimics server latency. Also added a cross-check of the load generator against `vllm bench serve` with fixed tolerances. |
| A6 | `PREDICTIONS.md` SLO capacity "0.70-0.90 x saturation" | SLO capacity from a TPOT-under-load model (regenerated). | The TPOT SLO binds before saturation on this GPU; the old rule was a hand-wave. |

Not changed: hypotheses P1-P6 and their margins, the primary SLO, the statistical procedure,
the budget cap, and the stopping rule.
