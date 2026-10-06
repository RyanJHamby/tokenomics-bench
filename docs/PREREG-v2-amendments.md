# Amendments to pre-registration v2 (A1-A6 tag `prereg-v2.1`; A7-A13 tag `prereg-v2.2`; A14 recorded after)

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

## Second round (tag `prereg-v2.2`), after the pre-deployment review

Five further independent reviews (single-deployment data completeness, real-server protocol
conformance, landscape of professional benchmark projects, model/workload choice, energy and
compute telemetry) were run before deployment. Still before any GPU run; still none
outcome-dependent.

| # | Change | Why |
|---|---|---|
| A7 | Every engine setting vLLM would otherwise pick by GPU or version is **pinned** in every config: `--dtype float16`, `--generation-config vllm`, `--max-num-seqs 256`, `--max-num-batched-tokens 2048`, `--seed 0`, async scheduling and chunked prefill on, `--attention-backend FLASH_ATTN`, and model + tokenizer **revision SHAs**. Configs are now generated from `tokbench/configgen.py` and a test fails on drift. | Defaults differ by GPU (L40S 256/2048 vs H100 1024/8192) and by checkpoint dtype (`auto` ran Llama in bf16 while AWQ would run fp16). The baseline arm is now byte-identical everywhere it is reused. |
| A8 | **Exploratory** blocks added, run only after the core blocks and only if budget remains: shape loads (decode-heavy 128/512 and prefill-heavy 2048/32) on the baseline and the 70% / 55% lock arms; overload-then-recovery (B5); a Qwen3-30B-A3B-FP8 MoE arm (B6); an SGLang cross-check (B7). **No confirmatory claim** rests on any of them. | The reviews found the 512/128 headline shape is prefill-heavy (about 59% of saturated GPU time by the repo's own model), so the lock-vs-cap result may flip with shape; the MoE and engine arms test generality. Total plan is about 18.2 GPU-hours, so these are the first things the budget guard cuts. |
| A9 | The `fp8-kv8` arm is labelled a **kernel + dtype confound**: FlashAttention has no fp8 KV on SM89, so enabling it changes the attention backend (FlashInfer). It is exploratory. | A pure KV-dtype effect cannot be claimed. |
| A10 | Measurement fixes found by running the client against a real server: the final token can arrive in an empty-text chunk with `finish_reason` (`t_last` now moves to it); real-vLLM cells **fail** if no `usage` chunk or no vLLM `/metrics` arrived; the prefix-cache hit rate uses exact metric names (a substring match picked the always-zero `external_*` counters). | These would have silently understated TPOT, passed an unverified workload, and dropped the B2 hit rate. |
| A11 | v2 states prefix caching "defaults ON in vLLM V1". Sources conflict. Configs pass the explicit flag, so results are unaffected; the statement is to be corrected from the server's actual default (recorded at preflight). | Do not assert what has not been verified. |
| A12 | **Novelty framing:** the lock-beats-cap-on-decode result is already reported by "The Illusion of Power Capping in LLM Decode" (arXiv 2605.11999: one H200, vLLM, BF16, four ~4B dense attention variants, batch 1-32; verified against the full text on 2026-10-06; it reports no SLO or goodput metric and releases no code). This work **replicates and extends** it on GDDR6 Ada with an 8B model, adds an SLO and goodput-per-dollar layer, an MoE arm, and pre-registered equivalence tests. It is not claimed as first. | The landscape review located the prior art. |
| A13 | **Pre-run risk notes, not changes to claims.** By the repo's own roofline model, prefill is ~59% of saturated time at 512/128, so P2 (lock70 costs no goodput) may fail by construction, and P3 (cap80 does not bind) may fail because a cap can bind during prefill bursts. Both stay as stated and will be reported as measured. | Recording the failure risk before the data prevents a post hoc rewrite. |

| A14 | **Hardware figures verified.** The prediction model's L40S and H100-SXM figures were checked against NVIDIA's product pages on 2026-10-06 (the pages list sparsity figures; the dense values used are half). The H100-PCIe entry is still unverified (the page shows NVL). v2's statement that the figures are unverified recollections is superseded for L40S and H100-SXM. | Closes the open datasheet check. No prediction value changed. |

Not changed (both rounds): hypotheses P1-P6 and their margins, the primary SLO, the statistical procedure,
the budget cap, and the stopping rule.
