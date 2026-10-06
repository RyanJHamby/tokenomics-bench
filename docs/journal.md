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

## 2026-10-05 19:09 EDT (at 8681f4a)

Pre-run amendments to prereg-v2 (all before any GPU data; none outcome-dependent): (1) DERIVED FINDING is evaluated at SATURATION (c=64), not 0.7x capacity: at 0.7x every arm meets the SLO so goodput equals offered load and cannot rank arms by cost. (2) P6 uses TPOT p50 (stable) as the metric. (3) Holm is applied across the superiority tests (P1a,P1b,P4,P6); equivalence tests P2,P3 are intersection-union TOST at alpha .05, not multiplicity-adjusted. (4) B3a split: FP8 capacity gets 3 launches (my own rule: primary comparisons never <3 launches; the single-launch block violated it); fp8-kv8/AWQ are exploratory. (5) Model: SLO capacity now solved from TPOT-under-load (fp16 predicted ~5.8-12 req/s vs fp8 ~15-26, i.e. TPOT binds before saturation); regenerated PREDICTIONS.md before any run. Bugs found by tracing, not by users: launch_done skipped capacity repeats 2..n (pilot would have had no variance); load_cells crashed on capacity artifacts in the cell dir. Both fixed with regression tests; mock server never exposed them. Still unverified: all behaviour on real vLLM/NVML/providers.

## 2026-10-05 19:21 EDT (at 1f82639)

Tagged prereg-v2.1: amendments A1-A6 recorded in docs/PREREG-v2-amendments.md (PREREG-v2.md untouched). Still no GPU data.

## 2026-10-06 07:06 EDT (at bb43365)

Tagged prereg-v2.2 after 5 pre-deployment reviews (data completeness, real-server protocol, landscape, model/workload, energy telemetry). Fixed bugs: prefix hit rate matched always-zero external_* counters; t_last missed empty-text finish chunks; real-vLLM cells now fail without usage/metrics. Pinned all engine settings + model revisions via a config generator (drift test). User chose all four exploratory extras (shape loads, overload/recovery, MoE, SGLang): plan ~18.2 GPU-h; they are last-priority and budget-cut. Honest caveats: SGLang flags UNVERIFIED (no source offline); fp8-kv8 is a kernel+dtype confound; the repo's own model says prefill is ~59% of saturated time at 512/128 so P2/P3 may fail. STILL NOT BUILT: persistence layer (server log, per-request records, GPU/metrics series, env bundle), watchdog, off-pod sync, canary, schema v3, figures.

## 2026-10-06 07:28 EDT (at e1b87ec)

Built the pre-deployment P0 set from the five reviews: per-launch server.log; per-cell .req/.gpu/.metrics artifacts (+raw metrics text, wall-clock anchors); events.jsonl; _env bundle (pip freeze, nvidia-smi -q hashed ids, lscpu, cgroup, HF snapshot, allow-listed env only); crash bundle; detached watchdog (tested with a real SIGKILL); post-health canary; off-pod sync loop; telemetry probe; sampler jitter stats; metric stats now in the SAME window as energy. Tests re-derive the latency summary and the energy figure from the saved artifacts alone. 201 tests. NOT yet built: make figures, result-schema v3 doc/JSON schema, accuracy-parity gate wiring, per-run provenance index, NVML instant-vs-average power fields (only the defensive optional ones). Still unverified on real hardware: NVML optional fields, viol counters, watchdog nvidia-smi calls, SGLang flags, everything vLLM-side.

## 2026-10-06 07:39 EDT (at 4ee48da)

README rewritten for external readers (experts/recruiters): no measured numbers (none exist), novelty framed as replicate-and-extend of arXiv 2605.11999, design-principles table with file links, experiment plan + P1-P6, limitations, accurate AI disclosure. Flags for author: no LICENSE file yet; 'How this was built' needs the author's own review status; prior-work attributions (Watt Counts, ML.ENERGY, InferenceX) come from agent web research and were not independently re-verified.

## 2026-10-06 07:45 EDT (at b9277b7)

Pre-expert-review fixes: (1) MIT LICENSE added (author chose MIT). (2) README disclosure now says author directed the work and has NOT yet read every file; review in progress. (3) Prior-work claims verified this session against sources: Illusion of Power Capping (arXiv 2605.11999) full text confirms single H200, vLLM BF16, ~4B dense (Minitron/TransMLA/Qwen3.5/Nemotron/Qwen3-4B), batch 1-32, no SLO/goodput, no code released, limitations name MoE + other GPU generations; Watt Counts (2604.09048) = 5000+ experiments, 50 LLMs, 10 NVIDIA GPUs; ML.ENERGY = NeurIPS D&B 2025 spotlight; InferenceX formerly InferenceMAX, has power_model/. Removed Zeus/MLPerf phrasings I could not confirm. (4) Hardware: L40S (48GB GDDR6, 864GB/s, FP16 362.05/FP8 733 dense, 350W) and H100 SXM (80GB, 3.35TB/s, FP16 989.5/FP8 1979 dense = half of listed sparsity, 700W) verified vs NVIDIA pages 2026-10-06; H100-PCIe still unverified. Amendment A14.

## 2026-10-06 07:56 EDT (at 2fe33c1)

make figures built (8 figures, saved-data only, skip-never-fake, synthetic watermark in-image). Used the dataviz method: validated palette slots 1-3 all-pairs (aqua contrast WARN covered by direct labels), fixed order by arm family, no dual axes. Render check found and I fixed: footer/watermark collision, legend over data labels, clipped y-label, sci-notation log ticks, flat-zero panels, overlapping p99 labels. Tests caught: generator truthiness (any(d.glob) always True) and plotting an all-zero clock. README now embeds ONE watermarked synthetic example with an explicit not-a-measurement caption. 207 tests. Still not built: result-schema doc, parity gate wiring, provenance index.
