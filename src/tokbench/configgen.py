"""Single source of truth for every experiment config (configs/*.yaml are generated).

    python -m tokbench.configgen          # rewrite configs/*.yaml
    pytest tests/test_configs.py          # fails if the committed YAML differs from this

Why generated: the baseline arm must be byte-identical wherever it is reused, and every
setting vLLM would otherwise choose by GPU or version (dtype, scheduler limits, attention
backend, generation config, seed, model revision) must be pinned identically everywhere.
Defaults on L40S differ from H100 (max_num_seqs/max_num_batched_tokens 256/2048 vs 1024/8192),
so an unpinned run is not reproducible across GPUs or vLLM versions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Model:
    repo: str
    rev: str  # HF commit SHA, looked up 2026-10-06; preflight verifies it still resolves


LLAMA = Model("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659")
AWQ = Model(
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "db1f81ad4b8c7e39777509fac66c652eb0a52f91"
)
QWEN_MOE = Model("Qwen/Qwen3-30B-A3B-FP8", "d206ba732169f29bb77fbf80fc2c4b81d4d30782")

COMMON = {
    "seed": 0,
    "startup_seconds": 240,
    "soak_s": 60,
    "idle_s": 10,
    "slo": {"ttft_s": 1.0, "tpot_s": 0.05},
    "capacity_file": "results/capacity/capacity.json",
}
WL = {"input_tokens": 512, "output_tokens": 128, "prefix_share": 0.0}
SHAPES = {  # compute- vs memory-bound regimes: where lock-vs-cap should flip
    "decode_heavy": {"input_tokens": 128, "output_tokens": 512, "prefix_share": 0.0},
    "prefill_heavy": {"input_tokens": 2048, "output_tokens": 32, "prefix_share": 0.0},
}
CAP_SEARCH = {"mode": "capacity_search", "qps_lo": 1, "qps_hi": 32, "resolution": 0.08}


def pinned(model: Model, dtype: str = "float16", backend: str = "FLASH_ATTN", extra=()) -> list:
    """Every engine setting that vLLM would otherwise pick by GPU/version, pinned."""
    return [
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
        "--dtype", dtype, "--generation-config", "vllm",
        "--max-num-seqs", "256", "--max-num-batched-tokens", "2048", "--seed", "0",
        "--async-scheduling", "--enable-chunked-prefill", "--no-enable-prefix-caching",
        "--attention-backend", backend,
        "--revision", model.rev, "--tokenizer-revision", model.rev,
        *extra,
    ]  # fmt: skip


def _fp16() -> dict:
    return {"name": "fp16-default", "model": LLAMA.repo, "server_args": pinned(LLAMA)}


def _cc(mode: str) -> list:
    return ["--compilation-config", f'{{"cudagraph_mode": "{mode}"}}']


def build() -> dict[str, tuple[str, dict]]:
    out: dict[str, tuple[str, dict]] = {}

    def add(name, header, **cfg):
        base = {"name": name, "model": LLAMA.repo, **COMMON, **cfg}
        out[name] = (header.strip(), base)

    add(
        "b1_capacity_pilot",
        """
B1 / PILOT. Baseline FP16: SLO capacity (doubling+bisection, 8%), batch-1 decode speed (c=1), saturated
throughput and uncapped power (c=64), and launch-to-launch variance over 3 independent launches. Variance
sizes later blocks (tokbench.pilot); saturated p95 power sets cap arms (tokbench.arms). All engine settings
are pinned (see tokbench.configgen). Prefix caching OFF everywhere except its own exploratory use.""",
        block="B1",
        workload=WL,
        repeats=3,
        warmup_s=30,
        measure_s=60,
        variants=[_fp16()],
        loads=[
            CAP_SEARCH,
            {"mode": "closed", "concurrency": 1},
            {"mode": "closed", "concurrency": 64},
        ],
    )

    add(
        "b2_power_clock.template",
        """
B2 TEMPLATE (core finding): power cap (-pl) vs SM clock lock (-lgc) vs baseline, FP16. Do not run
directly: `python -m tokbench.arms <pilot dir> --template configs/b2_power_clock.template.yaml >
configs/b2_power_clock.yaml` fills the arms from the pilot's measured saturated power and the card's limits.
Main loads: saturation (c=64) and 0.7 x baseline capacity. `shape_loads` (EXPLORATORY) are added by the
generator only to the baseline and the 70% / 55% lock arms (not all 8): decode-heavy 128/512 and
prefill-heavy 2048/32, to test whether lock-vs-cap flips with workload shape and to make the
prefill-vs-decode energy regression identifiable.""",
        block="B2",
        workload=WL,
        repeats=3,
        warmup_s=30,
        measure_s=180,
        variants=[{"name": "uncapped", "model": LLAMA.repo, "server_args": pinned(LLAMA)}],
        loads=[
            {"mode": "closed", "concurrency": 64},
            {"mode": "open", "rel": 0.7, "ref": "fp16-default"},
        ],
        shape_loads=[
            {"mode": "closed", "concurrency": 64, "workload": SHAPES["decode_heavy"]},
            {"mode": "closed", "concurrency": 16, "workload": SHAPES["prefill_heavy"]},
        ],
    )

    fp8 = {
        "name": "fp8",
        "model": LLAMA.repo,
        "server_args": pinned(LLAMA, extra=["--quantization", "fp8"]),
    }
    fp8kv = {
        "name": "fp8-kv8",
        "model": LLAMA.repo,
        "server_args": pinned(
            LLAMA, backend="FLASHINFER", extra=["--quantization", "fp8", "--kv-cache-dtype", "fp8"]
        ),
    }
    awq = {
        "name": "awq-int4",
        "model": AWQ.repo,
        "server_args": pinned(AWQ, extra=["--quantization", "awq_marlin"]),
    }

    add(
        "b3a_fp8_capacity",
        """
B3a (primary, P4): SLO capacity of FP8 weights, 3 launches (pairs with B1's 3 FP16 capacities).""",
        block="B3a",
        workload=WL,
        repeats=3,
        warmup_s=30,
        measure_s=60,
        variants=[fp8],
        loads=[{**CAP_SEARCH, "qps_hi": 64}],
    )
    add(
        "b3a_extra_capacity",
        """
B3a-extra (EXPLORATORY): capacity of fp8-kv8 and AWQ-INT4, 1 launch each. fp8-kv8 changes the ATTENTION
BACKEND as well as the KV dtype (FlashAttention has no fp8 KV on SM89): a kernel+dtype confound, labelled
as such. AWQ is a third-party checkpoint: a model change AND a precision change.""",
        block="B3a-extra",
        workload=WL,
        repeats=1,
        warmup_s=30,
        measure_s=60,
        variants=[fp8kv, awq],
        loads=[{**CAP_SEARCH, "qps_hi": 64}],
    )
    add(
        "b3b_quant_fixed_load",
        """
B3b: variants at EQUAL absolute offered load (fractions of FP16 capacity) plus batch-1 and saturation,
3 launches each, for paired comparisons. Quality: full GSM8K, paired McNemar/TOST vs FP16.""",
        block="B3b",
        workload=WL,
        repeats=3,
        warmup_s=30,
        measure_s=180,
        variants=[_fp16(), fp8, fp8kv, awq],
        loads=[
            {"mode": "closed", "concurrency": 1},
            {"mode": "open", "rel": 0.5, "ref": "fp16-default"},
            {"mode": "open", "rel": 0.9, "ref": "fp16-default"},
            {"mode": "closed", "concurrency": 64},
        ],
    )

    graphs = [
        {
            "name": "graphs-none-compile-on",
            "model": LLAMA.repo,
            "server_args": pinned(LLAMA, extra=_cc("NONE")),
        },
        {
            "name": "graphs-piecewise",
            "model": LLAMA.repo,
            "server_args": pinned(LLAMA, extra=_cc("PIECEWISE")),
        },
        {
            "name": "graphs-full-and-piecewise",
            "model": LLAMA.repo,
            "server_args": pinned(LLAMA, extra=_cc("FULL_AND_PIECEWISE")),
        },
        {
            "name": "eager",
            "model": LLAMA.repo,
            "server_args": pinned(LLAMA, extra=["--enforce-eager"]),
        },
    ]
    add(
        "b4_graph_modes",
        """
B4: CUDA graph modes. 'graphs-none-compile-on' isolates graph capture from torch.compile (--enforce-eager
disables both, so 'eager' is a separate, coarser arm). Closed loop pins batch size; metric is TPOT/ITL.
The effective cudagraph mode is read back from the server log and the cell fails if it was downgraded.""",
        block="B4",
        workload=WL,
        repeats=3,
        warmup_s=20,
        measure_s=60,
        variants=graphs,
        loads=[{"mode": "closed", "concurrency": c} for c in (1, 8, 64)],
    )

    add(
        "b5_overload_recovery",
        """
B5 (EXPLORATORY; 'reliable' in the thesis): baseline FP16 driven past capacity then dropped back.
Phase 1: 1.3 x capacity for 60 s (queue builds, goodput collapses?); phase 2: 0.5 x capacity for 120 s.
Reported: per-phase goodput/SLO attainment, preemptions, and recovery time (first 5 s arrival window
whose p99 TTFT is back within the SLO with zero failures). No confirmatory claim.""",
        block="B5",
        workload=WL,
        repeats=3,
        warmup_s=0,
        measure_s=1,
        drain_s=60,
        variants=[_fp16()],
        loads=[
            {
                "mode": "phased_open",
                "phases": [
                    {"rel": 1.3, "ref": "fp16-default", "duration_s": 60},
                    {"rel": 0.5, "ref": "fp16-default", "duration_s": 120},
                ],
            }
        ],
    )

    moe = [
        {
            "name": "moe-fp8",
            "model": QWEN_MOE.repo,
            "server_args": pinned(QWEN_MOE, dtype="bfloat16"),
        },
    ]
    add(
        "b6_moe.template",
        """
B6 TEMPLATE (EXPLORATORY): Qwen3-30B-A3B-FP8 (MoE, 3B active). Generated variants: baseline + 70% clock
lock via `python -m tokbench.arms <pilot> --template configs/b6_moe.template.yaml --locks 0.70 --no-caps`.
Hypothesis: at batch 1 only active params stream (J/token far below dense FP8); by batch 64 nearly all
experts are touched and the advantage shrinks. Compare against Llama FP8 (B3b), NOT FP16 (the MoE does
not fit in bf16 on 48 GB). 2 launches per arm; first block to drop if money is short.""",
        block="B6",
        model=QWEN_MOE.repo,
        workload=WL,
        repeats=2,
        warmup_s=30,
        measure_s=100,
        variants=moe,
        startup_seconds=420,
        loads=[{"mode": "closed", "concurrency": c} for c in (1, 16, 64)],
    )

    sg = {"name": "sglang-fp16", "model": LLAMA.repo, "engine": "sglang",
          "server_args": ["--context-length", "4096", "--mem-fraction-static", "0.88",
                          "--dtype", "float16", "--max-running-requests", "256",
                          "--chunked-prefill-size", "2048", "--disable-radix-cache",
                          "--revision", LLAMA.rev, "--enable-metrics"]}  # fmt: skip
    add(
        "b7_sglang_crosscheck",
        """
B7 (EXPLORATORY, UNVERIFIED FLAGS, run LAST): one SGLang FP16 variant on the same workload and SLO as the
vLLM baseline, for a labelled engine cross-check. SGLang flag names were NOT checked against a source tree
(none available offline); `scripts/setup_sglang.sh` + preflight must confirm each before this runs.
Fairness: same dtype, max length, prefix cache off (radix off), scheduler limits, chunked-prefill size,
token-id prompts, greedy + ignore_eos. Not a claim about which engine is faster.""",
        block="B7",
        workload=WL,
        repeats=2,
        warmup_s=30,
        measure_s=60,
        variants=[sg],
        startup_seconds=300,
        loads=[
            CAP_SEARCH,
            {"mode": "closed", "concurrency": 1},
            {"mode": "closed", "concurrency": 64},
        ],
    )

    add(
        "crosscheck",
        """
Cross-check of tokbench's load generator vs `vllm bench serve` on the same running server
(scripts/crosscheck.sh). Not a result block.""",
        block="XCHECK",
        workload=WL,
        repeats=1,
        warmup_s=20,
        measure_s=120,
        variants=[_fp16()],
        loads=[{"mode": "open", "qps": 4}],
    )
    out["crosscheck"][1].pop("capacity_file")
    return out


def render(name: str, header: str, cfg: dict) -> str:
    head = "".join(f"# {line}\n" for line in header.splitlines())
    return head + yaml.safe_dump(cfg, sort_keys=False, width=150, default_flow_style=None)


def write_all(directory: str | Path = "configs") -> list[str]:
    written = []
    for name, (header, cfg) in build().items():
        path = Path(directory) / f"{name}.yaml"
        path.write_text(render(name, header, cfg))
        written.append(str(path))
    return written


if __name__ == "__main__":
    print("\n".join(write_all(sys.argv[1] if len(sys.argv) > 1 else "configs")))
