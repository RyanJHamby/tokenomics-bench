"""First-principles performance model for Llama-3.1-8B on one GPU (roofline + fluid batch).

Purpose: commit numeric predictions BEFORE any measurement (docs/PREDICTIONS.md), then
report prediction vs measurement, including the misses. A model that is wrong in an
interesting direction is a result; a model fitted after the fact is not.

Everything here is a deliberately simple upper-bound-ish model. Efficiency factors are
ranges, not point values, so predictions are intervals. Hardware numbers are from vendor
datasheets as recalled by the author and are NOT verified; `verified` says so, and the
generated document repeats it. Re-verify against the vendor datasheet before publishing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 2**30
# Llama-3.1-8B (HF config): 32 layers, 8 KV heads (GQA), head dim 128, vocab 128256, d=4096.
N_LAYERS, N_KV_HEADS, HEAD_DIM, VOCAB, D_MODEL = 32, 8, 128, 128256, 4096
N_PARAMS = 8.03e9
EMBED_PARAMS = 2 * VOCAB * D_MODEL  # untied embedding + lm_head (kept FP16 under FP8 weights)
BODY_PARAMS = N_PARAMS - EMBED_PARAMS


@dataclass(frozen=True)
class Hardware:
    name: str
    mem_gb: float
    mem_bw_gbs: float
    tflops_fp16: float  # dense tensor-core
    tflops_fp8: float  # dense; 0 if unsupported
    tdp_w: float
    verified: bool = False  # all entries below are unverified recollections


HARDWARE = {
    "L40S": Hardware("L40S", 48, 864, 362, 733, 350),
    "H100-PCIe": Hardware("H100-PCIe", 80, 2000, 756, 1513, 350),
    "H100-SXM": Hardware("H100-SXM", 80, 3350, 989, 1979, 700),
}

PRECISIONS = ("fp16", "fp8", "awq-int4")


def weight_bytes(prec: str) -> float:
    return {
        "fp16": N_PARAMS * 2,
        "fp8": BODY_PARAMS * 1 + EMBED_PARAMS * 2,  # weights-only FP8; embeddings stay FP16
        "awq-int4": BODY_PARAMS * 0.5 + EMBED_PARAMS * 2 + BODY_PARAMS * 0.5 * 0.05,
    }[prec]


def kv_bytes_per_token(kv_dtype: str = "fp16") -> int:
    per = 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM  # K and V
    return per * (1 if kv_dtype == "fp8" else 2)


def kv_capacity_tokens(
    hw: Hardware, prec: str, util: float = 0.90, overhead_gib: float = 2.5, kv_dtype: str = "fp16"
) -> int:
    free = hw.mem_gb * 1e9 * util - weight_bytes(prec) - overhead_gib * GIB
    return max(0, int(free / kv_bytes_per_token(kv_dtype)))


def _peak_flops(hw: Hardware, prec: str) -> float:
    # AWQ-INT4 (Marlin W4A16) does FP16 math: it saves bandwidth, not FLOPs.
    return (hw.tflops_fp8 if prec == "fp8" and hw.tflops_fp8 else hw.tflops_fp16) * 1e12


def decode_step_s(
    hw: Hardware, prec: str, batch: int, ctx: float, mem_eff: float, kv_dtype: str = "fp16"
) -> float:
    """Memory-bound decode step: stream the weights once plus every sequence's KV."""
    bytes_moved = weight_bytes(prec) + batch * ctx * kv_bytes_per_token(kv_dtype)
    return bytes_moved / (hw.mem_bw_gbs * 1e9 * mem_eff)


def prefill_s(hw: Hardware, prec: str, n_tokens: int, mfu: float) -> float:
    """Compute-bound prefill: 2 * params * tokens FLOPs (attention term ignored)."""
    return 2 * BODY_PARAMS * n_tokens / (_peak_flops(hw, prec) * mfu)


def batch1_decode_tok_s(hw, prec, ctx=300.0, mem_eff=0.75, kv_dtype="fp16") -> float:
    return 1.0 / decode_step_s(hw, prec, 1, ctx, mem_eff, kv_dtype)


def saturation_req_s(
    hw, prec, n_in, n_out, mem_eff=0.75, mfu=0.5, max_num_seqs=256, kv_dtype="fp16"
) -> dict:
    """Fluid model of a saturated continuous-batching server: batch is limited by KV
    capacity or max_num_seqs; each request costs its prefill plus its share of decode steps."""
    cap = kv_capacity_tokens(hw, prec, kv_dtype=kv_dtype)
    batch = max(1, min(cap // (n_in + n_out), max_num_seqs))
    step = decode_step_s(hw, prec, batch, n_in + n_out / 2, mem_eff, kv_dtype)
    per_req = prefill_s(hw, prec, n_in, mfu) + n_out * step / batch
    return {
        "batch": batch,
        "req_s": 1.0 / per_req,
        "tok_s": n_out / per_req,
        "kv_capacity_tokens": cap,
    }


def slo_capacity_req_s(
    hw, prec, n_in, n_out, tpot_slo_s, mem_eff=0.75, mfu=0.5, max_num_seqs=256, kv_dtype="fp16"
) -> float:
    """Highest arrival rate whose TPOT stays within the SLO, in a fluid model.

    At rate lam the server spends a fraction lam*prefill_s of its time in prefill, so the
    decode step stretches to step(B)/(1 - lam*prefill_s), with batch B = lam*n_out*stretched
    step (Little's law). Capacity is the largest lam where that stretched step <= tpot_slo and
    B fits the KV cache / max_num_seqs. TTFT is ignored here (it only binds near saturation).
    """
    p = prefill_s(hw, prec, n_in, mfu)
    cap_b = max(
        1, min(kv_capacity_tokens(hw, prec, kv_dtype=kv_dtype) // (n_in + n_out), max_num_seqs)
    )

    def feasible(lam: float) -> bool:
        if lam * p >= 1.0:
            return False
        eff = 0.0
        for _ in range(60):  # fixed point of eff = step(B)/(1-lam*p), B = lam*n_out*eff
            b = lam * n_out * eff
            nxt = decode_step_s(
                hw, prec, max(1, round(b)) if b >= 1 else 1, n_in + n_out / 2, mem_eff, kv_dtype
            ) / (1.0 - lam * p)
            if abs(nxt - eff) < 1e-9:
                break
            eff = nxt
        return eff <= tpot_slo_s and lam * n_out * eff <= cap_b

    lo, hi = 0.0, 1.0 / p
    if not feasible(1e-6):
        return 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


def interval(fn, *, mem_effs=(0.65, 0.85), mfus=(0.35, 0.60)) -> tuple[float, float]:
    """Min/max of fn(mem_eff, mfu) over the corners of the efficiency ranges."""
    vals = [fn(m, f) for m in mem_effs for f in mfus]
    return min(vals), max(vals)


def ttft_unloaded_s(hw, prec, n_in, mem_eff=0.75, mfu=0.5) -> float:
    """TTFT at near-zero load: one prefill plus one decode step."""
    return prefill_s(hw, prec, n_in, mfu) + decode_step_s(hw, prec, 1, n_in, mem_eff)


TPOT_SLO_S = 0.05  # primary SLO (docs/PREREG-v2.md)


def prediction_intervals(hw_name: str, n_in: int = 512, n_out: int = 128) -> dict[str, tuple]:
    """Machine-readable predictions: key -> (lo, hi, unit). Corners of the efficiency ranges."""
    hw = HARDWARE[hw_name]
    out: dict[str, tuple] = {}
    for prec in PRECISIONS:
        lo, hi = interval(
            lambda m, f, prec=prec: batch1_decode_tok_s(hw, prec, n_in + n_out / 2, m)
        )
        out[f"batch1_decode_tok_s:{prec}"] = (lo, hi, "tok/s")
        lo, hi = interval(
            lambda m, f, prec=prec: saturation_req_s(hw, prec, n_in, n_out, m, f)["req_s"]
        )
        out[f"saturation_req_s:{prec}"] = (lo, hi, "req/s")
        lo, hi = interval(
            lambda m, f, prec=prec: slo_capacity_req_s(hw, prec, n_in, n_out, TPOT_SLO_S, m, f)
        )
        out[f"slo_capacity_req_s:{prec}"] = (lo, hi, "req/s")
        lo, hi = interval(lambda m, f, prec=prec: ttft_unloaded_s(hw, prec, n_in, m, f))
        out[f"unloaded_ttft_s:{prec}"] = (lo, hi, "s")
    return out


def predictions(hw_name: str, n_in: int = 512, n_out: int = 128) -> list[tuple[str, str, str]]:
    hw = HARDWARE[hw_name]
    iv = prediction_intervals(hw_name, n_in, n_out)
    rows = []
    for prec in PRECISIONS:
        lo, hi, _ = iv[f"batch1_decode_tok_s:{prec}"]
        rows.append((f"{prec} batch-1 decode", f"{lo:.0f} - {hi:.0f}", "tok/s"))
    for prec in PRECISIONS:
        lo, hi, _ = iv[f"saturation_req_s:{prec}"]
        rows.append((f"{prec} saturation rate ({n_in}/{n_out})", f"{lo:.1f} - {hi:.1f}", "req/s"))
        lo, hi, _ = iv[f"slo_capacity_req_s:{prec}"]
        rows.append(
            (
                f"{prec} SLO capacity (TPOT <= {TPOT_SLO_S * 1000:.0f} ms)",
                f"{lo:.1f} - {hi:.1f}",
                "req/s",
            )
        )
    for prec in PRECISIONS:
        lo, hi, _ = iv[f"unloaded_ttft_s:{prec}"]
        rows.append(
            (f"{prec} unloaded TTFT ({n_in} in)", f"{lo * 1000:.0f} - {hi * 1000:.0f}", "ms")
        )
    for prec, kvd in (("fp16", "fp16"), ("fp8", "fp16"), ("fp8", "fp8")):
        cap = kv_capacity_tokens(hw, prec, kv_dtype=kvd)
        rows.append(
            (
                f"KV capacity, weights {prec}, KV {kvd}",
                f"{cap // 1000}k",
                f"tokens ({cap // (n_in + n_out)} seqs)",
            )
        )
    return rows


def render_markdown(hw_name: str = "L40S", n_in: int = 512, n_out: int = 128) -> str:
    hw = HARDWARE[hw_name]
    lines = [
        f"# Predictions: {hw_name}, Llama-3.1-8B, {n_in} in / {n_out} out",
        "",
        "Committed before any GPU run. Generated by `python -m tokbench.model`; reproducible.",
        "",
        (
            "**Hardware figures are unverified recollections** "
            f"({hw.mem_gb:.0f} GB, {hw.mem_bw_gbs:.0f} GB/s, {hw.tflops_fp16:.0f} / "
            f"{hw.tflops_fp8:.0f} TFLOPS fp16/fp8 dense, {hw.tdp_w:.0f} W). "
            "Check the vendor datasheet before publishing."
        ),
        "",
        (
            "Model: memory-bound decode (weights + per-sequence KV streamed each step, memory "
            "efficiency 0.65-0.85), compute-bound prefill (MFU 0.35-0.60), fluid continuous "
            "batching limited by KV capacity (90% utilisation, 2.5 GiB overhead) or "
            "max_num_seqs=256. SLO capacity solves for the arrival rate at which TPOT, stretched "
            "by the time spent in prefill, reaches the SLO (TTFT is ignored). Weights-only FP8 keeps embeddings FP16; AWQ-INT4 saves "
            "bandwidth, not FLOPs. Ranges are the corners of the efficiency ranges."
        ),
        "",
        "| Quantity | Predicted | Unit |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in predictions(hw_name, n_in, n_out)],
        "",
        (
            "How these will be used: the README reports each as predicted-vs-measured, marks "
            "which fell inside the interval, and explains the misses. The model is not refitted "
            "to the data."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="L40S", choices=sorted(HARDWARE))
    ap.add_argument("--n-in", type=int, default=512)
    ap.add_argument("--n-out", type=int, default=128)
    ap.add_argument("--write")
    a = ap.parse_args(argv)
    md = render_markdown(a.hw, a.n_in, a.n_out)
    if a.write:
        with open(a.write, "w") as f:
            f.write(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
