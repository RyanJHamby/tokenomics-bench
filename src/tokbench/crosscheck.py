"""Compare this load generator with `vllm bench serve` on the SAME server instance.

If two tools disagree beyond tolerance on identical load, the discrepancy is client-side
(timestamping, event-loop saturation, window definition) and must be understood before any
latency number from either is quoted.

    python -m tokbench.crosscheck <my cell.json> <vllm bench result.json>

Known, expected differences: my numbers use a steady-state window (warmup excluded) while
vllm bench averages the whole run, so throughput can differ by the warmup fraction; tolerances
below allow for that and for run-to-run noise. They are pre-set here, not tuned to a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (label, my summary key, vllm bench key, scale, relative tolerance)
CHECKS = (
    ("output throughput (tok/s)", "throughput_tok_s", "output_throughput", 1.0, 0.10),
    ("TTFT p50 (ms)", "ttft_p50", "median_ttft_ms", 1000.0, 0.15),
    ("TTFT p99 (ms)", "ttft_p99", "p99_ttft_ms", 1000.0, 0.30),
    ("TPOT p50 (ms)", "tpot_p50", "median_tpot_ms", 1000.0, 0.15),
    ("TPOT p99 (ms)", "tpot_p99", "p99_tpot_ms", 1000.0, 0.30),
    ("ITL p99 (ms)", "itl_p99", "p99_itl_ms", 1000.0, 0.30),
)


def compare(mine: dict, bench: dict) -> list[dict]:
    rows = []
    for label, mk, bk, scale, tol in CHECKS:
        if bk not in bench:
            rows.append({"metric": label, "verdict": "missing_in_bench", "bench_key": bk})
            continue
        a, b = mine["summary"][mk] * scale, float(bench[bk])
        rel = (a - b) / b if b else float("inf")
        rows.append(
            {
                "metric": label,
                "mine": a,
                "bench": b,
                "rel_diff": rel,
                "tol": tol,
                "verdict": "agree" if abs(rel) <= tol else "DISAGREE",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("my_cell")
    ap.add_argument("bench_result")
    a = ap.parse_args(argv)
    rows = compare(
        json.loads(Path(a.my_cell).read_text()), json.loads(Path(a.bench_result).read_text())
    )
    for r in rows:
        if "mine" in r:
            print(
                f"{r['metric']:<28} mine={r['mine']:.3f} bench={r['bench']:.3f} "
                f"diff={r['rel_diff']:+.1%} (tol {r['tol']:.0%}) {r['verdict']}"
            )
        else:
            print(f"{r['metric']:<28} {r['verdict']} ({r['bench_key']})")
    return 0 if all(r["verdict"] == "agree" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
