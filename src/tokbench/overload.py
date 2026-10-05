"""Derive B7 overload loads from B1 results, so overload QPS is a pre-registered
multiple of *measured* SLO capacity rather than a number guessed in advance.

    python -m tokbench.overload results/b1-... --variant fp16-default \
        --ttft-slo 2 --tpot-slo 0.1 --multiples 1.2 1.5 2.0
"""

from __future__ import annotations

import argparse

import yaml

from .report import aggregate, load_cells


def capacity_qps(rows: list[dict], variant: str, ttft_slo: float, tpot_slo: float) -> float | None:
    """Highest open-loop QPS (no failed or invalid cells) whose p99 TTFT and TPOT meet the
    SLO. Assumes latency is monotone in load; stops at the first violation going up.
    NaN or missing measurements are violations, never passes."""
    pts = sorted(
        (r for r in rows if r["variant"] == variant and r["load"]["mode"] == "open"),
        key=lambda r: r["load"]["qps"],
    )
    best = None
    for r in pts:
        meets = r["ttft_p99"] <= ttft_slo and r["tpot_p99"] <= tpot_slo  # NaN -> False
        if r["any_failures"] or r.get("invalid") or not meets:
            break
        best = r["load"]["qps"]
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("b1_dir")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ttft-slo", type=float, default=2.0)
    ap.add_argument("--tpot-slo", type=float, default=0.1)
    ap.add_argument("--multiples", type=float, nargs="+", default=[1.2, 1.5, 2.0])
    ap.add_argument("--template", default="configs/b7_overload.template.yaml")
    a = ap.parse_args(argv)
    cap = capacity_qps(aggregate(load_cells(a.b1_dir)), a.variant, a.ttft_slo, a.tpot_slo)
    if cap is None:
        raise SystemExit("no open-loop load met the SLO in B1; cannot derive overload loads")
    with open(a.template) as f:
        cfg = yaml.safe_load(f)
    cfg["loads"] = [{"mode": "open", "qps": round(cap * m, 3)} for m in a.multiples]
    cfg["derived_from"] = {
        "b1_dir": a.b1_dir,
        "variant": a.variant,
        "capacity_qps": cap,
        "multiples": a.multiples,
    }
    print(yaml.safe_dump(cfg, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
