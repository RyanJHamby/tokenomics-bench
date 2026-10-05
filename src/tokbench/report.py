"""Aggregate per-cell JSON results (schema 2) into frontier points and the chart."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from .analysis import Config, cheapest_feasible, pareto
from .config import load_label


def load_cells(results_dir: str | Path) -> list[dict]:
    cells = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name == "manifest.json":
            continue
        d = json.loads(p.read_text())
        if d.get("schema_version") != 2:
            raise ValueError(f"{p}: unsupported schema {d.get('schema_version')}")
        cells.append(d)
    return cells


def _med(vals: list[float | None]) -> float:
    """Median over finite values only; NaN if none. Order-independent (plain median of a
    list containing NaN depends on where the NaN sits)."""
    xs = [v for v in vals if v is not None and not math.isnan(v)]
    return median(xs) if xs else float("nan")


def aggregate(cells: list[dict]) -> list[dict]:
    """Group by (variant, load); median across repeats. Cells with failed requests are
    flagged, never silently dropped."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["variant"], load_label(c["load"]))].append(c)
    rows = []
    for (variant, label), cs in sorted(groups.items()):
        s = lambda k, cs=cs: _med([c["summary"].get(k) for c in cs])
        rows.append(
            {
                "variant": variant,
                "label": label,
                "load": cs[0]["load"],
                "n_repeats": len(cs),
                "throughput_tok_s": s("throughput_tok_s"),
                "goodput_tok_s": s("goodput_tok_s"),
                "slo_attainment": s("slo_attainment"),
                "ttft_p99": s("ttft_p99"),
                "tpot_p99": s("tpot_p99"),
                "itl_p99": s("itl_p99"),
                "j_per_token": _med([c["energy"]["j_per_output_token"] for c in cs]),
                "n_failed": sum(c["summary"]["n_failed"] for c in cs),
                "any_failures": any(c["summary"]["n_failed"] for c in cs),
                "bad_throttle": any(c["power"]["bad_throttle_seen"] for c in cs),
                "invalid": not all(
                    c["workload_ok"] and c["sampler_ok"] and c["client"]["client_ok"] for c in cs
                ),
                "gpu_usd_per_hr": cs[0]["gpu_usd_per_hr"],
                "synthetic": any(c["synthetic"] for c in cs),
            }
        )
    return rows


def to_configs(rows: list[dict]) -> list[Config]:
    return [
        Config(
            name=f"{r['variant']}@{r['label']}",
            mode=r["load"]["mode"],
            throughput_tok_s=r["throughput_tok_s"],
            goodput_tok_s=r["goodput_tok_s"],
            ttft_p99_s=r["ttft_p99"],
            tpot_p99_s=r["tpot_p99"],
            j_per_token=r["j_per_token"],
            # an invalid cell (workload mismatch / dead sampler) counts as failed
            n_failed=r["n_failed"] + (1 if r["invalid"] else 0),
            gpu_usd_per_hr=r["gpu_usd_per_hr"],
        )
        for r in rows
    ]


def frontier_chart(rows: list[dict], out_png: str, ttft_slo: float, tpot_slo: float) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synthetic = any(r["synthetic"] for r in rows)
    cfgs = to_configs(rows)
    best = cheapest_feasible(cfgs, ttft_slo, tpot_slo)
    open_rows = [r for r in rows if r["load"]["mode"] == "open" and not math.isnan(r["ttft_p99"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for v in sorted({r["variant"] for r in open_rows}):
        pts = sorted((r for r in open_rows if r["variant"] == v), key=lambda r: r["ttft_p99"])
        ax.plot([p["ttft_p99"] for p in pts], [p["j_per_token"] for p in pts], marker="o", label=v)
    ax.axvline(ttft_slo, color="gray", ls="--", lw=1)
    ax.annotate(
        "TTFT SLO",
        (ttft_slo, 1),
        xycoords=("data", "axes fraction"),
        ha="right",
        va="top",
        color="gray",
        rotation=90,
    )
    if best:
        b = next(r for r in open_rows if f"{r['variant']}@{r['label']}" == best.name)
        ax.scatter([b["ttft_p99"]], [b["j_per_token"]], s=160, facecolors="none", edgecolors="k")
    ax.set_xscale("log")
    ax.set_xlabel("p99 TTFT (s, log scale), open-loop cells only")
    ax.set_ylabel("GPU energy per output token (J)")
    ax.set_title("Energy per token vs tail latency" + ("  [SYNTHETIC DATA]" if synthetic else ""))
    ax.legend(title="variant")
    fig.tight_layout()
    fig.savefig(out_png)
    return {
        "cheapest_feasible": best.name if best else None,
        "pareto": [c.name for c in pareto(cfgs)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--png", required=True)
    ap.add_argument("--ttft-slo", type=float, default=2.0)
    ap.add_argument("--tpot-slo", type=float, default=0.1)
    a = ap.parse_args(argv)
    rows = aggregate(load_cells(a.results_dir))
    print(json.dumps(frontier_chart(rows, a.png, a.ttft_slo, a.tpot_slo), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
