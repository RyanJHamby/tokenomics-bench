"""Aggregate per-cell JSON results into frontier points and the headline chart."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .analysis import Config, cheapest_feasible, pareto


def load_cells(results_dir: str | Path) -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in sorted(Path(results_dir).glob("*.json"))
        if p.name != "manifest.json"
    ]


def _med(cs: list[dict], f) -> float:
    return median(f(c) for c in cs)


def aggregate(cells: list[dict]) -> list[dict]:
    """Group by (variant, load); median across repeats. A cell with failed requests
    is kept but flagged, never silently dropped."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["variant"], json.dumps(c["load"], sort_keys=True))].append(c)
    rows = []
    for (variant, _), cs in groups.items():
        rows.append(
            {
                "variant": variant,
                "load": cs[0]["load"],
                "n_repeats": len(cs),
                "throughput_tok_s": _med(cs, lambda c: c["summary"]["throughput_tok_s"]),
                "ttft_p99": _med(cs, lambda c: c["summary"]["ttft_p99"]),
                "tpot_p99": _med(cs, lambda c: c["summary"]["tpot_p99"]),
                "j_per_token": _med(cs, lambda c: c["j_per_token"]),
                "gpu_usd_per_hr": cs[0]["gpu_usd_per_hr"],
                "any_failures": any(c["summary"]["n_failed"] for c in cs),
                "synthetic": any(c["synthetic"] for c in cs),
            }
        )
    return rows


def to_configs(rows: list[dict]) -> list[Config]:
    return [
        Config(
            name=f"{r['variant']}@{r['load'].get('qps', r['load'].get('concurrency'))}",
            throughput_tok_s=r["throughput_tok_s"],
            ttft_p99_s=r["ttft_p99"],
            tpot_p99_s=r["tpot_p99"],
            energy_j=r["j_per_token"] * 1000,
            output_tokens=1000,
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
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for v in sorted({r["variant"] for r in rows}):
        pts = sorted((r for r in rows if r["variant"] == v), key=lambda r: r["ttft_p99"])
        ax.plot([p["ttft_p99"] for p in pts], [p["j_per_token"] for p in pts], marker="o", label=v)
    ax.axvline(ttft_slo, color="gray", ls="--", lw=1)
    ax.text(ttft_slo, ax.get_ylim()[1], " SLO", va="top", color="gray")
    if best:
        b = next(
            r
            for r in rows
            if f"{r['variant']}@{r['load'].get('qps', r['load'].get('concurrency'))}" == best.name
        )
        ax.scatter(
            [b["ttft_p99"]], [b["j_per_token"]], s=160, facecolors="none", edgecolors="black"
        )
    ax.set_xscale("log")
    ax.set_xlabel("p99 TTFT (s, log scale)")
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
    info = frontier_chart(rows, a.png, a.ttft_slo, a.tpot_slo)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
