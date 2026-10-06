"""Publication figures from saved run data only (no network, no recomputation of results).

    python -m tokbench.figures results/raw/<stamp>-b2 [more dirs] --out results/figures \
        [--analysis results/analysis.json]

Every figure reads the artifacts the runner saved (cell summaries, `.req.jsonl.gz`,
`.gpu.csv.gz`, capacity probes). A figure whose inputs are missing is skipped with a message,
never faked. Anything produced from the mock server or a fake GPU is stamped SYNTHETIC inside
the image so it cannot be mistaken for a measurement.

Design: one axis per chart (no dual axes); identity colours are the first slots of a palette
validated for colour-vision deficiency (blue, orange, aqua), assigned in fixed order; arm
families (baseline / power cap / clock lock) carry the colour in scatter charts, with every
point also labelled directly, so identity is never colour-alone; text is always neutral ink.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .capture import load_gpu_series, load_requests
from .config import load_label
from .loadgen.stats import percentile
from .report import aggregate, load_cells

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated slots 1-4
FAMILY_COLOR = {"baseline": SERIES[0], "power cap": SERIES[1], "clock lock": SERIES[2]}
SYNTH = "SYNTHETIC DATA: mock server and fake GPU. Pipeline demo, not a measurement."


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
        "legend.frameon": False, "legend.labelcolor": INK2, "figure.dpi": 150,
    })  # fmt: skip


def _plain_log(ax, axis: str = "both") -> None:
    """Log axes with plain tick labels at 1-2-5 steps (0.1, 0.2, 0.5, 1), not 1e-1 notation."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for name in ("x", "y") if axis == "both" else (axis,):
        a = getattr(ax, f"{name}axis")
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.set_major_formatter(fmt)
        a.set_minor_formatter(NullFormatter())


def _finish(fig, path: Path, synthetic: bool, note: str = "") -> None:
    if note:
        fig.text(0.012, 0.052 if synthetic else 0.014, note, fontsize=7, color=INK2, ha="left")
    if synthetic:
        fig.text(0.012, 0.012, SYNTH, fontsize=7.5, color=INK, fontweight="bold",
                 bbox={"boxstyle": "square,pad=0.25", "fc": SURFACE, "ec": INK2, "lw": 0.8})  # fmt: skip
    fig.tight_layout(rect=(0, 0.095 if synthetic else 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def family(cell: dict) -> str:
    if cell.get("power_cap_w"):
        return "power cap"
    if cell.get("clock_lock_mhz"):
        return "clock lock"
    return "baseline"


def _by_launch(cells: list[dict], label: str, getter) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for c in cells:
        if load_label(c["load"]) == label and c["load"]["mode"] != "phased_open":
            v = getter(c)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out.setdefault(c["variant"], {})[c["repeat"]] = v
    return out


def _saturation_label(cells: list[dict]) -> str | None:
    closed = [c["load"]["concurrency"] for c in cells if c["load"]["mode"] == "closed"
              and "workload" not in c["load"]]  # fmt: skip
    return f"c{max(closed)}" if closed else None


def _variant_family(cells: list[dict]) -> dict[str, str]:
    return {c["variant"]: family(c) for c in cells}


def _jpt(c):
    return c["energy"]["j_per_output_token"]


def _good(c):
    return c["summary"]["goodput_tok_s"]


# ---------------------------------------------------------------- figures


def fig_latency_vs_load(cells, out: Path, synthetic: bool) -> str | None:
    rows = [
        r for r in aggregate(cells) if r["load"]["mode"] == "open" and not math.isnan(r["ttft_p99"])
    ]
    names = sorted({r["variant"] for r in rows}, key=lambda n: (n != "fp16-default", n))[:4]
    if not rows or not names:
        return None
    slo = cells[0]["slo"]["ttft_s"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for color, v in zip(SERIES, names, strict=False):
        pts = sorted((r for r in rows if r["variant"] == v), key=lambda r: r["load"]["qps"])
        x = [p["load"]["qps"] for p in pts]
        y = [p["ttft_p99"] for p in pts]
        lo = [
            max(0.0, p["ttft_p99"] - p["ttft_p99_lo"])
            if p["ttft_p99_lo"] == p["ttft_p99_lo"]
            else 0
            for p in pts
        ]
        hi = [
            max(0.0, p["ttft_p99_hi"] - p["ttft_p99"])
            if p["ttft_p99_hi"] == p["ttft_p99_hi"]
            else 0
            for p in pts
        ]
        ax.errorbar(x, y, yerr=[lo, hi], color=color, marker="o", ms=5, capsize=2, elinewidth=1,
                    mec=SURFACE, mew=1.2, label=v)  # fmt: skip
        ax.annotate(v, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK2,
                    fontsize=8, va="center")  # fmt: skip
    ax.axhline(slo, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log(ax)
    ax.annotate(f"TTFT SLO {slo:g} s", (1, slo), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", color=INK2, fontsize=8, ha="right")  # fmt: skip
    ax.set_xlabel("offered load (requests/s, open loop)")
    ax.set_ylabel("p99 time to first token (s)")
    ax.set_title("Tail latency vs offered load")
    if len(names) > 1:
        ax.legend(loc="lower right", fontsize=8)
    extra = len({r["variant"] for r in rows}) - len(names)
    _finish(fig, out / "01_latency_vs_load.png", synthetic,
            f"median across launches; bars = 95% CI on p99 (order statistics){f'; {extra} more variants not shown' if extra > 0 else ''}")  # fmt: skip
    return "01_latency_vs_load.png"


def _pareto(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Non-dominated: lower J/token and higher goodput."""
    return sorted(
        (
            p
            for p in points
            if not any(
                o[1] <= p[1] and o[2] >= p[2] and (o[1], o[2]) != (p[1], p[2]) for o in points
            )
        ),
        key=lambda p: p[1],
    )


def fig_goodput_vs_energy(cells, out: Path, synthetic: bool) -> str | None:
    label = _saturation_label(cells)
    if not label:
        return None
    j, g = _by_launch(cells, label, _jpt), _by_launch(cells, label, _good)
    fam = _variant_family(cells)
    names = [n for n in j if n in g]
    if len(names) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = []
    for n in names:
        jv, gv = list(j[n].values()), list(g[n].values())
        mj, mg = median(jv), median(gv)
        pts.append((n, mj, mg))
        ax.errorbar(mj, mg, xerr=[[mj - min(jv)], [max(jv) - mj]], yerr=[[mg - min(gv)], [max(gv) - mg]],
                    fmt="o", ms=7, color=FAMILY_COLOR[fam[n]], mec=SURFACE, mew=1.5, elinewidth=1, capsize=2)  # fmt: skip
        ax.annotate(n, (mj, mg), xytext=(7, 5), textcoords="offset points", color=INK2, fontsize=8)
    front = _pareto(pts)
    if len(front) > 1:
        ax.step(
            [p[1] for p in front], [p[2] for p in front], where="post", color=INK2, lw=1, ls=":"
        )
    for f, color in FAMILY_COLOR.items():
        if any(fam[n] == f for n in names):
            ax.scatter([], [], color=color, label=f, s=30)
    ax.margins(0.14)
    ax.legend(loc="upper right", title="arm family", title_fontsize=8, fontsize=8)
    ax.set_xlabel("GPU energy per output token (J), lower is better")
    ax.set_ylabel("goodput (tok/s within SLO), higher is better")
    ax.set_title(f"Goodput vs energy per token, saturation ({label})")
    _finish(fig, out / "02_goodput_vs_energy.png", synthetic,
            "points = median across launches; bars = min-max; dotted line = Pareto front")  # fmt: skip
    return "02_goodput_vs_energy.png"


def fig_energy_vs_clock(cells, out: Path, synthetic: bool) -> str | None:
    label = _saturation_label(cells)
    if not label:
        return None
    fam = _variant_family(cells)
    clk = _by_launch(
        cells, label, lambda c: c["power"]["sm_clock_mhz_mean"] or None
    )  # 0 = not measured
    j = _by_launch(cells, label, _jpt)
    names = [n for n in j if clk.get(n)]
    if len(names) < 3 or not any(fam[n] != "baseline" for n in names):
        return None
    if len({round(median(clk[n].values())) for n in names}) < 2:
        return None  # all arms at one clock: nothing to plot against
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    locks = sorted(
        (median(clk[n].values()), median(j[n].values()), n)
        for n in names
        if fam[n] in ("clock lock", "baseline")
    )
    ax.plot([p[0] for p in locks], [p[1] for p in locks], color=SERIES[2], lw=1.4, alpha=0.9)
    for n in names:
        x, y = median(clk[n].values()), median(j[n].values())
        ax.scatter([x], [y], s=46, color=FAMILY_COLOR[fam[n]], edgecolor=SURFACE, linewidth=1.2,
                   marker="o" if fam[n] != "power cap" else "D", zorder=3)  # fmt: skip
        ax.annotate(n, (x, y), xytext=(6, 5), textcoords="offset points", color=INK2, fontsize=8)
    for f, color in FAMILY_COLOR.items():
        if any(fam[n] == f for n in names):
            ax.scatter([], [], color=color, label=f, s=30, marker="D" if f == "power cap" else "o")
    ax.legend(loc="best")
    ax.set_xlabel("mean SM clock during the measurement window (MHz)")
    ax.set_ylabel("GPU energy per output token (J)")
    ax.set_title(f"Energy per token vs achieved SM clock ({label})")
    _finish(fig, out / "03_energy_vs_clock.png", synthetic,
            "x is the MEASURED clock, not the nominal lock or cap; line joins baseline and lock arms")  # fmt: skip
    return "03_energy_vs_clock.png"


def fig_power_timeline(
    cells, out: Path, run_dir: Path, synthetic: bool, repeat: int = 0
) -> str | None:
    label = _saturation_label(cells)
    if not label:
        return None
    chosen = [c for c in cells if load_label(c["load"]) == label and c["repeat"] == repeat]
    chosen.sort(key=lambda c: (family(c) != "baseline", c["variant"]))
    chosen = [
        c for c in chosen if (run_dir / f"{c['variant']}__{label}__r{repeat}.gpu.csv.gz").exists()
    ][:3]
    if not chosen:
        return None
    series = {
        c["variant"]: load_gpu_series(run_dir / f"{c['variant']}__{label}__r{repeat}.gpu.csv.gz")
        for c in chosen
    }
    candidates = (
        ("power_w", "board power (W)"),
        ("sm_clock_mhz", "SM clock (MHz)"),
        ("temp_c", "GPU temperature (C)"),
    )
    # a panel whose values are all zero was not measured (fake GPU): omit it, never plot zeros
    panels = [
        (col, yl)
        for col, yl in candidates
        if any(float(r[col]) != 0 for rows in series.values() for r in rows)
    ]
    if not panels:
        return None
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7.2, 2.4 * len(panels) + 0.8), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for color, c in zip(SERIES, chosen, strict=False):
        rows = series[c["variant"]]
        x = [float(r["t_mono"]) - c["clock"]["t0"] for r in rows]
        for ax, (col, _) in zip(axes, panels, strict=True):
            ax.plot(x, [float(r[col]) for r in rows], color=color, lw=1.4, label=c["variant"])
    c0 = chosen[0]
    for ax, (_, yl) in zip(axes, panels, strict=True):
        ax.axvspan(c0["clock"]["ws"] - c0["clock"]["t0"], c0["clock"]["we"] - c0["clock"]["t0"],
                   color=GRID, alpha=0.55, lw=0, zorder=0)  # fmt: skip
        ax.set_ylabel(yl)
    axes[0].set_title(f"Power, clock and temperature through one {label} cell")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axes[-1].set_xlabel("seconds since the load started (shaded = measurement window)")
    _finish(
        fig,
        out / "04_power_timeline.png",
        synthetic,
        f"launch {repeat}; panels with no measurement (all zero) are omitted",
    )
    return "04_power_timeline.png"


def fig_ttft_cdf(cells, out: Path, run_dir: Path, synthetic: bool) -> str | None:
    opens = sorted(
        {load_label(c["load"]) for c in cells if c["load"]["mode"] == "open"},
        key=lambda s: float(s[1:].split("-")[0]),
    )
    if not opens:
        return None
    label = opens[-1]  # heaviest open load: where tails differ
    files = sorted(run_dir.glob(f"*__{label}__r0.req.jsonl.gz"))[:4]
    if not files:
        return None
    slo = cells[0]["slo"]["ttft_s"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for color, f in zip(SERIES, files, strict=False):
        header, recs = load_requests(f)
        ttft = sorted(r.ttft for r in recs if r.ok and header["ws"] <= r.t_sched < header["we"])
        if not ttft:
            continue
        name = f.name.split("__")[0]
        ax.plot(ttft, [(i + 1) / len(ttft) for i in range(len(ttft))], color=color,
                label=f"{name} (p99 {percentile(ttft, 99):.3g} s)")  # fmt: skip
    ax.axvline(slo, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    _plain_log(ax, "x")
    ax.set_xlabel("time to first token (s)")
    ax.set_ylabel("fraction of requests")
    ax.set_title(f"TTFT distribution at {label} requests/s (launch 0)")
    ax.annotate(f"TTFT SLO {slo:g} s", (slo, 0.02), xytext=(-5, 0), textcoords="offset points",
                color=INK2, fontsize=8, rotation=90, ha="right", va="bottom")  # fmt: skip
    ax.legend(loc="lower right", bbox_to_anchor=(0.9, 0.04), fontsize=8)
    _finish(fig, out / "05_ttft_cdf.png", synthetic, "every request in the measurement window")
    return "05_ttft_cdf.png"


def fig_launch_variance(cells, out: Path, synthetic: bool) -> str | None:
    label = _saturation_label(cells)
    if not label:
        return None
    g = _by_launch(cells, label, _good)
    fam = _variant_family(cells)
    if not g or max(len(v) for v in g.values()) < 2:
        return None
    names = sorted(g)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.8))
    for i, n in enumerate(names):
        vals = list(g[n].values())
        ax.scatter(
            vals,
            [i] * len(vals),
            s=40,
            color=FAMILY_COLOR[fam[n]],
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, zorder=1)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("goodput (output tok/s meeting the SLO); one dot per independent launch")
    ax.set_title(f"Launch-to-launch spread at saturation ({label})")
    _finish(
        fig, out / "06_launch_variance.png", synthetic, "the unit of replication is a server launch"
    )
    return "06_launch_variance.png"


def fig_model_check(analysis: dict, out: Path, synthetic: bool) -> str | None:
    rows = analysis.get("model_check") or []
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.8))
    for i, r in enumerate(rows):
        lo, hi = r["predicted"]
        ax.plot(
            [lo, hi], [i, i], color=SERIES[0], lw=5, solid_capstyle="butt", alpha=0.85, zorder=2
        )
        ax.scatter([r["measured"]], [i], marker="D" if r["in_interval"] else "X", s=55,
                   color=INK, zorder=3)  # fmt: skip
        ax.annotate("inside" if r["in_interval"] else "MISS", (max(hi, r["measured"]), i),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)  # fmt: skip
    ax.set_yticks(range(len(rows)), [f"{r['quantity']} ({r['unit']})" for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    _plain_log(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("value (log scale); bar = predicted interval, marker = measured")
    ax.set_title("Predicted vs measured (model not refit)")
    _finish(
        fig, out / "07_model_check.png", synthetic, "diamond = inside the interval, cross = miss"
    )
    return "07_model_check.png"


def fig_capacity_probes(run_dirs: list[Path], out: Path, synthetic: bool) -> str | None:
    probes: dict[str, list[tuple[float, bool]]] = {}
    for d in run_dirs:
        for p in sorted(d.glob("capacity__*__r*.json")):
            name = p.name.split("__")[1]
            probes.setdefault(f"{name} r{p.stem.rsplit('__r', 1)[1]}", []).extend(
                (q, bool(ok)) for q, ok in json.loads(p.read_text()).get("probes", [])
            )
    if not probes:
        return None
    names = sorted(probes)
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.8))
    for i, n in enumerate(names):
        for q, ok in probes[n]:
            ax.scatter([q], [i], marker="o" if ok else "X", s=34, color=SERIES[0] if ok else INK,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)  # fmt: skip
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    _plain_log(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("probed offered load (requests/s, log); circle = SLO held, cross = violated")
    ax.set_title("Capacity search: doubling then bisection")
    _finish(fig, out / "08_capacity_probes.png", synthetic, "each marker is one measured probe")
    return "08_capacity_probes.png"


def make_figures(run_dirs: list[Path], out: Path, analysis_path: Path | None = None) -> dict:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    cells = [c for d in run_dirs for c in load_cells(d)]
    if not cells and not any(list(d.glob("capacity__*")) for d in run_dirs):
        raise SystemExit("no cells found in the given directories")
    synthetic = any(c.get("synthetic") for c in cells)
    primary = run_dirs[0]
    made, skipped = [], []
    steps = [
        ("latency_vs_load", lambda: fig_latency_vs_load(cells, out, synthetic) if cells else None),
        (
            "goodput_vs_energy",
            lambda: fig_goodput_vs_energy(cells, out, synthetic) if cells else None,
        ),
        ("energy_vs_clock", lambda: fig_energy_vs_clock(cells, out, synthetic) if cells else None),
        (
            "power_timeline",
            lambda: fig_power_timeline(cells, out, primary, synthetic) if cells else None,
        ),
        ("ttft_cdf", lambda: fig_ttft_cdf(cells, out, primary, synthetic) if cells else None),
        ("launch_variance", lambda: fig_launch_variance(cells, out, synthetic) if cells else None),
        ("capacity_probes", lambda: fig_capacity_probes(run_dirs, out, synthetic)),
    ]
    if analysis_path and analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        steps.append(("model_check", lambda: fig_model_check(analysis, out, synthetic)))
    for name, fn in steps:
        res = fn()
        (made if res else skipped).append(res or name)
    (out / "INDEX.md").write_text(
        "# Figures\n\n" + ("**SYNTHETIC: pipeline demo, not a result.**\n\n" if synthetic else "")
        + "\n".join(f"- ![{m}]({m})" for m in made) + ("\n\nSkipped (inputs missing): " + ", ".join(skipped) if skipped else "") + "\n")  # fmt: skip
    return {"made": made, "skipped": skipped, "synthetic": synthetic}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis")
    a = ap.parse_args(argv)
    info = make_figures(
        [Path(d) for d in a.run_dirs], Path(a.out), Path(a.analysis) if a.analysis else None
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
