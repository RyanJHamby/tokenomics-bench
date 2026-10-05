"""Execute the frozen analysis plan (docs/PREREG-v2.md) mechanically: P1-P6, the derived
finding, the model check, and the list of excluded cells. No choices are made here that
the pre-registration did not already make; arms are picked by rule, not by looking.

    python -m tokbench.analyze --b1 DIR --b2 DIR --b3a-fp8 DIR --b3b DIR --b4 DIR \
        --gsm8k-dir results/gates/<stamp> --out results/analysis.json

Any input may be omitted; tests that need it report `insufficient_data` instead of guessing.
Holm correction is applied across the superiority tests (P1a, P1b, P4, P6). Equivalence tests
(P2, P3) are intersection-union (TOST) at alpha=.05 each and are not multiplicity-adjusted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from .analysis import holm, paired_log_ratio
from .config import load_label
from .model import prediction_intervals
from .quality import paired_accuracy
from .report import load_cells

BASELINE_B2 = "uncapped"
FP16 = "fp16-default"


def partition_valid(cells: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validity rules from the pre-registration. Excluded cells are returned, not hidden."""
    good, bad = [], []
    for c in cells:
        reasons = []
        if not c["workload_ok"]:
            reasons.append("workload_mismatch")
        if not c["sampler_ok"]:
            reasons.append("sampler")
        if not c["client"]["client_ok"]:
            reasons.append("client_saturated")
        if c["power"]["bad_throttle_seen"]:
            reasons.append("thermal_or_hw_throttle")
        hard = [e for e in c["summary"]["errors"] if e != "incomplete"]
        if hard:
            reasons.append(f"errors:{hard}")
        (bad if reasons else good).append(
            c
            if not reasons
            else {
                "variant": c["variant"],
                "load": c["load"],
                "repeat": c["repeat"],
                "reasons": reasons,
            }
        )
    return good, bad


def by_repeat(cells: list[dict], variant: str, label: str, getter) -> dict[int, float]:
    out = {}
    for c in cells:
        if c["variant"] == variant and load_label(c["load"]) == label:
            v = getter(c)
            if v is not None:
                out[c["repeat"]] = v
    return out


def _pair(a: dict[int, float], b: dict[int, float]) -> tuple[list[float], list[float]]:
    common = sorted(set(a) & set(b))
    return [a[r] for r in common], [b[r] for r in common]


def run_test(tid, claim, a: dict, b: dict, margin: float, expect: tuple[str, ...]) -> dict:
    xs, ys = _pair(a, b)
    base = {"id": tid, "claim": claim, "margin": margin, "expect": list(expect), "n": len(xs)}
    if len(xs) < 2:
        return {**base, "status": "insufficient_data", "supported": None}
    r = paired_log_ratio(xs, ys, margin)
    return {
        **base,
        "status": "ok",
        "ratio": r.ratio,
        "ci": [r.ci_lo, r.ci_hi],
        "p_value": r.p_value,
        "verdict": r.verdict,
        "supported": r.verdict in expect,
    }


def _gp(c):
    return c["summary"]["goodput_tok_s"]  # noqa: E704


def _jpt(c):
    return c["energy"]["j_per_output_token"]  # noqa: E704


def _tpot50(c):
    return c["summary"]["tpot_p50"]  # noqa: E704


def _open_label(cells: list[dict]) -> str | None:
    labels = sorted({load_label(c["load"]) for c in cells if c["load"]["mode"] == "open"})
    return labels[0] if len(labels) == 1 else None


def power_clock_tests(cells: list[dict]) -> tuple[list[dict], dict]:
    """P1-P3 and the derived finding. Arms are chosen by the pre-registered rule."""
    tests: list[dict] = []
    info: dict = {}
    base_c64 = [c for c in cells if c["variant"] == BASELINE_B2 and load_label(c["load"]) == "c64"]
    if not base_c64:
        return tests, {"note": "no baseline c64 cells"}
    max_sm = base_c64[0]["manifest"].get("gpu_info", {}).get("clocks.max.sm")
    p95 = median(c["power"]["power_w_p95"] for c in base_c64)
    locks = {c["variant"]: c["clock_lock_mhz"] for c in cells if c.get("clock_lock_mhz")}
    caps = {c["variant"]: c["power_cap_w"] for c in cells if c.get("power_cap_w")}
    lock70 = min(locks, key=lambda v: abs(locks[v] - 0.70 * max_sm)) if locks and max_sm else None
    cap80 = min(caps, key=lambda v: abs(caps[v] - 0.80 * p95)) if caps else None
    info.update(lock70=lock70, cap80=cap80, saturated_p95_w=p95, max_sm_mhz=max_sm)
    loads = {"saturation": "c64"}
    ol = _open_label([c for c in cells if c["variant"] == BASELINE_B2])
    if ol:
        loads["0.7xcapacity"] = ol
    for tag, label in loads.items():
        b_j = by_repeat(cells, BASELINE_B2, label, _jpt)
        b_g = by_repeat(cells, BASELINE_B2, label, _gp)
        if lock70:
            tests.append(
                run_test(
                    f"P1[{tag}]",
                    f"{lock70} cuts J/token vs baseline",
                    by_repeat(cells, lock70, label, _jpt),
                    b_j,
                    0.10,
                    ("superior_lower",),
                )
            )
            tests.append(
                run_test(
                    f"P2[{tag}]",
                    f"{lock70} goodput equivalent to baseline",
                    by_repeat(cells, lock70, label, _gp),
                    b_g,
                    0.05,
                    ("equivalent",),
                )
            )
        if cap80:
            tests.append(
                run_test(
                    f"P3j[{tag}]",
                    f"{cap80} J/token equivalent to baseline",
                    by_repeat(cells, cap80, label, _jpt),
                    b_j,
                    0.05,
                    ("equivalent",),
                )
            )
            tests.append(
                run_test(
                    f"P3g[{tag}]",
                    f"{cap80} goodput equivalent to baseline",
                    by_repeat(cells, cap80, label, _gp),
                    b_g,
                    0.05,
                    ("equivalent",),
                )
            )
    # Derived finding at SATURATION (amended pre-run: at 0.7 x capacity every arm meets the SLO,
    # so goodput equals offered load and cannot rank arms by cost).
    arms = [BASELINE_B2, *locks]
    med = {
        a: (
            median(by_repeat(cells, a, "c64", _jpt).values() or [float("nan")]),
            median(by_repeat(cells, a, "c64", _gp).values() or [float("nan")]),
        )
        for a in arms
        if by_repeat(cells, a, "c64", _jpt)
    }
    if len(med) >= 2:
        e_opt = min(med, key=lambda a: med[a][0])
        c_opt = max(med, key=lambda a: med[a][1])
        d = {"energy_optimal": e_opt, "cost_optimal": c_opt, "differ": e_opt != c_opt}
        if e_opt != c_opt:
            d["j_per_token"] = run_test(
                "D[j]",
                "energy-optimal arm lower J/token than cost-optimal",
                by_repeat(cells, e_opt, "c64", _jpt),
                by_repeat(cells, c_opt, "c64", _jpt),
                0.10,
                ("superior_lower",),
            )
            d["goodput"] = run_test(
                "D[g]",
                "energy-optimal arm lower goodput than cost-optimal",
                by_repeat(cells, e_opt, "c64", _gp),
                by_repeat(cells, c_opt, "c64", _gp),
                0.05,
                ("superior_lower",),
            )
            d["holds"] = bool(d["j_per_token"]["supported"] and d["goodput"]["supported"])
        info["derived_finding"] = d
    return tests, info


def capacity_by_repeat(d: Path, variant: str) -> dict[int, float]:
    out = {}
    for p in sorted(d.glob(f"capacity__{variant}__r*.json")):
        j = json.loads(p.read_text())
        if j.get("capacity") is not None:
            out[int(p.stem.rsplit("__r", 1)[1])] = j["capacity"]
    return out


def model_check(b1_dir, b1_cells, b3b_cells, b3a_dir, hw: str) -> list[dict]:
    iv = prediction_intervals(hw)
    rows = []

    def add(name, key, measured):
        if measured is None:
            return
        lo, hi, unit = iv[key]
        rows.append(
            {
                "quantity": name,
                "predicted": [lo, hi],
                "unit": unit,
                "measured": measured,
                "in_interval": lo <= measured <= hi,
            }
        )

    def med_of(cells, variant, label, fn):
        v = list(by_repeat(cells, variant, label, fn).values())
        return median(v) if v else None

    for variant, prec, cells in (
        (FP16, "fp16", b1_cells),
        (FP16, "fp16", b3b_cells),
        ("fp8", "fp8", b3b_cells),
        ("awq-int4", "awq-int4", b3b_cells),
    ):
        t = med_of(cells, variant, "c1", _tpot50)
        add(f"{variant} batch-1 decode", f"batch1_decode_tok_s:{prec}", 1.0 / t if t else None)
        ttft = med_of(cells, variant, "c1", lambda c: c["summary"]["ttft_p50"])
        add(f"{variant} unloaded TTFT", f"unloaded_ttft_s:{prec}", ttft)
    for d, variant, prec in ((b1_dir, FP16, "fp16"), (b3a_dir, "fp8", "fp8")):
        if d:
            caps = capacity_by_repeat(Path(d), variant)
            add(
                f"{variant} SLO capacity",
                f"slo_capacity_req_s:{prec}",
                median(caps.values()) if caps else None,
            )
    return rows


def analyze(args) -> dict:
    res: dict = {"tests": [], "excluded": {}}
    load = lambda d: partition_valid(load_cells(d)) if d else ([], [])
    b1, b1x = load(args.b1)
    b2, b2x = load(args.b2)
    b3b, b3bx = load(args.b3b)
    b4, b4x = load(args.b4)
    res["excluded"] = {"b1": b1x, "b2": b2x, "b3b": b3bx, "b4": b4x}

    if b2:
        t, info = power_clock_tests(b2)
        res["tests"] += t
        res["b2"] = info
    if args.b1 and args.b3a_fp8:  # P4: capacity, fp8 vs fp16, paired by launch
        res["tests"].append(
            run_test(
                "P4",
                "FP8 raises SLO capacity vs FP16",
                capacity_by_repeat(Path(args.b3a_fp8), "fp8"),
                capacity_by_repeat(Path(args.b1), FP16),
                0.20,
                ("superior_higher",),
            )
        )
    if args.gsm8k_dir:  # P5
        g = Path(args.gsm8k_dir)
        try:
            base = json.loads((g / f"gsm8k-{FP16}.json").read_text())
            cand = json.loads((g / "gsm8k-fp8.json").read_text())
            if base["items"] != cand["items"]:
                raise ValueError("different item sets")
            r = paired_accuracy(base["correct"], cand["correct"], 0.02)
            res["tests"].append(
                {
                    "id": "P5",
                    "claim": "FP8 non-inferior on GSM8K (+/-2pp)",
                    "status": "ok",
                    **r,
                    "supported": r["verdict"] in ("non_inferior", "equivalent"),
                }
            )
        except (OSError, ValueError, KeyError) as e:
            res["tests"].append(
                {"id": "P5", "status": "insufficient_data", "supported": None, "reason": str(e)}
            )
    if b4:  # P6: benefit larger at c1 than c64 (paired difference of log ratios)
        none, full = "graphs-none-compile-on", "graphs-full-and-piecewise"
        r1 = {
            r: by_repeat(b4, none, "c1", _tpot50).get(r) / by_repeat(b4, full, "c1", _tpot50)[r]
            for r in by_repeat(b4, none, "c1", _tpot50)
            if r in by_repeat(b4, full, "c1", _tpot50)
        }
        r64 = {
            r: by_repeat(b4, none, "c64", _tpot50).get(r) / by_repeat(b4, full, "c64", _tpot50)[r]
            for r in by_repeat(b4, none, "c64", _tpot50)
            if r in by_repeat(b4, full, "c64", _tpot50)
        }
        res["tests"].append(
            run_test(
                "P6", "graph benefit larger at c=1 than c=64", r1, r64, 0.0, ("superior_higher",)
            )
        )
    sup = [
        t for t in res["tests"] if t["id"].startswith(("P1", "P4", "P6")) and t["status"] == "ok"
    ]
    rej = holm([t["p_value"] for t in sup]) if sup else []
    for t, r in zip(sup, rej, strict=True):
        t["holm_reject"] = r
        t["supported_after_holm"] = bool(t["supported"] and r)
    if args.b1 or args.b3b:
        hw = args.hw
        res["model_check"] = model_check(args.b1, b1, b3b, args.b3a_fp8, hw)
    return res


def render(res: dict) -> str:
    L = [
        "# Analysis (pre-registered procedure, docs/PREREG-v2.md)",
        "",
        "| Test | Claim | n | Ratio (95% CI) | Verdict | Supported |",
        "|---|---|---|---|---|---|",
    ]
    for t in res["tests"]:
        if t["status"] != "ok":
            L.append(
                f"| {t['id']} | {t.get('claim', '')} | {t.get('n', '-')} | - | insufficient data | - |"
            )
        elif "ratio" in t:
            ci = f"{t['ci'][0]:.3f}-{t['ci'][1]:.3f}"
            sup = (
                t["supported"]
                if "supported_after_holm" not in t
                else f"{t['supported']} (Holm: {t['supported_after_holm']})"
            )
            L.append(
                f"| {t['id']} | {t['claim']} | {t['n']} | {t['ratio']:.3f} ({ci}) | {t['verdict']} | {sup} |"
            )
        else:
            L.append(
                f"| {t['id']} | {t['claim']} | {t['n']} | diff {t['diff']:+.3f} ({t['ci'][0]:+.3f}..{t['ci'][1]:+.3f}) | {t['verdict']} | {t['supported']} |"
            )
    L += [
        "",
        "`inconclusive` means the data cannot distinguish the claim from its negation; it is not 'no difference'.",
    ]
    d = res.get("b2", {}).get("derived_finding")
    if d:
        L += [
            "",
            f"Derived finding (saturation): energy-optimal arm `{d['energy_optimal']}`, cost-optimal arm `{d['cost_optimal']}`, differ: {d['differ']}, claim holds: {d.get('holds')}",
        ]
    if res.get("model_check"):
        L += [
            "",
            "## Model check (docs/PREDICTIONS.md, not refit)",
            "",
            "| Quantity | Predicted | Measured | In interval |",
            "|---|---|---|---|",
        ]
        for r in res["model_check"]:
            L.append(
                f"| {r['quantity']} | {r['predicted'][0]:.3g}-{r['predicted'][1]:.3g} {r['unit']} | {r['measured']:.3g} | {r['in_interval']} |"
            )
    ex = {k: v for k, v in res["excluded"].items() if v}
    if ex:
        L += ["", "## Excluded cells (validity rules)", ""]
        for k, v in ex.items():
            L += [
                f"- {k}: {x['variant']} {load_label(x['load'])} r{x['repeat']}: {x['reasons']}"
                for x in v
            ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("b1", "b2", "b3a-fp8", "b3b", "b4", "gsm8k-dir"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--hw", default="L40S")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    a.b3a_fp8, a.gsm8k_dir = a.b3a_fp8, a.gsm8k_dir
    res = analyze(a)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
