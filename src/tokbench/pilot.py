"""Size the later blocks from the pilot's launch-to-launch variance.

    python -m tokbench.pilot results/raw/<stamp>-b1

For each metric, the SD of log(value) across independent launches of the same variant. A
paired difference of two variants has SD ~ sqrt(2) x that (independent launches), which
feeds `required_repeats`. If the numbers say more launches are needed than the budget
allows, the pre-registered response is to widen the margin, never to narrow it afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import stdev

from .analysis import required_repeats

MARGINS = (0.05, 0.10, 0.20)


def _by_launch(d: Path, pattern: str, getter) -> list[float]:
    return [getter(json.loads(p.read_text())) for p in sorted(d.glob(pattern))]


def sd_log(values: list[float]) -> float | None:
    vals = [v for v in values if v and v > 0 and not math.isnan(v)]
    return stdev([math.log(v) for v in vals]) if len(vals) >= 2 else None


def pilot_report(d: Path) -> dict:
    metrics = {
        "saturated goodput (c64)": _by_launch(
            d, "*__c64__r*.json", lambda c: c["summary"]["goodput_tok_s"]
        ),
        "saturated J/token (c64)": _by_launch(
            d, "*__c64__r*.json", lambda c: c["energy"]["j_per_output_token"]
        ),
        "saturated p95 power (c64)": _by_launch(
            d, "*__c64__r*.json", lambda c: c["power"]["power_w_p95"]
        ),
        "batch-1 TPOT p50 (c1)": _by_launch(
            d, "*__c1__r*.json", lambda c: c["summary"]["tpot_p50"]
        ),
    }
    caps = _by_launch(d, "capacity__*__r*.json", lambda c: c["capacity"] or float("nan"))
    metrics["SLO capacity (qps)"] = caps
    out = {}
    for name, vals in metrics.items():
        s = sd_log(vals)
        out[name] = {
            "n_launches": len(vals),
            "sd_log": s,
            "required_repeats": (
                {f"{m:.0%}": required_repeats(math.sqrt(2) * s, m) for m in MARGINS} if s else None
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_dir")
    a = ap.parse_args(argv)
    for name, r in pilot_report(Path(a.pilot_dir)).items():
        print(f"{name}: launches={r['n_launches']} sd_log={r['sd_log']}")
        if r["required_repeats"]:
            print(f"    paired launches needed for margin: {r['required_repeats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
