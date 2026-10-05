"""Generate the power-cap and clock-lock arms from PILOT measurements (pre-registered rule).

Caps and locks are not hand-picked: a cap above the GPU's natural decode draw never binds
(so it tests nothing), and one below the card's minimum is refused by the driver. The rule,
fixed in docs/PREREG.md before any run:

  caps   = {90, 80, 70, 60}% of the median saturated p95 board power from the pilot's
           closed-loop c=64 cells, rounded to 1 W, dropped if below the card's min limit
  locks  = {85, 70, 55}% of the card's max SM clock, rounded to 15 MHz
  plus an unmodified baseline arm.

    python -m tokbench.arms results/raw/<pilot> --template configs/b2_power_clock.template.yaml \
        > configs/b2_power_clock.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import yaml

CAP_FRACS = (0.90, 0.80, 0.70, 0.60)
LOCK_FRACS = (0.85, 0.70, 0.55)
CLOCK_STEP_MHZ = 15


def power_caps(p95_w: float, min_w: float, max_w: float) -> tuple[list[int], list[int]]:
    """Returns (caps to run, caps dropped for being outside [min, max])."""
    keep, dropped = [], []
    for f in CAP_FRACS:
        w = round(p95_w * f)
        if min_w <= w <= max_w:
            if w not in keep:
                keep.append(w)
        else:
            dropped.append(w)
    return keep, dropped


def clock_locks(max_sm_mhz: float) -> list[int]:
    out = []
    for f in LOCK_FRACS:
        mhz = int(round(max_sm_mhz * f / CLOCK_STEP_MHZ) * CLOCK_STEP_MHZ)
        if mhz not in out:
            out.append(mhz)
    return out


def saturated_p95_power(pilot_dir: Path) -> float:
    vals = []
    for p in sorted(pilot_dir.glob("*__c64__r*.json")):
        d = json.loads(p.read_text())
        vals.append(d["power"]["power_w_p95"])
    if not vals:
        raise SystemExit(f"no closed-loop c64 cells in {pilot_dir}; run the pilot first")
    return median(vals)


def build(template: dict, p95_w: float, info: dict) -> dict:
    base = template["variants"][0]
    caps, dropped = power_caps(p95_w, info["power.min_limit"], info["power.max_limit"])
    variants = [base]
    variants += [{**base, "name": f"cap{w}w", "power_cap_w": w} for w in caps]
    variants += [
        {**base, "name": f"lock{m}", "clock_lock_mhz": m}
        for m in clock_locks(info["clocks.max.sm"])
    ]
    return {
        **template,
        "variants": variants,
        "derived_from": {
            "saturated_p95_w": p95_w,
            "caps_dropped_below_min": dropped,
            "gpu_info": info,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_dir")
    ap.add_argument("--template", required=True)
    a = ap.parse_args(argv)
    d = Path(a.pilot_dir)
    info = json.loads((d / "manifest.json").read_text()).get("gpu_info")
    if not info:
        raise SystemExit("pilot manifest has no gpu_info (was it run on a GPU box?)")
    cfg = build(yaml.safe_load(Path(a.template).read_text()), saturated_p95_power(d), info)
    print(yaml.safe_dump(cfg, sort_keys=False, width=140))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
