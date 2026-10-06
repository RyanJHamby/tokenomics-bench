"""Report which telemetry this machine actually exposes, before any billable run.

    python -m tokbench.telemetry.probe        # prints JSON; preflight stores it in _env/

Optional fields vary by GPU, driver and container: this answers 'what will the sampler really
capture here' instead of discovering gaps after the run.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict


def probe() -> dict:
    out: dict = {"dcgmi": shutil.which("dcgmi"), "nv_hostengine": shutil.which("nv-hostengine")}
    rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"
    out["rapl_readable"] = os.access(rapl, os.R_OK) if os.path.exists(rapl) else False
    try:
        from .gpu import NvmlBackend

        b = NvmlBackend(0)
        s = asdict(b.read())
        out["nvml"] = "ok"
        out["fields"] = {k: ("ok" if v is not None else "UNSUPPORTED") for k, v in s.items()}
        out["energy_counter"] = b._has_energy
        out["sample"] = s
    except Exception as e:  # noqa: BLE001 - the whole point is to report any failure
        out["nvml"] = f"unavailable: {type(e).__name__}: {e}"
    return out


def main() -> int:
    print(json.dumps(probe(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
