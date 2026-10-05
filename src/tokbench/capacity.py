"""Find a variant's SLO capacity: the highest open-loop rate that still meets the SLO.

Replaces coarse QPS grids, where "cheapest feasible" would just be the highest grid point
that happens to pass. Doubling to bracket the knee, then bisection to a fixed relative
resolution (pre-registered), then a confirmation probe, because a single passing probe at
the boundary can be luck.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

Probe = Callable[[float], Awaitable[bool]]


async def find_capacity(
    passes: Probe, lo: float, hi: float, resolution: float, max_confirm_steps: int = 3
) -> dict:
    """`passes(qps)` runs one measured probe and returns whether the SLO held with zero
    failures. Assumes pass/fail is monotone in qps (checked by the confirmation probe)."""
    probes: list[tuple[float, bool]] = []

    async def test(q: float) -> bool:
        ok = await passes(q)
        probes.append((q, ok))
        return ok

    if not await test(lo):
        return {"capacity": None, "bracketed": False, "probes": probes}
    last_ok, first_bad = lo, None
    while last_ok < hi:
        nxt = min(last_ok * 2, hi)
        if await test(nxt):
            last_ok = nxt
        else:
            first_bad = nxt
            break
    if first_bad is None:  # never failed up to hi: capacity is at least hi
        return {"capacity": last_ok, "bracketed": False, "probes": probes}
    a, b = last_ok, first_bad
    while (b - a) / a > resolution:
        mid = (a + b) / 2
        if await test(mid):
            a = mid
        else:
            b = mid
    for _ in range(max_confirm_steps):  # re-test the claimed capacity; back off if it was luck
        if await test(a):
            return {"capacity": a, "bracketed": True, "upper_fail": b, "probes": probes}
        a *= 1 - resolution
    return {
        "capacity": a,
        "bracketed": True,
        "upper_fail": b,
        "probes": probes,
        "unconfirmed": True,
    }


def update_capacity_file(path: str | Path, variant: str, qps: float) -> None:
    """Record capacity conservatively: keep the minimum across repeats, write atomically."""
    p = Path(path)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[variant] = min(qps, cur.get(variant, qps))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, p)
