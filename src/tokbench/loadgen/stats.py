"""Latency statistics over request records. Pure Python, no numpy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import RequestRecord


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(records: Sequence[RequestRecord], wall_s: float, warmup: int = 0) -> dict:
    """Aggregate records. The first `warmup` requests by send time are discarded."""
    recs = sorted(records, key=lambda r: r.t_send)[warmup:]
    ok = [r for r in recs if r.ok]
    ttft = [r.ttft for r in ok]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    e2e = [r.e2e for r in ok]
    out_tokens = sum(r.n_out for r in ok)
    return {
        "n_requests": len(recs),
        "n_ok": len(ok),
        "n_failed": len(recs) - len(ok),
        "wall_s": wall_s,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / wall_s if wall_s > 0 else float("nan"),
        "ttft_p50": percentile(ttft, 50),
        "ttft_p99": percentile(ttft, 99),
        "tpot_p50": percentile(tpot, 50),
        "tpot_p99": percentile(tpot, 99),
        "e2e_p50": percentile(e2e, 50),
        "e2e_p99": percentile(e2e, 99),
    }
