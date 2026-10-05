"""Latency statistics over request records. Pure Python, no numpy."""

from __future__ import annotations

import math
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


def steady_window(recs: Sequence[RequestRecord]) -> float:
    """Seconds from first kept send to last kept completion. Tokens and time use the
    same requests, so throughput is not biased by discarded warmup or the drain."""
    if not recs:
        return float("nan")
    return max(r.t_last for r in recs) - min(r.t_send for r in recs)


def summarize(
    records: Sequence[RequestRecord],
    wall_s: float | None = None,
    warmup: int = 0,
    ttft_slo: float | None = None,
    tpot_slo: float | None = None,
) -> dict:
    """Aggregate records. The first `warmup` requests by send time are discarded.

    Throughput is over the steady window of the kept requests unless `wall_s` is given.
    With SLOs, `slo_attainment` is the fraction of ALL kept requests (failures count as
    misses) that met both, and `goodput_tok_s` counts only tokens from those requests.
    """
    recs = sorted(records, key=lambda r: r.t_send)[warmup:]
    ok = [r for r in recs if r.ok]
    window = wall_s if wall_s is not None else steady_window(recs)
    out_tokens = sum(r.n_out for r in ok)
    ttft = [r.ttft for r in ok]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    itl = [g for r in ok for g in r.itls]
    out = {
        "n_requests": len(recs),
        "n_ok": len(ok),
        "n_failed": len(recs) - len(ok),
        "errors": sorted({r.error for r in recs if r.error}),
        "window_s": window,
        "output_tokens": out_tokens,
        "mean_prompt_tokens": (
            sum(r.prompt_tokens for r in ok if r.prompt_tokens) / max(1, len(ok))
            if any(r.prompt_tokens for r in ok)
            else None
        ),
        "throughput_tok_s": out_tokens / window if window and window > 0 else float("nan"),
        "ttft_p50": percentile(ttft, 50),
        "ttft_p99": percentile(ttft, 99),
        "tpot_p50": percentile(tpot, 50),
        "tpot_p99": percentile(tpot, 99),
        "itl_p99": percentile(itl, 99),
        "e2e_p50": percentile([r.e2e for r in ok], 50),
        "e2e_p99": percentile([r.e2e for r in ok], 99),
    }
    if ttft_slo is not None and tpot_slo is not None:
        good = [r for r in ok if r.ttft <= ttft_slo and (r.tpot is None or r.tpot <= tpot_slo)]
        out["slo_attainment"] = len(good) / len(recs) if recs else float("nan")
        gt = sum(r.n_out for r in good)
        out["goodput_tok_s"] = gt / window if window and window > 0 else float("nan")
        out["goodput_req_s"] = len(good) / window if window and window > 0 else float("nan")
    return out


def summarize_window(
    records: Sequence[RequestRecord],
    ws: float,
    we: float,
    ttft_slo: float,
    tpot_slo: float,
) -> dict:
    """Steady-state summary over the measurement window [ws, we) (perf_counter seconds).

    Latency/attainment cover requests that *arrived* in the window; every one that did
    not finish cleanly (including 'incomplete' at the drain deadline) is a failure and an
    SLO miss, so overload cannot hide. Throughput and goodput count tokens of requests
    that *completed* in the window, divided by the window length, so under overload they
    show service rate rather than offered load.
    """
    span = we - ws
    arrived = [r for r in records if ws <= r.t_sched < we]
    ok = [r for r in arrived if r.ok]
    done = [r for r in records if r.ok and ws <= r.t_last <= we]

    def good(r: RequestRecord) -> bool:
        return r.ttft <= ttft_slo and (r.tpot is None or r.tpot <= tpot_slo)

    done_good = [r for r in done if good(r)]
    tokens = sum(r.n_out for r in done)
    ttft = [r.ttft for r in ok]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    itl = [g for r in ok for g in r.itls]
    pt = [r.prompt_tokens for r in ok if r.prompt_tokens]
    n = len(arrived)
    return {
        "n_requests": n,
        "n_ok": len(ok),
        "n_failed": n - len(ok),
        "n_completed": len(done),
        "errors": sorted({r.error for r in arrived if r.error}),
        "window_s": span,
        "offered_req_s": n / span,
        "completed_req_s": len(done) / span,
        "output_tokens": tokens,
        "mean_prompt_tokens": sum(pt) / len(pt) if pt else None,
        "throughput_tok_s": tokens / span,
        "goodput_tok_s": sum(r.n_out for r in done_good) / span,
        "goodput_req_s": len(done_good) / span,
        "slo_attainment": sum(good(r) for r in ok) / n if n else float("nan"),
        "ttft_p50": percentile(ttft, 50),
        "ttft_p99": percentile(ttft, 99),
        "tpot_p50": percentile(tpot, 50),
        "tpot_p99": percentile(tpot, 99),
        "itl_p99": percentile(itl, 99),
        "e2e_p50": percentile([r.e2e for r in ok], 50),
        "e2e_p99": percentile([r.e2e for r in ok], 99),
    }


def finite(x: float | None) -> bool:
    return x is not None and not math.isnan(x) and not math.isinf(x)
