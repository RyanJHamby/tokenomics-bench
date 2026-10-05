import math

import pytest

from tokbench.loadgen import percentile, summarize
from tokbench.loadgen.client import RequestRecord


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([7], 99) == 7
    assert math.isnan(percentile([], 50))


def _rec(t_send, ttft, n_out, tpot, ok=True):
    return RequestRecord(
        t_sched=t_send,
        t_send=t_send,
        t_first=t_send + ttft,
        t_last=t_send + ttft + tpot * (n_out - 1),
        n_out=n_out,
        ok=ok,
    )


def test_summarize_discards_warmup_and_counts_failures():
    recs = [_rec(0, 5.0, 10, 0.5), _rec(1, 0.1, 10, 0.02), _rec(2, 0.1, 10, 0.02)]
    recs.append(RequestRecord(t_sched=3, t_send=3, ok=False, error="http 500", t_last=3.1))
    s = summarize(recs, wall_s=10.0, warmup=1)
    assert s["n_requests"] == 3 and s["n_ok"] == 2 and s["n_failed"] == 1
    assert s["errors"] == ["http 500"]
    assert abs(s["ttft_p50"] - 0.1) < 1e-9  # the 5s warmup outlier is gone
    assert abs(s["tpot_p50"] - 0.02) < 1e-9
    assert s["throughput_tok_s"] == 2.0  # explicit wall_s respected


def test_throughput_uses_steady_window_of_kept_requests():
    # warmup request occupies [0, 12]; kept requests span [10, 12.5]. The window must not
    # include the warmup, or throughput is understated by warmup/n.
    recs = [_rec(0, 1.0, 10, 11.0), _rec(10, 0.5, 11, 0.15), _rec(10.5, 0.5, 11, 0.15)]
    s = summarize(recs, warmup=1)
    assert s["window_s"] == pytest.approx(12.5 - 10.0, abs=0.01)
    assert s["throughput_tok_s"] == pytest.approx(22 / s["window_s"])


def test_goodput_counts_failures_as_misses_and_only_good_tokens():
    good = _rec(0, 0.5, 11, 0.05)
    slow_ttft = _rec(1, 3.0, 11, 0.05)
    slow_tpot = _rec(2, 0.5, 11, 0.5)
    failed = RequestRecord(t_sched=3, t_send=3, ok=False, error="truncated", t_last=3.5)
    s = summarize([good, slow_ttft, slow_tpot, failed], ttft_slo=2.0, tpot_slo=0.1)
    assert s["slo_attainment"] == pytest.approx(1 / 4)  # failure is a miss, not dropped
    assert s["goodput_tok_s"] == pytest.approx(11 / s["window_s"])
    assert s["throughput_tok_s"] == pytest.approx(33 / s["window_s"])  # 3 ok requests


def test_p99_sees_a_2pct_tail_and_ignores_failures():
    fast = [_rec(i, 0.1, 5, 0.01) for i in range(196)]
    slow = [_rec(196 + i, 4.0, 5, 0.01) for i in range(4)]  # 2% of 200
    failed = RequestRecord(t_sched=0, t_send=999, ok=False, error="x", t_last=1000)
    assert summarize(fast + slow + [failed])["ttft_p99"] > 1.0
    # one outlier in 99 samples barely moves p99: small-n p99 is nearly the max, so a
    # single stall decides it. This is why cells need thousands of requests.
    few = [_rec(i, 0.1, 5, 0.01) for i in range(98)] + [_rec(98, 4.0, 5, 0.01)]
    assert summarize(few)["ttft_p99"] < 0.5
