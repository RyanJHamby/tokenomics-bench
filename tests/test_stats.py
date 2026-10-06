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


def _w(t_sched, t_first, t_last, n_out=10, ok=True):
    return RequestRecord(
        t_sched=t_sched, t_send=t_sched, t_first=t_first, t_last=t_last, n_out=n_out, ok=ok
    )


def test_window_attribution_latency_by_arrival_throughput_by_completion():
    from tokbench.loadgen import summarize_window

    recs = [
        _w(5, 5.2, 5.9),  # arrived before the window, completes inside: counts for tokens only
        _w(10, 10.2, 10.9),  # inside entirely: counts for both
        _w(19, 19.3, 20.6),  # arrives inside, completes after: latency yes, tokens no
        _w(25, 25.1, 25.5),  # after the window: neither
    ]
    s = summarize_window(recs, 10.0, 20.0, ttft_slo=1.0, tpot_slo=1.0)
    assert s["n_requests"] == 2 and s["n_ok"] == 2
    assert s["output_tokens"] == 10 and s["throughput_tok_s"] == pytest.approx(1.0)
    assert s["offered_req_s"] == pytest.approx(0.2) and s["completed_req_s"] == pytest.approx(0.1)


def test_window_counts_unfinished_as_failures_and_slo_misses():
    from tokbench.loadgen import summarize_window

    inc = RequestRecord(t_sched=11, t_send=11, ok=False, error="incomplete", t_last=21)
    s = summarize_window([_w(10, 10.1, 10.5), inc], 10.0, 20.0, 1.0, 1.0)
    assert s["n_failed"] == 1 and s["errors"] == ["incomplete"]
    assert s["slo_attainment"] == 0.5


def test_percentile_ci_matches_scipy_binomial_ranks():
    from tokbench.loadgen.stats import percentile_ci

    xs = list(range(1, 1001))  # value == rank
    lo, hi = percentile_ci(xs, 99)  # scipy.stats.binom.ppf(.025/.975, 1000, .99) = 983, 996
    assert (lo, hi) == (983, 996)


def test_percentile_ci_is_wide_for_small_n_and_clips_to_the_max():
    from tokbench.loadgen.stats import percentile_ci

    xs = [float(i) for i in range(1, 201)]
    lo, hi = percentile_ci(xs, 99)
    assert hi == 200.0 and lo < 199  # p99 from 200 samples is essentially the maximum
    assert math.isnan(percentile_ci([], 99)[0])


def test_recovery_time_is_the_start_of_the_first_stable_healthy_run_of_bins():
    from tokbench.loadgen import recovery_time_s

    def rec(t, ttft, ok=True):
        return RequestRecord(
            t_sched=t, t_send=t, t_first=t + ttft, t_last=t + ttft + 0.1, n_out=5, ok=ok
        )

    # overload ends at t=100. bins of 5 s: [100,105) bad, [105,110) bad, [110,115) good,
    # [115,120) good, [120,125) good
    recs = [rec(101, 3.0), rec(103, 4.0), rec(106, 2.5), rec(108, 2.0)] + [
        rec(t, 0.2) for t in (111, 113, 116, 118, 121, 123)
    ]
    assert recovery_time_s(recs, 100.0, 125.0, ttft_slo=1.0, tpot_slo=0.5) == 10.0
    # a lone healthy bin followed by a bad one is not recovery
    flap = [rec(101, 3.0), rec(106, 0.2), rec(111, 3.0), rec(116, 3.0), rec(121, 3.0)]
    assert recovery_time_s(flap, 100.0, 125.0, 1.0, 0.5) is None
    # a failed request in a bin makes it unhealthy even with good latency
    failed = [rec(t, 0.2) for t in (101, 106, 111, 116, 121)] + [rec(102, 0.1, ok=False)]
    assert recovery_time_s(failed, 100.0, 125.0, 1.0, 0.5) == 5.0  # first bin unhealthy, rest fine
