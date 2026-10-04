import math

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
    assert abs(s["ttft_p50"] - 0.1) < 1e-9  # the 5s warmup outlier is gone
    assert abs(s["tpot_p50"] - 0.02) < 1e-9
    assert s["throughput_tok_s"] == 2.0
