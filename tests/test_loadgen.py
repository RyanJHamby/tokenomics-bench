"""Validate the load generator against the mock server's analytic latencies, and that it
refuses to call broken streams successes."""

import pytest
from aiohttp.test_utils import TestServer

from tokbench.loadgen import run_closed_loop, run_open_loop, summarize, summarize_window
from tokbench.mockserver import make_app

TPOT = 0.01
N_OUT = 20


def body(i):
    return {"prompt": list(range(1000, 1016)), "max_tokens": N_OUT, "ignore_eos": True}


async def _serve(**kw):
    server = TestServer(make_app(**{"prefill_s_per_token": 0.0, "tpot_s": TPOT, "slots": 4, **kw}))
    await server.start_server()
    return server, str(server.make_url("/v1/completions"))


@pytest.fixture
async def url():
    server, u = await _serve()
    yield u
    await server.close()


async def test_closed_loop_below_capacity_matches_model(url):
    recs, wall = await run_closed_loop(url, body, concurrency=2, n_requests=12, expect_out=N_OUT)
    s = summarize(recs, wall)
    assert s["n_failed"] == 0 and s["output_tokens"] == 12 * N_OUT
    assert s["tpot_p50"] == pytest.approx(TPOT, rel=0.5)  # asyncio sleep granularity
    assert s["ttft_p50"] == pytest.approx(TPOT, rel=1.0)  # first token = one decode step
    assert s["mean_prompt_tokens"] == 16  # server-reported usage is recorded
    assert s["itl_p99"] > 0 and all(len(r.itls) == N_OUT - 1 for r in recs)


async def test_open_loop_counts_queueing_delay(url):
    # Capacity: 4 slots / (20 tokens * 10ms) = 20 req/s. Offer 2x.
    recs, wall = await run_open_loop(url, body, rate_qps=40, n_requests=60, seed=1)
    over = summarize(recs, wall)
    recs2, wall2 = await run_open_loop(url, body, rate_qps=5, n_requests=20, seed=1)
    under = summarize(recs2, wall2)
    assert over["ttft_p99"] > 5 * under["ttft_p99"]  # overload must show a latency wall
    assert under["ttft_p99"] < 0.2


async def test_http_errors_are_recorded_not_raised():
    recs, _ = await run_closed_loop("http://127.0.0.1:1/v1/completions", body, 1, 2, timeout_s=2)
    assert len(recs) == 2 and not any(r.ok for r in recs)


async def test_truncated_stream_is_a_failure_not_a_success():
    server, u = await _serve(fail_after=5)  # drops after 5 tokens: no usage, no [DONE]
    try:
        recs, wall = await run_closed_loop(u, body, 2, 4, expect_out=N_OUT)
    finally:
        await server.close()
    assert not any(r.ok for r in recs)
    assert {r.error for r in recs} == {"truncated"}
    s = summarize(recs, wall)
    assert s["n_failed"] == 4 and s["errors"] == ["truncated"]


async def test_short_completion_fails_when_length_is_pinned(url):
    recs, _ = await run_closed_loop(url, body, 1, 2, expect_out=N_OUT + 5)
    assert not any(r.ok for r in recs)
    assert all(r.error.startswith("short") for r in recs)


async def test_duration_open_loop_stops_arrivals_at_deadline(url):
    t0 = __import__("time").perf_counter()
    recs, wall = await run_open_loop(url, body, rate_qps=20, duration_s=2.0, seed=3)
    assert 20 <= len(recs) <= 65  # Poisson(40) with wide tolerance
    assert all(r.t_sched - t0 < 2.05 for r in recs)
    assert all(r.ok for r in recs) and wall < 4.0


async def test_overload_with_drain_deadline_is_recorded_as_incomplete_failures():
    # 1 slot, 0.5 s per request => capacity 2 req/s. Offer 10 req/s for 2 s, drain only 0.3 s.
    server, u = await _serve(slots=1, tpot_s=0.05)
    try:
        t0 = __import__("time").perf_counter()
        recs, _ = await run_open_loop(
            u,
            lambda i: {**body(i), "max_tokens": 10},
            rate_qps=10,
            duration_s=2.0,
            drain_s=0.3,
            seed=1,
            expect_out=10,
        )
    finally:
        await server.close()
    s = summarize_window(recs, t0, t0 + 2.0, ttft_slo=1.0, tpot_slo=0.2)
    assert "incomplete" in s["errors"] and s["n_failed"] > 0
    assert s["completed_req_s"] < 0.5 * s["offered_req_s"]  # service rate, not offered load
    assert s["slo_attainment"] < 0.5  # failures are misses
    assert not any(r.t_last is None for r in recs)


async def test_closed_loop_duration_stops_starting_new_requests(url):
    t0 = __import__("time").perf_counter()
    recs, _ = await run_closed_loop(url, body, 2, duration_s=1.0, expect_out=N_OUT)
    assert recs and all(r.t_send - t0 < 1.05 for r in recs) and all(r.ok for r in recs)


async def test_exactly_one_stop_condition_required(url):
    with pytest.raises(ValueError):
        await run_open_loop(url, body, 5)
    with pytest.raises(ValueError):
        await run_closed_loop(url, body, 1, 5, duration_s=1.0)
