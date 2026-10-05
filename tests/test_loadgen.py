"""Validate the load generator against the mock server's analytic latencies, and that it
refuses to call broken streams successes."""

import pytest
from aiohttp.test_utils import TestServer

from tokbench.loadgen import run_closed_loop, run_open_loop, summarize
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
