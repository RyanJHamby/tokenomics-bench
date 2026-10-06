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


async def test_last_token_in_an_empty_text_finish_chunk_still_sets_t_last():
    """vLLM can emit the final token as a chunk with text='' (mid-UTF-8, special token,
    ignore_eos past EOS) carrying finish_reason; t_last must move to that chunk."""
    import asyncio
    import json as _json

    from aiohttp import web

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(3):
            await asyncio.sleep(0.02)
            await resp.write(f"data: {_json.dumps({'choices': [{'text': f't{i}'}]})}\n\n".encode())
        await asyncio.sleep(0.2)  # the last token takes a while, then arrives with no text
        await resp.write(
            f"data: {_json.dumps({'choices': [{'text': '', 'finish_reason': 'length'}]})}\n\n".encode()
        )
        await resp.write(
            f"data: {_json.dumps({'choices': [], 'usage': {'prompt_tokens': 4, 'completion_tokens': 4}})}\n\n".encode()
        )
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        recs, _ = await run_closed_loop(
            str(server.make_url("/v1/completions")),
            lambda i: {"prompt": [1, 2, 3, 4]},
            1,
            1,
            expect_out=4,
        )
    finally:
        await server.close()
    r = recs[0]
    assert r.ok and r.n_out == 4
    assert r.t_last - r.t_first >= 0.23  # 2 x 20 ms + the 200 ms before the empty-text last token


async def test_http_error_bodies_are_kept_for_diagnosis():
    from aiohttp import web

    async def handler(request):
        return web.json_response({"error": "token id 99999 out of vocab"}, status=400)

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        recs, _ = await run_closed_loop(
            str(server.make_url("/v1/completions")), lambda i: {"prompt": [1]}, 1, 1
        )
    finally:
        await server.close()
    assert recs[0].error.startswith("http 400") and "out of vocab" in recs[0].error


async def test_phased_open_loop_changes_arrival_rate_between_phases(url):
    import time as _t

    from tokbench.loadgen import run_phased_open_loop

    t0 = _t.perf_counter()
    recs, _ = await run_phased_open_loop(url, body, [(40, 1.0), (5, 2.0)], seed=2, expect_out=N_OUT)
    first = [r for r in recs if r.t_sched - t0 < 1.0]
    second = [r for r in recs if r.t_sched - t0 >= 1.0]
    assert len(first) > 2 * len(second) / 2 and len(first) >= 15  # ~40/s for 1 s vs ~5/s for 2 s
    assert 2 <= len(second) <= 25 and all(r.t_sched - t0 < 3.1 for r in recs)
