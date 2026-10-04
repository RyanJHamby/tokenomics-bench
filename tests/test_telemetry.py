import time

import pytest

from tokbench.telemetry import (
    FakeBackend,
    GpuSample,
    MetricsScraper,
    PowerSampler,
    energy_joules,
    parse_prometheus,
)

PROM = """# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 12.0
vllm:num_requests_waiting{model_name="m"} 3.0
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:num_preemptions_total{model_name="m",engine="0"} 2.0
vllm:num_preemptions_total{model_name="m",engine="1"} 5.0
bad_line NaN
"""


def test_parse_prometheus_sums_labelsets_and_skips_noise():
    m = parse_prometheus(PROM)
    assert m["vllm:num_requests_running"] == 12.0
    assert m["vllm:num_preemptions_total"] == 7.0
    assert "bad_line" not in m and "vllm:kv_cache_usage_perc" in m


def test_energy_trapezoid_exact():
    s = [GpuSample(t=0, power_w=100), GpuSample(t=1, power_w=200), GpuSample(t=3, power_w=200)]
    assert energy_joules(s) == pytest.approx(150 + 400)


def test_power_sampler_with_fake_backend_and_cap():
    uncapped = FakeBackend(idle_w=50, peak_w=300)
    capped = FakeBackend(idle_w=50, peak_w=300, cap_w=200)
    for be, expect in ((uncapped, 300.0), (capped, 200.0)):
        with PowerSampler(be, hz=50) as ps:
            time.sleep(0.2)
        assert len(ps.samples) >= 5
        wall = ps.samples[-1].t - ps.samples[0].t
        assert energy_joules(ps.samples) == pytest.approx(expect * wall, rel=1e-6)


async def test_metrics_scraper_against_local_server():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    app = web.Application()

    async def metrics(_):
        return web.Response(text=PROM)

    app.router.add_get("/metrics", metrics)
    server = TestServer(app)
    await server.start_server()
    import asyncio

    async with MetricsScraper(str(server.make_url("/metrics")), interval_s=0.05) as sc:
        await asyncio.sleep(0.3)
    await server.close()
    assert len(sc.rows) >= 3 and sc.rows[0][1]["vllm:num_requests_waiting"] == 3.0
