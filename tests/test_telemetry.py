import time

import pytest

from tokbench.telemetry import (
    FakeBackend,
    GpuSample,
    MetricsScraper,
    PowerSampler,
    energy_between,
    energy_joules,
    parse_prometheus,
    throttle_summary,
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


def test_energy_between_interpolates_window_ends_exactly():
    # constant 100 W sampled at t=0,1,2,3: window [0.5, 2.5] must be exactly 200 J
    s = [GpuSample(t=float(t), power_w=100.0) for t in range(4)]
    e = energy_between(s, 0.5, 2.5)
    assert e["energy_j"] == pytest.approx(200.0)
    assert e["energy_j_counter"] is None


def test_energy_between_prefers_counter_and_reports_both():
    # power says 100 W (100 J over 1 s) but the counter says 150 J: the counter wins.
    s = [
        GpuSample(t=0.0, power_w=100, energy_mj=0.0),
        GpuSample(t=1.0, power_w=100, energy_mj=150_000.0),
        GpuSample(t=2.0, power_w=100, energy_mj=300_000.0),
    ]
    e = energy_between(s, 0.0, 1.0)
    assert e["energy_j"] == pytest.approx(150.0)
    assert e["energy_j_power_integral"] == pytest.approx(100.0)


def test_energy_between_rejects_bad_window():
    with pytest.raises(ValueError):
        energy_between([GpuSample(t=0, power_w=1)], 0, 1)


def test_throttle_power_cap_bit_is_not_a_fault_but_thermal_is():
    cap_only = [GpuSample(t=i, power_w=200, throttle_reasons=0x4) for i in range(4)]
    s = throttle_summary(cap_only)
    assert s["cap_bound_fraction"] == 1.0 and s["bad_throttle_seen"] is False
    hot = cap_only + [GpuSample(t=9, power_w=200, throttle_reasons=0x4 | 0x40)]
    assert throttle_summary(hot)["bad_throttle_seen"] is True
    assert throttle_summary(hot)["throttle_reasons_or"] == 0x44


def test_sampler_death_is_loud_not_silent():
    class Dying:
        n = 0

        def read(self):
            Dying.n += 1
            if Dying.n > 3:
                raise OSError("GPU is lost")
            return GpuSample(t=time.perf_counter(), power_w=100)

    with pytest.raises(RuntimeError, match="sampler died"), PowerSampler(Dying(), hz=100):
        time.sleep(0.2)


def test_prometheus_histogram_buckets_are_not_summed():
    m = parse_prometheus(
        'h_bucket{le="1"} 5\nh_bucket{le="+Inf"} 9\nh_sum 3.5\nh_count 9\nh_created 1e9\n'
    )
    assert "h_bucket" not in m and "h_created" not in m
    assert m["h_sum"] == 3.5 and m["h_count"] == 9


def test_sampler_quality_counts_gaps_and_stale_counter_reads():
    from tokbench.telemetry import sampler_quality

    ts = [0.0, 0.1, 0.2, 0.3, 0.9, 1.0]  # a 0.6 s gap: a dropped stretch at 10 Hz
    samples = [
        GpuSample(t=t, power_w=100, energy_mj=e)
        for t, e in zip(ts, [0, 0, 5, 5, 9, 12], strict=True)
    ]
    q = sampler_quality(samples, hz=10)
    assert q["n"] == 6 and q["dropped"] == 1
    assert q["dt_ms_max"] == pytest.approx(600) and q["dt_ms_p50"] == pytest.approx(100, abs=1)
    assert q["stale_frac"] == pytest.approx(2 / 5)  # two of five consecutive pairs repeat
    assert sampler_quality(samples[:1], 10)["dt_ms_max"] is None
