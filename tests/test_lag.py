import asyncio
import time

from tokbench.loadgen import LoopLagMonitor


async def test_idle_loop_has_small_lag_and_is_ok():
    async with LoopLagMonitor(0.005) as m:
        await asyncio.sleep(0.4)
    s = m.summary(max_p99_ms=50)
    assert s["n_samples"] > 20 and s["client_ok"] and s["loop_lag_p99_ms"] < 50


async def test_a_blocked_loop_is_detected_and_flags_the_client_invalid():
    async def hog():
        for _ in range(6):
            time.sleep(0.05)  # noqa: ASYNC251  CPU-bound work in the event loop: what a saturated client does
            await asyncio.sleep(0)

    async with LoopLagMonitor(0.005) as m:
        await hog()
        await asyncio.sleep(0.05)
    s = m.summary(max_p99_ms=10)
    assert s["loop_lag_max_ms"] >= 40 and s["loop_lag_p99_ms"] > 10 and not s["client_ok"]


async def test_no_samples_is_not_ok():
    m = LoopLagMonitor()
    assert not m.summary(10)["client_ok"]
