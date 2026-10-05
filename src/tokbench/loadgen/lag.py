"""Detect a saturated load-generator.

A single asyncio client parsing many SSE streams (and sharing CPU with vLLM's API server)
can fall behind: timestamps are then taken late, inflating TTFT and distorting inter-token
latency, and the artefact looks like server latency. The monitor sleeps a short interval and
records how late it wakes; that lateness bounds the client's timestamp error. A cell whose
p99 lag exceeds a threshold is flagged invalid rather than trusted.
"""

from __future__ import annotations

import asyncio
import time
from typing import Self

from .stats import percentile


class LoopLagMonitor:
    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.lags: list[float] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while True:
            t = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            self.lags.append(max(0.0, time.perf_counter() - t - self.interval_s))

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    def summary(self, max_p99_ms: float) -> dict:
        p99 = percentile(self.lags, 99) * 1000
        return {
            "loop_lag_p99_ms": p99,
            "loop_lag_max_ms": max(self.lags, default=float("nan")) * 1000,
            "n_samples": len(self.lags),
            "client_ok": bool(self.lags) and p99 <= max_p99_ms,
        }
