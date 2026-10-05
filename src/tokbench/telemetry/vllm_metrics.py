"""Scrape vLLM's Prometheus /metrics endpoint to explain latency by server state.

Metric names vary across vLLM versions (e.g. gpu_cache_usage_perc was renamed
kv_cache_usage_perc), so callers pass the names they want and the exact names seen
are recorded with each run's hardware manifest.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Self

import aiohttp

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples per metric name across label sets. Skips comments and NaN."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name.endswith(("_bucket", "_created")):  # histogram buckets are not sums
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if math.isnan(v):
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


class MetricsScraper:
    def __init__(self, url: str, interval_s: float = 0.5):
        self.url, self.interval_s = url, interval_s
        self.rows: list[tuple[float, dict[str, float]]] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        async with aiohttp.ClientSession() as s:
            while True:
                try:
                    async with s.get(self.url) as r:
                        self.rows.append((time.perf_counter(), parse_prometheus(await r.text())))
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    pass
                await asyncio.sleep(self.interval_s)

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc) -> None:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
