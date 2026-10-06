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
    """Polls /metrics. `rows` are parsed (summed across label sets) at `interval_s`; `raw`
    keeps the exposition text (labels, histogram buckets) every `raw_every_s` and at the end,
    so anything the parsed rows dropped can still be recovered."""

    def __init__(self, url: str, interval_s: float = 0.5, raw_every_s: float = 15.0):
        self.url, self.interval_s, self.raw_every_s = url, interval_s, raw_every_s
        self.rows: list[tuple[float, dict[str, float]]] = []
        self.raw: list[tuple[float, str]] = []
        self.errors = 0
        self._task: asyncio.Task | None = None

    async def _scrape(self, session: aiohttp.ClientSession) -> None:
        try:
            async with session.get(self.url) as r:
                text = await r.text()
            t = time.perf_counter()
            self.rows.append((t, parse_prometheus(text)))
            if not self.raw or t - self.raw[-1][0] >= self.raw_every_s:
                self.raw.append((t, text))
        except (aiohttp.ClientError, TimeoutError, ValueError):
            self.errors += 1

    async def _run(self) -> None:
        async with aiohttp.ClientSession() as s:
            while True:
                await self._scrape(s)
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
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
            await self._scrape(s)  # end-of-load state
            if self.rows and (not self.raw or self.raw[-1][0] != self.rows[-1][0]):
                pass  # last parsed row is kept; raw text only at intervals + first
