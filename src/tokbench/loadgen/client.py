"""Streaming load generator for OpenAI-compatible chat endpoints.

Closed loop: N workers each send the next request when the last finishes. Offered
load drops as the server slows, which hides queueing delay (coordinated omission).

Open loop: requests arrive on a Poisson schedule regardless of server state, and
TTFT/e2e are measured from the *scheduled* arrival time, so queueing delay is
counted. Use open loop for latency claims.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp


@dataclass
class RequestRecord:
    t_sched: float  # when the request should have started (open loop) or did start
    t_send: float
    t_first: float | None = None
    t_last: float | None = None
    n_out: int = 0
    ok: bool = False
    error: str = ""

    @property
    def ttft(self) -> float:
        return self.t_first - self.t_sched

    @property
    def e2e(self) -> float:
        return self.t_last - self.t_sched

    @property
    def tpot(self) -> float | None:
        if self.n_out < 2 or self.t_first is None or self.t_last is None:
            return None
        return (self.t_last - self.t_first) / (self.n_out - 1)


PromptFn = Callable[[int], dict]  # request index -> chat-completions JSON body


async def _one(
    session: aiohttp.ClientSession, url: str, body: dict, t_sched: float
) -> RequestRecord:
    rec = RequestRecord(t_sched=t_sched, t_send=time.perf_counter())
    chunks = 0
    usage_tokens: int | None = None
    payload = {**body, "stream": True, "stream_options": {"include_usage": True}}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec.error = f"http {resp.status}"
                rec.t_last = time.perf_counter()
                return rec
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                evt = json.loads(data)
                if evt.get("usage"):
                    usage_tokens = evt["usage"].get("completion_tokens")
                choices = evt.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if rec.t_first is None:
                        rec.t_first = now
                    rec.t_last = now
                    chunks += 1
        rec.n_out = usage_tokens if usage_tokens is not None else chunks
        rec.ok = rec.t_first is not None
        if not rec.ok:
            rec.error = "no tokens"
            rec.t_last = rec.t_last or time.perf_counter()
    except Exception as e:  # noqa: BLE001 - record any transport failure
        rec.error = type(e).__name__
        rec.t_last = time.perf_counter()
    return rec


def _session(timeout_s: float, limit: int) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_s),
        connector=aiohttp.TCPConnector(limit=limit),
    )


async def run_closed_loop(
    url: str, make_body: PromptFn, concurrency: int, n_requests: int, timeout_s: float = 300.0
) -> tuple[list[RequestRecord], float]:
    """Returns (records, wall_seconds)."""
    records: list[RequestRecord] = []
    counter = iter(range(n_requests))

    async with _session(timeout_s, concurrency) as session:

        async def worker() -> None:
            for i in counter:
                records.append(await _one(session, url, make_body(i), time.perf_counter()))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        return records, time.perf_counter() - t0


async def run_open_loop(
    url: str,
    make_body: PromptFn,
    rate_qps: float,
    n_requests: int,
    seed: int = 0,
    timeout_s: float = 300.0,
    max_in_flight: int = 4096,
) -> tuple[list[RequestRecord], float]:
    """Poisson arrivals at rate_qps. Returns (records, wall_seconds)."""
    rng = random.Random(seed)
    arrivals, t = [], 0.0
    for _ in range(n_requests):
        t += rng.expovariate(rate_qps)
        arrivals.append(t)

    async with _session(timeout_s, max_in_flight) as session:
        t0 = time.perf_counter()
        tasks = []
        for i, offset in enumerate(arrivals):
            delay = t0 + offset - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(_one(session, url, make_body(i), t0 + offset)))
        records = await asyncio.gather(*tasks)
        return list(records), time.perf_counter() - t0
