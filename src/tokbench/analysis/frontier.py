"""Cost/energy accounting and the SLO-feasible frontier."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str  # unique: "<variant>@<load label>", e.g. "fp8@q8" or "fp8@c8"
    mode: str  # "open" or "closed"
    throughput_tok_s: float
    goodput_tok_s: float  # tokens/s from requests that met both SLOs
    ttft_p99_s: float
    tpot_p99_s: float
    j_per_token: float  # board energy / output tokens, same window as throughput
    n_failed: int
    gpu_usd_per_hr: float

    @property
    def usd_per_mtok(self) -> float:
        return usd_per_mtok(self.gpu_usd_per_hr, self.throughput_tok_s)

    @property
    def usd_per_mtok_good(self) -> float:
        """Cost per million tokens that actually met the SLO (the number to quote)."""
        return usd_per_mtok(self.gpu_usd_per_hr, self.goodput_tok_s)

    def meets(self, ttft_slo_s: float, tpot_slo_s: float) -> bool:
        """Inclusive SLO check. NaN never meets (a missing measurement is not a pass)."""
        return self.ttft_p99_s <= ttft_slo_s and self.tpot_p99_s <= tpot_slo_s


def joules_per_token(energy_j: float, output_tokens: int) -> float:
    return energy_j / output_tokens if output_tokens else float("inf")


def usd_per_mtok(gpu_usd_per_hr: float, tokens_per_s: float) -> float:
    if not tokens_per_s or math.isnan(tokens_per_s) or tokens_per_s <= 0:
        return float("inf")
    return gpu_usd_per_hr / (tokens_per_s * 3600.0) * 1e6


def _eligible(c: Config) -> bool:
    """Open-loop, zero-failure cells only: closed-loop TTFT excludes queueing delay, and
    p99 over survivors would hide failed requests."""
    return c.mode == "open" and c.n_failed == 0


def pareto(configs: list[Config]) -> list[Config]:
    """Non-dominated set over (ttft_p99, usd_per_mtok_good, j_per_token); lower is better.
    Only eligible cells compete."""
    pool = [c for c in configs if _eligible(c)]

    def key(c: Config) -> tuple[float, float, float]:
        return (c.ttft_p99_s, c.usd_per_mtok_good, c.j_per_token)

    return [
        c
        for c in pool
        if not any(
            all(x <= y for x, y in zip(key(o), key(c), strict=True)) and key(o) != key(c)
            for o in pool
        )
    ]


def cheapest_feasible(
    configs: list[Config], ttft_slo_s: float, tpot_slo_s: float, by: str = "usd_per_mtok_good"
) -> Config | None:
    """Lowest `by` among eligible configs meeting the SLO; None if none do."""
    ok = [c for c in configs if _eligible(c) and c.meets(ttft_slo_s, tpot_slo_s)]
    return min(ok, key=lambda c: getattr(c, by)) if ok else None
