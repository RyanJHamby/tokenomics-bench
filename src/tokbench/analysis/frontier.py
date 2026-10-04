"""Cost/energy accounting and the SLO-feasible frontier (the headline result)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str
    throughput_tok_s: float
    ttft_p99_s: float
    tpot_p99_s: float
    energy_j: float
    output_tokens: int
    gpu_usd_per_hr: float

    @property
    def j_per_token(self) -> float:
        return joules_per_token(self.energy_j, self.output_tokens)

    @property
    def usd_per_mtok(self) -> float:
        return usd_per_mtok(self.gpu_usd_per_hr, self.throughput_tok_s)

    def meets(self, ttft_slo_s: float, tpot_slo_s: float) -> bool:
        return self.ttft_p99_s <= ttft_slo_s and self.tpot_p99_s <= tpot_slo_s


def joules_per_token(energy_j: float, output_tokens: int) -> float:
    return energy_j / output_tokens if output_tokens else float("inf")


def usd_per_mtok(gpu_usd_per_hr: float, throughput_tok_s: float) -> float:
    if throughput_tok_s <= 0:
        return float("inf")
    return gpu_usd_per_hr / (throughput_tok_s * 3600.0) * 1e6


def pareto(configs: list[Config]) -> list[Config]:
    """Non-dominated set over (ttft_p99, usd_per_mtok, j_per_token); lower is better."""

    def key(c: Config) -> tuple[float, float, float]:
        return (c.ttft_p99_s, c.usd_per_mtok, c.j_per_token)

    return [
        c
        for c in configs
        if not any(
            all(x <= y for x, y in zip(key(o), key(c))) and key(o) != key(c) for o in configs
        )
    ]


def cheapest_feasible(
    configs: list[Config], ttft_slo_s: float, tpot_slo_s: float, by: str = "usd_per_mtok"
) -> Config | None:
    """Lowest `by` among configs meeting the SLO; None if none do."""
    ok = [c for c in configs if c.meets(ttft_slo_s, tpot_slo_s)]
    return min(ok, key=lambda c: getattr(c, by)) if ok else None
