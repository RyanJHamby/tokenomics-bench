"""Pre-registered decision rule (same as measured-speedup-harness): a difference is
'real' only if it beats a minimum effect AND a Welch's t-test. Pure Python.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean, variance


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def welch_p_value(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Welch's t-test p-value."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = variance(a) / na, variance(b) / nb
    if va + vb == 0:
        return 1.0 if mean(a) == mean(b) else 0.0
    t = (mean(a) - mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return stats[int(n_boot * alpha / 2)], stats[int(n_boot * (1 - alpha / 2)) - 1]


@dataclass(frozen=True)
class DecisionResult:
    verdict: str  # "confirmed" | "not_distinguishable"
    rel_change: float  # (candidate - baseline) / baseline
    p_value: float


def decide(
    baseline: Sequence[float],
    candidate: Sequence[float],
    min_effect: float = 0.05,
    alpha: float = 0.05,
) -> DecisionResult:
    """Compare per-repeat values (e.g. p50 TTFT per repeat). Direction-agnostic."""
    base = mean(baseline)
    rel = (mean(candidate) - base) / base
    p = welch_p_value(baseline, candidate)
    real = abs(rel) >= min_effect and p < alpha
    return DecisionResult("confirmed" if real else "not_distinguishable", rel, p)
