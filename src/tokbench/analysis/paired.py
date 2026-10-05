"""Pre-registered comparison procedure (prereg-v2).

Estimand: for variants A and B at the same load, the per-launch log ratio
d_r = log(m_A,r / m_B,r) of a metric (p99 TTFT, goodput, J/token, ...). A launch (server
restart) is the unit of replication: cells inside one launch share its state, so they are
not independent samples. We summarise d_r with a t interval (exact under normal d_r; with
n_launches this small the bootstrap has poor coverage and is not used), and issue a verdict
against a pre-registered margin rather than a bare significance test:

  superior_*     CI lies entirely beyond +/- log(1+margin)
  equivalent     CI lies entirely inside (-log(1+margin), +log(1+margin))   (TOST-equivalent)
  inconclusive   anything else; this is reported as such, never as "no difference"

`required_repeats` sizes the design from a pilot's launch-to-launch SD. Pure Python.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, stdev


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


def betainc(a: float, b: float, x: float) -> float:
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


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student t with df degrees of freedom."""
    p_two = betainc(df / 2.0, 0.5, df / (df + t * t))  # P(|T| > |t|)
    return p_two / 2.0 if t >= 0 else 1.0 - p_two / 2.0


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student t (bisection on the survival function)."""
    if not 0.5 <= p < 1.0:
        raise ValueError("t_ppf supports 0.5 <= p < 1")
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > 1.0 - p:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError("t_ppf did not bracket")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_sf(mid, df) > 1.0 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_log_ratio: float
    ratio: float  # exp(mean): A relative to B
    ci_lo: float  # CI on the ratio scale
    ci_hi: float
    p_value: float  # two-sided paired t, for Holm across the pre-registered family only
    verdict: str


def paired_log_ratio(
    a: Sequence[float], b: Sequence[float], margin: float = 0.05, conf: float = 0.95
) -> PairedResult:
    """A vs B from per-launch values, paired by launch index (same repeat of each)."""
    if len(a) != len(b):
        raise ValueError("a and b must be paired (same number of launches)")
    n = len(a)
    if n < 2:
        raise ValueError("need >= 2 paired launches")
    if any(x <= 0 or y <= 0 or math.isnan(x) or math.isnan(y) for x, y in zip(a, b, strict=True)):
        raise ValueError("values must be positive and finite")
    d = [math.log(x / y) for x, y in zip(a, b, strict=True)]
    m, sd = mean(d), stdev(d)
    se = sd / math.sqrt(n)
    tcrit = t_ppf(1 - (1 - conf) / 2, n - 1)
    lo, hi = m - tcrit * se, m + tcrit * se
    p = 1.0 if se == 0 and m == 0 else (0.0 if se == 0 else 2 * t_sf(abs(m / se), n - 1))
    thr = math.log(1 + margin)
    if lo > thr:
        verdict = "superior_higher"
    elif hi < -thr:
        verdict = "superior_lower"
    elif lo > -thr and hi < thr:
        verdict = "equivalent"
    else:
        verdict = "inconclusive"
    return PairedResult(n, m, math.exp(m), math.exp(lo), math.exp(hi), p, verdict)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns reject flags in input order. Apply only across
    the small pre-registered family of primary comparisons."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    m = len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def required_repeats(sd_log: float, margin: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Paired launches needed to detect a `margin` relative effect given the pilot SD of
    per-launch log ratios. Normal-approximation power with the t critical value, iterated
    over n (agrees with the exact noncentral-t answer to within one launch)."""
    if sd_log <= 0 or margin <= 0:
        raise ValueError("sd_log and margin must be positive")
    delta = math.log(1 + margin)
    z_beta = _norm_ppf(power)
    for n in range(2, 1000):
        tcrit = t_ppf(1 - alpha / 2, n - 1)
        if n >= ((tcrit + z_beta) * sd_log / delta) ** 2:
            return n
    raise ValueError("effect too small to detect with < 1000 launches")


def _norm_ppf(p: float) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
