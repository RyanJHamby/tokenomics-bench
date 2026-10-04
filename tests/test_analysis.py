import pytest

from tokbench.analysis import (
    Config,
    bootstrap_ci,
    cheapest_feasible,
    decide,
    joules_per_token,
    pareto,
    usd_per_mtok,
    welch_p_value,
)

A = [100.474, 106.25, 95.343, 104.962, 98.704]
B = [105.646, 125.098, 109.418, 107.614]


def test_welch_matches_scipy_reference():
    # scipy.stats.ttest_ind(A, B, equal_var=False).pvalue == 0.08809802235816977
    assert welch_p_value(A, B) == pytest.approx(0.08809802235816977, rel=1e-6)


def test_decision_requires_effect_and_significance():
    # ~8% slower but p ~ 0.09: not distinguishable under alpha=0.05
    assert decide(A, B).verdict == "not_distinguishable"
    # tiny effect, hugely significant: still rejected by the min-effect floor
    base = [100.0, 100.1, 99.9, 100.0, 100.05]
    cand = [101.0, 101.1, 100.9, 101.0, 101.05]
    assert decide(base, cand).verdict == "not_distinguishable"
    # big effect, tight variance: confirmed
    cand = [120.0, 120.1, 119.9, 120.0, 120.05]
    r = decide(base, cand)
    assert r.verdict == "confirmed" and r.rel_change == pytest.approx(0.2, rel=0.01)


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    lo, hi = bootstrap_ci(A)
    assert lo < sum(A) / len(A) < hi
    assert bootstrap_ci(A) == (lo, hi)


def test_cost_and_energy_math():
    assert joules_per_token(500.0, 100) == 5.0
    # $2/hr at 1000 tok/s = 3.6M tok/hr -> $0.5556 per Mtok
    assert usd_per_mtok(2.0, 1000.0) == pytest.approx(2.0 / 3.6, rel=1e-9)
    assert usd_per_mtok(2.0, 0.0) == float("inf")


def _c(name, tput, ttft, tpot, energy, usd=2.0):
    return Config(name, tput, ttft, tpot, energy, 1000, usd)


def test_frontier_and_slo_choice_differ_from_max_throughput():
    fast = _c("fp16-uncapped", 2000, 2.5, 0.05, 900)  # fastest, violates TTFT SLO
    fp8 = _c("fp8", 1800, 1.5, 0.05, 500)
    capped = _c("fp8-cap70", 1500, 1.6, 0.06, 300)
    slow = _c("fp16-eager", 900, 3.0, 0.12, 900)
    cfgs = [fast, fp8, capped, slow]
    assert slow not in pareto(cfgs) and fp8 in pareto(cfgs)
    assert cheapest_feasible(cfgs, 2.0, 0.1) is fp8
    assert cheapest_feasible(cfgs, 2.0, 0.1, by="j_per_token") is capped
    assert cheapest_feasible(cfgs, 0.1, 0.01) is None
