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


def _c(name, tput, ttft, tpot, jpt, usd=2.0, mode="open", good=None, failed=0):
    return Config(name, mode, tput, tput if good is None else good, ttft, tpot, jpt, failed, usd)


def test_frontier_and_slo_choice_differ_from_max_throughput():
    fast = _c("fp16-uncapped@q9", 2000, 2.5, 0.05, 0.9)  # fastest, violates TTFT SLO
    fp8 = _c("fp8@q8", 1800, 1.5, 0.05, 0.5)
    capped = _c("fp8-cap70@q8", 1500, 1.6, 0.06, 0.3)
    slow = _c("fp16-eager@q4", 900, 3.0, 0.12, 0.9)
    cfgs = [fast, fp8, capped, slow]
    assert slow not in pareto(cfgs) and fp8 in pareto(cfgs)
    assert cheapest_feasible(cfgs, 2.0, 0.1) is fp8
    assert cheapest_feasible(cfgs, 2.0, 0.1, by="j_per_token") is capped
    assert cheapest_feasible(cfgs, 0.1, 0.01) is None


def test_slo_boundary_is_inclusive_for_both_latencies():
    c = _c("a@q1", 100, 2.0, 0.1, 1.0)
    assert c.meets(2.0, 0.1)
    assert not c.meets(1.999, 0.1) and not c.meets(2.0, 0.099)


def test_nan_latency_never_meets_slo():
    nan = float("nan")
    assert not _c("a@q1", 100, nan, 0.05, 1.0).meets(2.0, 0.1)
    assert not _c("a@q1", 100, 1.0, nan, 1.0).meets(2.0, 0.1)


def test_failed_and_closed_loop_cells_are_not_eligible():
    failed = _c("bad@q1", 5000, 0.1, 0.01, 0.1, failed=120)  # great p99 over survivors only
    closed = _c("c@c4", 4000, 0.1, 0.01, 0.1, mode="closed")  # TTFT excludes queueing
    good = _c("ok@q4", 1000, 1.0, 0.05, 0.5)
    assert cheapest_feasible([failed, closed, good], 2.0, 0.1) is good
    assert pareto([failed, closed, good]) == [good]


def test_cost_is_per_goodput_when_it_differs_from_throughput():
    c = _c("a@q1", 1000, 1.0, 0.05, 1.0, usd=3.6, good=500)
    assert c.usd_per_mtok == pytest.approx(1.0)  # $3.6/hr at 1000 tok/s
    assert c.usd_per_mtok_good == pytest.approx(2.0)  # but only half the tokens met the SLO
    assert usd_per_mtok(1.0, float("nan")) == float("inf")
