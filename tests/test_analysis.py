import math

import pytest

from tokbench.analysis import (
    Config,
    cheapest_feasible,
    holm,
    joules_per_token,
    paired_log_ratio,
    pareto,
    required_repeats,
    t_ppf,
    usd_per_mtok,
)


def test_t_quantiles_match_scipy_reference():
    # scipy.stats.t.ppf(0.975, df)
    for df, ref in ((2, 4.302653), (4, 2.776445), (9, 2.262157)):
        assert t_ppf(0.975, df) == pytest.approx(ref, rel=1e-5)


A = [1.10, 1.18, 1.05, 1.12]
B = [1.00, 1.02, 0.98, 1.01]


def test_paired_log_ratio_matches_scipy_reference_interval_and_p():
    # numpy/scipy: mean log ratio 0.103348, 95% CI [0.052699, 0.153998], p = 0.0074151
    r = paired_log_ratio(A, B, margin=0.05)
    assert r.n == 4 and r.mean_log_ratio == pytest.approx(0.10334830, abs=1e-7)
    assert math.log(r.ci_lo) == pytest.approx(0.05269860, abs=1e-6)
    assert math.log(r.ci_hi) == pytest.approx(0.15399800, abs=1e-6)
    assert r.p_value == pytest.approx(0.00741514, rel=1e-4)


def test_verdicts_superior_equivalent_inconclusive():
    tight = [1.0, 1.001, 0.999, 1.0]
    assert paired_log_ratio([x * 1.3 for x in tight], tight).verdict == "superior_higher"
    assert paired_log_ratio([x * 0.7 for x in tight], tight).verdict == "superior_lower"
    assert paired_log_ratio([x * 1.01 for x in tight], tight, margin=0.05).verdict == "equivalent"
    noisy_a, noisy_b = [1.0, 1.4, 0.8, 1.2], [1.0, 1.0, 1.0, 1.0]
    assert paired_log_ratio(noisy_a, noisy_b).verdict == "inconclusive"  # wide CI: say so


def test_paired_rejects_unpaired_or_invalid_input():
    for a, b in (([1, 2], [1]), ([1], [1]), ([1, -2], [1, 2]), ([1, float("nan")], [1, 2])):
        with pytest.raises(ValueError):
            paired_log_ratio(a, b)


def test_holm_stepdown():
    assert holm([0.001, 0.04, 0.03]) == [True, False, False]  # 0.03 > 0.05/2: stop
    assert holm([0.001, 0.01, 0.02]) == [True, True, True]
    assert holm([0.2]) == [False]


def test_required_repeats_matches_exact_power_calculation():
    # statsmodels TTestPower (exact noncentral t): sd .05/delta .05 -> 9.94, .10/.05 -> 33.4,
    # .03/.10 -> 2.97. Normal-approx with t critical must land within one launch.
    assert abs(required_repeats(0.05, 0.05) - 10) <= 1
    assert abs(required_repeats(0.10, 0.05) - 34) <= 1
    assert required_repeats(0.03, 0.10) == 3
    with pytest.raises(ValueError):
        required_repeats(0, 0.1)


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
