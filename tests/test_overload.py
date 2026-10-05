from tokbench.overload import capacity_qps


def _r(qps, ttft, tpot=0.02, fail=False, variant="v", invalid=False):
    return {
        "variant": variant,
        "load": {"mode": "open", "qps": qps},
        "ttft_p99": ttft,
        "tpot_p99": tpot,
        "any_failures": fail,
        "invalid": invalid,
    }


def test_capacity_is_last_load_before_first_violation():
    rows = [_r(1, 0.1), _r(2, 0.3), _r(4, 1.5), _r(8, 9.0), _r(16, 0.2)]  # 16 ignored: past knee
    assert capacity_qps(rows, "v", 2.0, 0.1) == 4


def test_failures_invalid_and_tpot_count_as_violations_and_none_when_nothing_meets():
    assert capacity_qps([_r(1, 0.1), _r(2, 0.1, fail=True)], "v", 2.0, 0.1) == 1
    assert capacity_qps([_r(1, 0.1), _r(2, 0.1, invalid=True)], "v", 2.0, 0.1) == 1
    assert capacity_qps([_r(1, 0.1, tpot=0.5)], "v", 2.0, 0.1) is None
    assert capacity_qps([_r(1, 0.1)], "other", 2.0, 0.1) is None


def test_slo_boundary_inclusive_and_nan_is_a_violation_not_a_pass():
    assert capacity_qps([_r(1, 2.0, tpot=0.1)], "v", 2.0, 0.1) == 1
    nan = float("nan")
    assert capacity_qps([_r(1, 0.1), _r(2, nan)], "v", 2.0, 0.1) == 1  # nan must stop the climb
    assert capacity_qps([_r(1, nan)], "v", 2.0, 0.1) is None


def test_closed_loop_rows_never_define_capacity():
    row = {**_r(1, 0.1), "load": {"mode": "closed", "concurrency": 8}}
    assert capacity_qps([row], "v", 2.0, 0.1) is None
