from tokbench.overload import capacity_qps


def _r(qps, ttft, tpot=0.02, fail=False, variant="v"):
    return {
        "variant": variant,
        "load": {"mode": "open", "qps": qps},
        "ttft_p99": ttft,
        "tpot_p99": tpot,
        "any_failures": fail,
    }


def test_capacity_is_last_load_before_first_violation():
    rows = [_r(1, 0.1), _r(2, 0.3), _r(4, 1.5), _r(8, 9.0), _r(16, 0.2)]  # 16 ignored: past knee
    assert capacity_qps(rows, "v", 2.0, 0.1) == 4


def test_failures_and_tpot_count_as_violations_and_none_when_nothing_meets():
    assert capacity_qps([_r(1, 0.1), _r(2, 0.1, fail=True)], "v", 2.0, 0.1) == 1
    assert capacity_qps([_r(1, 0.1, tpot=0.5)], "v", 2.0, 0.1) is None
    assert capacity_qps([_r(1, 0.1)], "other", 2.0, 0.1) is None
