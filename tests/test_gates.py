import pytest

from tokbench.gates import gate_verdict, n_mismatches


def test_mismatch_count_and_length_check():
    assert n_mismatches(["a", "b", "c"], ["a", "x", "c"]) == 1
    with pytest.raises(ValueError):
        n_mismatches(["a"], ["a", "b"])


def test_verdict_is_judged_against_noise_floor():
    assert gate_verdict(0, 0) == "pass"
    assert gate_verdict(1, 0) == "fail"  # any disagreement fails when the config is self-consistent
    assert gate_verdict(2, 3) == "pass"  # within the same-config noise
    assert gate_verdict(4, 3) == "fail"
