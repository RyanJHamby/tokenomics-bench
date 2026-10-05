import json
import math

import pytest

from tokbench.pilot import pilot_report, sd_log


def test_sd_log_is_standard_deviation_of_logs_and_needs_two_launches():
    vals = [100.0, 105.0, 95.0]
    expect = __import__("statistics").stdev([math.log(v) for v in vals])
    assert sd_log(vals) == pytest.approx(expect)
    assert sd_log([100.0]) is None and sd_log([float("nan"), 1.0]) is None


def _cell(good, jpt, p95, tpot=0.025):
    return {
        "summary": {"goodput_tok_s": good, "tpot_p50": tpot},
        "energy": {"j_per_output_token": jpt},
        "power": {"power_w_p95": p95},
    }


def test_pilot_report_sizes_repeats_from_launch_noise(tmp_path):
    for r, (g, j, p) in enumerate([(1000, 0.50, 240), (1030, 0.52, 250), (980, 0.48, 245)]):
        (tmp_path / f"v__c64__r{r}.json").write_text(json.dumps(_cell(g, j, p)))
        (tmp_path / f"v__c1__r{r}.json").write_text(json.dumps(_cell(g, j, p)))
        (tmp_path / f"capacity__v__r{r}.json").write_text(json.dumps({"capacity": 10 + r}))
    rep = pilot_report(tmp_path)
    cap = rep["SLO capacity (qps)"]
    assert cap["n_launches"] == 3 and cap["sd_log"] > 0
    # noisier metric needs more launches for the same margin; wider margin needs fewer
    rr = rep["saturated goodput (c64)"]["required_repeats"]
    assert rr["5%"] >= rr["10%"] >= rr["20%"]
    assert rep["saturated p95 power (c64)"]["sd_log"] < cap["sd_log"]
