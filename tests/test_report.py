import json
import math

import pytest

from tokbench.report import _med, aggregate, frontier_chart, load_cells, to_configs


def _cell(
    variant,
    load,
    repeat=0,
    ttft=0.5,
    tpot=0.03,
    tput=1000.0,
    good=1000.0,
    jpt=0.4,
    n_failed=0,
    invalid=False,
    usd=2.0,
):
    return {
        "schema_version": 2,
        "variant": variant,
        "load": load,
        "repeat": repeat,
        "summary": {
            "throughput_tok_s": tput,
            "goodput_tok_s": good,
            "slo_attainment": 1.0,
            "ttft_p99": ttft,
            "tpot_p99": tpot,
            "itl_p99": 0.02,
            "n_failed": n_failed,
        },
        "energy": {"j_per_output_token": jpt},
        "power": {"bad_throttle_seen": False},
        "workload_ok": not invalid,
        "sampler_ok": True,
        "gpu_usd_per_hr": usd,
        "synthetic": True,
    }


def test_median_is_order_independent_and_skips_nan():
    nan = float("nan")
    assert _med([nan, 1, 2]) == _med([1, nan, 2]) == _med([1, 2, nan]) == 1.5
    assert math.isnan(_med([nan, None]))


def test_closed_and_open_cells_with_same_number_do_not_collide():
    cells = [
        _cell("v", {"mode": "closed", "concurrency": 4}),
        _cell("v", {"mode": "open", "qps": 4}),
    ]
    names = [c.name for c in to_configs(aggregate(cells))]
    assert sorted(names) == ["v@c4", "v@q4"]


def test_aggregate_medians_across_repeats_and_flags_failures():
    load = {"mode": "open", "qps": 8}
    rows = aggregate([_cell("v", load, r, ttft=t) for r, t in enumerate([1.0, 9.0, 1.2])])
    assert len(rows) == 1 and rows[0]["ttft_p99"] == 1.2 and rows[0]["n_repeats"] == 3
    rows = aggregate([_cell("v", load, 0), _cell("v", load, 1, n_failed=7)])
    assert rows[0]["any_failures"] and rows[0]["n_failed"] == 7


def test_invalid_cell_is_ineligible_even_with_great_numbers():
    from tokbench.analysis import cheapest_feasible

    load = {"mode": "open", "qps": 8}
    cfgs = to_configs(aggregate([_cell("bad", load, ttft=0.01, invalid=True), _cell("ok", load)]))
    assert cheapest_feasible(cfgs, 2.0, 0.1).name == "ok@q8"


def test_load_cells_rejects_wrong_schema_and_ignores_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "a.json").write_text(json.dumps(_cell("v", {"mode": "open", "qps": 1})))
    assert len(load_cells(tmp_path)) == 1
    (tmp_path / "old.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="unsupported schema"):
        load_cells(tmp_path)


def test_chart_picks_cheapest_feasible_open_cell_and_labels_synthetic(tmp_path):
    rows = aggregate(
        [
            _cell("fast", {"mode": "open", "qps": 8}, tput=2000, good=2000, ttft=1.0),
            _cell("slow", {"mode": "open", "qps": 8}, tput=500, good=500, ttft=0.5),
            _cell("fast", {"mode": "closed", "concurrency": 64}, tput=9000, good=9000, ttft=0.1),
        ]
    )
    info = frontier_chart(rows, str(tmp_path / "f.png"), 2.0, 0.1)
    assert info["cheapest_feasible"] == "fast@q8"  # closed-loop 9000 tok/s cell is ignored
    assert (tmp_path / "f.png").stat().st_size > 1000
