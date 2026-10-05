import json

import pytest

from tokbench.capacity import find_capacity, update_capacity_file


def _probe(true_cap, log=None, flaky=()):
    async def passes(q):
        if log is not None:
            log.append(q)
        return q <= true_cap and q not in flaky

    return passes


async def test_finds_capacity_within_resolution_by_bracketing_then_bisecting():
    res = await find_capacity(_probe(13.7), lo=1, hi=64, resolution=0.05)
    assert res["bracketed"] and not res.get("unconfirmed")
    assert 13.7 * 0.94 <= res["capacity"] <= 13.7  # within ~5% below the true knee
    assert res["upper_fail"] > 13.7


async def test_uses_logarithmic_number_of_probes_not_a_grid():
    log = []
    await find_capacity(_probe(40.0, log), lo=1, hi=64, resolution=0.05)
    assert len(log) <= 16  # ~6 doublings + ~5 bisections + confirm


async def test_fails_fast_when_even_the_lowest_load_violates_the_slo():
    res = await find_capacity(_probe(0.5), lo=1, hi=64, resolution=0.05)
    assert res["capacity"] is None and len(res["probes"]) == 1


async def test_reports_unbracketed_when_slo_holds_up_to_hi():
    res = await find_capacity(_probe(1e9), lo=1, hi=32, resolution=0.05)
    assert res["capacity"] == 32 and not res["bracketed"]


async def test_confirmation_probe_backs_off_when_the_boundary_pass_was_luck():
    seen = {}

    async def passes(q):  # passes the first time at ~12.5 but fails when re-tested
        first = q not in seen
        seen[q] = seen.get(q, 0) + 1
        return q < 12.4 or (first and q < 13)

    res = await find_capacity(passes, lo=1, hi=64, resolution=0.05)
    assert res["capacity"] < 12.4 and res["bracketed"]


def test_capacity_file_keeps_the_conservative_minimum_across_repeats(tmp_path):
    p = tmp_path / "capacity.json"
    update_capacity_file(p, "fp16", 10.0)
    update_capacity_file(p, "fp16", 8.0)
    update_capacity_file(p, "fp16", 12.0)
    update_capacity_file(p, "fp8", 15.0)
    assert json.loads(p.read_text()) == {"fp16": 8.0, "fp8": 15.0}
    assert not list(tmp_path.glob("*.tmp"))
    assert pytest.approx(8.0) == json.loads(p.read_text())["fp16"]
