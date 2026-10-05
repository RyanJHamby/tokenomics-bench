import pytest
import yaml

from tokbench.config import (
    Launch,
    estimate_cost,
    load_config,
    load_order,
    plan,
    resolve_loads,
)
from tokbench.workloads import make_prompt_fn, salt_for


def _write(tmp_path, **over):
    cfg = {
        "name": "t",
        "block": "B1",
        "model": "m",
        "workload": {"input_tokens": 10, "output_tokens": 5},
        "variants": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        "loads": [{"mode": "closed", "concurrency": 4}, {"mode": "open", "qps": 2}],
        "repeats": 3,
        "measure_s": 100,
        "warmup_s": 20,
    }
    cfg.update(over)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_plan_runs_every_variant_each_repeat_in_varying_order(tmp_path):
    cfg = load_config(_write(tmp_path))
    launches = plan(cfg)
    assert len(launches) == 9
    for r in range(3):
        assert sorted(la.variant["name"] for la in launches if la.repeat == r) == ["a", "b", "c"]
    orders = {tuple(la.variant["name"] for la in launches if la.repeat == r) for r in range(3)}
    assert len(orders) > 1  # interleaved, not the same fixed order every repeat
    assert plan(cfg) == launches  # deterministic for a given seed


def test_cost_estimate_counts_windows_soak_idle_and_search_probes(tmp_path):
    loads = [
        {"mode": "open", "qps": 2},
        {"mode": "capacity_search", "qps_lo": 1, "qps_hi": 64, "resolution": 0.05},
    ]
    cfg = load_config(_write(tmp_path, loads=loads, startup_seconds=100, soak_s=60, idle_s=10))
    one = 20 + 100 + 15
    probes = 6 + 5 + 2  # log2(64) + log2(1/0.05) rounded up + bracket/confirm allowance
    expect = (100 + 60 + 10 + one + one * probes) * 9 / 3600
    est = estimate_cost(cfg, usd_per_hr=3.6)
    assert est["launches"] == 9 and est["gpu_hours"] == pytest.approx(expect)
    assert est["usd"] == pytest.approx(expect * 3.6)
    assert estimate_cost(cfg, 3.6, launches=2)["launches"] == 2


@pytest.mark.parametrize(
    "over",
    [
        {"repeats": 0},
        {"measure_s": 0},
        {"variants": [{"name": "a"}, {"name": "a"}]},
        {"variants": [{"name": "a", "power_cap_w": 200, "clock_lock_mhz": 1500}]},
        {"loads": [{"mode": "closed", "qps": 3}]},
        {"loads": [{"mode": "open", "qps": 1, "rel": 0.5, "ref": "a"}]},
        {"loads": [{"mode": "open", "rel": 0.5}]},
        {"loads": [{"mode": "capacity_search", "qps_lo": 1}]},
        {"loads": [{"mode": "bogus"}]},
    ],
)
def test_invalid_configs_rejected(tmp_path, over):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, **over))


def test_rel_loads_resolve_against_measured_capacity_for_every_variant(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            loads=[
                {"mode": "open", "rel": 0.7, "ref": "a"},
                {"mode": "open", "rel": 1.5, "ref": "a"},
            ],
        )
    )
    out = resolve_loads(cfg, {"a": 10.0})
    assert [ld["qps"] for ld in out["loads"]] == [7.0, 15.0]  # same absolute qps for all variants
    with pytest.raises(ValueError, match="run the capacity block first"):
        resolve_loads(cfg, None)
    with pytest.raises(ValueError, match="run the capacity block first"):
        resolve_loads(cfg, {"other": 3.0})


def test_load_order_is_a_seeded_shuffle_that_differs_across_launches(tmp_path):
    cfg = load_config(_write(tmp_path, loads=[{"mode": "open", "qps": q} for q in range(1, 9)]))
    la = Launch({"name": "a"}, 0)
    o1 = load_order(cfg, la, 8)
    assert sorted(o1) == list(range(8)) and load_order(cfg, la, 8) == o1
    assert o1 != list(range(8))  # not the cool-to-hot ascent
    assert (
        len({tuple(load_order(cfg, Launch({"name": n}, r), 8)) for n in "abc" for r in range(3)})
        > 3
    )
    fixed = load_config(_write(tmp_path, randomize_load_order=False))
    assert load_order(fixed, la, 5) == [0, 1, 2, 3, 4]


def test_prompts_are_exact_length_and_prefix_shared():
    f = make_prompt_fn(100, 8, prefix_share=0.9)
    a, b = f(0), f(1)
    assert len(a["prompt"]) == len(b["prompt"]) == 100  # exact token count, no tokenizer
    assert a["prompt"][:90] == b["prompt"][:90]  # identical shared prefix
    assert a["prompt"][90:] != b["prompt"][90:]
    assert f(0) == make_prompt_fn(100, 8, prefix_share=0.9)(0)  # deterministic
    assert a["temperature"] == 0 and a["max_tokens"] == 8 and a["ignore_eos"] is True
    assert make_prompt_fn(100, 8, 0.0)(0)["prompt"] != make_prompt_fn(100, 8, 0.0)(1)["prompt"]


def test_salt_makes_loads_and_repeats_disjoint_so_prefix_cache_starts_cold():
    """Regression: unsalted prompts replayed across loads hit the KV prefix cache."""
    seen = set()
    for repeat in range(3):
        for load_idx in range(4):
            f = make_prompt_fn(64, 4, 0.5, seed=0, salt=salt_for(repeat, load_idx))
            body = f(0)["prompt"]
            assert tuple(body[:32]) not in seen  # shared prefix differs per (repeat, load)
            assert tuple(body) not in seen
            seen.add(tuple(body[:32]))
    assert len({salt_for(r, i) for r in range(10) for i in range(20)}) == 200


def test_prefix_share_out_of_range_rejected():
    with pytest.raises(ValueError):
        make_prompt_fn(10, 1, prefix_share=1.5)
