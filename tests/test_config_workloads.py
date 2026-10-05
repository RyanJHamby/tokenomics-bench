import pytest
import yaml

from tokbench.config import estimate_cost, load_config, plan
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
        "n_requests": 10,
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


def test_cost_estimate(tmp_path):
    cfg = load_config(_write(tmp_path, startup_seconds=100, est_seconds_per_load=50))
    est = estimate_cost(cfg, usd_per_hr=3.6)
    assert est["launches"] == 9
    assert est["gpu_hours"] == pytest.approx(9 * 200 / 3600)
    assert est["usd"] == pytest.approx(est["gpu_hours"] * 3.6)


@pytest.mark.parametrize(
    "over",
    [
        {"repeats": 0},
        {"variants": [{"name": "a"}, {"name": "a"}]},
        {"loads": [{"mode": "closed", "qps": 3}]},
        {"loads": [{"mode": "bogus"}]},
    ],
)
def test_invalid_configs_rejected(tmp_path, over):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, **over))


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
