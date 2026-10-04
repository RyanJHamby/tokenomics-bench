import pytest
import yaml

from tokbench.config import estimate_cost, load_config, plan
from tokbench.workloads import make_prompt_fn


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


def test_prefix_share_shapes_prompts():
    f = make_prompt_fn(100, 8, prefix_share=0.9)
    a, b = f(0), f(1)
    assert a["messages"][0] == b["messages"][0]  # identical shared system prompt
    assert len(a["messages"][0]["content"].split()) == 90
    assert a["messages"][1]["content"] != b["messages"][1]["content"]
    assert f(0) == make_prompt_fn(100, 8, prefix_share=0.9)(0)  # deterministic
    assert a["temperature"] == 0 and a["max_tokens"] == 8
    assert make_prompt_fn(100, 8, 0.0)(0)["messages"][0]["content"] == ""
