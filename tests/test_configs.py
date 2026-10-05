"""Every committed experiment config must parse, and rel loads must resolve."""

import glob
import json
import pathlib

import pytest
import yaml

from tokbench import arms
from tokbench.config import estimate_cost, load_config, resolve_loads

FILES = sorted(f for f in glob.glob("configs/*.yaml"))


@pytest.mark.parametrize("path", FILES)
def test_config_parses_and_has_pinned_prefix_cache_state(path):
    cfg = load_config(path)
    resolve_loads(cfg, {"fp16-default": 10.0})
    if cfg["block"] not in ("DEMO",):
        for v in cfg["variants"]:
            args = v["server_args"]
            # prefix caching defaults ON in V1: every non-prefix block must switch it off
            assert "--no-enable-prefix-caching" in args or "--enable-prefix-caching" in args, v[
                "name"
            ]
        assert cfg["slo"] == {"ttft_s": 1.0, "tpot_s": 0.05}  # one pre-registered primary SLO
        assert cfg["randomize_load_order"] is True


def test_baseline_arm_is_identical_wherever_it_is_reused():
    """B1, B3b and the B2 template share the fp16 baseline; unequal args would make
    'fp16-default' mean different things in different blocks."""
    args = {}
    for f in ("b1_capacity_pilot", "b3b_quant_fixed_load"):
        v = next(
            x for x in load_config(f"configs/{f}.yaml")["variants"] if x["name"] == "fp16-default"
        )
        args[f] = v["server_args"]
    b2 = load_config("configs/b2_power_clock.template.yaml")["variants"][0]["server_args"]
    assert args["b1_capacity_pilot"] == args["b3b_quant_fixed_load"] == b2


def test_generated_b2_is_valid_and_arms_are_separate(tmp_path):
    tpl = yaml.safe_load(pathlib.Path("configs/b2_power_clock.template.yaml").read_text())
    info = {"power.min_limit": 150.0, "power.max_limit": 350.0, "clocks.max.sm": 2520.0}
    out = tmp_path / "b2.yaml"
    out.write_text(yaml.safe_dump(arms.build(tpl, 250.0, info)))
    cfg = load_config(out)
    assert len(cfg["variants"]) == 8
    est = estimate_cost(resolve_loads(cfg, {"fp16-default": 10.0}), 1.0)
    assert 3 < est["gpu_hours"] < 8 and json.dumps(cfg)  # sane size for the budget


def test_total_planned_gpu_hours_fit_the_budget_with_headroom():
    total = 0.0
    for f in (
        "b1_capacity_pilot",
        "b3a_fp8_capacity",
        "b3a_extra_capacity",
        "b3b_quant_fixed_load",
        "b4_graph_modes",
    ):
        cfg = resolve_loads(load_config(f"configs/{f}.yaml"), {"fp16-default": 10.0})
        total += estimate_cost(cfg, 1.0)["gpu_hours"]
    total += 5.1  # b2 with 8 arms, 3 launches each
    assert total < 16  # at the $60 cap this leaves >= 3x headroom at ~$1/hr, ~1.4x at ~$2.7/hr
