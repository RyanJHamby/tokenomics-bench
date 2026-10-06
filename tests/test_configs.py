"""Every committed experiment config must be exactly what the generator emits, must parse,
and the pinned engine settings must be identical wherever a variant is reused."""

import glob
import pathlib

import pytest
import yaml

from tokbench import arms, configgen
from tokbench.config import estimate_cost, load_config, loads_for, resolve_loads

FILES = sorted(glob.glob("configs/*.yaml"))
GENERATED = set(configgen.build())


def test_committed_configs_equal_the_generator_output():
    """Edit tokbench/configgen.py and rerun `python -m tokbench.configgen`; never hand-edit."""
    for name, (header, cfg) in configgen.build().items():
        committed = pathlib.Path(f"configs/{name}.yaml").read_text()
        assert committed == configgen.render(name, header, cfg), f"configs/{name}.yaml drifted"
    extra = {pathlib.Path(f).stem for f in FILES} - GENERATED - {"demo_synthetic"}
    assert not extra, f"hand-written configs not in the generator: {extra}"


@pytest.mark.parametrize("path", FILES)
def test_config_parses_and_resolves(path):
    cfg = load_config(path)
    resolve_loads(cfg, {"fp16-default": 10.0})
    if cfg["block"] != "DEMO":
        assert cfg["slo"] == {"ttft_s": 1.0, "tpot_s": 0.05}  # one pre-registered primary SLO
        assert cfg["randomize_load_order"] is True


def _flag(args, name):
    return args[args.index(name) + 1]


@pytest.mark.parametrize("path", [f for f in FILES if "demo" not in f and "sglang" not in f])
def test_every_vllm_variant_pins_everything_vllm_would_choose_by_gpu_or_version(path):
    cfg = load_config(path)
    for v in cfg["variants"]:
        a = v["server_args"]
        for flag in (
            "--dtype",
            "--generation-config",
            "--max-num-seqs",
            "--max-num-batched-tokens",
            "--seed",
            "--attention-backend",
            "--revision",
            "--tokenizer-revision",
        ):
            assert flag in a, f"{v['name']} does not pin {flag}"
        assert (
            _flag(a, "--max-num-seqs") == "256" and _flag(a, "--max-num-batched-tokens") == "2048"
        )
        assert _flag(a, "--generation-config") == "vllm" and _flag(a, "--revision") == _flag(
            a, "--tokenizer-revision"
        )
        assert (
            "--no-enable-prefix-caching" in a
            and "--async-scheduling" in a
            and "--enable-chunked-prefill" in a
        )
        assert len(_flag(a, "--revision")) == 40  # a commit SHA, not a branch name


def test_baseline_arm_is_identical_wherever_it_is_reused():
    args = {}
    for f in ("b1_capacity_pilot", "b3b_quant_fixed_load", "b5_overload_recovery"):
        v = next(
            x for x in load_config(f"configs/{f}.yaml")["variants"] if x["name"] == "fp16-default"
        )
        args[f] = v["server_args"]
    b2 = load_config("configs/b2_power_clock.template.yaml")["variants"][0]["server_args"]
    assert len({tuple(v) for v in args.values()} | {tuple(b2)}) == 1


def test_fp8_kv_arm_is_flagged_as_a_kernel_confound_and_uses_a_different_backend():
    cfg = load_config("configs/b3a_extra_capacity.yaml")
    kv = next(v for v in cfg["variants"] if v["name"] == "fp8-kv8")
    base = load_config("configs/b1_capacity_pilot.yaml")["variants"][0]["server_args"]
    assert _flag(kv["server_args"], "--attention-backend") != _flag(base, "--attention-backend")
    assert "kernel" in pathlib.Path("configs/b3a_extra_capacity.yaml").read_text().lower()


def test_generated_b2_has_shape_loads_only_on_baseline_and_the_70_55_locks(tmp_path):
    tpl = yaml.safe_load(pathlib.Path("configs/b2_power_clock.template.yaml").read_text())
    info = {"power.min_limit": 150.0, "power.max_limit": 350.0, "clocks.max.sm": 2520.0}
    out = tmp_path / "b2.yaml"
    out.write_text(yaml.safe_dump(arms.build(tpl, 250.0, info)))
    cfg = load_config(out)
    assert len(cfg["variants"]) == 8 and "shape_loads" not in cfg
    with_shapes = [v["name"] for v in cfg["variants"] if v.get("extra_loads")]
    assert with_shapes == ["uncapped", "lock1770", "lock1380"]  # 100%, 70%, 55%
    resolved = resolve_loads(cfg, {"fp16-default": 10.0})
    assert len(loads_for(resolved, resolved["variants"][0])) == 4
    assert (
        len(loads_for(resolved, next(v for v in resolved["variants"] if v["name"] == "cap200w")))
        == 2
    )


def test_moe_template_yields_baseline_plus_one_lock_and_no_caps(tmp_path):
    tpl = yaml.safe_load(pathlib.Path("configs/b6_moe.template.yaml").read_text())
    info = {"power.min_limit": 150.0, "power.max_limit": 350.0, "clocks.max.sm": 2520.0}
    cfg = arms.build(tpl, 250.0, info, lock_fracs=(0.70,), with_caps=False)
    assert [v["name"] for v in cfg["variants"]] == ["moe-fp8", "lock1770"]
    assert all("extra_loads" not in v for v in cfg["variants"])


def _total_hours(tmp_path) -> float:
    total = 0.0
    for f in (
        "b1_capacity_pilot",
        "b3a_fp8_capacity",
        "b3a_extra_capacity",
        "b3b_quant_fixed_load",
        "b4_graph_modes",
        "b5_overload_recovery",
        "b7_sglang_crosscheck",
    ):
        cfg = resolve_loads(load_config(f"configs/{f}.yaml"), {"fp16-default": 10.0})
        total += estimate_cost(cfg, 1.0)["gpu_hours"]
    # B2 (8 arms; shape loads on 3) and B6 (2 arms) are generated from the pilot: build them here
    info = {"power.min_limit": 150.0, "power.max_limit": 350.0, "clocks.max.sm": 2520.0}
    b2 = arms.build(
        yaml.safe_load(pathlib.Path("configs/b2_power_clock.template.yaml").read_text()),
        250.0,
        info,
    )
    b6 = arms.build(
        yaml.safe_load(pathlib.Path("configs/b6_moe.template.yaml").read_text()),
        250.0,
        info,
        lock_fracs=(0.70,),
        with_caps=False,
    )
    for cfg in (b2, b6):
        p = tmp_path / "cfg_budget.yaml"
        p.write_text(yaml.safe_dump(cfg))
        total += estimate_cost(resolve_loads(load_config(p), {"fp16-default": 10.0}), 1.0)[
            "gpu_hours"
        ]
    return total


def test_total_planned_gpu_hours_are_within_what_the_budget_can_carry(tmp_path):
    total = _total_hours(tmp_path)
    assert 17 < total < 20  # ~18.2 h with all exploratory blocks: ~$18 at $1/h, ~$49 at $2.7/h


def test_capacity_search_brackets_cover_the_models_predicted_upper_bound_with_margin():
    """If the true capacity exceeds qps_hi the search reports 'not bracketed' and the block
    wastes its launches. Require qps_hi >= 2x the predicted upper end of SLO capacity."""
    from tokbench.model import prediction_intervals

    iv = prediction_intervals("L40S")
    for cfgname, variant, prec in (
        ("b1_capacity_pilot", "fp16-default", "fp16"),
        ("b3a_fp8_capacity", "fp8", "fp8"),
    ):
        cfg = load_config(f"configs/{cfgname}.yaml")
        hi = next(ld["qps_hi"] for ld in cfg["loads"] if ld["mode"] == "capacity_search")
        assert hi >= 2 * iv[f"slo_capacity_req_s:{prec}"][1], (cfgname, hi)
