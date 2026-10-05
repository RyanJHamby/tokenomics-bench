import json

import yaml

from tokbench import arms, manifest


def test_caps_are_fractions_of_measured_draw_and_below_min_are_dropped_not_run():
    keep, dropped = arms.power_caps(250.0, 150.0, 350.0)
    assert keep == [225, 200, 175, 150] and dropped == []
    keep, dropped = arms.power_caps(200.0, 150.0, 350.0)  # decode draws little: tight range
    assert keep == [180, 160] and dropped == [140, 120]
    keep, dropped = arms.power_caps(500.0, 150.0, 350.0)  # 450 and 400 exceed the max
    assert keep == [350, 300] and dropped == [450, 400]


def test_clock_locks_round_to_15mhz_and_dedupe():
    # 0.85*2520=2142 -> 2145; 0.70*2520=1764 -> 1770; 0.55*2520=1386 -> 1380
    assert arms.clock_locks(2520) == [2145, 1770, 1380]
    assert arms.clock_locks(1500) == [1275, 1050, 825]
    # coarse rounding can collapse neighbours: no duplicate arms
    locks = arms.clock_locks(60)
    assert len(locks) == len(set(locks))


def test_build_adds_baseline_caps_and_locks_keeping_server_args():
    tpl = {"name": "b2", "variants": [{"name": "uncapped", "server_args": ["--x"]}]}
    info = {"power.min_limit": 150.0, "power.max_limit": 350.0, "clocks.max.sm": 2520.0}
    cfg = arms.build(tpl, 250.0, info)
    names = [v["name"] for v in cfg["variants"]]
    assert (
        names[0] == "uncapped" and "cap225w" in names and any(n.startswith("lock") for n in names)
    )
    assert all(v["server_args"] == ["--x"] for v in cfg["variants"])
    assert not any("power_cap_w" in v and "clock_lock_mhz" in v for v in cfg["variants"])
    assert cfg["derived_from"]["saturated_p95_w"] == 250.0


def test_saturated_power_is_the_median_across_pilot_repeats(tmp_path):
    for r, w in enumerate((240.0, 260.0, 250.0)):
        (tmp_path / f"v__c64__r{r}.json").write_text(json.dumps({"power": {"power_w_p95": w}}))
    (tmp_path / "v__c1__r0.json").write_text(json.dumps({"power": {"power_w_p95": 90.0}}))
    assert arms.saturated_p95_power(tmp_path) == 250.0


def test_cli_end_to_end(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "gpu_info": {
                    "power.min_limit": 150.0,
                    "power.max_limit": 350.0,
                    "clocks.max.sm": 2520.0,
                }
            }
        )
    )
    (tmp_path / "v__c64__r0.json").write_text(json.dumps({"power": {"power_w_p95": 250.0}}))
    tpl = tmp_path / "t.yaml"
    tpl.write_text(yaml.safe_dump({"name": "b2", "variants": [{"name": "uncapped"}]}))
    assert arms.main([str(tmp_path), "--template", str(tpl)]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert len(out["variants"]) == 1 + 4 + 3


def test_gpu_info_parses_nvidia_smi_csv(monkeypatch):
    monkeypatch.setattr(manifest.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        manifest,
        "_run",
        lambda cmd: (
            "NVIDIA L40S, 580.65, 46068 MiB, 200.00 W, 350.00 W, 350.00 W, 2520 MHz, 9001 MHz, 8.9, GPU-abc"
        ),
    )
    info = manifest.gpu_info()
    assert info["power.min_limit"] == 200.0 and info["clocks.max.sm"] == 2520.0
    assert info["name"] == "NVIDIA L40S" and info["compute_cap"] == 8.9
