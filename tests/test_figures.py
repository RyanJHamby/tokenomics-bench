"""Figures are built from saved run data only; missing inputs are skipped, never faked."""

import json
import shutil
import socket
import struct
from pathlib import Path

import pytest
import yaml

from tokbench import figures, runner

CFG = {
    "name": "t",
    "block": "T",
    "model": "mock",
    "seed": 0,
    "repeats": 2,
    "warmup_s": 0.3,
    "measure_s": 1.5,
    "drain_s": 3.0,
    "startup_seconds": 10,
    "idle_s": 0.4,
    "slo": {"ttft_s": 0.5, "tpot_s": 0.1},
    "workload": {"input_tokens": 32, "output_tokens": 6},
    "variants": [
        {"name": "base", "server_args": ["--tpot", "0.006"], "slots": 4},
        {"name": "cap", "server_args": ["--tpot", "0.007"], "slots": 4, "power_cap_w": 200},
        {"name": "lock", "server_args": ["--tpot", "0.008"], "slots": 4, "clock_lock_mhz": 1500},
    ],
    "loads": [{"mode": "open", "qps": 10}, {"mode": "closed", "concurrency": 4}],
}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("figs")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    old, runner.PORT = runner.PORT, port
    try:
        p = tmp / "c.yaml"
        p.write_text(yaml.safe_dump(CFG))
        out = tmp / "out"
        assert runner.main([str(p), "--server", "mock", "--gpu", "fake", "--usd-per-hr", "1",
                            "--out", str(out)]) == 0  # fmt: skip
    finally:
        runner.PORT = old
    return out


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_figures_are_made_from_saved_artifacts_and_stamped_synthetic(run_dir, tmp_path):
    info = figures.make_figures([run_dir], tmp_path / "f")
    assert info["synthetic"] is True
    assert {"01_latency_vs_load.png", "02_goodput_vs_energy.png", "04_power_timeline.png",
            "05_ttft_cdf.png", "06_launch_variance.png"} <= set(info["made"])  # fmt: skip
    for name in info["made"]:
        f = tmp_path / "f" / name
        w, h = _png_size(f)
        assert f.stat().st_size > 8_000 and w >= 900 and h >= 400, name
    idx = (tmp_path / "f" / "INDEX.md").read_text()
    assert "SYNTHETIC" in idx and "01_latency_vs_load.png" in idx


def test_missing_inputs_are_skipped_not_faked(run_dir, tmp_path):
    info = figures.make_figures([run_dir], tmp_path / "f")
    # fake GPU reports no SM clock, so the energy-vs-clock figure has no x data
    assert "energy_vs_clock" in info["skipped"] and "03_energy_vs_clock.png" not in info["made"]
    assert (
        "capacity_probes" in info["skipped"] and "model_check" not in info["skipped"] + info["made"]
    )
    assert "Skipped (inputs missing)" in (tmp_path / "f" / "INDEX.md").read_text()


def test_energy_vs_clock_appears_once_the_cells_carry_a_measured_clock(run_dir, tmp_path):
    d = tmp_path / "withclk"
    shutil.copytree(run_dir, d)
    for p in d.glob("*__c4__r*.json"):
        c = json.loads(p.read_text())
        c["power"]["sm_clock_mhz_mean"] = {"base": 2400.0, "cap": 2300.0, "lock": 1500.0}[
            c["variant"]
        ]
        p.write_text(json.dumps(c))
    info = figures.make_figures([d], tmp_path / "f")
    assert "03_energy_vs_clock.png" in info["made"]


def test_capacity_probe_and_model_check_figures(run_dir, tmp_path):
    d = tmp_path / "cap"
    shutil.copytree(run_dir, d)
    (d / "capacity__base__r0.json").write_text(
        json.dumps(
            {"capacity": 6.0, "probes": [[1, True], [2, True], [4, True], [8, False], [6, True]]}
        )
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "model_check": [
                    {
                        "quantity": "fp16 batch-1 decode",
                        "predicted": [35, 46],
                        "unit": "tok/s",
                        "measured": 40,
                        "in_interval": True,
                    },
                    {
                        "quantity": "fp16 SLO capacity",
                        "predicted": [5.8, 12.2],
                        "unit": "req/s",
                        "measured": 3.1,
                        "in_interval": False,
                    },
                ]
            }
        )
    )
    info = figures.make_figures([d], tmp_path / "f", analysis)
    assert {"08_capacity_probes.png", "07_model_check.png"} <= set(info["made"])


def test_no_data_is_an_error_not_an_empty_success(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no cells"):
        figures.make_figures([tmp_path / "empty"], tmp_path / "f")


def test_families_and_palette_follow_the_validated_slots():
    assert figures.family({"power_cap_w": 200}) == "power cap"
    assert figures.family({"clock_lock_mhz": 1500}) == "clock lock"
    assert figures.family({}) == "baseline"
    assert figures.SERIES[:3] == ["#2a78d6", "#eb6834", "#1baf7a"]  # validated all-pairs set
    assert [
        figures.FAMILY_COLOR[k] for k in ("baseline", "power cap", "clock lock")
    ] == figures.SERIES[:3]
