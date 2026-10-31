"""Spec bench-readiness-report: a no-GPU pre-rental readiness report.

Run before renting: `python -m tokbench.readiness --price <live usd/hr>`.
"""

from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from tokbench import configgen

REPO = Path(__file__).resolve().parents[1]
XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-readiness-report (remove this marker when implemented)"
)


def _r():
    return importlib.import_module("tokbench.readiness")


def _root(tmp_path: Path, cap: float = 60.0, ledger_usd: float = 0.0) -> Path:
    """A throwaway repo root: generated configs, a budget policy, the GPU requirements."""
    root = tmp_path / "root"
    shutil.copytree(REPO / "configs", root / "configs")
    (root / "budget").mkdir()
    (root / "budget" / "budget.yaml").write_text(
        yaml.safe_dump({"cap_usd": cap, "safety_margin": 0.2})
    )
    ledger = root / "budget" / "ledger.jsonl"
    ledger.write_text("")
    if ledger_usd:
        ledger.write_text(
            json.dumps({"kind": "invoice", "usd": ledger_usd, "session": "pod1"}) + "\n"
        )
    shutil.copy(REPO / "requirements-gpu.txt", root / "requirements-gpu.txt")
    return root


@XFAIL
def test_real_repo_report_covers_every_generated_config_and_scales_with_price():
    a = _r().report(price=1.0, root=REPO)
    b = _r().report(price=2.0, root=REPO)
    names = {n for n in configgen.build()}
    assert {blk["config"] for blk in a["blocks"]} == names
    for blk in a["blocks"]:
        assert blk["gpu_hours"] > 0 and blk["usd"] > 0
        assert blk["usd"] == pytest.approx(blk["gpu_hours"] * 1.0)
    assert a["total_usd"] == pytest.approx(sum(x["usd"] for x in a["blocks"]))
    assert b["total_usd"] == pytest.approx(2 * a["total_usd"])
    assert a["total_gpu_hours"] == pytest.approx(b["total_gpu_hours"])
    # capacity-dependent loads use a stand-in until B1 exists; the report must say so
    assert a["capacity_is_standin"] is True
    assert a["config_drift"] == [] and a["unpinned_variants"] == []
    assert a["cap_usd"] == 60.0 and a["safety_margin"] == 0.2


@XFAIL
def test_budget_fit_uses_padded_total_against_remaining(tmp_path):
    cheap = _r().report(price=0.5, root=_root(tmp_path))
    assert cheap["fits_budget"] is True and cheap["ready"] is True
    assert cheap["padded_total_usd"] == pytest.approx(cheap["total_usd"] * 1.2)
    assert cheap["remaining_usd"] == pytest.approx(60.0)
    # same price, but most of the cap is already spent (invoice in the ledger)
    spent = _r().report(price=0.5, root=_root(tmp_path / "b", ledger_usd=55.0))
    assert spent["spent_usd"] == pytest.approx(55.0)
    assert spent["remaining_usd"] == pytest.approx(5.0)
    assert spent["fits_budget"] is False and spent["ready"] is False
    # an expensive pod does not fit the full cap either
    dear = _r().report(price=50.0, root=_root(tmp_path / "c"))
    assert dear["fits_budget"] is False and dear["ready"] is False


@XFAIL
def test_config_drift_is_named(tmp_path):
    root = _root(tmp_path)
    p = root / "configs" / "b4_graph_modes.yaml"
    p.write_text(p.read_text() + "# hand edit\n")
    rep = _r().report(price=0.5, root=root)
    assert rep["config_drift"] == ["b4_graph_modes"]
    assert rep["ready"] is False


@XFAIL
def test_unpinned_model_revision_is_named_and_blocks_ready(tmp_path):
    root = _root(tmp_path)
    p = root / "configs" / "b1_capacity_pilot.yaml"
    cfg = yaml.safe_load(p.read_text())
    args = cfg["variants"][0]["server_args"]
    i = args.index("--revision")
    del args[i : i + 2]
    p.write_text(yaml.safe_dump(cfg))
    rep = _r().report(price=0.5, root=root)
    assert f"b1_capacity_pilot:{cfg['variants'][0]['name']}" in rep["unpinned_variants"]
    assert rep["ready"] is False


@XFAIL
def test_vllm_pin_must_be_exact(tmp_path):
    root = _root(tmp_path)
    assert _r().report(price=0.5, root=root)["vllm_pin"] == "0.30.0"
    (root / "requirements-gpu.txt").write_text("vllm>=0.30.0\nnvidia-ml-py\n")
    rep = _r().report(price=0.5, root=root)
    assert rep["vllm_pin"] is None and rep["ready"] is False


@XFAIL
def test_cli_prints_json_and_exit_code_follows_readiness(tmp_path, capsys):
    root = _root(tmp_path)
    assert _r().main(["--price", "0.5", "--root", str(root), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ready"] is True and out["price_usd_per_hr"] == 0.5
    # human output names the verdict and every block
    assert _r().main(["--price", "50", "--root", str(root)]) == 1
    text = capsys.readouterr().out
    assert "NOT READY" in text and "b2_power_clock.template" in text


@XFAIL
def test_a_root_without_configs_is_an_error_not_a_green_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        _r().report(price=1.0, root=tmp_path)
