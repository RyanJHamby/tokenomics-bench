import json

import pytest
import yaml

from tokbench import runner
from tokbench.runner import CellFailed, cell_done, write_atomic

CFG = {
    "name": "t",
    "block": "T",
    "model": "mock",
    "seed": 0,
    "repeats": 1,
    "n_requests": 12,
    "warmup": 2,
    "startup_seconds": 10,
    "workload": {"input_tokens": 32, "output_tokens": 6},
    "variants": [{"name": "v", "server_args": ["--tpot", "0.004"], "slots": 4}],
    "loads": [{"mode": "closed", "concurrency": 2}, {"mode": "open", "qps": 20}],
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(runner, "PORT", s.getsockname()[1])

    def make(cfg=CFG):
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p

    return make, tmp_path / "out"


def _run(cfgpath, out, *extra):
    return runner.main(
        [
            str(cfgpath),
            "--server",
            "mock",
            "--gpu",
            "fake",
            "--usd-per-hr",
            "1",
            "--out",
            str(out),
            "--quiet-server",
            *extra,
        ]
    )


def test_end_to_end_mock_run_writes_valid_cells_and_energy_is_consistent(setup):
    make, out = setup
    assert _run(make(), out) == 0
    cells = sorted(out.glob("v__*.json"))
    assert len(cells) == 2 and (out / "manifest.json").exists()
    for c in cells:
        d = json.loads(c.read_text())
        s, e = d["summary"], d["energy"]
        assert d["schema_version"] == 2 and d["synthetic"] is True
        assert s["n_failed"] == 0 and s["n_requests"] == 10  # warmup=2 discarded
        assert s["mean_prompt_tokens"] == 32 and d["workload_ok"] and d["sampler_ok"]
        # J/token and throughput use the same requests and the same window
        assert e["j_per_output_token"] == pytest.approx(e["energy_j"] / s["output_tokens"])
        assert e["window_s"] == pytest.approx(s["window_s"], rel=0.01)
        assert d["power"]["bad_throttle_seen"] is False
    assert not list(out.glob("*.tmp"))


def test_resume_skips_finished_cells_without_launching_a_server(setup, capsys):
    make, out = setup
    p = make()
    _run(p, out)
    mtimes = {c.name: c.stat().st_mtime_ns for c in out.glob("v__*.json")}
    capsys.readouterr()
    _run(p, out)
    assert "[skip]" in capsys.readouterr().out and "[launch]" not in capsys.readouterr().out
    assert {c.name: c.stat().st_mtime_ns for c in out.glob("v__*.json")} == mtimes


def test_torn_or_old_schema_cells_are_redone(setup, tmp_path):
    make, out = setup
    p = make()
    _run(p, out)
    victim = next(out.glob("v__c2__r0.json"))
    victim.write_text('{"half": ')  # torn write
    assert not cell_done(victim)
    victim.write_text(json.dumps({"schema_version": 1}))
    assert not cell_done(victim)
    _run(p, out)
    assert cell_done(victim)


def test_total_failure_aborts_with_a_clear_error_instead_of_burning_gpu_time(setup):
    make, out = setup
    cfg = {**CFG, "variants": [{"name": "v", "server_args": ["--fail-after", "2"]}]}
    with pytest.raises(CellFailed, match="no successful requests|requests failed"):
        _run(make(cfg), out)
    assert not list(out.glob("v__*.json"))  # nothing invalid written as if it were data


def test_write_atomic_replaces_whole_file(tmp_path):
    p = tmp_path / "x.json"
    write_atomic(p, "a")
    write_atomic(p, "bb")
    assert p.read_text() == "bb" and not list(tmp_path.glob("*.tmp"))
