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
    "warmup_s": 0.5,
    "measure_s": 2.0,
    "drain_s": 5.0,
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
        assert s["n_failed"] == 0 and s["n_requests"] > 5  # only the measure window counts
        assert s["mean_prompt_tokens"] == 32 and d["workload_ok"] and d["sampler_ok"]
        # J/token and throughput use the same requests and the same window
        assert e["j_per_output_token"] == pytest.approx(e["energy_j"] / s["output_tokens"])
        assert e["window_s"] == pytest.approx(s["window_s"]) == pytest.approx(2.0)
        assert s["throughput_tok_s"] == pytest.approx(s["output_tokens"] / 2.0)
        assert d["power"]["bad_throttle_seen"] is False
    assert not list(out.glob("*.tmp"))


def test_resume_skips_finished_cells_without_launching_a_server(setup, capsys):
    make, out = setup
    p = make()
    _run(p, out)
    mtimes = {c.name: c.stat().st_mtime_ns for c in out.glob("v__*.json")}
    capsys.readouterr()
    _run(p, out)
    text = capsys.readouterr().out
    assert "0/1 server launches pending" in text and "[launch]" not in text
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


def test_real_runs_are_refused_when_estimate_exceeds_remaining_budget(setup, tmp_path, monkeypatch):
    from tokbench import budget

    make, out = setup
    b = tmp_path / "b"
    b.mkdir()
    (b / "budget.yaml").write_text(yaml.safe_dump({"cap_usd": 1.0, "safety_margin": 0.2}))
    monkeypatch.setattr(budget, "DIR", b)
    big = {**CFG, "repeats": 5, "startup_seconds": 600}  # far more than $1 at $3/hr
    rc = runner.main(
        [
            str(make(big)),
            "--server",
            "vllm",
            "--gpu",
            "nvml",
            "--usd-per-hr",
            "3",
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    assert rc == 3 and not out.exists()  # refused before touching GPU or creating output
    (b / "budget.yaml").write_text(yaml.safe_dump({"cap_usd": 500.0, "safety_margin": 0.2}))
    assert (
        runner.main(
            [
                str(make(big)),
                "--server",
                "vllm",
                "--gpu",
                "nvml",
                "--usd-per-hr",
                "3",
                "--out",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )


def test_mock_runs_ignore_the_budget(setup, tmp_path, monkeypatch):
    from tokbench import budget

    make, out = setup
    monkeypatch.setattr(budget, "DIR", tmp_path / "nonexistent")  # would raise if consulted
    assert _run(make(), out, "--dry-run") == 0


def test_capacity_search_on_mock_finds_the_analytic_knee_and_rel_loads_use_it(setup):
    """Mock capacity is known: 2 slots / (0.02 s/token * 5 tokens) = 20 req/s."""
    make, out = setup
    cfg = {
        **CFG,
        "workload": {"input_tokens": 16, "output_tokens": 5},
        "measure_s": 3.0,
        "warmup_s": 0.5,
        "drain_s": 3.0,
        "slo": {"ttft_s": 0.5, "tpot_s": 0.2},
        "variants": [
            {
                "name": "m",
                "server_args": ["--tpot", "0.02", "--slots", "2", "--prefill", "0"],
                "slots": 2,
            }
        ],
        "loads": [{"mode": "capacity_search", "qps_lo": 4, "qps_hi": 64, "resolution": 0.15}],
    }
    assert _run(make(cfg), out) == 0
    cap = json.loads((out / "capacity.json").read_text())["m"]
    assert 10 <= cap <= 21  # true knee 20 req/s; p99 TTFT<=0.5s needs headroom below it
    probes = sorted(out.glob("m__q*__r0.json"))
    assert len(probes) >= 4 and (out / "capacity__m__r0.json").exists()
    # resume: capacity already known -> launch skipped without starting a server
    # (second run must not rewrite the capacity file)
    mt = (out / "capacity.json").stat().st_mtime_ns
    assert _run(make(cfg), out) == 0 and (out / "capacity.json").stat().st_mtime_ns == mt


def test_rel_loads_run_at_fractions_of_the_measured_capacity(setup, tmp_path):
    make, out = setup
    capfile = tmp_path / "cap.json"
    capfile.write_text(json.dumps({"ref": 10.0}))
    cfg = {
        **CFG,
        "capacity_file": str(capfile),
        "loads": [
            {"mode": "open", "rel": 0.5, "ref": "ref"},
            {"mode": "open", "rel": 1.0, "ref": "ref"},
        ],
    }
    assert _run(make(cfg), out) == 0
    assert sorted(p.name for p in out.glob("v__*.json")) == ["v__q10__r0.json", "v__q5__r0.json"]


def test_soak_and_idle_baseline_are_recorded(setup):
    make, out = setup
    cfg = {
        **CFG,
        "soak_s": 1.0,
        "soak_concurrency": 2,
        "idle_s": 0.5,
        "loads": [{"mode": "open", "qps": 10}],
    }
    assert _run(make(cfg), out) == 0
    d = json.loads(next(out.glob("v__*.json")).read_text())
    assert d["soak_s"] == 1.0 and d["idle_power_w"] == pytest.approx(60.0)  # fake idle watts


def test_deliberate_overload_is_recorded_not_aborted(setup):
    """>=50% 'incomplete' is the expected signature of 2x-capacity load; only real errors
    (http/truncation) trigger fail-fast."""
    make, out = setup
    cfg = {
        **CFG,
        "drain_s": 0.3,
        "measure_s": 2.5,
        "warmup_s": 0.3,
        "workload": {"input_tokens": 16, "output_tokens": 10},
        "variants": [{"name": "v", "server_args": ["--tpot", "0.05", "--slots", "1"], "slots": 1}],
        "loads": [{"mode": "open", "qps": 12}],
    }
    assert _run(make(cfg), out) == 0
    s = json.loads(next(out.glob("v__*.json")).read_text())["summary"]
    assert "incomplete" in s["errors"] and s["n_failed"] / s["n_requests"] >= 0.5
    assert s["completed_req_s"] < s["offered_req_s"]


def test_cells_record_client_loop_lag_and_flag_saturation(setup):
    make, out = setup
    assert _run(make(), out) == 0
    d = json.loads(next(out.glob("v__*.json")).read_text())
    assert d["client"]["client_ok"] and d["client"]["n_samples"] > 50
    # an impossible threshold must mark the cell invalid in the aggregate (not crash)
    from tokbench.report import aggregate, load_cells

    cells = load_cells(out)
    for c in cells:
        c["client"]["client_ok"] = False
    assert all(r["invalid"] for r in aggregate(cells))


def test_every_repeat_of_a_capacity_search_runs_not_just_the_first(setup):
    """Regression: launch_done keyed on the shared capacity file marked repeats 2..n done
    after repeat 1, so the pilot would have had one capacity sample and no variance."""
    make, out = setup
    cfg = {
        **CFG,
        "repeats": 2,
        "workload": {"input_tokens": 16, "output_tokens": 5},
        "measure_s": 2.0,
        "warmup_s": 0.3,
        "drain_s": 2.0,
        "slo": {"ttft_s": 0.5, "tpot_s": 0.2},
        "variants": [
            {
                "name": "m",
                "server_args": ["--tpot", "0.02", "--slots", "2", "--prefill", "0"],
                "slots": 2,
            }
        ],
        "loads": [{"mode": "capacity_search", "qps_lo": 4, "qps_hi": 32, "resolution": 0.3}],
    }
    assert _run(make(cfg), out) == 0
    assert (out / "capacity__m__r0.json").exists() and (out / "capacity__m__r1.json").exists()
    mt = {p.name: p.stat().st_mtime_ns for p in out.glob("capacity__*.json")}
    assert _run(make(cfg), out) == 0  # resume: nothing rerun
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("capacity__*.json")} == mt
