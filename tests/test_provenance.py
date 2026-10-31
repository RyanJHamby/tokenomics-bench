"""Spec bench-run-provenance: a hash index for a saved run directory, and a verifier."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import socket
from pathlib import Path

import pytest
import yaml

from tokbench import runner

XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-run-provenance (remove this marker when implemented)"
)
CFG = {
    "name": "t",
    "block": "T",
    "model": "mock",
    "seed": 0,
    "repeats": 2,
    "warmup_s": 0.3,
    "measure_s": 1.0,
    "drain_s": 3.0,
    "startup_seconds": 10,
    "idle_s": 0.3,
    "slo": {"ttft_s": 0.5, "tpot_s": 0.1},
    "workload": {"input_tokens": 32, "output_tokens": 6},
    "variants": [{"name": "base", "server_args": ["--tpot", "0.006"], "slots": 4}],
    "loads": [{"mode": "closed", "concurrency": 4}],
}


def _p():
    return importlib.import_module("tokbench.provenance")


@pytest.fixture(scope="module")
def master(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prov")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    old, runner.PORT = runner.PORT, port
    try:
        (tmp / "c.yaml").write_text(yaml.safe_dump(CFG))
        out = tmp / "out"
        assert runner.main([str(tmp / "c.yaml"), "--server", "mock", "--gpu", "fake",
                            "--usd-per-hr", "1", "--out", str(out)]) == 0  # fmt: skip
    finally:
        runner.PORT = old
    return out


@pytest.fixture
def run(master, tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(master, dst)
    return dst


def _files(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.name != "PROVENANCE.json")


@XFAIL
def test_index_lists_every_file_with_its_sha256_and_size(run):
    idx = _p().build_index(run)
    on_disk = _files(run)
    assert len(on_disk) > 10  # the mock run really produced artifacts
    assert set(idx["files"]) == {p.relative_to(run).as_posix() for p in on_disk}
    for p in on_disk:
        e = idx["files"][p.relative_to(run).as_posix()]
        assert e["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert e["bytes"] == p.stat().st_size


@XFAIL
def test_index_summarises_cells_launches_and_manifest(run):
    idx = _p().build_index(run)
    cell_files = [
        p for p in run.glob("*.json") if p.name != "manifest.json" and "capacity" not in p.name
    ]
    assert len(idx["cells"]) == len(cell_files) == 2
    assert {c["repeat"] for c in idx["cells"]} == {0, 1}
    for c in idx["cells"]:
        assert c["variant"] == "base" and c["file"].endswith(".json")
        assert c["synthetic"] is True and c["schema_version"] == 2
    assert idx["synthetic"] is True
    assert sorted(idx["launches"]) == ["base__r0", "base__r1"]
    man = json.loads((run / "manifest.json").read_text())
    assert idx["manifest"]["tokbench_git_sha"] == man["tokbench_git_sha"]
    assert idx["manifest"]["gpu"] == man["gpu"]


@XFAIL
def test_write_is_deterministic_and_excludes_itself(run):
    path = _p().write_index(run)
    assert path == run / "PROVENANCE.json"
    first = path.read_bytes()
    assert _p().write_index(run) == path and path.read_bytes() == first
    assert "PROVENANCE.json" not in json.loads(first)["files"]


@XFAIL
def test_verify_clean_then_detects_modified_missing_and_unlisted(run):
    _p().write_index(run)
    assert _p().verify(run) == []
    victim = next(run.glob("*.req.jsonl.gz"))
    victim.write_bytes(victim.read_bytes() + b"x")
    gone = next(run.glob("*.gpu.csv.gz"))
    gone_name = gone.name
    gone.unlink()
    (run / "extra.txt").write_text("hi")
    problems = _p().verify(run)
    assert len(problems) == 3
    joined = "\n".join(problems)
    assert f"modified: {victim.name}" in joined
    assert f"missing: {gone_name}" in joined
    assert "unlisted: extra.txt" in joined


@XFAIL
def test_errors_instead_of_vacuous_success(tmp_path, run):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "note.txt").write_text("not a run")
    with pytest.raises(ValueError):
        _p().build_index(empty)  # no cells: not a run
    with pytest.raises(FileNotFoundError):
        _p().verify(run)  # never indexed


@XFAIL
def test_cli(run, capsys):
    assert _p().main([str(run)]) == 0
    assert (run / "PROVENANCE.json").is_file()
    assert _p().main([str(run), "--verify"]) == 0
    (run / "extra.txt").write_text("hi")
    assert _p().main([str(run), "--verify"]) == 1
    assert "unlisted: extra.txt" in capsys.readouterr().out
