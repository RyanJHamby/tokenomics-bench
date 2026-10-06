import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest
import yaml

from tokbench import runner
from tokbench import watchdog as wd


def test_decide_table():
    assert wd.decide(False, True, 1.0, 30) == "exit_clean"  # lease removed: finished cleanly
    assert wd.decide(True, False, 1.0, 30) == "cleanup"  # runner died
    assert wd.decide(True, True, 31.0, 30) == "cleanup"  # runner alive but stopped heartbeating
    assert wd.decide(True, True, None, 30) == "cleanup"  # no heartbeat info at all
    assert wd.decide(True, True, 5.0, 30) == "ok"


class FakeRun:
    def __init__(self, default="350"):
        self.cmds, self.default = [], default

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        out = self.default if "--query-gpu=power.default_limit" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def test_cleanup_resets_clocks_restores_default_power_limit_and_kills_groups_in_order():
    run, signals = FakeRun("350"), []

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == signal.SIGKILL:
            raise ProcessLookupError  # already dead after SIGTERM would stop earlier

    acts = wd.cleanup({"gpu_index": 1, "server_pgids": [4242]}, run=run, killpg=killpg,
                      sleep=lambda s: None, log=lambda m: None)  # fmt: skip
    smi = [c for c in run.cmds if c[0] == "nvidia-smi"]
    assert smi[0][:4] == ["nvidia-smi", "-i", "1", "-rgc"]
    assert ["nvidia-smi", "-i", "1", "-pl", "350"] in smi  # DEFAULT limit, not the capped one
    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert acts[0] == "reset_clocks" and "restore_power_limit=350" in acts[1]


def test_cleanup_survives_every_failure_and_still_kills_the_server():
    def broken_run(cmd, **kw):
        raise OSError("nvidia-smi: not found")

    killed = []
    acts = wd.cleanup({"server_pgids": [7]}, run=broken_run, killpg=lambda p, s: killed.append(p),
                      sleep=lambda s: None, log=lambda m: None)  # fmt: skip
    assert 7 in killed and not any(a.startswith("restore") for a in acts)


def test_a_sigkilled_runner_triggers_cleanup_and_the_orphaned_server_group_dies(tmp_path):
    runner_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              start_new_session=True)  # fmt: skip
    lease = wd.Lease(tmp_path / "lease.json", gpu_index=0)
    lease.start()
    lease._write_override = None
    (tmp_path / "lease.json").write_text(json.dumps({"runner_pid": runner_proc.pid, "gpu_index": 0,
                                                     "server_pgids": [server.pid]}))  # fmt: skip
    run = FakeRun("350")
    try:
        assert wd.watch(tmp_path / "lease.json", stale_s=30, poll_s=0.05, run=run, once=True,
                        log=lambda m: None) == "ok"  # runner alive, heartbeat fresh  # fmt: skip
        runner_proc.send_signal(signal.SIGKILL)  # the case the context managers cannot handle
        runner_proc.wait(timeout=5)
        outcome = wd.watch(tmp_path / "lease.json", stale_s=30, poll_s=0.05, run=run,
                           log=lambda m: None)  # fmt: skip
        assert outcome == "cleaned"
        server.wait(timeout=15)
        assert server.returncode is not None  # killed by the watchdog, not left holding the GPU
        assert ["nvidia-smi", "-i", "0", "-rgc"] in [c[:4] for c in run.cmds]
        assert not (tmp_path / "lease.json").exists()
    finally:
        lease._stop.set()
        for p in (runner_proc, server):
            if p.poll() is None:
                p.kill()


def test_stale_heartbeat_with_a_live_runner_still_cleans_up(tmp_path):
    runner_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "lease.json").write_text(json.dumps({"runner_pid": runner_proc.pid,
                                                         "gpu_index": 0, "server_pgids": []}))  # fmt: skip
        (tmp_path / "lease.hb").write_text(str(time.time() - 120))  # hung runner
        run = FakeRun()
        assert wd.watch(tmp_path / "lease.json", stale_s=30, poll_s=0.05, run=run,
                        log=lambda m: None) == "cleaned"  # fmt: skip
        assert run.cmds
    finally:
        runner_proc.kill()


def test_clean_exit_means_no_gpu_commands(tmp_path):
    lease = wd.Lease(tmp_path / "lease.json")
    lease.start()
    lease.add_server(1234)
    assert json.loads((tmp_path / "lease.json").read_text())["server_pgids"] == [1234]
    lease.remove_server(1234)
    assert json.loads((tmp_path / "lease.json").read_text())["server_pgids"] == []
    lease.stop()
    run = FakeRun()
    assert wd.watch(tmp_path / "lease.json", run=run, once=True, log=lambda m: None) == "exit_clean"
    assert run.cmds == [] and not (tmp_path / "lease.hb").exists()


def test_runner_with_watchdog_flag_spawns_a_detached_watchdog_that_exits_when_the_run_ends(
    tmp_path, monkeypatch
):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(runner, "PORT", s.getsockname()[1])
    cfg = {"name": "t", "block": "T", "model": "mock", "seed": 0, "repeats": 1, "warmup_s": 0.3,
           "measure_s": 1.0, "drain_s": 3.0, "startup_seconds": 10,
           "workload": {"input_tokens": 16, "output_tokens": 4},
           "variants": [{"name": "v", "server_args": ["--tpot", "0.004"], "slots": 4}],
           "loads": [{"mode": "open", "qps": 10}]}  # fmt: skip
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    rc = runner.main([str(p), "--server", "mock", "--gpu", "fake", "--usd-per-hr", "1",
                      "--out", str(out), "--watchdog"])  # fmt: skip
    assert rc == 0 and not (out / "lease.json").exists()  # clean shutdown removed the lease
    log = out / "watchdog.log"
    deadline = time.time() + 15
    while time.time() < deadline and "done: exit_clean" not in log.read_text():
        time.sleep(0.3)
    assert "watching" in log.read_text() and "done: exit_clean" in log.read_text()
    canary = json.loads((out / "launches" / "v__r0" / "launch.json").read_text())["canary"]
    assert canary["ok"] and canary["prompt_tokens"] == 16


def test_canary_fails_the_launch_when_the_server_cannot_serve_a_real_completion(
    tmp_path, monkeypatch
):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(runner, "PORT", s.getsockname()[1])
    cfg = {"name": "t", "block": "T", "model": "mock", "seed": 0, "repeats": 1, "warmup_s": 0.3,
           "measure_s": 1.0, "drain_s": 3.0, "startup_seconds": 10,
           "workload": {"input_tokens": 16, "output_tokens": 4},
           "variants": [{"name": "v", "server_args": ["--fail-after", "1"], "slots": 4}],
           "loads": [{"mode": "open", "qps": 10}]}  # fmt: skip
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(runner.CellFailed, match="canary"):
        runner.main([str(p), "--server", "mock", "--gpu", "fake", "--usd-per-hr", "1",
                     "--out", str(tmp_path / "o")])  # fmt: skip
    assert os.path.exists(tmp_path / "o" / "launches" / "v__r0" / "crash" / "exception.txt")
