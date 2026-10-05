import socket
import subprocess
import sys

import pytest

from tokbench.power import CapUnavailable, PowerCap, validate_cap
from tokbench.server import ServerProcess, kill_group, port_free, server_cmd


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_port_free_detects_a_listener():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert not port_free(s.getsockname()[1])
    assert port_free(_free_port())


def test_kill_group_on_dead_process_does_not_raise():
    p = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    p.wait()
    kill_group(p)  # used to raise ProcessLookupError and mask the real error


def test_kill_group_escalates_to_sigkill_when_sigterm_is_ignored():
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('r', flush=True); time.sleep(60)"
    p = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE
    )
    assert p.stdout.readline().strip() == b"r"  # handler installed
    kill_group(p, term_wait_s=0.5)
    assert p.poll() is not None


async def test_server_process_refuses_busy_port_and_cleans_up():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(RuntimeError, match="in use"):
            async with ServerProcess(["true"], s.getsockname()[1], 5):
                pass


async def test_server_process_starts_serves_and_frees_port():
    port = _free_port()
    cmd = server_cmd("mock", {}, {"server_args": []}, port)
    async with ServerProcess(cmd, port, 30, quiet=True) as srv:
        assert not port_free(port)
        assert srv.proc.poll() is None
    assert srv.proc.poll() is not None and port_free(port)


async def test_server_that_dies_on_startup_raises_real_error_not_cleanup_error():
    port = _free_port()
    with pytest.raises(RuntimeError, match="exited early"):
        async with ServerProcess([sys.executable, "-c", "raise SystemExit(3)"], port, 10):
            pass


LIMITS = {"min": 150.0, "max": 350.0, "default": 350.0, "current": 350.0}


def test_validate_cap_range():
    validate_cap(150, LIMITS)
    validate_cap(350, LIMITS)
    for bad in (149.9, 350.1, 0):
        with pytest.raises(CapUnavailable):
            validate_cap(bad, LIMITS)


def test_disabled_or_absent_cap_is_a_noop_and_never_shells_out(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("nvidia-smi must not be called")

    monkeypatch.setattr(subprocess, "run", boom)
    with PowerCap(250, enabled=False), PowerCap(None, enabled=True):
        pass


def test_cap_restores_default_not_current_and_failure_does_not_leave_cap(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = {
            "power.min_limit": "150",
            "power.max_limit": "350",
            "power.default_limit": "350",
            "power.limit": "99",
        }  # stale cap currently set
        for f, v in out.items():
            if f"--query-gpu={f}" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout=v + "\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # readback says 99 W (stale), cap request is 250 -> "did not stick" -> restore to default
    with pytest.raises(CapUnavailable, match="did not stick"), PowerCap(250, enabled=True):
        pass
    restores = [c for c in calls if "-pl" in c]
    assert restores[-1][-1] == "350"  # default limit, not the stale 99
