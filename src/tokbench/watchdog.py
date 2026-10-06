"""Out-of-process watchdog: undo what the runner did to the GPU if the runner dies.

The runner's context managers and signal handlers restore clocks, power limit and the
server on SIGTERM/SIGHUP/SIGINT/exceptions. They cannot run on SIGKILL, an OOM kill or a pod
kill, which would leave the (rented, shared) GPU locked or capped and vLLM orphaned holding
the GPU. The runner starts this process detached (own session) so it survives the runner. It
watches a lease file and heartbeat; on runner death or a stale heartbeat it resets clocks,
restores the DEFAULT power limit, and kills the recorded server process groups.

    python -m tokbench.watchdog --lease <dir>/lease.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

HEARTBEAT_EVERY_S = 5.0
STALE_AFTER_S = 30.0
POLL_S = 3.0


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def decide(lease_exists: bool, runner_alive: bool, hb_age_s: float | None, stale_s: float) -> str:
    """'exit_clean' (lease removed: the runner finished normally), 'cleanup' (runner died or
    stopped heartbeating), or 'ok'."""
    if not lease_exists:
        return "exit_clean"
    if not runner_alive:
        return "cleanup"
    if hb_age_s is None or hb_age_s > stale_s:
        return "cleanup"
    return "ok"


def _smi(run, *args: str) -> str:
    p = run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=False)
    return (p.stdout or "") + (p.stderr or "")


def cleanup(
    lease: dict, run=subprocess.run, killpg=os.killpg, sleep=time.sleep, log=print
) -> list[str]:
    """Reset the GPU and kill the server groups. Every step is attempted; failures are
    logged, never raised. Returns the actions taken (for the log and tests)."""
    actions = []
    idx = str(lease.get("gpu_index", 0))
    try:
        _smi(run, "-i", idx, "-rgc")
        actions.append("reset_clocks")
        out = _smi(
            run, "-i", idx, "--query-gpu=power.default_limit", "--format=csv,noheader,nounits"
        )
        default = int(float(out.strip().splitlines()[0]))
        _smi(run, "-i", idx, "-pl", str(default))
        actions.append(f"restore_power_limit={default}")
        log(
            f"[watchdog] readback: {_smi(run, '-i', idx, '--query-gpu=power.limit,clocks.applications.graphics', '--format=csv,noheader').strip()}"
        )
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        log(f"[watchdog] GPU reset failed: {type(e).__name__}: {e}")
    for pgid in lease.get("server_pgids", []):
        for sig, wait in ((signal.SIGTERM, 8.0), (signal.SIGKILL, 0.0)):
            try:
                killpg(pgid, sig)
                actions.append(f"killpg({pgid},{sig.name})")
            except ProcessLookupError:
                break
            except PermissionError as e:
                log(f"[watchdog] cannot signal group {pgid}: {e}")
                break
            sleep(wait)
    return actions


class Lease:
    """Runner-side handle: writes the lease and a heartbeat thread; removing it (stop) tells
    the watchdog the shutdown was clean."""

    def __init__(self, path: Path, gpu_index: int = 0):
        self.path, self.gpu_index = Path(path), gpu_index
        self.hb = self.path.with_suffix(".hb")
        self.server_pgids: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"runner_pid": os.getpid(), "gpu_index": self.gpu_index,
                                   "server_pgids": self.server_pgids}))  # fmt: skip
        os.replace(tmp, self.path)

    def _beat(self) -> None:
        while not self._stop.is_set():
            self.hb.write_text(str(time.time()))
            self._stop.wait(HEARTBEAT_EVERY_S)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self.hb.write_text(str(time.time()))
        self._thread.start()

    def add_server(self, pgid: int) -> None:
        self.server_pgids.append(pgid)
        self._write()

    def remove_server(self, pgid: int) -> None:
        self.server_pgids = [p for p in self.server_pgids if p != pgid]
        self._write()

    def stop(self) -> None:
        self._stop.set()
        for f in (self.path, self.hb):
            f.unlink(missing_ok=True)


def spawn(lease_path: Path, log_path: Path) -> subprocess.Popen:
    """Start the watchdog in its own session so it outlives a SIGKILLed runner."""
    import sys

    return subprocess.Popen(
        [sys.executable, "-m", "tokbench.watchdog", "--lease", str(lease_path)],
        start_new_session=True, stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
    )  # fmt: skip


def watch(
    lease_path: Path,
    stale_s: float = STALE_AFTER_S,
    poll_s: float = POLL_S,
    run=subprocess.run,
    once: bool = False,
    log=print,
) -> str:
    """Loop until the runner finishes cleanly or cleanup has run. Returns the outcome."""
    hb = lease_path.with_suffix(".hb")
    while True:
        exists = lease_path.exists()
        lease, alive, age = {}, False, None
        if exists:
            try:
                lease = json.loads(lease_path.read_text())
                alive = pid_alive(lease["runner_pid"])
                age = time.time() - float(hb.read_text())
            except (OSError, ValueError, KeyError):
                pass  # half-written lease/heartbeat: treat as no information this round
        verdict = decide(exists, alive, age, stale_s)
        if verdict == "exit_clean":
            return "exit_clean"
        if verdict == "cleanup":
            log(
                f"[watchdog] runner pid {lease.get('runner_pid')} gone or stale (age={age}); cleaning up"
            )
            acts = cleanup(lease, run=run, log=log)
            log(f"[watchdog] actions: {acts}")
            lease_path.unlink(missing_ok=True)
            hb.unlink(missing_ok=True)
            return "cleaned"
        if once:
            return "ok"
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lease", required=True)
    ap.add_argument("--stale-s", type=float, default=STALE_AFTER_S)
    a = ap.parse_args(argv)
    print(f"[watchdog] watching {a.lease}", flush=True)
    outcome = watch(Path(a.lease), a.stale_s, log=lambda m: print(m, flush=True))
    print(f"[watchdog] done: {outcome}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
