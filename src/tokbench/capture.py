"""Persist everything needed to explain, re-slice and re-plot a run after the pod is gone.

Only per-cell summaries used to be saved. The audit's point: the GPU is rented once, and
anything not written to disk during the run cannot be recovered. Artifacts (all small,
gzip-compressed text so no numpy/parquet dependency):

  <cell>.req.jsonl.gz      every request: scheduled/sent/first/last times, tokens, chunks,
                           request id, output crc, inter-chunk gaps (0.1 ms ints)
  <cell>.gpu.csv.gz        the full GPU sample series (power, clocks, temp, throttle bits,
                           energy counter, enforced limit, P-state, memory, violation counters)
  <cell>.metrics.jsonl.gz  the vLLM /metrics series (every key, 2 Hz); raw exposition text
                           with labels and histogram buckets at intervals in .metrics.raw.txt.gz
  launches/<variant>__r<k>/server.log, launch.json, soak/idle series
  events.jsonl             wall + monotonic anchored event log
  _env/                    pip freeze, nvidia-smi -q (identifiers hashed), lscpu, cgroup, HF
                           snapshot, allow-listed environment. NEVER the raw environment.

All times in the files are perf_counter seconds; `Anchor` converts them to wall-clock so GPU
traces can be aligned with vLLM's own log lines.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .loadgen.client import RequestRecord
from .telemetry.gpu import GpuSample

ITL_UNIT_S = 1e-4  # inter-chunk gaps stored as ints in 0.1 ms units
ENV_PREFIXES = (
    "CUDA_",
    "VLLM_",
    "NCCL",
    "HF_HOME",
    "HF_HUB_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
    "NVIDIA_",
    "LD_LIBRARY_PATH",
    "PYTHONUNBUFFERED",
)
SECRET_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASS", "CREDENTIAL")


@dataclass(frozen=True)
class Anchor:
    t_mono0: float
    t_wall0: float

    @classmethod
    def now(cls) -> Anchor:
        return cls(time.perf_counter(), time.time())

    def wall(self, t_mono: float) -> float:
        return self.t_wall0 + (t_mono - self.t_mono0)

    def to_dict(self) -> dict:
        return {"t_mono0": self.t_mono0, "t_wall0": self.t_wall0}


class EventLog:
    """Append-only JSONL event log; every line carries monotonic and wall time."""

    def __init__(self, path: Path, anchor: Anchor):
        self.path, self.anchor = path, anchor

    def emit(self, kind: str, **fields) -> None:
        t = time.perf_counter()
        line = {"kind": kind, "t_mono": t, "t_wall": self.anchor.wall(t), **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")


def _gz_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        f.write(text)
    os.replace(tmp, path)


def write_requests(path: Path, records: list[RequestRecord], header: dict) -> None:
    lines = [json.dumps({"header": header})]
    for r in records:
        lines.append(
            json.dumps(
                {
                    "i": r.idx,
                    "ts": r.t_sched,
                    "tsd": r.t_send,
                    "tf": r.t_first,
                    "tl": r.t_last,
                    "no": r.n_out,
                    "np": r.prompt_tokens,
                    "nc": r.n_chunks,
                    "ok": int(r.ok),
                    "e": r.error,
                    "id": r.req_id,
                    "crc": r.crc,
                    "itl": [min(65535, round(g / ITL_UNIT_S)) for g in r.itls],
                }
            )
        )
    _gz_write(path, "\n".join(lines) + "\n")


def load_requests(path: Path) -> tuple[dict, list[RequestRecord]]:
    """Inverse of write_requests (inter-chunk gaps are quantised to 0.1 ms)."""
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    header, recs = rows[0]["header"], []
    for d in rows[1:]:
        recs.append(
            RequestRecord(
                t_sched=d["ts"],
                t_send=d["tsd"],
                t_first=d["tf"],
                t_last=d["tl"],
                n_out=d["no"],
                prompt_tokens=d["np"],
                n_chunks=d["nc"],
                ok=bool(d["ok"]),
                error=d["e"],
                req_id=d["id"],
                crc=d["crc"],
                idx=d["i"],
                itls=[u * ITL_UNIT_S for u in d["itl"]],
            )
        )
    return header, recs


GPU_COLUMNS = (
    "t_mono",
    "t_wall",
    "power_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_c",
    "util_pct",
    "throttle_reasons",
    "energy_mj",
    "enforced_limit_w",
    "pstate",
    "mem_used_mib",
    "viol_power_ns",
    "viol_thermal_ns",
)


def write_gpu_series(path: Path, samples: list[GpuSample], anchor: Anchor) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(GPU_COLUMNS)
    for s in samples:
        w.writerow(
            [
                f"{s.t:.6f}",
                f"{anchor.wall(s.t):.6f}",
                s.power_w,
                s.sm_clock_mhz,
                s.mem_clock_mhz,
                s.temp_c,
                s.util_pct,
                s.throttle_reasons,
                "" if s.energy_mj is None else s.energy_mj,
                *("" if getattr(s, c) is None else getattr(s, c) for c in GPU_COLUMNS[9:]),
            ]
        )
    _gz_write(path, buf.getvalue())


def load_gpu_series(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def write_metrics_series(path: Path, rows: list[tuple[float, dict]], anchor: Anchor) -> None:
    lines = [json.dumps({"t": t, "tw": anchor.wall(t), "m": m}) for t, m in rows]
    _gz_write(path, "\n".join(lines) + ("\n" if lines else ""))


def write_raw_metrics(path: Path, snaps: list[tuple[float, str]], anchor: Anchor) -> None:
    parts = [f"#### t_mono={t:.3f} t_wall={anchor.wall(t):.3f}\n{text}" for t, text in snaps]
    _gz_write(path, "\n".join(parts))


# ---------------------------------------------------------------- environment bundle

_UUID = re.compile(
    r"^(\s*(?:GPU UUID|MIG UUID|Serial Number|Board Part Number)\s*:\s*)(\S.*)$", re.MULTILINE
)


def scrub_identifiers(text: str) -> str:
    """Replace device UUIDs/serials with a short hash: still joinable across files, not
    reversible, safe to publish."""
    return _UUID.sub(
        lambda m: f"{m.group(1)}<{hashlib.sha256(m.group(2).encode()).hexdigest()[:8]}>", text
    )


def allowlisted_env(env: dict[str, str]) -> dict[str, str]:
    """Only reproducibility-relevant variables, and never anything secret-looking."""
    return {
        k: v
        for k, v in sorted(env.items())
        if k.startswith(ENV_PREFIXES) and not any(w in k.upper() for w in SECRET_WORDS)
    }


def _run_text(cmd: list[str], timeout: float = 30) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.stdout + (f"\n# stderr:\n{p.stderr}" if p.stderr.strip() else "")
    except (OSError, subprocess.SubprocessError) as e:
        return f"# FAILED: {type(e).__name__}: {e}\n"


_GPU_STATIC = (
    "name,driver_version,vbios_version,compute_cap,persistence_mode,ecc.mode.current,"
    "pcie.link.gen.current,pcie.link.width.current,power.min_limit,power.max_limit,"
    "power.default_limit,clocks.max.sm,clocks.max.mem"
)

ENV_COMMANDS: dict[str, list[str]] = {
    "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
    "nvidia-smi-q.txt": ["nvidia-smi", "-q"],
    "nvidia-smi-clocks.txt": ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
    "nvidia-smi-topo.txt": ["nvidia-smi", "topo", "-m"],
    "gpu-static.csv": ["nvidia-smi", f"--query-gpu={_GPU_STATIC}", "--format=csv"],
    "lscpu.txt": ["lscpu"],
    "free.txt": ["free", "-h"],
    "uname.txt": ["uname", "-a"],
    "dmesg_tail.txt": ["sh", "-c", "dmesg 2>&1 | tail -200"],
    "nproc.txt": ["nproc"],
}


def hf_snapshot(models: list[tuple[str, str]], hf_home: Path | None = None) -> dict:
    """What is actually in the HF cache for each pinned (repo, revision): commit, file list,
    and sha256 of the small config/tokenizer files (weights' blob names are already hashes)."""
    home = hf_home or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    out = {}
    for repo, rev in models:
        snap = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
        entry: dict = {"revision": rev, "present": snap.is_dir(), "files": [], "sha256": {}}
        if snap.is_dir():
            entry["files"] = sorted(p.name for p in snap.iterdir())
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                f = snap / name
                if f.exists():
                    entry["sha256"][name] = hashlib.sha256(f.read_bytes()).hexdigest()
        out[repo] = entry
    return out


def capture_env(
    out_dir: Path, models: list[tuple[str, str]] | None = None, env: dict[str, str] | None = None
) -> list[str]:
    """Write the environment bundle. Every command is best-effort; a failure is recorded in
    the file rather than aborting the run. Returns the files written."""
    d = out_dir / "_env"
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cmd in ENV_COMMANDS.items():
        text = _run_text(cmd)
        if name.startswith("nvidia-smi") or name == "gpu-static.csv":
            text = scrub_identifiers(text)
        (d / name).write_text(text)
        written.append(name)
    for name, src in (
        ("cgroup_cpu.txt", "/sys/fs/cgroup/cpu.max"),
        ("cgroup_mem.txt", "/sys/fs/cgroup/memory.max"),
        ("os-release.txt", "/etc/os-release"),
    ):
        try:
            (d / name).write_text(Path(src).read_text())
            written.append(name)
        except OSError:
            pass
    (d / "env_allowlist.json").write_text(
        json.dumps(allowlisted_env(dict(env or os.environ)), indent=2)
    )
    (d / "hf_snapshot.json").write_text(json.dumps(hf_snapshot(models or []), indent=2))
    written += ["env_allowlist.json", "hf_snapshot.json"]
    return written


def write_crash_bundle(dir_: Path, server_log: Path | None, exc: BaseException) -> None:
    """On failure, leave enough to diagnose without the pod: error, server log tail, GPU state,
    kernel messages, process table."""
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "exception.txt").write_text("".join(traceback.format_exception(exc)))
    if server_log and server_log.exists():
        lines = server_log.read_text(errors="replace").splitlines()[-300:]
        (dir_ / "server_log_tail.txt").write_text("\n".join(lines))
    (dir_ / "nvidia-smi-q.txt").write_text(scrub_identifiers(_run_text(["nvidia-smi", "-q"])))
    (dir_ / "dmesg_tail.txt").write_text(_run_text(["sh", "-c", "dmesg 2>&1 | tail -200"]))
    (dir_ / "ps.txt").write_text(_run_text(["ps", "aux"]))
