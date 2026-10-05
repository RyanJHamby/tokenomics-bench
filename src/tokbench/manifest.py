"""Hardware/software manifest recorded with every run."""

from __future__ import annotations

import platform
import shutil
import subprocess
from importlib import metadata


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


_GPU_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "power.min_limit",
    "power.max_limit",
    "power.default_limit",
    "clocks.max.sm",
    "clocks.max.mem",
    "compute_cap",
    "uuid",
)


def _num(x: str):
    try:
        return float(x.replace(" MiB", "").replace(" W", "").replace(" MHz", "").strip())
    except ValueError:
        return x.strip()


def gpu_info() -> dict:
    """Structured GPU facts the experiment arms depend on (power range, max clock)."""
    if not shutil.which("nvidia-smi"):
        return {}
    out = _run(
        ["nvidia-smi", "-i", "0", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader"]
    )
    parts = [x.strip() for x in out.splitlines()[0].split(",")] if out else []
    return {k: _num(v) for k, v in zip(_GPU_FIELDS, parts, strict=False)}


def manifest() -> dict:
    gpu = ""
    if shutil.which("nvidia-smi"):
        gpu = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm",
                "--format=csv,noheader",
            ]
        )
    try:
        vllm = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        vllm = ""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu or "none",
        "gpu_info": gpu_info(),
        "vllm_version": vllm or "not installed",
        "tokbench_git_sha": _run(["git", "rev-parse", "HEAD"]) or "unknown",
    }
