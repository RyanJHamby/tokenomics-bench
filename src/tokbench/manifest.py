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
        "vllm_version": vllm or "not installed",
        "tokbench_git_sha": _run(["git", "rev-parse", "HEAD"]) or "unknown",
    }
