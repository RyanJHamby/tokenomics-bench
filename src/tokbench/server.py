"""Server process lifecycle: start, wait healthy, and tear down so the GPU is really free.

Failure modes this guards against (each one wastes billed GPU time or corrupts a result):
- a stale server already on the port, so we'd benchmark the wrong variant;
- teardown that raises on an already-dead group and masks the real error;
- SIGTERM that vLLM ignores or is slow to honour, leaving GPU memory held for the next
  launch (OOM at gpu-memory-utilization 0.90);
- a runner killed by SIGTERM/SIGHUP (dropped ssh) orphaning the server.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Self

import aiohttp


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def server_cmd(kind: str, cfg: dict, variant: dict, port: int) -> list[str]:
    args = [str(a) for a in variant.get("server_args", [])]
    if kind == "mock":
        return [sys.executable, "-m", "tokbench.mockserver", "--port", str(port), *args]
    model = variant.get("model", cfg["model"])
    if variant.get("engine") == "sglang":
        py = os.environ.get("SGLANG_PYTHON", sys.executable)  # SGLang lives in its own venv
        return [py, "-m", "sglang.launch_server", "--model-path", model,
                "--host", "127.0.0.1", "--port", str(port), *args]  # fmt: skip
    return ["vllm", "serve", model, "--port", str(port), *args]


async def wait_healthy(base: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as s:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early with code {proc.returncode}")
            try:
                async with s.get(f"{base}/health") as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError("server did not become healthy")


def gpu_mem_used_mib(index: int = 0) -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def kill_group(proc: subprocess.Popen, term_wait_s: float = 60.0) -> None:
    """SIGTERM the process group, escalate to SIGKILL, never raise on a dead group."""
    for sig, wait in ((signal.SIGTERM, term_wait_s), (signal.SIGKILL, 30.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


class ServerProcess:
    """Async context manager owning one server subprocess (and its children)."""

    def __init__(
        self,
        cmd: list[str],
        port: int,
        startup_timeout_s: float,
        quiet: bool = False,
        wait_gpu_free: bool = False,
        gpu_index: int = 0,
        log_path: Path | None = None,
        on_spawn=None,
        on_exit=None,
    ):
        self.cmd, self.port, self.timeout = cmd, port, startup_timeout_s
        self.quiet, self.wait_gpu_free, self.gpu_index = quiet, wait_gpu_free, gpu_index
        self.proc: subprocess.Popen | None = None
        self.log_path = log_path
        self.on_spawn, self.on_exit = on_spawn, on_exit  # lease bookkeeping for the watchdog
        self._log = None
        self.t_spawn = self.t_healthy = None  # perf_counter anchors for launch.json
        self.base = f"http://127.0.0.1:{port}"

    async def __aenter__(self) -> Self:
        if not port_free(self.port):
            raise RuntimeError(
                f"port {self.port} is in use; a stale server would be benchmarked instead"
            )
        out_kw: dict = {"stdout": subprocess.DEVNULL if self.quiet else None}
        if self.log_path is not None:  # vLLM logs the effective config, KV size, graph capture
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = await asyncio.to_thread(open, self.log_path, "ab", 0)
            out_kw = {"stdout": self._log, "stderr": subprocess.STDOUT}
        self.t_spawn = time.perf_counter()
        self.proc = await asyncio.to_thread(
            subprocess.Popen,
            self.cmd,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            **out_kw,
        )
        if self.on_spawn:
            self.on_spawn(self.proc.pid)
        try:
            await wait_healthy(self.base, self.proc, self.timeout)
            self.t_healthy = time.perf_counter()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        if self.proc is not None:
            await asyncio.to_thread(kill_group, self.proc)
        if self._log is not None:
            self._log.close()
        if self.on_exit and self.proc is not None:
            self.on_exit(self.proc.pid)
        if self.wait_gpu_free:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                used = await asyncio.to_thread(gpu_mem_used_mib, self.gpu_index)
                if used is None or used < 2048:
                    return
                await asyncio.sleep(2)
            print("[warn] GPU memory still held 90 s after server exit", flush=True)


class AttachedServer:
    """Use an already-running server (no launch, no teardown). For cross-checking this load
    generator against another tool on the very same server instance."""

    def __init__(self, base: str, timeout_s: float = 30.0):
        self.base, self.timeout = base.rstrip("/"), timeout_s

    async def __aenter__(self) -> Self:
        async with aiohttp.ClientSession() as s:
            try:
                async with s.get(
                    f"{self.base}/health", timeout=aiohttp.ClientTimeout(self.timeout)
                ) as r:
                    if r.status != 200:
                        raise RuntimeError(f"attached server unhealthy: HTTP {r.status}")
            except aiohttp.ClientError as e:
                raise RuntimeError(f"cannot reach attached server {self.base}: {e}") from e
        return self

    async def __aexit__(self, *exc) -> None:
        return None
