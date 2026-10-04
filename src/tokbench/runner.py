"""Run an experiment config: launch a server per (variant, repeat), sweep loads, record
load-generator stats, GPU energy, and server metrics as one JSON file per cell.

    python -m tokbench.runner configs/b1_knee.yaml --server vllm --gpu nvml \
        --usd-per-hr 1.20 --out results/b1-2026-10-12 [--dry-run]

--server mock --gpu fake runs everything on a laptop; results are flagged synthetic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

from .config import estimate_cost, load_config, load_label, plan
from .loadgen import run_closed_loop, run_open_loop, summarize
from .manifest import manifest
from .telemetry import FakeBackend, MetricsScraper, NvmlBackend, PowerSampler, energy_joules
from .workloads import make_prompt_fn

PORT = 8000


def server_cmd(kind: str, cfg: dict, variant: dict, port: int) -> list[str]:
    args = [str(a) for a in variant.get("server_args", [])]
    if kind == "mock":
        return [sys.executable, "-m", "tokbench.mockserver", "--port", str(port), *args]
    return ["vllm", "serve", cfg["model"], "--port", str(port), *args]


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


def _nvsmi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


class PowerCap:
    """Set `nvidia-smi -pl` for the duration of a launch, then restore. Needs root."""

    def __init__(self, watts: float | None, enabled: bool):
        self.watts, self.enabled, self.prev = watts, enabled and watts is not None, None

    def __enter__(self):
        if self.enabled:
            self.prev = _nvsmi("--query-gpu=power.limit", "--format=csv,noheader,nounits").strip()
            _nvsmi("-pl", str(int(self.watts)))
        return self

    def __exit__(self, *exc):
        if self.enabled and self.prev:
            _nvsmi("-pl", str(int(float(self.prev))))


def _metric_stats(rows: list[tuple[float, dict]], t0: float, t1: float) -> dict:
    sel = [m for t, m in rows if t0 <= t <= t1]
    out: dict = {}
    keys = {k for m in sel for k in m}
    for k in sorted(keys):
        vals = [m[k] for m in sel if k in m]
        out[k] = {"max": max(vals), "mean": sum(vals) / len(vals), "last": vals[-1]}
    return out


async def run_load(cfg: dict, load: dict, base: str, variant: dict, args, repeat: int) -> dict:
    wl = cfg["workload"]
    make_body = make_prompt_fn(
        wl["input_tokens"],
        wl["output_tokens"],
        wl.get("prefix_share", 0.0),
        cfg["model"],
        cfg["seed"],
    )
    url = f"{base}/v1/chat/completions"
    scraper = MetricsScraper(f"{base}/metrics", interval_s=0.5)
    slots = max(1, int(variant.get("slots", 8)))

    def fake_load() -> float:
        return (
            (scraper.rows[-1][1].get("vllm:num_requests_running", 0) / slots) if scraper.rows else 0
        )

    backend = (
        FakeBackend(fake_load, cap_w=variant.get("power_cap_w"))
        if args.gpu == "fake"
        else NvmlBackend()
    )
    async with scraper:
        with PowerSampler(backend, hz=10) as power:
            t0 = time.perf_counter()
            if load["mode"] == "closed":
                recs, wall = await run_closed_loop(
                    url, make_body, load["concurrency"], cfg["n_requests"]
                )
            else:
                recs, wall = await run_open_loop(
                    url, make_body, load["qps"], cfg["n_requests"], seed=cfg["seed"] + repeat
                )
            t1 = time.perf_counter()
    summary = summarize(recs, wall, warmup=cfg["warmup"])
    # Energy over the whole load window (warmup included), so divide by all tokens served.
    window_tokens = sum(r.n_out for r in recs if r.ok)
    energy = energy_joules([s for s in power.samples if t0 <= s.t <= t1 + 1])
    return {
        "summary": summary,
        "energy_j": energy,
        "window_output_tokens": window_tokens,
        "j_per_token": energy / window_tokens if window_tokens else None,
        "throttle_seen": any(s.throttle_reasons & ~0x1 for s in power.samples),
        "mean_power_w": sum(s.power_w for s in power.samples) / len(power.samples),
        "server_metrics": _metric_stats(scraper.rows, t0, t1),
        "errors": sorted({r.error for r in recs if r.error}),
    }


async def run_launch(cfg: dict, launch, args, out: Path, man: dict) -> None:
    variant = launch.variant
    base = f"http://127.0.0.1:{PORT}"
    cmd = server_cmd(args.server, cfg, variant, PORT)
    print(f"[launch] {variant['name']} repeat={launch.repeat}: {' '.join(cmd)}", flush=True)
    proc = await asyncio.to_thread(
        subprocess.Popen,
        cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL if args.quiet_server else None,
    )
    try:
        await wait_healthy(base, proc, timeout_s=cfg["startup_seconds"] * 3)
        with PowerCap(variant.get("power_cap_w"), enabled=args.gpu == "nvml"):
            for load in cfg["loads"]:
                res = await run_load(cfg, load, base, variant, args, launch.repeat)
                res.update(
                    block=cfg["block"],
                    config=cfg["name"],
                    variant=variant["name"],
                    load=load,
                    repeat=launch.repeat,
                    gpu_usd_per_hr=args.usd_per_hr,
                    power_cap_w=variant.get("power_cap_w"),
                    synthetic=args.server == "mock" or args.gpu == "fake",
                    manifest=man,
                    workload=cfg["workload"],
                    seed=cfg["seed"],
                )
                name = f"{variant['name']}__{load_label(load)}__r{launch.repeat}.json"
                (out / name).write_text(json.dumps(res, indent=2))
                s = res["summary"]
                print(
                    f"  {load_label(load):>6} tput={s['throughput_tok_s']:.0f} tok/s "
                    f"ttft_p99={s['ttft_p99']:.3f}s tpot_p99={s['tpot_p99'] * 1000:.1f}ms "
                    f"J/tok={res['j_per_token']:.3f} fail={s['n_failed']}",
                    flush=True,
                )
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--server", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--gpu", choices=["nvml", "fake"], default="nvml")
    ap.add_argument("--usd-per-hr", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print cost estimate and exit")
    ap.add_argument("--quiet-server", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    est = estimate_cost(cfg, args.usd_per_hr)
    print(
        f"[plan] {est['launches']} server launches, ~{est['gpu_hours']:.2f} GPU-h, ~${est['usd']:.2f}"
    )
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = manifest()
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    for launch in plan(cfg):
        asyncio.run(run_launch(cfg, launch, args, out, man))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
