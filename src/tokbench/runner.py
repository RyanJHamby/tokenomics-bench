"""Run an experiment config: launch a server per (variant, repeat), sweep loads, record
load-generator stats, GPU energy, and server metrics as one JSON file per cell.

    python -m tokbench.runner configs/b1_knee.yaml --server vllm --gpu nvml \
        --usd-per-hr 1.20 --out results/raw/b1-2026-10-12 [--dry-run]

--server mock --gpu fake runs everything on a laptop; results are flagged synthetic.

Resumable: re-running with the same --out skips cells whose JSON already exists and
parses. A launch whose loads are all done is skipped without starting a server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

from .config import estimate_cost, load_config, load_label, plan
from .loadgen import run_closed_loop, run_open_loop, summarize
from .manifest import manifest
from .power import CapUnavailable, PowerCap
from .server import ServerProcess, server_cmd
from .telemetry import (
    FakeBackend,
    MetricsScraper,
    NvmlBackend,
    PowerSampler,
    energy_between,
    throttle_summary,
)
from .workloads import make_prompt_fn, salt_for

PORT = 8000
SCHEMA_VERSION = 2
SAMPLE_HZ = 10


class CellFailed(RuntimeError):
    """A load produced no usable data; continuing would only burn GPU time."""


def cell_path(out: Path, variant: dict, load: dict, repeat: int) -> Path:
    return out / f"{variant['name']}__{load_label(load)}__r{repeat}.json"


def cell_done(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("schema_version") == SCHEMA_VERSION
    except (OSError, ValueError):
        return False


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _metric_stats(rows: list[tuple[float, dict]], t0: float, t1: float) -> dict:
    sel = [m for t, m in rows if t0 <= t <= t1]
    out: dict = {}
    keys = {k for m in sel for k in m}
    for k in sorted(keys):
        vals = [m[k] for m in sel if k in m]
        out[k] = {
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "first": vals[0],
            "last": vals[-1],
        }
    return out


def counter_deltas(stats: dict) -> dict:
    """Deltas of cumulative counters over the window, plus prefix-cache hit rate."""
    d = {k: v["last"] - v["first"] for k, v in stats.items() if k.endswith("_total")}
    hits = next((v for k, v in d.items() if "prefix_cache_hits" in k), None)
    qs = next((v for k, v in d.items() if "prefix_cache_queries" in k), None)
    if hits is not None and qs:
        d["prefix_cache_hit_rate"] = hits / qs
    return d


async def run_load(
    cfg: dict, load: dict, load_idx: int, base: str, variant: dict, args, repeat: int
) -> dict:
    wl = variant.get("workload", cfg["workload"])
    model = variant.get("model", cfg["model"])
    make_body = make_prompt_fn(
        wl["input_tokens"],
        wl["output_tokens"],
        wl.get("prefix_share", 0.0),
        model,
        cfg["seed"],
        salt=salt_for(repeat, load_idx),
    )
    url = f"{base}/v1/completions"
    slo = cfg["slo"]
    scraper = MetricsScraper(f"{base}/metrics", interval_s=0.5)
    slots = max(1, int(variant.get("slots", 8)))

    def fake_load() -> float:
        run = scraper.rows[-1][1].get("vllm:num_requests_running", 0) if scraper.rows else 0
        return run / slots

    backend = (
        FakeBackend(fake_load, cap_w=variant.get("power_cap_w"))
        if args.gpu == "fake"
        else NvmlBackend(args.gpu_index)
    )
    expect = wl["output_tokens"]
    async with scraper:
        with PowerSampler(backend, hz=SAMPLE_HZ) as power:
            t0 = time.perf_counter()
            if load["mode"] == "closed":
                recs, _ = await run_closed_loop(
                    url, make_body, load["concurrency"], cfg["n_requests"], expect_out=expect
                )
            else:
                recs, _ = await run_open_loop(
                    url,
                    make_body,
                    load["qps"],
                    cfg["n_requests"],
                    seed=cfg["seed"] * 31 + repeat * 7 + load_idx,
                    expect_out=expect,
                )
            t1 = time.perf_counter()
    kept = sorted(recs, key=lambda r: r.t_send)[cfg["warmup"] :]
    summary = summarize(recs, warmup=cfg["warmup"], ttft_slo=slo["ttft_s"], tpot_slo=slo["tpot_s"])
    if summary["n_ok"] == 0:
        raise CellFailed(f"no successful requests; errors={summary['errors']}")

    mpt = summary["mean_prompt_tokens"]
    workload_ok = mpt is None or abs(mpt - wl["input_tokens"]) <= max(2, 0.01 * wl["input_tokens"])

    w0, w1 = min(r.t_send for r in kept), max(r.t_last for r in kept)
    energy = energy_between(power.samples, w0, w1)
    in_win = [s for s in power.samples if w0 <= s.t <= w1]
    n_ok, out_tokens = summary["n_ok"], summary["output_tokens"]
    sampler_ok = len(power.samples) >= 0.5 * SAMPLE_HZ * (t1 - t0)
    stats = _metric_stats(scraper.rows, t0, t1)
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "slo": slo,
        "energy": {
            **energy,
            "window_s": w1 - w0,
            "j_per_output_token": energy["energy_j"] / out_tokens if out_tokens else None,
            "j_per_request": energy["energy_j"] / n_ok,
        },
        "power": throttle_summary(in_win or power.samples),
        "sampler_ok": sampler_ok,
        "workload_ok": workload_ok,
        "server_metrics": stats,
        "counter_deltas": counter_deltas(stats),
    }


async def run_launch(cfg: dict, launch, args, out: Path, man: dict) -> None:
    variant = launch.variant
    todo = [
        (i, ld)
        for i, ld in enumerate(cfg["loads"])
        if not cell_done(cell_path(out, variant, ld, launch.repeat))
    ]
    if not todo:
        print(f"[skip] {variant['name']} r{launch.repeat}: all loads already done", flush=True)
        return
    cmd = server_cmd(args.server, cfg, variant, PORT)
    print(f"[launch] {variant['name']} repeat={launch.repeat}: {' '.join(cmd)}", flush=True)
    async with ServerProcess(
        cmd,
        PORT,
        cfg["startup_seconds"] * 3,
        quiet=args.quiet_server,
        wait_gpu_free=args.gpu == "nvml",
        gpu_index=args.gpu_index,
    ) as srv:
        with PowerCap(variant.get("power_cap_w"), enabled=args.gpu == "nvml", index=args.gpu_index):
            for load_idx, load in todo:
                res = await run_load(cfg, load, load_idx, srv.base, variant, args, launch.repeat)
                res.update(
                    block=cfg["block"],
                    config=cfg["name"],
                    variant=variant["name"],
                    load=load,
                    load_idx=load_idx,
                    repeat=launch.repeat,
                    gpu_usd_per_hr=args.usd_per_hr,
                    power_cap_w=variant.get("power_cap_w"),
                    synthetic=args.server == "mock" or args.gpu == "fake",
                    manifest=man,
                    workload=variant.get("workload", cfg["workload"]),
                    seed=cfg["seed"],
                )
                write_atomic(
                    cell_path(out, variant, load, launch.repeat), json.dumps(res, indent=2)
                )
                s, e = res["summary"], res["energy"]
                jpt = e["j_per_output_token"]
                print(
                    f"  {load_label(load):>6} tput={s['throughput_tok_s']:.0f} tok/s "
                    f"ttft_p99={s['ttft_p99']:.3f}s tpot_p99={s['tpot_p99'] * 1000:.1f}ms "
                    f"J/tok={'n/a' if jpt is None else f'{jpt:.3f}'} "
                    f"slo={s.get('slo_attainment', float('nan')):.2f} fail={s['n_failed']}",
                    flush=True,
                )
                if not res["workload_ok"]:
                    raise CellFailed(
                        f"server saw {s['mean_prompt_tokens']} prompt tokens, config says "
                        f"{res['workload']['input_tokens']}: workload is not what it claims"
                    )
                if s["n_failed"] / s["n_requests"] >= 0.5:
                    raise CellFailed(
                        f"{s['n_failed']}/{s['n_requests']} requests failed: {s['errors']}"
                    )


def _raise_interrupt(signum, _frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--server", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--gpu", choices=["nvml", "fake"], default="nvml")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--usd-per-hr", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print cost estimate and exit")
    ap.add_argument("--quiet-server", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    est = estimate_cost(cfg, args.usd_per_hr)
    print(
        f"[plan] {est['launches']} server launches, ~{est['gpu_hours']:.2f} GPU-h, "
        f"~${est['usd']:.2f}"
    )
    if args.dry_run:
        return 0
    # A dropped ssh session (SIGHUP) or `kill` (SIGTERM) must still run our cleanup.
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = manifest()
    write_atomic(out / "manifest.json", json.dumps(man, indent=2))
    skipped = out / "skipped.jsonl"
    for launch in plan(cfg):
        try:
            asyncio.run(run_launch(cfg, launch, args, out, man))
        except CapUnavailable as e:
            print(f"[skip-launch] {launch.variant['name']} r{launch.repeat}: {e}", flush=True)
            with skipped.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "variant": launch.variant["name"],
                            "repeat": launch.repeat,
                            "reason": str(e),
                        }
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
