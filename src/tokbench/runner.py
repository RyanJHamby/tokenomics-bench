"""Run an experiment config: launch a server per (variant, repeat), sweep loads, record
load-generator stats, GPU energy, and server metrics as one JSON file per cell.

    python -m tokbench.runner configs/b1_capacity.yaml --server vllm --gpu nvml \
        --usd-per-hr 1.20 --out results/raw/20261012-0900-b1 [--dry-run]

--server mock --gpu fake runs everything on a laptop; results are flagged synthetic.

Measurement protocol per launch: start server -> (optional) saturating soak so clocks,
temperature and compile/graph state are steady -> idle power baseline -> loads in a seeded
random order, each a fixed `warmup_s` + `measure_s` window. Latency and SLO attainment are
over requests that arrive in the measure window; throughput, goodput and energy are over the
SAME window, so J/token and tokens/s describe identical work.

Resumable: re-running with the same --out skips cells whose JSON exists and parses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

from . import budget
from .capacity import find_capacity, update_capacity_file
from .config import (
    estimate_cost,
    load_config,
    load_label,
    load_order,
    per_launch_seconds,
    plan,
    read_capacity,
    resolve_loads,
)
from .loadgen import run_closed_loop, run_open_loop, summarize_window
from .manifest import manifest
from .power import CapUnavailable, ClockLock, PowerCap
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
SOAK_SALT = salt_for(999, 99)


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


def capacity_path(cfg: dict, out: Path) -> Path:
    return Path(cfg.get("capacity_file") or out / "capacity.json")


def launch_done(cfg: dict, out: Path, launch) -> bool:
    for load in cfg["loads"]:
        if load["mode"] == "capacity_search":
            cap = read_capacity(capacity_path(cfg, out)) or {}
            if launch.variant["name"] not in cap:
                return False
        elif not cell_done(cell_path(out, launch.variant, load, launch.repeat)):
            return False
    return True


def _metric_stats(rows: list[tuple[float, dict]], t0: float, t1: float) -> dict:
    sel = [m for t, m in rows if t0 <= t <= t1]
    out: dict = {}
    for k in sorted({k for m in sel for k in m}):
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


def make_backend(args, variant: dict, load_fn):
    if args.gpu == "fake":
        return FakeBackend(load_fn, cap_w=variant.get("power_cap_w"))
    return NvmlBackend(args.gpu_index)


def _body_fn(cfg: dict, variant: dict, salt: int):
    wl = variant.get("workload", cfg["workload"])
    model = variant.get("model", cfg["model"])
    return make_prompt_fn(
        wl["input_tokens"],
        wl["output_tokens"],
        wl.get("prefix_share", 0.0),
        model,
        cfg["seed"],
        salt=salt,
    ), wl["output_tokens"]


async def soak(cfg: dict, base: str, variant: dict) -> None:
    """Unrecorded saturating load: finishes compile/graph warm-up and brings the GPU to a
    steady thermal state, so the first measured load is not systematically cold."""
    if cfg["soak_s"] <= 0:
        return
    make_body, expect = _body_fn(cfg, variant, SOAK_SALT)
    await run_closed_loop(
        f"{base}/v1/completions",
        make_body,
        cfg["soak_concurrency"],
        duration_s=cfg["soak_s"],
        expect_out=expect,
        drain_s=cfg["drain_s"],
    )


async def measure_idle(cfg: dict, args, variant: dict) -> float | None:
    if cfg["idle_s"] <= 0:
        return None
    with PowerSampler(make_backend(args, variant, lambda: 0.0), hz=SAMPLE_HZ) as ps:
        await asyncio.sleep(cfg["idle_s"])
    return sum(s.power_w for s in ps.samples) / len(ps.samples)


async def run_load(
    cfg: dict, load: dict, load_idx: int, base: str, variant: dict, args, repeat: int
) -> dict:
    make_body, expect = _body_fn(cfg, variant, salt_for(repeat, load_idx))
    url = f"{base}/v1/completions"
    slo, drain = cfg["slo"], cfg["drain_s"]
    scraper = MetricsScraper(f"{base}/metrics", interval_s=0.5)
    slots = max(1, int(variant.get("slots", 8)))

    def fake_load() -> float:
        run = scraper.rows[-1][1].get("vllm:num_requests_running", 0) if scraper.rows else 0
        return run / slots

    total = cfg["warmup_s"] + cfg["measure_s"]
    async with scraper:
        with PowerSampler(make_backend(args, variant, fake_load), hz=SAMPLE_HZ) as power:
            t0 = time.perf_counter()
            ws, we = t0 + cfg["warmup_s"], t0 + total
            if load["mode"] == "closed":
                recs, _ = await run_closed_loop(
                    url,
                    make_body,
                    load["concurrency"],
                    duration_s=total,
                    expect_out=expect,
                    drain_s=drain,
                )
            else:
                recs, _ = await run_open_loop(
                    url,
                    make_body,
                    load["qps"],
                    duration_s=total,
                    drain_s=drain,
                    seed=cfg["seed"] * 31 + repeat * 7 + load_idx,
                    expect_out=expect,
                )
            t1 = time.perf_counter()
    summary = summarize_window(recs, ws, we, slo["ttft_s"], slo["tpot_s"])
    if summary["n_ok"] == 0:
        raise CellFailed(f"no successful requests; errors={summary['errors']}")

    mpt = summary["mean_prompt_tokens"]
    wl = variant.get("workload", cfg["workload"])
    workload_ok = mpt is None or abs(mpt - wl["input_tokens"]) <= max(2, 0.01 * wl["input_tokens"])
    energy = energy_between(power.samples, ws, we)
    in_win = [s for s in power.samples if ws <= s.t <= we]
    out_tokens, n_done = summary["output_tokens"], summary["n_completed"]
    stats = _metric_stats(scraper.rows, t0, t1)
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "slo": slo,
        "warmup_s": cfg["warmup_s"],
        "measure_s": cfg["measure_s"],
        "energy": {
            **energy,
            "window_s": we - ws,
            "j_per_output_token": energy["energy_j"] / out_tokens if out_tokens else None,
            "j_per_request": energy["energy_j"] / n_done if n_done else None,
        },
        "power": throttle_summary(in_win or power.samples),
        "sampler_ok": len(power.samples) >= 0.5 * SAMPLE_HZ * (t1 - t0),
        "workload_ok": workload_ok,
        "server_metrics": stats,
        "counter_deltas": counter_deltas(stats),
    }


def _annotate(res: dict, cfg, variant, load, load_idx, repeat, args, man, idle_w) -> dict:
    res.update(
        block=cfg["block"],
        config=cfg["name"],
        variant=variant["name"],
        load=load,
        load_idx=load_idx,
        repeat=repeat,
        gpu_usd_per_hr=args.usd_per_hr,
        power_cap_w=variant.get("power_cap_w"),
        clock_lock_mhz=variant.get("clock_lock_mhz"),
        idle_power_w=idle_w,
        soak_s=cfg["soak_s"],
        synthetic=args.server == "mock" or args.gpu == "fake",
        manifest=man,
        workload=variant.get("workload", cfg["workload"]),
        seed=cfg["seed"],
    )
    return res


def _print_cell(load: dict, res: dict) -> None:
    s, e = res["summary"], res["energy"]
    jpt = e["j_per_output_token"]
    print(
        f"  {load_label(load):>8} tput={s['throughput_tok_s']:.0f} tok/s "
        f"goodput={s['goodput_tok_s']:.0f} ttft_p99={s['ttft_p99']:.3f}s "
        f"tpot_p99={s['tpot_p99'] * 1000:.1f}ms J/tok={'n/a' if jpt is None else f'{jpt:.3f}'} "
        f"slo={s['slo_attainment']:.2f} fail={s['n_failed']}",
        flush=True,
    )


def _check_cell(res: dict) -> None:
    s = res["summary"]
    if not res["workload_ok"]:
        raise CellFailed(
            f"server saw {s['mean_prompt_tokens']} prompt tokens, config says "
            f"{res['workload']['input_tokens']}: workload is not what it claims"
        )
    # 'incomplete' is the expected signature of deliberate overload; real errors are not.
    hard = [e for e in s["errors"] if e != "incomplete"]
    if hard and s["n_failed"] / s["n_requests"] >= 0.5:
        raise CellFailed(f"{s['n_failed']}/{s['n_requests']} requests failed: {hard}")


async def run_capacity_search(cfg, load, base, variant, args, repeat, out, man, idle_w) -> None:
    slo, n = cfg["slo"], [0]

    async def passes(q: float) -> bool:
        n[0] += 1
        probe_load = {"mode": "open", "qps": round(q, 4)}
        try:
            res = await run_load(cfg, probe_load, 1000 + n[0], base, variant, args, repeat)
        except CellFailed as e:
            print(f"  probe q{q:.4g}: {e}", flush=True)
            return False
        _annotate(res, cfg, variant, probe_load, 1000 + n[0], repeat, args, man, idle_w)
        write_atomic(cell_path(out, variant, probe_load, repeat), json.dumps(res, indent=2))
        _print_cell(probe_load, res)
        _check_cell(res)
        s = res["summary"]
        return (
            s["n_failed"] == 0 and s["ttft_p99"] <= slo["ttft_s"] and s["tpot_p99"] <= slo["tpot_s"]
        )

    found = await find_capacity(passes, load["qps_lo"], load["qps_hi"], load["resolution"])
    write_atomic(out / f"capacity__{variant['name']}__r{repeat}.json", json.dumps(found, indent=2))
    if found["capacity"] is None:
        raise CellFailed(f"{variant['name']} fails the SLO even at qps_lo={load['qps_lo']}")
    update_capacity_file(capacity_path(cfg, out), variant["name"], found["capacity"])
    flag = "" if found["bracketed"] else " (NOT bracketed: raise qps_hi)"
    print(f"  capacity[{variant['name']}] = {found['capacity']:.4g} qps{flag}", flush=True)


async def run_launch(cfg: dict, launch, args, out: Path, man: dict) -> None:
    variant = launch.variant
    todo = [
        i
        for i in load_order(cfg, launch, len(cfg["loads"]))
        if cfg["loads"][i]["mode"] == "capacity_search"
        or not cell_done(cell_path(out, variant, cfg["loads"][i], launch.repeat))
    ]
    if launch_done(cfg, out, launch) or not todo:
        print(f"[skip] {variant['name']} r{launch.repeat}: already done", flush=True)
        return
    cmd = server_cmd(args.server, cfg, variant, PORT)
    print(f"[launch] {variant['name']} repeat={launch.repeat}: {' '.join(cmd)}", flush=True)
    real = args.gpu == "nvml"
    async with ServerProcess(
        cmd,
        PORT,
        cfg["startup_seconds"] * 3,
        quiet=args.quiet_server,
        wait_gpu_free=real,
        gpu_index=args.gpu_index,
    ) as srv:
        with (
            PowerCap(variant.get("power_cap_w"), enabled=real, index=args.gpu_index),
            ClockLock(variant.get("clock_lock_mhz"), enabled=real, index=args.gpu_index),
        ):
            await soak(cfg, srv.base, variant)
            idle_w = await measure_idle(cfg, args, variant)
            for i in todo:
                load = cfg["loads"][i]
                if load["mode"] == "capacity_search":
                    await run_capacity_search(
                        cfg, load, srv.base, variant, args, launch.repeat, out, man, idle_w
                    )
                    continue
                res = await run_load(cfg, load, i, srv.base, variant, args, launch.repeat)
                _annotate(res, cfg, variant, load, i, launch.repeat, args, man, idle_w)
                write_atomic(
                    cell_path(out, variant, load, launch.repeat), json.dumps(res, indent=2)
                )
                _print_cell(load, res)
                _check_cell(res)


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

    out = Path(args.out)
    cfg = load_config(args.config)
    cfg = resolve_loads(cfg, read_capacity(cfg.get("capacity_file")))
    real = args.server == "vllm" and args.gpu == "nvml"
    launches = plan(cfg)
    pending = [la for la in launches if not launch_done(cfg, out, la)]
    est = estimate_cost(cfg, args.usd_per_hr, launches=len(pending))
    print(
        f"[plan] {len(pending)}/{len(launches)} server launches pending, "
        f"~{est['gpu_hours']:.2f} GPU-h, ~${est['usd']:.2f}"
    )
    if real:
        try:
            info = budget.check(est["usd"])
        except budget.BudgetExceeded as e:
            print(f"[budget] REFUSING TO RUN: {e}", flush=True)
            return 3
        print(f"[budget] ok: ${info['remaining']:.2f} of ${info['cap']:.2f} remaining", flush=True)
    if args.dry_run:
        return 0
    # A dropped ssh session (SIGHUP) or `kill` (SIGTERM) must still run our cleanup.
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)

    out.mkdir(parents=True, exist_ok=True)
    man = manifest()
    write_atomic(out / "manifest.json", json.dumps(man, indent=2))
    skipped = out / "skipped.jsonl"
    started = time.monotonic()
    pad = budget.load_policy()["safety_margin"] if real else 0.0
    left_at_start = budget.remaining() if real else float("inf")
    per_launch_usd = per_launch_seconds(cfg) / 3600 * args.usd_per_hr
    try:
        for launch in pending:
            used = (time.monotonic() - started) / 3600 * args.usd_per_hr
            if real and used + per_launch_usd * (1 + pad) > left_at_start:
                print(
                    f"[budget] STOP: ${used:.2f} used this session; the next launch would "
                    "exceed the cap. Remaining launches not run.",
                    flush=True,
                )
                return 4
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
    finally:
        if real:
            hours = (time.monotonic() - started) / 3600
            budget.append(
                {
                    "kind": "runner_wall",
                    "usd": hours * args.usd_per_hr,
                    "hours": hours,
                    "usd_per_hr": args.usd_per_hr,
                    "session": out.name,
                    "note": "lower bound: runner wall time only",
                }
            )
            print(
                "[budget] REMINDER: stop the pod now, then record the invoice with "
                "`python -m tokbench.budget add ...`",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
