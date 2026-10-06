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

import aiohttp

from . import budget
from . import watchdog as wd
from .capacity import find_capacity, update_capacity_file
from .capture import (
    Anchor,
    EventLog,
    allowlisted_env,
    capture_env,
    write_crash_bundle,
    write_gpu_series,
    write_metrics_series,
    write_raw_metrics,
    write_requests,
)
from .config import (
    estimate_cost,
    load_config,
    load_label,
    load_order,
    loads_for,
    per_launch_seconds,
    plan,
    read_capacity,
    resolve_loads,
)
from .loadgen import (
    LoopLagMonitor,
    recovery_time_s,
    run_closed_loop,
    run_open_loop,
    run_phased_open_loop,
    summarize_window,
)
from .manifest import manifest
from .power import CapUnavailable, ClockLock, PowerCap
from .server import AttachedServer, ServerProcess, server_cmd
from .telemetry import (
    FakeBackend,
    MetricsScraper,
    NvmlBackend,
    PowerSampler,
    energy_between,
    sampler_quality,
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


def capacity_artifact(out: Path, variant: dict, repeat: int) -> Path:
    return out / f"capacity__{variant['name']}__r{repeat}.json"


def launch_done(cfg: dict, out: Path, launch) -> bool:
    """A launch is done when every one of its loads has its own artifact for THIS repeat.
    (The shared capacity file must not decide this: it is keyed by variant only, so it
    would mark repeats 2..n done after repeat 1 and the pilot would lose its variance.)"""
    for load in loads_for(cfg, launch.variant):
        if load["mode"] == "capacity_search":
            try:
                json.loads(capacity_artifact(out, launch.variant, launch.repeat).read_text())
            except (OSError, ValueError):
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


PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"


def counter_deltas(stats: dict) -> dict:
    """Deltas of cumulative counters over the window, plus prefix-cache hit rate.

    The hit rate uses EXACT metric names. vLLM also registers
    `vllm:external_prefix_cache_{hits,queries}_total` (always present, zero without a KV
    connector); a substring match picked those and silently dropped the hit rate."""
    d = {k: v["last"] - v["first"] for k, v in stats.items() if k.endswith("_total")}
    if d.get(PREFIX_QUERIES):
        d["prefix_cache_hit_rate"] = d.get(PREFIX_HITS, 0.0) / d[PREFIX_QUERIES]
    return d


def make_backend(args, variant: dict, load_fn):
    if args.gpu == "fake":
        return FakeBackend(load_fn, cap_w=variant.get("power_cap_w"))
    return NvmlBackend(args.gpu_index)


def workload_for(cfg: dict, variant: dict, load: dict | None = None) -> dict:
    """Load-level override > variant override > config default."""
    return (load or {}).get("workload") or variant.get("workload") or cfg["workload"]


def _body_fn(cfg: dict, variant: dict, salt: int, load: dict | None = None):
    wl = workload_for(cfg, variant, load)
    model = variant.get("model", cfg["model"])
    return make_prompt_fn(
        wl["input_tokens"],
        wl["output_tokens"],
        wl.get("prefix_share", 0.0),
        model,
        cfg["seed"],
        salt=salt,
    ), wl["output_tokens"]


async def soak(cfg: dict, base: str, variant: dict) -> list:
    """Unmeasured saturating load: finishes compile/graph warm-up and brings the GPU to a
    steady thermal state, so the first measured load is not systematically cold. Its records
    are returned and persisted (cold-start behaviour is data), never used in any summary."""
    if cfg["soak_s"] <= 0:
        return []
    make_body, expect = _body_fn(cfg, variant, SOAK_SALT)
    recs, _ = await run_closed_loop(
        f"{base}/v1/completions",
        make_body,
        cfg["soak_concurrency"],
        duration_s=cfg["soak_s"],
        expect_out=expect,
        drain_s=cfg["drain_s"],
    )
    return recs


async def measure_idle(cfg: dict, args, variant: dict) -> tuple[float | None, list]:
    """Idle board power, with the raw samples. The first seconds after the soak still carry
    its heat and power, so the first 20% of samples are excluded from the mean."""
    if cfg["idle_s"] <= 0:
        return None, []
    with PowerSampler(make_backend(args, variant, lambda: 0.0), hz=SAMPLE_HZ) as ps:
        await asyncio.sleep(cfg["idle_s"])
    tail = ps.samples[len(ps.samples) // 5 :] or ps.samples
    return sum(s.power_w for s in tail) / len(tail), ps.samples


def _phase_extras(recs, load: dict, t0: float, slo: dict) -> dict:
    """Per-phase summaries and recovery time for an overload-then-recovery cell."""
    out, start = [], t0
    for ph in load["phases"]:
        end = start + ph["duration_s"]
        out.append(
            {
                "qps": ph["qps"],
                "duration_s": ph["duration_s"],
                "summary": summarize_window(recs, start, end, slo["ttft_s"], slo["tpot_s"]),
            }
        )
        start = end
    over_end = t0 + load["phases"][0]["duration_s"]
    rec = recovery_time_s(recs, over_end, start, slo["ttft_s"], slo["tpot_s"])
    return {"kind": "phased", "phases": out, "recovery_time_s": rec}


def art_base(cell_json: Path) -> Path:
    """Artifact prefix for a cell: '<cell>.json' -> '<cell>' (labels may contain dots)."""
    return cell_json.with_name(cell_json.name[: -len(".json")])


def _persist(
    base: Path, args, cfg, variant, load, repeat, recs, power, scraper, marks
) -> list[str]:
    """Write the per-request, GPU and metrics artifacts next to the cell summary."""
    anchor = args.anchor
    header = {"anchor": anchor.to_dict(), "variant": variant["name"], "load": load,
              "repeat": repeat, "slo": cfg["slo"], **{k: v for k, v in marks.items()}}  # fmt: skip
    names = []
    for suffix, fn in (
        (".req.jsonl.gz", lambda p: write_requests(p, recs, header)),
        (".gpu.csv.gz", lambda p: write_gpu_series(p, power.samples, anchor)),
        (".metrics.jsonl.gz", lambda p: write_metrics_series(p, scraper.rows, anchor)),
        (".metrics.raw.txt.gz", lambda p: write_raw_metrics(p, scraper.raw, anchor)),
    ):
        path = base.with_name(base.name + suffix)
        fn(path)
        names.append(path.name)
    return names


async def run_load(
    cfg: dict,
    load: dict,
    load_idx: int,
    base: str,
    variant: dict,
    args,
    repeat: int,
    cell_json: Path | None = None,
) -> dict:
    make_body, expect = _body_fn(cfg, variant, salt_for(repeat, load_idx), load)
    url = f"{base}/v1/completions"
    slo, drain = cfg["slo"], cfg["drain_s"]
    scraper = MetricsScraper(f"{base}/metrics", interval_s=0.5)
    slots = max(1, int(variant.get("slots", 8)))

    def fake_load() -> float:
        run = scraper.rows[-1][1].get("vllm:num_requests_running", 0) if scraper.rows else 0
        return run / slots

    phased = load["mode"] == "phased_open"
    if phased:  # overload cells have no warmup: the soak already warmed the server
        phases = [(p["qps"], p["duration_s"]) for p in load["phases"]]
        warm, total = 0.0, sum(d for _, d in phases)
    else:
        warm, total = cfg["warmup_s"], cfg["warmup_s"] + cfg["measure_s"]
    lag = LoopLagMonitor()
    async with scraper, lag:
        with PowerSampler(make_backend(args, variant, fake_load), hz=SAMPLE_HZ) as power:
            t0 = time.perf_counter()
            ws, we = t0 + warm, t0 + total
            if phased:
                recs, _ = await run_phased_open_loop(
                    url,
                    make_body,
                    phases,
                    seed=cfg["seed"] * 31 + repeat * 7 + load_idx,
                    expect_out=expect,
                    drain_s=drain,
                )
            elif load["mode"] == "closed":
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
    wl = workload_for(cfg, variant, load)
    workload_ok = mpt is None or abs(mpt - wl["input_tokens"]) <= max(2, 0.01 * wl["input_tokens"])
    energy = energy_between(power.samples, ws, we)
    in_win = [s for s in power.samples if ws <= s.t <= we]
    out_tokens, n_done = summary["output_tokens"], summary["n_completed"]
    stats = _metric_stats(scraper.rows, ws, we)  # SAME window as energy and throughput
    marks = {"t0": t0, "ws": ws, "we": we, "t1": t1}
    artifacts = (
        _persist(art_base(cell_json), args, cfg, variant, load, repeat, recs, power, scraper, marks)
        if cell_json is not None
        else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "clock": {
            **args.anchor.to_dict(),
            "wall_ws": args.anchor.wall(ws),
            "wall_we": args.anchor.wall(we),
            **marks,
        },
        "metrics_scrape_errors": scraper.errors,
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
        "sampler": sampler_quality(power.samples, SAMPLE_HZ),
        "client": lag.summary(cfg["max_client_lag_ms"]),
        "workload_ok": workload_ok,
        "server_metrics": stats,
        "counter_deltas": counter_deltas(stats),
        **(_phase_extras(recs, load, t0, slo) if phased else {}),
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
        workload=workload_for(cfg, variant, load),
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


def _check_cell(res: dict, strict: bool = False, metrics_expected: bool = True) -> None:
    """`strict` (real vLLM): missing usage or empty server metrics mean the measurement is
    not what it claims (workload_ok would pass vacuously), so fail instead of proceeding."""
    s = res["summary"]
    if strict:
        if s["mean_prompt_tokens"] is None:
            raise CellFailed("no usage chunk from the server: prompt length is unverified")
        if metrics_expected and "vllm:num_requests_running" not in res["server_metrics"]:
            raise CellFailed("no vLLM /metrics scraped during the window (endpoint or name wrong)")
    if not res["client"]["client_ok"]:
        print(
            f"  WARN client saturated: loop lag p99 {res['client']['loop_lag_p99_ms']:.1f} ms; "
            "cell is flagged invalid (move the client to dedicated cores / reduce load)",
            flush=True,
        )
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
    strict = args.server != "mock"

    async def passes(q: float) -> bool:
        n[0] += 1
        probe_load = {"mode": "open", "qps": round(q, 4)}
        cell_json = cell_path(out, variant, probe_load, repeat)
        try:
            res = await run_load(
                cfg, probe_load, 1000 + n[0], base, variant, args, repeat, cell_json
            )
        except CellFailed as e:
            print(f"  probe q{q:.4g}: {e}", flush=True)
            return False
        _annotate(res, cfg, variant, probe_load, 1000 + n[0], repeat, args, man, idle_w)
        write_atomic(cell_json, json.dumps(res, indent=2))
        _print_cell(probe_load, res)
        _check_cell(res, strict, variant.get("engine") != "sglang")
        s = res["summary"]
        return (
            s["n_failed"] == 0 and s["ttft_p99"] <= slo["ttft_s"] and s["tpot_p99"] <= slo["tpot_s"]
        )

    found = await find_capacity(passes, load["qps_lo"], load["qps_hi"], load["resolution"])
    write_atomic(capacity_artifact(out, variant, repeat), json.dumps(found, indent=2))
    if found["capacity"] is None:
        raise CellFailed(f"{variant['name']} fails the SLO even at qps_lo={load['qps_lo']}")
    update_capacity_file(capacity_path(cfg, out), variant["name"], found["capacity"])
    flag = "" if found["bracketed"] else " (NOT bracketed: raise qps_hi)"
    print(f"  capacity[{variant['name']}] = {found['capacity']:.4g} qps{flag}", flush=True)


async def canary(cfg: dict, base: str, variant: dict, strict: bool, metrics_expected: bool) -> dict:
    """One tiny request after /health and before the soak. /health only says the API server is
    up; this proves a real completion works, usage reports the exact prompt length, and (vLLM)
    /metrics is being served. Cheap insurance against paying for a run that can never be valid."""
    n_in, n_out = 16, 8
    make_body, _ = _body_fn({**cfg, "workload": {"input_tokens": n_in, "output_tokens": n_out}},
                            {**variant, "workload": None}, salt_for(998, 98))  # fmt: skip
    recs, _ = await run_closed_loop(f"{base}/v1/completions", make_body, 1, 1, expect_out=n_out)
    rec = recs[0]
    problems = []
    if not rec.ok:
        problems.append(f"canary request failed: {rec.error}")
    elif strict and rec.prompt_tokens != n_in:
        problems.append(f"usage.prompt_tokens={rec.prompt_tokens}, expected {n_in}")
    if strict and metrics_expected:
        async with aiohttp.ClientSession() as s:
            try:
                async with s.get(f"{base}/metrics") as m:
                    if "vllm:num_requests_running" not in await m.text():
                        problems.append("/metrics has no vllm:num_requests_running")
            except aiohttp.ClientError as e:
                problems.append(f"/metrics unreachable: {e}")
    if problems:
        raise CellFailed("canary: " + "; ".join(problems))
    return {
        "ok": True,
        "ttft_s": rec.ttft,
        "prompt_tokens": rec.prompt_tokens,
        "req_id": rec.req_id,
    }


def launch_dir(out: Path, variant: dict, repeat: int) -> Path:
    d = out / "launches" / f"{variant['name']}__r{repeat}"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def run_launch(cfg: dict, launch, args, out: Path, man: dict) -> None:
    variant = launch.variant
    loads = loads_for(cfg, variant)
    todo = [
        i
        for i in load_order(cfg, launch, len(loads))
        if loads[i]["mode"] == "capacity_search"
        or not cell_done(cell_path(out, variant, loads[i], launch.repeat))
    ]
    if launch_done(cfg, out, launch) or not todo:
        print(f"[skip] {variant['name']} r{launch.repeat}: already done", flush=True)
        return
    real = args.gpu == "nvml"
    ldir = launch_dir(out, variant, launch.repeat)
    log_path = ldir / "server.log"
    ev = args.events
    cmd: list[str] = []
    if args.attach:
        print(f"[attach] {variant['name']} repeat={launch.repeat}: {args.attach}", flush=True)
        server_cm = AttachedServer(args.attach)
    else:
        cmd = server_cmd(args.server, cfg, variant, PORT)
        print(f"[launch] {variant['name']} repeat={launch.repeat}: {' '.join(cmd)}", flush=True)
        server_cm = ServerProcess(
            cmd, PORT, cfg["startup_seconds"] * 3, quiet=args.quiet_server, wait_gpu_free=real,
            gpu_index=args.gpu_index, log_path=log_path,
            on_spawn=args.lease.add_server if args.lease else None,
            on_exit=args.lease.remove_server if args.lease else None,
        )  # fmt: skip
    ev.emit("launch_start", variant=variant["name"], repeat=launch.repeat, cmd=cmd)
    try:
        async with server_cm as srv:
            probe = await canary(cfg, srv.base, variant, args.server != "mock",
                                 variant.get("engine") != "sglang")  # fmt: skip
            _write_launch_json(ldir, args, variant, launch, cmd, srv, probe)
            ev.emit("healthy", variant=variant["name"], repeat=launch.repeat, **probe)
            with (
                PowerCap(variant.get("power_cap_w"), enabled=real, index=args.gpu_index),
                ClockLock(variant.get("clock_lock_mhz"), enabled=real, index=args.gpu_index),
            ):
                ev.emit("soak_start", variant=variant["name"])
                soak_recs = await soak(cfg, srv.base, variant)
                if soak_recs:
                    write_requests(ldir / "soak.req.jsonl.gz", soak_recs,
                                   {"anchor": args.anchor.to_dict(), "kind": "soak"})  # fmt: skip
                idle_w, idle_samples = await measure_idle(cfg, args, variant)
                if idle_samples:
                    write_gpu_series(ldir / "idle.gpu.csv.gz", idle_samples, args.anchor)
                ev.emit("loads_start", variant=variant["name"], idle_power_w=idle_w)
                for i in todo:
                    load = loads[i]
                    if load["mode"] == "capacity_search":
                        await run_capacity_search(
                            cfg, load, srv.base, variant, args, launch.repeat, out, man, idle_w
                        )
                        continue
                    cell_json = cell_path(out, variant, load, launch.repeat)
                    res = await run_load(
                        cfg, load, i, srv.base, variant, args, launch.repeat, cell_json
                    )
                    _annotate(res, cfg, variant, load, i, launch.repeat, args, man, idle_w)
                    write_atomic(cell_json, json.dumps(res, indent=2))
                    ev.emit("cell", variant=variant["name"], load=load_label(load),
                            wall_ws=res["clock"]["wall_ws"], wall_we=res["clock"]["wall_we"])  # fmt: skip
                    _print_cell(load, res)
                    _check_cell(res, args.server != "mock", variant.get("engine") != "sglang")
    except Exception as e:
        # The server is already torn down by ServerProcess; the log and GPU state remain.
        write_crash_bundle(ldir / "crash", log_path, e)
        ev.emit("crash", variant=variant["name"], repeat=launch.repeat, error=repr(e))
        raise
    finally:
        ev.emit("launch_end", variant=variant["name"], repeat=launch.repeat)


def _write_launch_json(
    ldir: Path, args, variant: dict, launch, cmd: list[str], srv, canary_res: dict
) -> None:
    """Launch metadata with wall-clock anchors, so server.log lines align with GPU traces."""
    t_spawn, t_healthy = getattr(srv, "t_spawn", None), getattr(srv, "t_healthy", None)
    meta = {
        "variant": variant["name"], "repeat": launch.repeat, "cmd": cmd,
        "attach": args.attach, "anchor": args.anchor.to_dict(),
        "t_spawn_mono": t_spawn, "t_healthy_mono": t_healthy,
        "wall_spawn": args.anchor.wall(t_spawn) if t_spawn else None,
        "wall_healthy": args.anchor.wall(t_healthy) if t_healthy else None,
        "time_to_healthy_s": (t_healthy - t_spawn) if t_spawn and t_healthy else None,
        "env": allowlisted_env(dict(os.environ)),
        "canary": canary_res,
    }  # fmt: skip
    write_atomic(ldir / "launch.json", json.dumps(meta, indent=2))


def _capacity_refs(cfg: dict) -> set[str]:
    """Every variant name whose capacity some rel load / phase depends on."""
    all_loads = [*cfg["loads"], *(ld for v in cfg["variants"] for ld in v.get("extra_loads", []))]
    refs = {ld["ref"] for ld in all_loads if "ref" in ld}
    refs |= {ph["ref"] for ld in all_loads for ph in ld.get("phases", []) if "ref" in ph}
    return refs


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
    ap.add_argument(
        "--quiet-server",
        action="store_true",
        help="(server output always goes to launches/*/server.log)",
    )
    ap.add_argument(
        "--watchdog",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="detached watchdog that resets GPU clocks/power and kills the server if the "
        "runner dies (default: on for real GPU runs)",
    )
    ap.add_argument(
        "--capture-env",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write _env/ bundle (default: on unless --server mock)",
    )
    ap.add_argument(
        "--attach", help="use this already-running server (http://host:port); no launch"
    )
    ap.add_argument(
        "--assume-capacity",
        type=float,
        help="dry-run only: stand-in capacity (qps) for rel loads when no capacity file exists yet",
    )
    args = ap.parse_args(argv)
    if args.capture_env is None:
        args.capture_env = args.server != "mock"
    if args.watchdog is None:
        args.watchdog = args.gpu == "nvml"
    args.lease = None

    out = Path(args.out)
    cfg = load_config(args.config)
    capacity = read_capacity(cfg.get("capacity_file"))
    if capacity is None and args.dry_run and args.assume_capacity:
        refs = _capacity_refs(cfg)
        capacity = dict.fromkeys(refs, args.assume_capacity)
    cfg = resolve_loads(cfg, capacity)
    real = args.server == "vllm" and args.gpu == "nvml"
    launches = plan(cfg)
    pending = [la for la in launches if not launch_done(cfg, out, la)]
    est = estimate_cost(cfg, args.usd_per_hr, launches=pending)
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
    args.anchor = Anchor.now()
    args.events = EventLog(out / "events.jsonl", args.anchor)
    man = {**manifest(), "anchor": args.anchor.to_dict(), "config": args.config}
    write_atomic(out / "manifest.json", json.dumps(man, indent=2))
    if args.capture_env:  # pip freeze, nvidia-smi -q (ids hashed), lscpu, HF snapshot, ...
        pins = [
            (v.get("model", cfg["model"]), v["server_args"][v["server_args"].index("--revision") + 1])
            for v in cfg["variants"] if "--revision" in v.get("server_args", [])
        ]  # fmt: skip
        capture_env(out, pins)
    args.events.emit("run_start", config=args.config, argv=argv)
    if args.watchdog:
        args.lease = wd.Lease(out / "lease.json", args.gpu_index)
        args.lease.start()
        wd.spawn(out / "lease.json", out / "watchdog.log")
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
        if args.lease:
            args.lease.stop()  # lease removed = clean shutdown; the watchdog exits quietly
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
