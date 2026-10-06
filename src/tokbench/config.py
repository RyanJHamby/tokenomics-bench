"""Experiment config loading, validation, planning, and cost estimation."""

from __future__ import annotations

import json
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED = (
    "name",
    "block",
    "model",
    "workload",
    "variants",
    "loads",
    "repeats",
    "measure_s",
    "warmup_s",
)
MODES = ("closed", "open", "capacity_search", "phased_open")


@dataclass(frozen=True)
class Launch:
    """One server start: a variant at a repeat index. Runs every load."""

    variant: dict
    repeat: int


def _check_load(load: dict) -> None:
    mode = load.get("mode")
    if mode not in MODES:
        raise ValueError(f"load mode must be one of {MODES}: {load}")
    if mode == "closed" and "concurrency" not in load:
        raise ValueError(f"closed loads need concurrency: {load}")
    if mode == "open" and ("qps" in load) == ("rel" in load):
        raise ValueError(f"open loads need exactly one of qps or rel(+ref): {load}")
    if mode == "open" and "rel" in load and "ref" not in load:
        raise ValueError(f"rel loads need ref (the variant whose capacity is the unit): {load}")
    if mode == "capacity_search" and not {"qps_lo", "qps_hi", "resolution"} <= load.keys():
        raise ValueError(f"capacity_search needs qps_lo, qps_hi, resolution: {load}")
    if mode == "phased_open":
        phases = load.get("phases")
        if not phases or len(phases) < 2:
            raise ValueError(f"phased_open needs >= 2 phases: {load}")
        for ph in phases:
            if "duration_s" not in ph or ph["duration_s"] <= 0 or (("qps" in ph) == ("rel" in ph)):
                raise ValueError(
                    f"each phase needs duration_s and exactly one of qps / rel+ref: {ph}"
                )
            if "rel" in ph and "ref" not in ph:
                raise ValueError(f"rel phases need ref: {ph}")
    if mode not in ("closed", "phased_open") and "concurrency" in load:
        raise ValueError(f"concurrency only applies to closed loads: {load}")
    if "workload" in load:
        wl = load["workload"]
        if mode not in ("closed", "open") or not {"input_tokens", "output_tokens"} <= wl.keys():
            raise ValueError(
                f"load workload override needs input/output tokens on closed/open: {load}"
            )


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["repeats"] < 1 or cfg["measure_s"] <= 0 or cfg["warmup_s"] < 0:
        raise ValueError("repeats>=1, measure_s>0, warmup_s>=0 required")
    names = [v["name"] for v in cfg["variants"]]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    for v in cfg["variants"]:
        if "power_cap_w" in v and "clock_lock_mhz" in v:
            raise ValueError(f"{v['name']}: cap and lock are separate arms, not combined")
    for load in cfg["loads"]:
        _check_load(load)
    for v in cfg["variants"]:
        for load in v.get("extra_loads", []):
            _check_load(load)
    cfg.setdefault("seed", 0)
    cfg.setdefault("slo", {"ttft_s": 2.0, "tpot_s": 0.1})
    cfg.setdefault("drain_s", 60.0)
    cfg.setdefault("soak_s", 0.0)  # saturating load before measuring (thermal + compile warm)
    cfg.setdefault("soak_concurrency", 32)
    cfg.setdefault("idle_s", 0.0)  # idle power baseline measured per launch
    cfg.setdefault("randomize_load_order", True)
    cfg.setdefault("max_client_lag_ms", 10.0)  # p99 event-loop lag above this => cell invalid
    cfg.setdefault("startup_seconds", 180)
    return cfg


def _resolve_one(load: dict, capacity: dict[str, float] | None) -> dict:
    def need(ref: str) -> float:
        if not capacity or ref not in capacity:
            raise ValueError(f"load {load} needs capacity of {ref!r}; run the capacity block first")
        return capacity[ref]

    if load["mode"] == "open" and "rel" in load:
        return {**load, "qps": round(load["rel"] * need(load["ref"]), 4)}
    if load["mode"] == "phased_open":
        phases = [
            {**ph, "qps": round(ph["rel"] * need(ph["ref"]), 4)} if "rel" in ph else ph
            for ph in load["phases"]
        ]
        return {**load, "phases": phases}
    return load


def resolve_loads(cfg: dict, capacity: dict[str, float] | None) -> dict:
    """Turn `rel` loads (a multiple of a reference variant's measured capacity) into absolute
    qps. Absolute qps is shared by all variants, so they are compared at equal offered load.
    Applies to shared loads, per-variant extra loads, and the phases of phased loads."""
    variants = [
        {**v, "extra_loads": [_resolve_one(ld, capacity) for ld in v["extra_loads"]]}
        if v.get("extra_loads")
        else v
        for v in cfg["variants"]
    ]
    return {
        **cfg,
        "loads": [_resolve_one(ld, capacity) for ld in cfg["loads"]],
        "variants": variants,
    }


def loads_for(cfg: dict, variant: dict) -> list[dict]:
    """The loads a launch of `variant` runs: the shared loads plus its own extra loads."""
    return [*cfg["loads"], *variant.get("extra_loads", [])]


def read_capacity(path: str | Path | None) -> dict[str, float] | None:
    if path is None or not Path(path).exists():
        return None
    return {k: float(v) for k, v in json.loads(Path(path).read_text()).items()}


def plan(cfg: dict) -> list[Launch]:
    """Launch order: per repeat, variants in a seeded shuffled order, so slow drift
    (thermals, neighbours) hits all variants roughly equally."""
    launches = []
    for r in range(cfg["repeats"]):
        order = list(cfg["variants"])
        random.Random(cfg["seed"] * 7919 + r).shuffle(order)
        launches += [Launch(v, r) for v in order]
    return launches


def load_order(cfg: dict, launch: Launch, n: int) -> list[int]:
    """Order in which a launch runs its loads: seeded shuffle (default), so a fixed
    cool-to-hot ascent cannot confound load level with thermal state."""
    idx = list(range(n))
    if cfg["randomize_load_order"]:
        seed = cfg["seed"] * 101 + launch.repeat * 13 + zlib.crc32(launch.variant["name"].encode())
        random.Random(seed).shuffle(idx)
    return idx


def load_label(load: dict) -> str:
    if load["mode"] == "closed":
        base = f"c{load['concurrency']}"
    elif load["mode"] == "phased_open":
        base = "phased"
    else:
        base = f"q{load['qps']:.5g}"
    wl = load.get("workload")
    return f"{base}-i{wl['input_tokens']}o{wl['output_tokens']}" if wl else base


def n_probes(load: dict) -> int:
    ratio = load["qps_hi"] / load["qps_lo"]
    return math.ceil(math.log2(max(ratio, 1.0))) + math.ceil(math.log2(1 / load["resolution"])) + 2


def est_load_seconds(cfg: dict, load: dict) -> float:
    one = cfg["warmup_s"] + cfg["measure_s"] + 15  # + drain/settle allowance
    if load["mode"] == "phased_open":
        return sum(p["duration_s"] for p in load["phases"]) + cfg["drain_s"] / 2
    return one * (n_probes(load) if load["mode"] == "capacity_search" else 1)


def per_launch_seconds(cfg: dict, variant: dict | None = None) -> float:
    """Seconds for one launch. Without `variant`, the most expensive variant (conservative)."""
    variants = [variant] if variant else cfg["variants"]
    return max(
        cfg["startup_seconds"]
        + cfg["soak_s"]
        + cfg["idle_s"]
        + sum(est_load_seconds(cfg, ld) for ld in loads_for(cfg, v))
        for v in variants
    )


def estimate_cost(cfg: dict, usd_per_hr: float, launches: int | list | None = None) -> dict:
    """`launches` may be a count (priced at the average launch) or the pending Launch list
    (priced per variant, so extra loads on some arms are costed correctly)."""
    if launches is None:
        launches = plan(cfg)
    if isinstance(launches, int):
        avg = sum(per_launch_seconds(cfg, v) for v in cfg["variants"]) / len(cfg["variants"])
        seconds, n = launches * avg, launches
    else:
        seconds, n = sum(per_launch_seconds(cfg, la.variant) for la in launches), len(launches)
    hours = seconds / 3600
    return {"launches": n, "gpu_hours": hours, "usd": hours * usd_per_hr}
