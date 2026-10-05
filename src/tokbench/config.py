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
MODES = ("closed", "open", "capacity_search")


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
    if mode != "closed" and "concurrency" in load:
        raise ValueError(f"concurrency only applies to closed loads: {load}")


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


def resolve_loads(cfg: dict, capacity: dict[str, float] | None) -> dict:
    """Turn `rel` loads (a multiple of a reference variant's measured capacity) into
    absolute qps. Absolute qps is shared by all variants, so they are compared at equal
    offered load."""
    out = []
    for load in cfg["loads"]:
        if load["mode"] == "open" and "rel" in load:
            if not capacity or load["ref"] not in capacity:
                raise ValueError(
                    f"load {load} needs capacity of {load['ref']!r}; run the capacity block first"
                )
            out.append(
                {
                    "mode": "open",
                    "qps": round(load["rel"] * capacity[load["ref"]], 4),
                    "rel": load["rel"],
                    "ref": load["ref"],
                }
            )
        else:
            out.append(load)
    return {**cfg, "loads": out}


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
        return f"c{load['concurrency']}"
    return f"q{load['qps']:.5g}"


def n_probes(load: dict) -> int:
    ratio = load["qps_hi"] / load["qps_lo"]
    return math.ceil(math.log2(max(ratio, 1.0))) + math.ceil(math.log2(1 / load["resolution"])) + 2


def est_load_seconds(cfg: dict, load: dict) -> float:
    one = cfg["warmup_s"] + cfg["measure_s"] + 15  # + drain/settle allowance
    return one * (n_probes(load) if load["mode"] == "capacity_search" else 1)


def per_launch_seconds(cfg: dict) -> float:
    return (
        cfg["startup_seconds"]
        + cfg["soak_s"]
        + cfg["idle_s"]
        + sum(est_load_seconds(cfg, ld) for ld in cfg["loads"])
    )


def estimate_cost(cfg: dict, usd_per_hr: float, launches: int | None = None) -> dict:
    """`launches` overrides the planned count (e.g. only the not-yet-done ones on resume)."""
    launches = len(plan(cfg)) if launches is None else launches
    hours = launches * per_launch_seconds(cfg) / 3600
    return {"launches": launches, "gpu_hours": hours, "usd": hours * usd_per_hr}
