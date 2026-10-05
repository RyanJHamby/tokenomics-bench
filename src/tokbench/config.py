"""Experiment config loading, validation, and cost estimation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED = ("name", "block", "model", "workload", "variants", "loads", "repeats", "n_requests")


@dataclass(frozen=True)
class Launch:
    """One server start: a variant at a repeat index. Runs every load."""

    variant: dict
    repeat: int


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["repeats"] < 1 or cfg["n_requests"] < 1:
        raise ValueError("repeats and n_requests must be >= 1")
    names = [v["name"] for v in cfg["variants"]]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    for load in cfg["loads"]:
        if load.get("mode") not in ("closed", "open"):
            raise ValueError(f"load mode must be closed|open: {load}")
        if (load["mode"] == "closed") != ("concurrency" in load):
            raise ValueError(f"closed loads need concurrency, open loads need qps: {load}")
    cfg.setdefault("warmup", 0)
    cfg.setdefault("seed", 0)
    cfg.setdefault("est_request_s", 4.0)  # rough per-request service time
    cfg.setdefault("startup_seconds", 180)
    return cfg


def plan(cfg: dict) -> list[Launch]:
    """Launch order: per repeat, variants in a seeded shuffled order, so slow drift
    (thermals, neighbours) hits all variants roughly equally."""
    launches = []
    for r in range(cfg["repeats"]):
        order = list(cfg["variants"])
        random.Random(cfg["seed"] * 7919 + r).shuffle(order)
        launches += [Launch(v, r) for v in order]
    return launches


def load_label(load: dict) -> str:
    return f"c{load['concurrency']}" if load["mode"] == "closed" else f"q{load['qps']}"


def est_load_seconds(cfg: dict, load: dict) -> float:
    """Open loop is paced by arrivals; closed loop by service time / concurrency.
    Plus a 15 s drain/settle allowance. `est_seconds_per_load` overrides if set."""
    if "est_seconds_per_load" in cfg:
        return cfg["est_seconds_per_load"]
    if load["mode"] == "open":
        return cfg["n_requests"] / load["qps"] + 15
    return cfg["n_requests"] * cfg["est_request_s"] / load["concurrency"] + 15


def estimate_cost(cfg: dict, usd_per_hr: float) -> dict:
    launches = len(plan(cfg))
    per_launch = cfg["startup_seconds"] + sum(est_load_seconds(cfg, ld) for ld in cfg["loads"])
    seconds = launches * per_launch
    hours = seconds / 3600
    return {"launches": launches, "gpu_hours": hours, "usd": hours * usd_per_hr}
