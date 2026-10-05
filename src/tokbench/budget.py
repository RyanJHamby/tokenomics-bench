"""Spend ledger and hard cap for GPU rental.

This is a guard, not enforcement: it cannot see the provider's bill. Two entry kinds:
  invoice      what the provider actually charged (user-recorded; authoritative)
  runner_wall  billable hours the runner itself observed (a lower bound: setup, idle
               and download time on the pod are invisible to it)
For a given `session`, an invoice supersedes the runner's lower bound. Also set a spending
limit on the provider's side; that is the only real enforcement.

    python -m tokbench.budget status
    python -m tokbench.budget add --provider runpod --gpu L40S --usd-per-hr 0.99 --hours 2.5 \
        --session pod1 --note "smoke test"
    python -m tokbench.budget check --estimate 12.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

DIR = Path("budget")  # tests point this at a temp dir


class BudgetExceeded(RuntimeError):
    pass


def load_policy(d: Path | None = None) -> dict:
    d = d or DIR
    p = yaml.safe_load((d / "budget.yaml").read_text())
    if p["cap_usd"] <= 0 or not 0 <= p["safety_margin"] < 5:
        raise ValueError("invalid budget policy")
    return p


def read_ledger(d: Path | None = None) -> list[dict]:
    d = d or DIR
    f = d / "ledger.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def append(entry: dict, d: Path | None = None) -> None:
    d = d or DIR
    if entry["usd"] < 0:
        raise ValueError("usd must be >= 0")
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}
    with (d / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def spent(entries: list[dict]) -> float:
    """Total spend. Invoices count in full; a runner_wall entry counts only if no invoice
    covers its session."""
    invoiced = {e["session"] for e in entries if e["kind"] == "invoice" and e.get("session")}
    total = 0.0
    for e in entries:
        if e["kind"] == "invoice" or (
            e["kind"] == "runner_wall" and e.get("session") not in invoiced
        ):
            total += e["usd"]
    return total


def remaining(d: Path | None = None) -> float:
    d = d or DIR
    return load_policy(d)["cap_usd"] - spent(read_ledger(d))


def check(estimate_usd: float, d: Path | None = None) -> dict:
    """Raise BudgetExceeded unless the padded estimate fits in the remaining budget."""
    d = d or DIR
    pol = load_policy(d)
    padded = estimate_usd * (1 + pol["safety_margin"])
    left = remaining(d)
    info = {"cap": pol["cap_usd"], "remaining": left, "estimate": estimate_usd, "padded": padded}
    if padded > left:
        raise BudgetExceeded(
            f"estimate ${estimate_usd:.2f} (+{pol['safety_margin']:.0%} margin = ${padded:.2f}) "
            f"exceeds remaining ${left:.2f} of the ${pol['cap_usd']:.2f} cap"
        )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    a_ = sub.add_parser("add")
    a_.add_argument("--provider", required=True)
    a_.add_argument("--gpu", required=True)
    a_.add_argument("--usd-per-hr", type=float, required=True)
    a_.add_argument("--hours", type=float, required=True)
    a_.add_argument("--session", required=True)
    a_.add_argument("--note", default="")
    c = sub.add_parser("check")
    c.add_argument("--estimate", type=float, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        pol, ent = load_policy(), read_ledger()
        s = spent(ent)
        print(f"cap ${pol['cap_usd']:.2f}  spent ${s:.2f}  remaining ${pol['cap_usd'] - s:.2f}")
    elif a.cmd == "add":
        usd = a.usd_per_hr * a.hours
        append(
            {
                "kind": "invoice",
                "provider": a.provider,
                "gpu": a.gpu,
                "usd_per_hr": a.usd_per_hr,
                "hours": a.hours,
                "usd": usd,
                "session": a.session,
                "note": a.note,
            }
        )
        print(f"recorded ${usd:.2f}; remaining ${remaining():.2f}")
    else:
        try:
            print(json.dumps(check(a.estimate)))
        except BudgetExceeded as e:
            print(f"BUDGET: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
