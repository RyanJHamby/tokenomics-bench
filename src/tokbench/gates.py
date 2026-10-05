"""Quality-gate CLI. Run on the GPU box, one server at a time.

    python -m tokbench.gates collect configs/b2_prefix_cache.yaml prefix-on-s90 out/on.json
    python -m tokbench.gates collect configs/b2_prefix_cache.yaml prefix-on-s90 out/on2.json
    python -m tokbench.gates collect configs/b2_prefix_cache.yaml prefix-off-s90 out/off.json
    python -m tokbench.gates compare out/on.json out/off.json --noise out/on.json out/on2.json
    python -m tokbench.gates gsm8k configs/b3_quantization.yaml fp8 data/gsm8k_test.jsonl --n 200

Why a noise floor: greedy decoding is not bit-reproducible across runs on a GPU (batching
and kernel selection change numerics), so "identical outputs" must be judged against how
often the SAME config disagrees with itself. Pre-registered in docs/PREREG.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp

from .config import load_config
from .quality import gsm8k_accuracy, load_gsm8k, paired_accuracy
from .runner import PORT
from .server import ServerProcess, server_cmd
from .workloads import make_prompt_fn


def n_mismatches(a: list[str], b: list[str]) -> int:
    if len(a) != len(b):
        raise ValueError("output lists differ in length")
    return sum(x != y for x, y in zip(a, b, strict=True))


def gate_verdict(candidate_mismatches: int, noise_mismatches: int) -> str:
    """Pass if the cross-config disagreement is no worse than the same-config noise floor."""
    return "pass" if candidate_mismatches <= noise_mismatches else "fail"


def _variant(cfg: dict, name: str) -> dict:
    return next(v for v in cfg["variants"] if v["name"] == name)


async def _with_server(cfg: dict, variant: dict, fn):
    cmd = server_cmd("vllm", cfg, variant, PORT)
    async with ServerProcess(cmd, PORT, cfg["startup_seconds"] * 3, wait_gpu_free=True) as srv:
        return await fn(srv.base)


async def collect(cfg_path: str, variant_name: str, out: str, n: int) -> None:
    cfg = load_config(cfg_path)
    v = _variant(cfg, variant_name)
    wl = v.get("workload", cfg["workload"])
    mk = make_prompt_fn(
        wl["input_tokens"],
        wl["output_tokens"],
        wl.get("prefix_share", 0.0),
        v.get("model", cfg["model"]),
        cfg["seed"],
    )

    async def run(base: str) -> list[str]:
        url = f"{base}/v1/completions"
        outs = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
            for i in range(n):  # sequential: batch composition is the same every run
                async with s.post(url, json={**mk(i), "stream": False}) as r:
                    r.raise_for_status()
                    outs.append((await r.json())["choices"][0]["message"]["content"])
        return outs

    outs = await _with_server(cfg, v, run)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"variant": variant_name, "outputs": outs}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("config"), c.add_argument("variant"), c.add_argument("out")
    c.add_argument("--n", type=int, default=50)
    k = sub.add_parser("compare")
    k.add_argument("a"), k.add_argument("b")
    k.add_argument("--noise", nargs=2, required=True, metavar=("RUN1", "RUN2"))
    g = sub.add_parser("gsm8k")
    g.add_argument("config"), g.add_argument("variant"), g.add_argument("data")
    g.add_argument("out")
    g.add_argument("--n", type=int, default=1319)
    q = sub.add_parser("gsm8k-compare")
    q.add_argument("base"), q.add_argument("cand")
    q.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args(argv)

    if a.cmd == "collect":
        asyncio.run(collect(a.config, a.variant, a.out, a.n))
    elif a.cmd == "compare":
        load = lambda p: json.loads(Path(p).read_text())["outputs"]
        cand = n_mismatches(load(a.a), load(a.b))
        noise = n_mismatches(load(a.noise[0]), load(a.noise[1]))
        res = {
            "n": len(load(a.a)),
            "cross_config_mismatches": cand,
            "noise_floor_mismatches": noise,
            "verdict": gate_verdict(cand, noise),
        }
        print(json.dumps(res, indent=2))
        return 0 if res["verdict"] == "pass" else 1
    elif a.cmd == "gsm8k-compare":
        base, cand = (json.loads(Path(x).read_text()) for x in (a.base, a.cand))
        if base["items"] != cand["items"]:
            raise SystemExit("baseline and candidate were scored on different items")
        res = paired_accuracy(base["correct"], cand["correct"], a.margin)
        print(json.dumps({"base": base["variant"], "cand": cand["variant"], **res}, indent=2))
        return 0 if res["verdict"] in ("equivalent", "non_inferior") else 1
    else:
        cfg = load_config(a.config)
        v = _variant(cfg, a.variant)
        model = v.get("model", cfg["model"])
        items = load_gsm8k(a.data, a.n, cfg["seed"])
        res = asyncio.run(
            _with_server(
                cfg, v, lambda base: gsm8k_accuracy(f"{base}/v1/chat/completions", model, items)
            )
        )
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(
            json.dumps({"variant": a.variant, "items": [it["question"] for it in items], **res})
        )
        print(
            json.dumps(
                {"variant": a.variant, **{k: v for k, v in res.items() if k != "correct"}}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
