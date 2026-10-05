"""The analyzer must reproduce known, planted effects (and refuse to find ones that are not
there). Cells are fabricated with controlled values; no GPU, no mock server."""

import json
from argparse import Namespace

from tokbench import analyze
from tokbench.config import load_label

NOISE = (0.998, 1.0, 1.002)  # three launches, +/-0.2% launch noise


def cell(
    variant,
    label,
    repeat,
    *,
    jpt=1.0,
    good=1000.0,
    tpot=0.03,
    ttft=0.07,
    p95=250.0,
    lock=None,
    cap=None,
    throttle=False,
    errors=(),
    max_sm=2520.0,
):
    mode = (
        {"mode": "closed", "concurrency": int(label[1:])}
        if label[0] == "c"
        else {"mode": "open", "qps": float(label[1:])}
    )
    k = NOISE[repeat]
    return {
        "schema_version": 2,
        "variant": variant,
        "load": mode,
        "repeat": repeat,
        "synthetic": True,
        "summary": {
            "goodput_tok_s": good * k,
            "throughput_tok_s": good * k,
            "tpot_p50": tpot * k,
            "ttft_p50": ttft * k,
            "ttft_p99": ttft,
            "tpot_p99": tpot,
            "itl_p99": tpot,
            "n_failed": 0,
            "errors": list(errors),
        },
        "energy": {"j_per_output_token": jpt * k},
        "power": {"power_w_p95": p95, "bad_throttle_seen": throttle},
        "workload_ok": True,
        "sampler_ok": True,
        "client": {"client_ok": True},
        "gpu_usd_per_hr": 1.0,
        "clock_lock_mhz": lock,
        "power_cap_w": cap,
        "manifest": {"gpu_info": {"clocks.max.sm": max_sm}},
    }


def write(d, cells):
    d.mkdir(parents=True, exist_ok=True)
    for c in cells:
        (d / f"{c['variant']}__{load_label(c['load'])}__r{c['repeat']}.json").write_text(
            json.dumps(c)
        )


def args(**kw):
    base = {
        "b1": None,
        "b2": None,
        "b3a_fp8": None,
        "b3b": None,
        "b4": None,
        "gsm8k_dir": None,
        "hw": "L40S",
    }
    return Namespace(**{**base, **kw})


def b2_cells(lock_effect=0.75, lock_good=0.99, throttle_repeat=None):
    cs = []
    for r in range(3):
        for label, jb, gb in (("c64", 1.0, 1000.0), ("q5", 1.2, 500.0)):
            cs.append(cell("uncapped", label, r, jpt=jb, good=gb))
            cs.append(
                cell(
                    "lock1770",
                    label,
                    r,
                    jpt=jb * lock_effect,
                    good=gb * lock_good,
                    lock=1770,
                    throttle=(r == throttle_repeat),
                )
            )
            cs.append(cell("lock1380", label, r, jpt=jb * 0.7, good=gb * 0.6, lock=1380))
            cs.append(cell("cap200w", label, r, jpt=jb, good=gb, cap=200))
            cs.append(cell("cap150w", label, r, jpt=jb * 0.97, good=gb * 0.9, cap=150))
    return cs


def test_planted_clock_lock_effect_confirms_p1_p2_and_no_effect_cap_confirms_p3(tmp_path):
    write(tmp_path / "b2", b2_cells())
    res = analyze.analyze(args(b2=str(tmp_path / "b2")))
    t = {x["id"]: x for x in res["tests"]}
    assert res["b2"]["lock70"] == "lock1770" and res["b2"]["cap80"] == "cap200w"
    for tid in ("P1[saturation]", "P1[0.7xcapacity]"):
        assert t[tid]["verdict"] == "superior_lower" and t[tid]["supported"]
        assert t[tid]["supported_after_holm"] is True
    for tid in ("P2[saturation]", "P3j[saturation]", "P3g[0.7xcapacity]"):
        assert t[tid]["verdict"] == "equivalent" and t[tid]["supported"]
    d = res["b2"]["derived_finding"]
    assert d["energy_optimal"] == "lock1380" and d["cost_optimal"] == "uncapped" and d["holds"]


def test_no_planted_effect_is_not_called_an_effect(tmp_path):
    write(tmp_path / "b2", b2_cells(lock_effect=1.0, lock_good=1.0))
    res = analyze.analyze(args(b2=str(tmp_path / "b2")))
    t = {x["id"]: x for x in res["tests"]}
    assert (
        t["P1[saturation]"]["supported"] is False and t["P1[saturation]"]["verdict"] == "equivalent"
    )


def test_invalid_cells_are_excluded_and_listed_not_hidden(tmp_path):
    write(tmp_path / "b2", b2_cells(throttle_repeat=1))
    res = analyze.analyze(args(b2=str(tmp_path / "b2")))
    bad = res["excluded"]["b2"]
    assert bad and all(x["variant"] == "lock1770" and x["repeat"] == 1 for x in bad)
    assert "thermal_or_hw_throttle" in bad[0]["reasons"]
    t = {x["id"]: x for x in res["tests"]}
    assert t["P1[saturation]"]["n"] == 2  # the throttled launch is dropped from the pairing
    assert "Excluded cells" in analyze.render(res)


def test_too_few_launches_reports_insufficient_data_not_a_verdict(tmp_path):
    cs = [c for c in b2_cells() if c["repeat"] == 0]
    write(tmp_path / "b2", cs)
    res = analyze.analyze(args(b2=str(tmp_path / "b2")))
    assert {x["status"] for x in res["tests"]} == {"insufficient_data"}


def test_p4_capacity_p5_quality_p6_graphs_and_model_check(tmp_path):
    b1, b3a, b4, g = (tmp_path / n for n in ("b1", "b3a", "b4", "g"))
    for d, variant, caps in (
        (b1, "fp16-default", [10.0, 10.2, 9.9]),
        (b3a, "fp8", [15.0, 15.3, 14.8]),
    ):
        d.mkdir()
        for r, c in enumerate(caps):
            (d / f"capacity__{variant}__r{r}.json").write_text(
                json.dumps({"capacity": c, "probes": []})
            )
    write(b1, [cell("fp16-default", "c1", r, tpot=0.025, ttft=0.07) for r in range(3)])
    cs = []
    for r in range(3):
        for variant, c1, c64 in (
            ("graphs-none-compile-on", 0.030, 0.050),
            ("graphs-full-and-piecewise", 0.022, 0.049),
        ):
            cs += [cell(variant, "c1", r, tpot=c1), cell(variant, "c64", r, tpot=c64)]
    write(b4, cs)
    g.mkdir()
    base = [i % 5 != 0 for i in range(400)]
    cand = list(base)
    cand[3] = not cand[3]  # one discordant item: well within 2pp
    for name, corr in (("fp16-default", base), ("fp8", cand)):
        (g / f"gsm8k-{name}.json").write_text(
            json.dumps({"variant": name, "items": list(range(400)), "correct": corr})
        )
    res = analyze.analyze(args(b1=str(b1), b3a_fp8=str(b3a), b4=str(b4), gsm8k_dir=str(g)))
    t = {x["id"]: x for x in res["tests"]}
    assert t["P4"]["verdict"] == "superior_higher" and t["P4"]["supported_after_holm"]
    assert t["P5"]["supported"] and t["P5"]["verdict"] in ("equivalent", "non_inferior")
    assert t["P6"]["verdict"] == "superior_higher" and t["P6"]["supported"]
    mc = {r["quantity"]: r for r in res["model_check"]}
    assert (
        mc["fp16-default batch-1 decode"]["in_interval"]
        and mc["fp16-default SLO capacity"]["in_interval"]
    )
    md = analyze.render(res)
    assert "not 'no difference'" in md and "Model check" in md


def test_p5_with_different_item_sets_refuses_to_compare(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    (g / "gsm8k-fp16-default.json").write_text(
        json.dumps({"items": [1, 2], "correct": [True, False]})
    )
    (g / "gsm8k-fp8.json").write_text(json.dumps({"items": [3, 4], "correct": [True, False]}))
    res = analyze.analyze(args(gsm8k_dir=str(g)))
    assert res["tests"][0]["status"] == "insufficient_data"
