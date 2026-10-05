import json

import pytest
from aiohttp.test_utils import TestServer

from tokbench.mockserver import make_app
from tokbench.quality import greedy_equivalence, last_number, load_gsm8k
from tokbench.workloads import make_prompt_fn


async def _srv(salt=""):
    s = TestServer(make_app(salt=salt))
    await s.start_server()
    return s, str(s.make_url("/v1/chat/completions"))


async def test_equivalence_passes_for_identical_servers_and_fails_when_outputs_differ():
    bodies = [make_prompt_fn(20, 6, 0.5)(i) for i in range(5)]
    (sa, a), (sb, b), (sc, c) = await _srv(), await _srv(), await _srv(salt="x")
    try:
        ok = await greedy_equivalence(a, b, bodies)
        bad = await greedy_equivalence(a, c, bodies)
    finally:
        for s in (sa, sb, sc):
            await s.close()
    assert ok.passed and ok.n == 5
    assert not bad.passed and bad.first_diff == 0


async def test_equivalence_refuses_non_greedy_bodies():
    with pytest.raises(ValueError):
        await greedy_equivalence("x", "y", [{"temperature": 0.7}])


def test_last_number_and_loader(tmp_path):
    assert last_number("so it is 1,234.") == "1234"
    assert last_number("The answer is 42.") == "42"
    assert last_number("no digits") is None
    p = tmp_path / "g.jsonl"
    p.write_text(
        "\n".join(json.dumps({"question": f"q{i}", "answer": f"w #### {i},000"}) for i in range(10))
    )
    a, b = load_gsm8k(p, 4, seed=1), load_gsm8k(p, 4, seed=1)
    assert a == b and len(a) == 4 and a[0]["answer"].endswith("000")


def test_paired_accuracy_matches_hand_and_scipy_references():
    from tokbench.quality import paired_accuracy

    # 100 items: baseline-only correct 10, candidate-only correct 2, 88 concordant.
    base = [True] * 10 + [False] * 2 + [True] * 44 + [False] * 44
    cand = [False] * 10 + [True] * 2 + [True] * 44 + [False] * 44
    r = paired_accuracy(base, cand, margin=0.02)
    assert r["diff"] == pytest.approx(-0.08)
    assert r["ci"][0] == pytest.approx(-0.146060, abs=1e-5)  # d -/+ 1.96 * paired SE
    assert r["ci"][1] == pytest.approx(-0.013940, abs=1e-5)
    assert r["mcnemar_p"] == pytest.approx(0.03857421875)  # scipy binomtest(2, 12, .5)
    assert r["verdict"] == "inconclusive"  # CI spans -2pp: cannot call inferior or fine


def test_paired_accuracy_verdicts():
    from tokbench.quality import paired_accuracy

    same = [True, False] * 500
    assert paired_accuracy(same, same)["verdict"] == "equivalent"  # identical -> CI is [0, 0]
    worse = [True] * 600 + [False] * 400
    assert (
        paired_accuracy(worse, [False] * 200 + [True] * 400 + [False] * 400)["verdict"]
        == "inferior"
    )
    with pytest.raises(ValueError):
        paired_accuracy([True], [True, False])


async def test_gsm8k_runner_scores_every_item_concurrently_and_keeps_item_order():
    from tokbench.quality import gsm8k_accuracy

    server, u = await _srv()
    try:
        items = [{"question": f"q{i}", "answer": "999"} for i in range(40)]
        res = await gsm8k_accuracy(u, "m", items, concurrency=8)
    finally:
        await server.close()
    assert res["n"] == 40 and len(res["correct"]) == 40 and res["accuracy"] == 0.0
