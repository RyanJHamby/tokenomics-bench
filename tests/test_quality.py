import json

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
    import pytest

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
