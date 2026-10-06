import pytest

from tokbench import model as m

L40S = m.HARDWARE["L40S"]


def test_kv_bytes_per_token_is_128kib_for_llama31_8b():
    assert m.kv_bytes_per_token() == 131072  # 2 * 32 layers * 8 kv heads * 128 dim * 2 B
    assert m.kv_bytes_per_token("fp8") == 65536


def test_weight_bytes_ordering_and_fp8_keeps_embeddings_fp16():
    fp16, fp8, int4 = (m.weight_bytes(p) for p in m.PRECISIONS)
    assert fp16 == pytest.approx(16.06e9, rel=1e-3)
    assert int4 < fp8 < fp16
    assert fp8 == pytest.approx(m.BODY_PARAMS + 2 * m.EMBED_PARAMS * 1.0, rel=1e-6)


def test_l40s_fp16_kv_capacity_and_batch1_decode_are_in_the_expected_range():
    cap = m.kv_capacity_tokens(L40S, "fp16")
    assert 180_000 <= cap <= 195_000
    assert 35 <= m.batch1_decode_tok_s(L40S, "fp16", mem_eff=0.75) <= 45  # ~864GB/s / 16GB * 0.75
    assert m.kv_capacity_tokens(L40S, "fp8", kv_dtype="fp8") > 1.8 * cap  # lighter weights+KV


def test_quantization_helps_decode_but_int4_does_not_help_prefill():
    d = {p: m.batch1_decode_tok_s(L40S, p) for p in m.PRECISIONS}
    assert d["awq-int4"] > d["fp8"] > d["fp16"]
    assert m.prefill_s(L40S, "fp8", 1000, 0.5) < m.prefill_s(L40S, "fp16", 1000, 0.5)
    assert m.prefill_s(L40S, "awq-int4", 1000, 0.5) == m.prefill_s(L40S, "fp16", 1000, 0.5)


def test_saturation_is_batch_limited_by_kv_and_prefill_dominated_for_long_prompts():
    short = m.saturation_req_s(L40S, "fp16", 512, 128)
    long = m.saturation_req_s(L40S, "fp16", 2600, 128)
    assert long["req_s"] < short["req_s"] and long["batch"] < short["batch"]
    assert short["batch"] == min(short["kv_capacity_tokens"] // 640, 256)


def test_interval_brackets_the_point_estimate_and_h100_beats_l40s():
    lo, hi = m.interval(lambda me, f: m.saturation_req_s(L40S, "fp16", 512, 128, me, f)["req_s"])
    pt = m.saturation_req_s(L40S, "fp16", 512, 128)["req_s"]
    assert lo < pt < hi
    h100 = m.saturation_req_s(m.HARDWARE["H100-SXM"], "fp16", 512, 128)["req_s"]
    assert h100 > 2 * pt


def test_rendered_predictions_state_what_is_verified_and_what_is_not():
    md = m.render_markdown()  # L40S: checked against NVIDIA's page on 2026-10-06
    assert "Committed before any GPU run" in md and "**verified**" in md and "2026-10-06" in md
    assert "unverified" not in md.lower()
    assert md == m.render_markdown()
    pcie = m.render_markdown("H100-PCIe")  # not checked (the page lists NVL, not PCIe)
    assert "unverified recollections" in pcie and "**verified**" not in pcie
    assert m.HARDWARE["H100-SXM"].verified and not m.HARDWARE["H100-PCIe"].verified


def test_slo_capacity_is_below_saturation_and_monotone_in_the_slo():
    sat = m.saturation_req_s(L40S, "fp16", 512, 128)["req_s"]
    caps = [m.slo_capacity_req_s(L40S, "fp16", 512, 128, slo) for slo in (0.03, 0.05, 0.1, 1.0)]
    assert caps == sorted(caps) and caps[1] < sat  # TPOT SLO binds before saturation
    assert caps[-1] <= sat * 1.001  # a loose SLO converges to (at most) saturation
    assert m.slo_capacity_req_s(L40S, "fp16", 512, 128, 0.001) == 0.0  # infeasible: below one step


def test_fp8_has_materially_more_slo_capacity_than_fp16():
    f16 = m.slo_capacity_req_s(L40S, "fp16", 512, 128, 0.05)
    f8 = m.slo_capacity_req_s(L40S, "fp8", 512, 128, 0.05)
    assert f8 > 1.2 * f16


def test_prediction_intervals_are_ordered_and_match_the_rendered_table():
    iv = m.prediction_intervals("L40S")
    assert all(lo <= hi for lo, hi, _ in iv.values())
    lo, hi, _ = iv["slo_capacity_req_s:fp16"]
    assert f"{lo:.1f} - {hi:.1f}" in m.render_markdown()
