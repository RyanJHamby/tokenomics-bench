import json
import socket
import subprocess
import sys
import time

import pytest
import yaml

from tokbench import crosscheck, runner


def _cell(**kw):
    s = {
        "throughput_tok_s": 500.0,
        "ttft_p50": 0.10,
        "ttft_p99": 0.30,
        "tpot_p50": 0.030,
        "tpot_p99": 0.045,
        "itl_p99": 0.05,
    }
    s.update(kw)
    return {"summary": s}


BENCH = {
    "output_throughput": 505.0,
    "median_ttft_ms": 105.0,
    "p99_ttft_ms": 310.0,
    "median_tpot_ms": 31.0,
    "p99_tpot_ms": 46.0,
    "p99_itl_ms": 52.0,
}


def test_agreeing_tools_pass_and_a_biased_client_is_caught():
    assert all(r["verdict"] == "agree" for r in crosscheck.compare(_cell(), BENCH))
    biased = crosscheck.compare(_cell(ttft_p50=0.14), BENCH)  # client 33% slower to timestamp
    assert [r["metric"] for r in biased if r["verdict"] == "DISAGREE"] == ["TTFT p50 (ms)"]


def test_missing_bench_metric_is_reported_not_silently_ok():
    bench = {k: v for k, v in BENCH.items() if k != "p99_itl_ms"}
    rows = crosscheck.compare(_cell(), bench)
    assert any(r["verdict"] == "missing_in_bench" for r in rows)


def test_cli_exit_code_reflects_agreement(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps(_cell()))
    (tmp_path / "b.json").write_text(json.dumps(BENCH))
    assert crosscheck.main([str(tmp_path / "m.json"), str(tmp_path / "b.json")]) == 0
    (tmp_path / "m2.json").write_text(json.dumps(_cell(throughput_tok_s=300.0)))
    assert crosscheck.main([str(tmp_path / "m2.json"), str(tmp_path / "b.json")]) == 1


def test_attach_mode_uses_an_externally_started_server_and_leaves_it_running(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen(
        [sys.executable, "-m", "tokbench.mockserver", "--port", str(port), "--tpot", "0.004"],
        start_new_session=True,
    )
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        cfg = {
            "name": "x",
            "block": "X",
            "model": "mock",
            "seed": 0,
            "repeats": 1,
            "warmup_s": 0.3,
            "measure_s": 1.5,
            "startup_seconds": 5,
            "workload": {"input_tokens": 16, "output_tokens": 5},
            "variants": [{"name": "v", "slots": 4}],
            "loads": [{"mode": "open", "qps": 10}],
        }
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        rc = runner.main(
            [
                str(p),
                "--server",
                "mock",
                "--gpu",
                "fake",
                "--usd-per-hr",
                "1",
                "--out",
                str(tmp_path / "o"),
                "--attach",
                f"http://127.0.0.1:{port}",
            ]
        )
        assert rc == 0 and (tmp_path / "o" / "v__q10__r0.json").exists()
        assert srv.poll() is None  # runner must not kill a server it did not start
    finally:
        srv.terminate()
        srv.wait(timeout=10)


def test_attaching_to_nothing_fails_with_a_clear_error(tmp_path):
    cfg = {
        "name": "x",
        "block": "X",
        "model": "mock",
        "seed": 0,
        "repeats": 1,
        "warmup_s": 0.1,
        "measure_s": 0.5,
        "startup_seconds": 5,
        "workload": {"input_tokens": 8, "output_tokens": 2},
        "variants": [{"name": "v"}],
        "loads": [{"mode": "open", "qps": 5}],
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="cannot reach attached server"):
        runner.main(
            [
                str(p),
                "--server",
                "mock",
                "--gpu",
                "fake",
                "--usd-per-hr",
                "1",
                "--out",
                str(tmp_path / "o"),
                "--attach",
                "http://127.0.0.1:9",
            ]
        )
