"""The persistence layer: a result must be re-derivable from the saved artifacts alone."""

import asyncio
import gzip
import json
import socket
import sys
import types
from types import SimpleNamespace

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestServer

from tokbench import capture, runner
from tokbench.loadgen import summarize_window
from tokbench.runner import CellFailed
from tokbench.telemetry import GpuSample, MetricsScraper, NvmlBackend, energy_between

CFG = {
    "name": "t",
    "block": "T",
    "model": "mock",
    "seed": 0,
    "repeats": 1,
    "warmup_s": 0.5,
    "measure_s": 2.0,
    "drain_s": 5.0,
    "startup_seconds": 10,
    "soak_s": 1.0,
    "soak_concurrency": 2,
    "idle_s": 0.6,
    "slo": {"ttft_s": 0.5, "tpot_s": 0.1},
    "workload": {"input_tokens": 32, "output_tokens": 6},
    "variants": [{"name": "v", "server_args": ["--tpot", "0.004"], "slots": 4}],
    "loads": [{"mode": "open", "qps": 20}],
}


def _gz_text(path):
    with gzip.open(path, "rt") as f:
        return f.read()


def _gz_jsonl(path):
    return [json.loads(line) for line in _gz_text(path).splitlines() if line.strip()]


@pytest.fixture
def run(tmp_path, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(runner, "PORT", s.getsockname()[1])
    p = tmp_path / "c.yaml"

    def go(cfg=CFG, *extra):
        p.write_text(yaml.safe_dump(cfg))
        out = tmp_path / "out"
        rc = runner.main([str(p), "--server", "mock", "--gpu", "fake", "--usd-per-hr", "1",
                          "--out", str(out), *extra])  # fmt: skip
        return rc, out

    return go


def test_summary_is_reproducible_from_the_saved_request_records(run):
    rc, out = run()
    assert rc == 0
    cell = json.loads((out / "v__q20__r0.json").read_text())
    header, recs = capture.load_requests(out / "v__q20__r0.req.jsonl.gz")
    again = summarize_window(recs, header["ws"], header["we"], ttft_slo=0.5, tpot_slo=0.1)
    for k in ("n_requests", "n_ok", "n_failed", "n_completed", "output_tokens", "throughput_tok_s",
              "goodput_tok_s", "slo_attainment", "ttft_p50", "ttft_p99", "tpot_p50"):  # fmt: skip
        assert again[k] == pytest.approx(cell["summary"][k], rel=1e-6, abs=1e-9), k
    assert again["itl_p99"] == pytest.approx(cell["summary"]["itl_p99"], abs=2e-4)  # 0.1 ms quantum
    assert all(r.req_id == "" and r.n_chunks == r.n_out for r in recs if r.ok)  # mock: no ids
    assert {r.idx for r in recs} == set(range(len(recs)))


def test_energy_is_reproducible_from_the_saved_gpu_series(run):
    _, out = run()
    cell = json.loads((out / "v__q20__r0.json").read_text())
    header, _ = capture.load_requests(out / "v__q20__r0.req.jsonl.gz")
    rows = capture.load_gpu_series(out / "v__q20__r0.gpu.csv.gz")
    assert list(rows[0]) == list(capture.GPU_COLUMNS)
    samples = [GpuSample(t=float(r["t_mono"]), power_w=float(r["power_w"])) for r in rows]
    e = energy_between(samples, header["ws"], header["we"])
    assert e["energy_j"] == pytest.approx(cell["energy"]["energy_j"], rel=1e-4)
    # wall-clock column is anchored consistently with the monotonic column
    a = header["anchor"]
    r0 = rows[0]
    assert float(r0["t_wall"]) == pytest.approx(
        a["t_wall0"] + float(r0["t_mono"]) - a["t_mono0"], abs=1e-3
    )


def test_metrics_series_launch_files_events_and_soak_idle_are_all_written(run):
    _, out = run()
    base = out / "v__q20__r0"
    rows = _gz_jsonl(f"{base}.metrics.jsonl.gz")
    assert rows and "vllm:num_requests_running" in rows[0]["m"] and rows[0]["tw"] > 1.7e9
    raw = _gz_text(f"{base}.metrics.raw.txt.gz")
    assert raw.startswith("#### t_mono=") and "vllm:num_requests_waiting" in raw
    ld = out / "launches" / "v__r0"
    assert (ld / "server.log").exists()
    meta = json.loads((ld / "launch.json").read_text())
    assert meta["time_to_healthy_s"] > 0 and meta["wall_healthy"] > meta["wall_spawn"]
    assert "HF_TOKEN" not in meta["env"]
    assert (ld / "soak.req.jsonl.gz").stat().st_size > 50 and (ld / "idle.gpu.csv.gz").exists()
    kinds = [json.loads(line)["kind"] for line in (out / "events.jsonl").read_text().splitlines()]
    assert kinds[0] == "run_start" and {"launch_start", "healthy", "cell", "launch_end"} <= set(
        kinds
    )
    cell = json.loads((out / "v__q20__r0.json").read_text())
    assert cell["clock"]["wall_we"] - cell["clock"]["wall_ws"] == pytest.approx(2.0, abs=1e-6)
    assert set(cell["artifacts"]) == {"v__q20__r0.req.jsonl.gz", "v__q20__r0.gpu.csv.gz",
                                      "v__q20__r0.metrics.jsonl.gz", "v__q20__r0.metrics.raw.txt.gz"}  # fmt: skip


def test_metrics_stats_use_the_same_window_as_energy_and_throughput(run):
    _, out = run()
    cell = json.loads((out / "v__q20__r0.json").read_text())
    rows = _gz_jsonl(out / "v__q20__r0.metrics.jsonl.gz")
    ws, we = cell["clock"]["ws"], cell["clock"]["we"]
    inside = [r["m"]["vllm:num_requests_running"] for r in rows if ws <= r["t"] <= we]
    assert cell["server_metrics"]["vllm:num_requests_running"]["max"] == max(inside)


def test_a_failed_launch_leaves_a_crash_bundle_with_the_server_log(run, tmp_path):
    cfg = {
        **CFG,
        "soak_s": 0,
        "idle_s": 0,
        "variants": [{"name": "v", "server_args": ["--fail-after", "2"]}],
    }
    with pytest.raises(CellFailed):
        run(cfg)
    crash = tmp_path / "out" / "launches" / "v__r0" / "crash"
    assert "CellFailed" in (crash / "exception.txt").read_text()
    assert (crash / "server_log_tail.txt").exists() and (crash / "ps.txt").exists()
    kinds = [
        json.loads(x)["kind"] for x in (tmp_path / "out" / "events.jsonl").read_text().splitlines()
    ]
    assert "crash" in kinds and kinds[-1] == "launch_end"


def test_crash_bundle_contents(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"line {i}" for i in range(500)))
    try:
        raise RuntimeError("engine died")
    except RuntimeError as e:
        capture.write_crash_bundle(tmp_path / "crash", log, e)
    d = tmp_path / "crash"
    assert "engine died" in (d / "exception.txt").read_text()
    tail = (d / "server_log_tail.txt").read_text().splitlines()
    assert tail[-1] == "line 499" and len(tail) == 300
    assert (
        (d / "ps.txt").exists()
        and (d / "nvidia-smi-q.txt").exists()
        and (d / "dmesg_tail.txt").exists()
    )


def test_env_allowlist_never_contains_secrets():
    env = {"HF_TOKEN": "hf_secret", "AWS_SECRET_ACCESS_KEY": "x", "CUDA_VISIBLE_DEVICES": "0",
           "VLLM_LOGGING_LEVEL": "INFO", "HF_HOME": "/data/hf", "HOME": "/root", "PATH": "/bin",
           "VLLM_API_KEY": "k", "NCCL_DEBUG": "INFO", "GITHUB_TOKEN": "t"}  # fmt: skip
    got = capture.allowlisted_env(env)
    assert got == {"CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/data/hf", "NCCL_DEBUG": "INFO",
                   "VLLM_LOGGING_LEVEL": "INFO"}  # fmt: skip
    assert "secret" not in json.dumps(got)


def test_identifiers_are_hashed_not_published():
    text = "    GPU UUID                              : GPU-abc123-def\n    Serial Number : 1650123\n    Product Name : L40S\n"
    out = capture.scrub_identifiers(text)
    assert "abc123" not in out and "1650123" not in out and "L40S" in out
    assert out == capture.scrub_identifiers(text)  # deterministic: still joinable across files


def test_capture_env_bundle_is_written_best_effort_and_snapshots_hf_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    hub = tmp_path / "hf" / "hub" / "models--org--m" / "snapshots" / ("a" * 40)
    hub.mkdir(parents=True)
    (hub / "config.json").write_text('{"x": 1}')
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    written = capture.capture_env(
        tmp_path / "run", [("org/m", "a" * 40), ("org/missing", "b" * 40)]
    )
    d = tmp_path / "run" / "_env"
    assert {"pip_freeze.txt", "nvidia-smi-q.txt", "lscpu.txt", "env_allowlist.json"} <= set(written)
    for f in d.iterdir():
        assert "hf_supersecret" not in f.read_text(errors="ignore"), f.name
    assert (
        "# FAILED" in (d / "nvidia-smi-q.txt").read_text()
        or "NVIDIA" in (d / "nvidia-smi-q.txt").read_text()
    )
    snap = json.loads((d / "hf_snapshot.json").read_text())
    assert snap["org/m"]["present"] and "config.json" in snap["org/m"]["sha256"]
    assert snap["org/missing"]["present"] is False


def _fake_pynvml(monkeypatch, reasons_name="nvmlDeviceGetCurrentClocksEventReasons"):
    class NVMLError(Exception):
        pass

    def unsupported(*a):
        raise NVMLError("not supported")

    m = types.ModuleType("pynvml")
    m.NVMLError = NVMLError
    m.NVML_CLOCK_SM, m.NVML_CLOCK_MEM, m.NVML_TEMPERATURE_GPU = 0, 1, 0
    m.nvmlInit = lambda: None
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetHandleByIndex = lambda i: "h"
    m.nvmlDeviceGetTotalEnergyConsumption = lambda h: 5_000
    m.nvmlDeviceGetPowerUsage = lambda h: 250_000
    m.nvmlDeviceGetClockInfo = lambda h, k: 1500 if k == 0 else 9000
    m.nvmlDeviceGetTemperature = lambda h, k: 61
    m.nvmlDeviceGetUtilizationRates = lambda h: SimpleNamespace(gpu=97)
    setattr(m, reasons_name, lambda h: 0x4)
    m.nvmlDeviceGetEnforcedPowerLimit = unsupported  # optional field: must become None
    m.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(used=3 * 2**30)
    m.nvmlDeviceGetPerformanceState = lambda h: 2
    m.nvmlDeviceGetViolationStatus = lambda h, k: SimpleNamespace(violationTime=1234 + k)
    monkeypatch.setitem(sys.modules, "pynvml", m)


def test_nvml_backend_reads_optional_fields_defensively_and_accepts_renamed_reasons_api(
    monkeypatch,
):
    _fake_pynvml(monkeypatch)  # only the NEW EventReasons name exists
    s = NvmlBackend(0).read()
    assert s.power_w == 250.0 and s.sm_clock_mhz == 1500 and s.throttle_reasons == 0x4
    assert s.enforced_limit_w is None  # unsupported -> None, never an exception
    assert s.mem_used_mib == 3072.0 and s.pstate == 2 and s.energy_mj == 5000.0
    assert s.viol_power_ns == 1234 and s.viol_thermal_ns == 1235
    _fake_pynvml(monkeypatch, reasons_name="nvmlDeviceGetCurrentClocksThrottleReasons")  # old name
    assert NvmlBackend(0).read().throttle_reasons == 0x4


async def test_metrics_scraper_keeps_raw_text_at_intervals_and_scrape_errors():
    calls = {"n": 0}

    async def metrics(_):
        calls["n"] += 1
        return web.Response(text=f'vllm:x{{l="a"}} {calls["n"]}\nh_bucket{{le="1"}} 3\n')

    app = web.Application()
    app.router.add_get("/metrics", metrics)
    server = TestServer(app)
    await server.start_server()
    try:
        async with MetricsScraper(
            str(server.make_url("/metrics")), interval_s=0.05, raw_every_s=0.2
        ) as sc:
            await asyncio.sleep(0.7)
    finally:
        await server.close()
    assert len(sc.rows) >= 8 and 2 <= len(sc.raw) <= len(sc.rows)
    assert (
        'h_bucket{le="1"}' in sc.raw[0][1] and "h_bucket" not in sc.rows[0][1]
    )  # raw keeps buckets
    bad = MetricsScraper("http://127.0.0.1:9/metrics", interval_s=0.05)
    async with bad:
        await asyncio.sleep(0.2)
    assert bad.errors >= 2 and not bad.rows


async def test_server_process_writes_its_output_to_the_log_file(tmp_path):
    from tokbench.server import ServerProcess

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    code = ("import http.server,sys\nprint('hello-from-server', flush=True)\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
            f"http.server.HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n")  # fmt: skip
    log = tmp_path / "launch" / "server.log"
    async with ServerProcess([sys.executable, "-c", code], port, 20, log_path=log) as srv:
        assert srv.t_healthy > srv.t_spawn
    assert "hello-from-server" in log.read_text()
