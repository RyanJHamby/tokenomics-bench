import json
import os
import subprocess
import sys
import tarfile
import time
import types
from types import SimpleNamespace

SCRIPT = os.path.abspath("scripts/sync_loop.sh")


def _sync(root, dest):
    env = {**os.environ, "ONCE": "1", "SYNC_DEST": str(dest)}
    r = subprocess.run(
        ["bash", SCRIPT, str(root)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return r.stdout + r.stderr


def _members(tar_path):
    with tarfile.open(tar_path) as t:
        return sorted(t.getnames())


def test_sync_ships_only_new_files_each_pass_and_never_the_sync_dir_itself(tmp_path):
    root, dest = tmp_path / "results", tmp_path / "dest"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "a.json").write_text("{}")
    out1 = _sync(root, dest)
    parts = sorted(dest.glob("*.tar.gz"))
    assert "shipped part0001" in out1 and len(parts) == 1
    assert [n for n in _members(parts[0]) if n.endswith("a.json")] and not any(
        "/_sync/" in n for n in _members(parts[0])
    )
    assert "shipped" not in _sync(root, dest)  # nothing changed: nothing shipped
    time.sleep(1.1)  # file mtime resolution
    (root / "raw" / "b.json").write_text("{}")
    (root / "raw" / "half.tmp").write_text("partial")  # in-flight atomic-write temp: skipped
    out3 = _sync(root, dest)
    parts = sorted(dest.glob("*.tar.gz"))
    assert len(parts) == 2 and "shipped" in out3
    names = _members(parts[1])
    assert any(n.endswith("b.json") for n in names) and not any(n.endswith("a.json") for n in names)
    assert not any(n.endswith(".tmp") for n in names)


def test_sync_without_a_destination_fails_loudly_and_retries_the_same_files(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "x.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if k not in ("SYNC_DEST", "SYNC_RELEASE")}
    env["ONCE"] = "1"
    out = subprocess.run(
        ["bash", SCRIPT, str(root)], env=env, capture_output=True, text=True, check=False
    ).stdout
    assert "no destination configured" in out and "FAILED to ship" in out
    # marker was NOT advanced, so a later pass with a destination ships the file
    assert "shipped" in _sync(root, tmp_path / "dest")


def _const(v):
    return lambda *a: v


def _fake_nvml(monkeypatch, fail_optional=True):
    class NVMLError(Exception):
        pass

    def unsupported(*a):
        raise NVMLError("n/a")

    m = types.ModuleType("pynvml")
    m.NVMLError = NVMLError
    m.NVML_CLOCK_SM = m.NVML_CLOCK_MEM = m.NVML_TEMPERATURE_GPU = 0
    for name, val in (("nvmlInit", None), ("nvmlDeviceGetHandleByIndex", "h"),
                      ("nvmlDeviceGetTotalEnergyConsumption", 1), ("nvmlDeviceGetPowerUsage", 200000),
                      ("nvmlDeviceGetClockInfo", 1000), ("nvmlDeviceGetTemperature", 50),
                      ("nvmlDeviceGetCurrentClocksEventReasons", 0), ("nvmlDeviceGetPerformanceState", 2)):  # fmt: skip
        setattr(m, name, _const(val))
    m.nvmlDeviceGetUtilizationRates = lambda h: SimpleNamespace(gpu=1)
    m.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(used=2**30)
    m.nvmlDeviceGetEnforcedPowerLimit = unsupported if fail_optional else (lambda h: 350000)
    m.nvmlDeviceGetViolationStatus = unsupported
    monkeypatch.setitem(sys.modules, "pynvml", m)


def test_probe_reports_which_optional_fields_this_gpu_does_not_expose(monkeypatch):
    from tokbench.telemetry.probe import probe

    _fake_nvml(monkeypatch)
    d = probe()
    assert d["nvml"] == "ok" and d["energy_counter"] is True
    assert (
        d["fields"]["enforced_limit_w"] == "UNSUPPORTED"
        and d["fields"]["viol_power_ns"] == "UNSUPPORTED"
    )
    assert d["fields"]["power_w"] == "ok" and d["fields"]["mem_used_mib"] == "ok"
    json.dumps(d, default=str)  # serialisable for _env/


def test_probe_degrades_to_a_message_when_nvml_is_absent(monkeypatch):
    from tokbench.telemetry.probe import probe

    monkeypatch.setitem(sys.modules, "pynvml", None)  # import fails
    d = probe()
    assert d["nvml"].startswith("unavailable") and "fields" not in d
