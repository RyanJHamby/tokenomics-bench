"""GPU telemetry: power, energy counter, clocks, temperature, utilization, throttling.

Backends are swappable so the harness runs on a laptop (FakeBackend) and on a GPU box
(NvmlBackend, needs `pip install nvidia-ml-py`).

Energy: `nvmlDeviceGetPowerUsage` is a ~1 s average on recent data-center GPUs, so
integrating 10 Hz samples of it lags and smooths real power. Where available we use the
cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`, mJ) and report the
power-integral alongside it as a cross-check. Board power only: host CPU and cooling are
excluded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, Self

# NVML clocks-throttle-reason bits (nvmlClocksThrottleReasons*).
THROTTLE_SW_POWER_CAP = 0x4  # expected under a power cap or at TDP; not a fault
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80
THROTTLE_BAD_MASK = (
    THROTTLE_HW_SLOWDOWN | THROTTLE_SW_THERMAL | THROTTLE_HW_THERMAL | THROTTLE_HW_POWER_BRAKE
)


@dataclass
class GpuSample:
    t: float  # seconds, time.perf_counter() clock
    power_w: float
    sm_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    temp_c: int = 0
    util_pct: int = 0
    throttle_reasons: int = 0  # NVML clocks-throttle-reasons bitmask
    energy_mj: float | None = None  # cumulative energy counter, if the backend has one
    # Optional extras; None where a field is unsupported. Persisted in the per-cell series.
    enforced_limit_w: float | None = None
    pstate: int | None = None
    mem_used_mib: float | None = None
    viol_power_ns: int | None = None  # cumulative time held back by the power policy
    viol_thermal_ns: int | None = None


class GpuBackend(Protocol):
    def read(self) -> GpuSample: ...


class NvmlBackend:
    def __init__(self, index: int = 0):
        import pynvml  # imported lazily so laptops don't need it

        self._n = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
            self._has_energy = True
        except pynvml.NVMLError:
            self._has_energy = False

    def _opt(self, fn, *args):
        """Optional field: None if this GPU/driver does not support it (never raises)."""
        try:
            return fn(*args)
        except self._n.NVMLError:
            return None

    def _reasons(self) -> int:
        """Newer nvidia-ml-py renamed ThrottleReasons to EventReasons; accept either."""
        n = self._n
        fn = (
            getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None)
            or n.nvmlDeviceGetCurrentClocksThrottleReasons
        )
        return fn(self._h)

    def read(self) -> GpuSample:
        n, h = self._n, self._h
        limit = self._opt(n.nvmlDeviceGetEnforcedPowerLimit, h)
        mem = self._opt(n.nvmlDeviceGetMemoryInfo, h)
        vp = self._opt(n.nvmlDeviceGetViolationStatus, h, getattr(n, "NVML_PERF_POLICY_POWER", 0))
        vt = self._opt(n.nvmlDeviceGetViolationStatus, h, getattr(n, "NVML_PERF_POLICY_THERMAL", 1))
        return GpuSample(
            t=time.perf_counter(),
            power_w=n.nvmlDeviceGetPowerUsage(h) / 1000.0,  # mW -> W
            sm_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
            mem_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
            temp_c=n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            util_pct=n.nvmlDeviceGetUtilizationRates(h).gpu,
            throttle_reasons=self._reasons(),
            energy_mj=(
                float(n.nvmlDeviceGetTotalEnergyConsumption(h)) if self._has_energy else None
            ),
            enforced_limit_w=limit / 1000.0 if limit is not None else None,
            pstate=self._opt(n.nvmlDeviceGetPerformanceState, h),
            mem_used_mib=mem.used / 2**20 if mem is not None else None,
            viol_power_ns=vp.violationTime if vp is not None else None,
            viol_thermal_ns=vt.violationTime if vt is not None else None,
        )

    def close(self) -> None:
        self._n.nvmlShutdown()


class FakeBackend:
    """Deterministic stand-in: power = idle + (peak-idle) * load(), capped."""

    def __init__(
        self,
        load: Callable[[], float] = lambda: 1.0,
        idle_w: float = 60.0,
        peak_w: float = 300.0,
        cap_w: float | None = None,
    ):
        self.load, self.idle_w, self.peak_w, self.cap_w = load, idle_w, peak_w, cap_w

    def read(self) -> GpuSample:
        p = self.idle_w + (self.peak_w - self.idle_w) * min(max(self.load(), 0.0), 1.0)
        if self.cap_w is not None:
            p = min(p, self.cap_w)
        return GpuSample(t=time.perf_counter(), power_w=p)


class PowerSampler:
    """Samples a backend on a background thread. Use as a context manager.

    A failing backend read stops sampling, and the exception is re-raised on exit so a
    dead sampler can never produce a silently-valid cell.
    """

    def __init__(self, backend: GpuBackend, hz: float = 10.0):
        self.backend, self.period = backend, 1.0 / hz
        self.samples: list[GpuSample] = []
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.backend.read())
            except BaseException as e:  # noqa: BLE001 - surfaced in __exit__
                self.error = e
                return
            self._stop.wait(self.period)

    def __enter__(self) -> Self:
        self.samples.append(self.backend.read())  # opening sample bounds the interval
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        if self.error is not None:
            raise RuntimeError(f"GPU sampler died: {self.error!r}") from self.error
        self.samples.append(self.backend.read())  # closing sample bounds the interval


def energy_joules(samples: Sequence[GpuSample]) -> float:
    """Trapezoidal integral of power over the sampled interval."""
    return sum(0.5 * (a.power_w + b.power_w) * (b.t - a.t) for a, b in pairwise(samples))


def _lerp_at(samples: Sequence[GpuSample], t: float, field: str) -> float:
    """Linear interpolation of `field` at time t (clamped to the sampled range)."""
    if t <= samples[0].t:
        return getattr(samples[0], field)
    if t >= samples[-1].t:
        return getattr(samples[-1], field)
    for a, b in pairwise(samples):
        if a.t <= t <= b.t:
            w = (t - a.t) / (b.t - a.t) if b.t > a.t else 0.0
            return getattr(a, field) * (1 - w) + getattr(b, field) * w
    raise AssertionError("unreachable")


def energy_between(samples: Sequence[GpuSample], t0: float, t1: float) -> dict:
    """Energy over exactly [t0, t1] by interpolating at both ends.

    Returns the power-integral and, if every sample has the cumulative counter, the
    counter delta. `energy_j` prefers the counter. Both are board energy only.
    """
    if len(samples) < 2 or t1 <= t0:
        raise ValueError("need >=2 samples and t1 > t0")
    inner = [s for s in samples if t0 < s.t < t1]
    ends = [GpuSample(t=t0, power_w=_lerp_at(samples, t0, "power_w"))]
    ends_t1 = GpuSample(t=t1, power_w=_lerp_at(samples, t1, "power_w"))
    integral = energy_joules([*ends, *inner, ends_t1])
    out = {"energy_j_power_integral": integral, "energy_j_counter": None}
    if all(s.energy_mj is not None for s in samples):
        out["energy_j_counter"] = (
            _lerp_at(samples, t1, "energy_mj") - _lerp_at(samples, t0, "energy_mj")
        ) / 1000.0
    out["energy_j"] = out["energy_j_counter"] if out["energy_j_counter"] is not None else integral
    return out


def throttle_summary(samples: Sequence[GpuSample]) -> dict:
    n = max(1, len(samples))
    powers = sorted(s.power_w for s in samples)
    return {
        "throttle_reasons_or": _or_all(s.throttle_reasons for s in samples),
        "cap_bound_fraction": sum(bool(s.throttle_reasons & THROTTLE_SW_POWER_CAP) for s in samples)
        / n,
        "bad_throttle_seen": any(s.throttle_reasons & THROTTLE_BAD_MASK for s in samples),
        "power_w_mean": sum(powers) / n if samples else float("nan"),
        "power_w_p95": powers[int(0.95 * (len(powers) - 1))] if powers else float("nan"),
        "power_w_max": powers[-1] if powers else float("nan"),
        "sm_clock_mhz_mean": sum(s.sm_clock_mhz for s in samples) / n,
        "temp_c_max": max((s.temp_c for s in samples), default=0),
    }


def _or_all(xs) -> int:
    v = 0
    for x in xs:
        v |= x
    return v
