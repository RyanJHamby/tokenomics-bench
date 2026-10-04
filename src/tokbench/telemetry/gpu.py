"""GPU telemetry: power, clocks, temperature, utilization, throttle reasons.

Backends are swappable so the harness runs on a laptop (FakeBackend) and on a GPU
box (NvmlBackend, needs `pip install nvidia-ml-py`). Energy is the trapezoidal
integral of board power over time; host and cooling power are NOT included.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol


@dataclass
class GpuSample:
    t: float  # seconds, time.perf_counter() clock
    power_w: float
    sm_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    temp_c: int = 0
    util_pct: int = 0
    throttle_reasons: int = 0  # NVML clocks-throttle-reasons bitmask


class GpuBackend(Protocol):
    def read(self) -> GpuSample: ...


class NvmlBackend:
    def __init__(self, index: int = 0):
        import pynvml  # imported lazily so laptops don't need it

        self._n = pynvml
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def read(self) -> GpuSample:
        n, h = self._n, self._h
        return GpuSample(
            t=time.perf_counter(),
            power_w=n.nvmlDeviceGetPowerUsage(h) / 1000.0,  # mW -> W
            sm_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
            mem_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
            temp_c=n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            util_pct=n.nvmlDeviceGetUtilizationRates(h).gpu,
            throttle_reasons=n.nvmlDeviceGetCurrentClocksThrottleReasons(h),
        )


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
    """Samples a backend on a background thread. Use as a context manager."""

    def __init__(self, backend: GpuBackend, hz: float = 10.0):
        self.backend, self.period = backend, 1.0 / hz
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.backend.read())
            self._stop.wait(self.period)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(self.backend.read())  # closing sample bounds the interval


def energy_joules(samples: Sequence[GpuSample]) -> float:
    """Trapezoidal integral of power over the sampled interval."""
    return sum(0.5 * (a.power_w + b.power_w) * (b.t - a.t) for a, b in pairwise(samples))
