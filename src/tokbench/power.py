"""Power cap control via nvidia-smi (needs root; many containers forbid it).

Always restores to the card's *default* limit, never to whatever limit was set when we
started: after a hard kill mid-sweep the "current" limit may be a stale cap, and
restoring to that would silently run every later block capped.
"""

from __future__ import annotations

import subprocess


class CapUnavailable(RuntimeError):
    """The requested cap cannot be applied (out of range, no permission, did not stick)."""


def _q(index: int, field: str) -> float:
    out = subprocess.run(
        ["nvidia-smi", "-i", str(index), f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout
    return float(out.strip().splitlines()[0])


def power_limits(index: int = 0) -> dict:
    return {
        "min": _q(index, "power.min_limit"),
        "max": _q(index, "power.max_limit"),
        "default": _q(index, "power.default_limit"),
        "current": _q(index, "power.limit"),
    }


def validate_cap(watts: float, limits: dict) -> None:
    if not limits["min"] <= watts <= limits["max"]:
        raise CapUnavailable(
            f"cap {watts} W outside this GPU's [{limits['min']}, {limits['max']}] W range"
        )


class PowerCap:
    """Context manager applying `-pl` for a launch. No-op when disabled or watts is None."""

    def __init__(self, watts: float | None, enabled: bool, index: int = 0):
        self.watts, self.index = watts, index
        self.enabled = enabled and watts is not None
        self.limits: dict | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            self.limits = power_limits(self.index)
            validate_cap(self.watts, self.limits)
            subprocess.run(
                ["nvidia-smi", "-i", str(self.index), "-pl", str(int(self.watts))],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            got = _q(self.index, "power.limit")
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            self._restore()
            raise CapUnavailable(f"could not apply {self.watts} W cap: {e}") from e
        if abs(got - self.watts) > 1.5:
            self._restore()
            raise CapUnavailable(f"cap did not stick: asked {self.watts} W, GPU reports {got} W")
        return self

    def _restore(self) -> None:
        if self.limits:
            subprocess.run(
                ["nvidia-smi", "-i", str(self.index), "-pl", str(int(self.limits["default"]))],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

    def __exit__(self, *exc) -> None:
        if self.enabled:
            self._restore()
