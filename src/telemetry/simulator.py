from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class TelemetryPoint:
    ts: float
    values: Dict[str, float]


class TelemetrySimulator:
    """Simple synthetic telemetry generator.

    Channels:
      - temp_c: diurnal-like sinusoid + noise (temperature sensor)
      - voltage: slow drift and random ripple (power supply)
      - current: zero-mean jitter + occasional spike (current sensor)
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        rate_hz: float = 5.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.channels = channels or ["temp_c", "voltage", "current"]
        self.period_s = 1.0 / max(rate_hz, 1e-6)
        self.rng = random.Random(seed)
        self._t0 = time.time()
        self._tick = 0

    def _gen(self, t: float) -> Dict[str, float]:
        r = self.rng
        vals: Dict[str, float] = {}

        if "temp_c" in self.channels:
            base = 20.0 + 5.0 * math.sin(t / 60.0)
            noise = r.gauss(0, 0.2)
            vals["temp_c"] = base + noise

        if "voltage" in self.channels:
            base = 3.3 + 0.1 * math.sin(t / 45.0)  # voltage fluctuation
            ripple = 0.02 * math.sin(t / 15.0) + r.gauss(0, 0.005)
            vals["voltage"] = base + ripple

        if "current" in self.channels:
            base = 1.0 + 0.05 * math.sin(t / 30.0)  # current variation
            jitter = r.gauss(0, 0.01)
            spike = 0.0
            if self._tick % 200 == 0 and self._tick > 0 and r.random() < 0.3:
                spike = r.choice([-0.1, 0.1]) * (1.0 + r.random() * 0.5)
            vals["current"] = base + jitter + spike

        return vals

    def stream(self, duration_sec: Optional[float] = None) -> Iterator[TelemetryPoint]:
        start = time.time()
        while True:
            now = time.time()
            t = now - self._t0
            values = self._gen(t)
            self._tick += 1
            yield TelemetryPoint(ts=now, values=values)

            if duration_sec is not None and (now - start) >= duration_sec:
                return

            # sleep to maintain rate
            to_sleep = self.period_s - (time.time() - now)
            if to_sleep > 0:
                time.sleep(to_sleep)
