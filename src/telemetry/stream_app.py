from __future__ import annotations

import time
from typing import Dict, List, Optional
import json
from pathlib import Path

import typer

from .simulator import TelemetrySimulator
from .anomaly import EWMAZScoreDetector, RangeGuard
from .forecast import make_forecast_summary

app = typer.Typer(help="Telemetry streaming demo: simulate, detect anomalies, forecast trends.")


@app.command()
def main(
    rate_hz: float = typer.Option(5.0, help="Sampling rate in Hz"),
    duration_sec: float = typer.Option(30.0, help="Run duration"),
    channels: str = typer.Option("battery_v,temp_c,gyro_x", help="Comma-separated channel names (ignored if --config used)"),
    z_threshold: float = typer.Option(3.0, help="Z-score threshold (ignored if --config used)"),
    alpha: float = typer.Option(0.05, help="EWMA smoothing factor (ignored if --config used)"),
    config: Optional[str] = typer.Option(None, help="Path to JSON config to override CLI defaults"),
) -> None:
    # Load configuration
    cfg = None
    if config:
        p = Path(config)
        if not p.exists():
            raise typer.BadParameter(f"Config not found: {config}")
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    if cfg is not None:
        rate_hz = float(cfg.get("rate_hz", rate_hz))
        det_cfg = cfg.get("detector", {})
        alpha = float(det_cfg.get("alpha", alpha))
        z_threshold = float(det_cfg.get("z_threshold", z_threshold))
        ch_defs = cfg.get("channels", [])
        ch_list = [c["name"] for c in ch_defs]
    else:
        ch_list = [c.strip() for c in channels.split(",") if c.strip()]

    sim = TelemetrySimulator(channels=ch_list, rate_hz=rate_hz)
    det = EWMAZScoreDetector(alpha=alpha, z_threshold=z_threshold)

    # Per-channel range guards and breach settings
    guards: Dict[str, RangeGuard] = {}
    breach: Dict[str, Dict[str, float]] = {}
    if cfg is not None:
        for cdef in cfg.get("channels", []):
            name = cdef["name"]
            guards[name] = RangeGuard(cdef.get("min"), cdef.get("max"))
            b = cdef.get("breach")
            if b:
                breach[name] = {"direction": b.get("direction", "above"), "threshold": float(b.get("threshold"))}
    else:
        guards = {
            "battery_v": RangeGuard(20.0, 32.0),
            "temp_c": RangeGuard(-40.0, 85.0),
            "gyro_x": RangeGuard(-5.0, 5.0),
        }
        breach = {
            "battery_v": {"direction": "below", "threshold": 24.0},
            "temp_c": {"direction": "above", "threshold": 60.0},
        }

    # Buffers for short history to forecast
    hist: Dict[str, List[float]] = {c: [] for c in ch_list}
    ts_hist: List[float] = []
    last_forecast_time = 0.0

    print("ts,channel,value,mean,std,z,is_anomaly,range_violation")

    for pt in sim.stream(duration_sec=duration_sec):
        ts = pt.ts
        ts_hist.append(ts)
        ts_hist = ts_hist[-600:]

        for c, v in pt.values.items():
            hist[c].append(v)
            hist[c] = hist[c][-600:]

            res = det.update(c, v)
            rg = guards.get(c)
            range_violation = rg.check(v) if rg else False
            print(f"{ts:.3f},{c},{v:.6f},{res.mean:.6f},{res.std:.6f},{res.zscore:.2f},{res.is_anomaly},{range_violation}")

        # Periodic forecast summary every ~5s
        if ts - last_forecast_time >= 5.0:
            last_forecast_time = ts
            for c in ch_list:
                series = hist[c]
                if len(series) < 30:
                    continue
                # Assume temp has weak seasonality; others simple trend
                seasonal_periods = 50 if c == "temp_c" else None
                thr = None
                direction = "above"
                if c in breach:
                    b = breach[c]
                    thr = float(b.get("threshold"))
                    direction = str(b.get("direction", "above"))
                summ = make_forecast_summary(series, ts_hist[-len(series):], horizon=60, seasonal_periods=seasonal_periods, threshold=thr, direction=direction)
                if summ.est_time_to_breach_sec is not None:
                    eta = summ.est_time_to_breach_sec
                    print(f"FORECAST,{c},threshold={summ.breach_threshold},eta_sec={eta:.1f},next={summ.forecast[:3]}")


if __name__ == "__main__":
    app()
