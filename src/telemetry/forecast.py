from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastSummary:
    horizon: int
    forecast: List[float]
    breach_threshold: Optional[float]
    est_time_to_breach_sec: Optional[float]


def fit_forecast(
    series: List[float],
    seasonal_periods: Optional[int] = None,
    trend: Optional[str] = None,
    seasonal: Optional[str] = None,
) -> Optional[ExponentialSmoothing]:
    y = np.asarray(series)
    n = len(y)
    if n < 10:
        return None
    use_seasonal = seasonal if seasonal_periods and seasonal and n >= 2 * seasonal_periods else None
    try:
        model = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=use_seasonal,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated",
        )
        return model.fit(optimized=True)
    except Exception:
        # Fallback to a simpler non-seasonal model
        try:
            model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
            return model.fit(optimized=True)
        except Exception:
            return None


def summarize_breach(
    yhat: List[float],
    timestamps_sec: List[float],
    current_time_sec: float,
    threshold: Optional[float],
    direction: str = "above",
) -> Tuple[Optional[float], Optional[float]]:
    if threshold is None or len(yhat) == 0:
        return None, None
    # Estimate time to threshold by linear scan
    idx = None
    if direction == "above":
        for i, v in enumerate(yhat):
            if v >= threshold:
                idx = i
                break
    else:
        for i, v in enumerate(yhat):
            if v <= threshold:
                idx = i
                break
    if idx is None:
        return threshold, None
    if idx < len(timestamps_sec):
        return threshold, max(0.0, timestamps_sec[idx] - current_time_sec)
    return threshold, None


def make_forecast_summary(
    series: List[float],
    timestamps_sec: List[float],
    horizon: int,
    seasonal_periods: Optional[int],
    threshold: Optional[float],
    direction: str,
) -> ForecastSummary:
    model = fit_forecast(series, seasonal_periods=seasonal_periods, trend="add", seasonal="add" if seasonal_periods else None)
    if model is None:
        return ForecastSummary(horizon=horizon, forecast=[], breach_threshold=threshold, est_time_to_breach_sec=None)

    steps = min(horizon, max(1, len(series) // 4))
    yhat = list(map(float, model.forecast(steps)))
    # Build future timestamps by extending the last interval
    if len(timestamps_sec) >= 2:
        dt = (timestamps_sec[-1] - timestamps_sec[0]) / max(1, len(timestamps_sec) - 1)
    else:
        dt = 1.0
    future_ts = [timestamps_sec[-1] + (i + 1) * dt for i in range(len(yhat))]
    thr, ttb = summarize_breach(yhat, future_ts, timestamps_sec[-1] if timestamps_sec else 0.0, threshold, direction)
    return ForecastSummary(horizon=len(yhat), forecast=yhat, breach_threshold=thr, est_time_to_breach_sec=ttb)
