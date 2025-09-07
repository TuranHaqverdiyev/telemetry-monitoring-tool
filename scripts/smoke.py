from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.telemetry.anomaly import EWMAZScoreDetector
from src.telemetry.forecast import make_forecast_summary


def test_anomaly_detector() -> None:
    det = EWMAZScoreDetector(alpha=0.1, z_threshold=2.5)
    for _ in range(100):
        det.update("x", 0.0)
    res = det.update("x", 3.0)
    assert res.is_anomaly or abs(res.zscore) > 2.0


def test_forecast_summary() -> None:
    series = [i * 0.1 for i in range(100)]
    ts = [float(i) for i in range(100)]
    summ = make_forecast_summary(series, ts, horizon=20, seasonal_periods=None, threshold=9.5, direction="above")
    assert summ.breach_threshold is not None


def main() -> int:
    try:
        test_anomaly_detector()
        test_forecast_summary()
    except Exception as e:
        print(f"SMOKE: FAIL: {e}")
        return 1
    print("SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
