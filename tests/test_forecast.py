from src.telemetry.forecast import make_forecast_summary


def test_basic_forecast_summary():
    # rising line
    series = [i * 0.1 for i in range(100)]
    ts = [float(i) for i in range(100)]
    summ = make_forecast_summary(series, ts, horizon=20, seasonal_periods=None, threshold=9.5, direction="above")
    # Should estimate breach soon
    assert summ.breach_threshold is not None
