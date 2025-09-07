from src.telemetry.anomaly import EWMAZScoreDetector


def test_detector_basic_shift():
    det = EWMAZScoreDetector(alpha=0.1, z_threshold=2.5)
    # Warmup near 0
    for _ in range(100):
        res = det.update("x", 0.0)
    # Inject a shift
    res = det.update("x", 3.0)
    assert res.is_anomaly or abs(res.zscore) > 2.0
