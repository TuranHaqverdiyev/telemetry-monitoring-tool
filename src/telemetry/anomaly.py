from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import time

from .detector_base import AnomalyDetector, AnomalyResult as BaseAnomalyResult, DetectorFactory


@dataclass
class LegacyAnomalyResult:
    """Legacy result structure for backward compatibility."""
    mean: float
    std: float
    zscore: float
    is_anomaly: bool


class EWMAZScoreDetector(AnomalyDetector):
    """Robust streaming anomaly detector using EWMA for mean and variance.

    - Keeps running estimates per channel.
    - Only updates statistics with non-anomalous points (robust learning).
    - Flags if |z| >= z_threshold and std is above a small epsilon.

    This prevents anomalies from polluting the baseline statistics, 
    keeping threshold bounds stable in real-world scenarios.
    """

    def __init__(self, alpha: float = 0.05, z_threshold: float = 3.0, eps: float = 1e-6, 
                 learning_period: int = 20, drift_update_rate: int = 100, channel: str = "default"):
        super().__init__(f"EWMA-ZScore-{channel}")
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.eps = eps
        self.learning_period = learning_period  # Initial points to learn unconditionally
        self.drift_update_rate = drift_update_rate  # Update every N-th point for drift
        self.channel = channel
        self._state: Dict[str, float] = {"mean": 0.0, "var": 0.0, "count": 0}

    def detect(self, value: float, timestamp: Optional[float] = None) -> BaseAnomalyResult:
        """Detect anomalies using Z-score method."""
        start_time = time.time()
        self.detection_count += 1
        
        # Get current state
        mean = self._state.get("mean", 0.0)
        var = self._state.get("var", 0.0)
        count = self._state.get("count", 0)
        
        if count <= self.learning_period:
            # During learning period - never anomalous
            std = max(var, 0.0) ** 0.5
            z_score = 0.0 if std < self.eps else (value - mean) / max(std, self.eps)
            is_anomaly = False
        else:
            # Calculate z-score using current statistics
            std = max(var, 0.0) ** 0.5
            z_score = 0.0 if std < self.eps else (value - mean) / max(std, self.eps)
            is_anomaly = abs(z_score) >= self.z_threshold and std >= self.eps
        
        if is_anomaly:
            self.anomaly_count += 1
        
        # Create explanation if anomaly detected
        explanation = None
        if is_anomaly:
            confidence = min(abs(z_score) / self.z_threshold, 1.0)
            severity = self._determine_severity(abs(z_score))
            
            explanation = self._create_explanation(
                summary=f"Z-score {z_score:.2f} exceeds threshold ±{self.z_threshold}",
                confidence=confidence,
                deviation_mag=abs(z_score) - self.z_threshold,
                deviation_dir="above" if z_score > 0 else "below",
                baseline=mean,
                suggested_actions=self._get_suggested_actions(severity, value, mean),
                severity=severity,
                method_info={
                    "z_score": f"{z_score:.3f}",
                    "threshold": f"±{self.z_threshold}",
                    "std_dev": f"{std:.3f}",
                    "data_points": str(count)
                }
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return BaseAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=min(abs(z_score) / self.z_threshold, 1.0) if std >= self.eps else 0.0,
            explanation=explanation,
            detector_method=self.name,
            processing_time_ms=processing_time,
            raw_data={
                "z_score": z_score,
                "mean": mean,
                "std": std,
                "threshold": self.z_threshold,
                "value": value,
                "learning_mode": count <= self.learning_period
            }
        )

    def update(self, value: float, timestamp: Optional[float] = None) -> None:
        """Update detector statistics with new value."""
        # Legacy update method - extract logic from original update
        result = self._legacy_update_and_detect(value)
        
        # Update internal state from legacy result
        self._state["mean"] = result.mean
        self._state["var"] = result.std ** 2
        self._state["count"] = self._state.get("count", 0) + 1

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.detection_count = 0
        self.anomaly_count = 0
        self._state = {"mean": 0.0, "var": 0.0, "count": 0}

    def get_parameters(self) -> Dict[str, float]:
        """Get current detector parameters."""
        return {
            "alpha": self.alpha,
            "z_threshold": self.z_threshold,
            "eps": self.eps,
            "learning_period": float(self.learning_period),
            "drift_update_rate": float(self.drift_update_rate)
        }

    def set_parameters(self, params: Dict[str, float]) -> None:
        """Update detector parameters."""
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "z_threshold" in params:
            self.z_threshold = params["z_threshold"]
        if "eps" in params:
            self.eps = params["eps"]
        if "learning_period" in params:
            self.learning_period = int(params["learning_period"])
        if "drift_update_rate" in params:
            self.drift_update_rate = int(params["drift_update_rate"])

    def _determine_severity(self, abs_z_score: float) -> str:
        """Determine severity based on Z-score magnitude."""
        if abs_z_score >= 5.0:
            return "critical"
        elif abs_z_score >= 4.0:
            return "high"
        elif abs_z_score >= 3.5:
            return "medium"
        else:
            return "low"

    def _get_suggested_actions(self, severity: str, value: float, mean: float) -> list[str]:
        """Get context-appropriate suggested actions."""
        actions = []
        
        if severity in ["critical", "high"]:
            actions.extend([
                "IMMEDIATE: Check sensor for malfunction",
                "Verify physical system integrity",
                "Review recent maintenance activities"
            ])
        else:
            actions.extend([
                "Monitor trend for pattern changes",
                "Check sensor calibration if pattern persists",
                "Review environmental conditions"
            ])
        
        # Add channel-specific advice
        if "temp" in self.channel.lower():
            actions.append("Check thermal management system")
        elif "voltage" in self.channel.lower() or "current" in self.channel.lower():
            actions.append("Verify electrical connections and power supply")
        
        return actions

    def _legacy_update_and_detect(self, x: float) -> LegacyAnomalyResult:
        """Legacy update method for backward compatibility."""
        st = self._state
        a = self.alpha
        
        if st.get("count", 0) == 0:
            # Initialize state for new channel
            mean = x
            var = 0.0
            count = 1
            self._state = {"mean": mean, "var": var, "count": count}
            std = 0.0
            z = 0.0
            is_anom = False
        else:
            mean_prev = st["mean"]
            var_prev = st["var"]
            count = st.get("count", self.learning_period + 1)
            
            # Calculate z-score using current statistics
            std_prev = max(var_prev, 0.0) ** 0.5
            z = 0.0 if std_prev < self.eps else (x - mean_prev) / max(std_prev, self.eps)
            is_anom_current = abs(z) >= self.z_threshold and std_prev >= self.eps
            
            # Robust update logic: only update statistics with non-anomalous points
            should_update = (
                count <= self.learning_period or  # Learning phase: update unconditionally
                not is_anom_current or            # Normal point: safe to update
                count % self.drift_update_rate == 0  # Drift handling: occasional update
            )
            
            if should_update:
                # Update statistics with this point
                mean = (1 - a) * mean_prev + a * x
                # EW variance of residuals (Welford-like EW update)
                resid = x - mean_prev
                var = (1 - a) * var_prev + a * (resid * resid)
            else:
                # Keep statistics unchanged (anomaly detected, don't pollute baseline)
                mean = mean_prev
                var = var_prev
            
            count += 1
            self._state = {"mean": mean, "var": var, "count": count}
            
            # Recalculate final z-score with potentially updated statistics
            std = max(var, 0.0) ** 0.5
            z = 0.0 if std < self.eps else (x - mean) / max(std, self.eps)
            is_anom = abs(z) >= self.z_threshold and std >= self.eps
        
        return LegacyAnomalyResult(mean=mean, std=std, zscore=z, is_anomaly=is_anom)

    # Legacy method for backward compatibility
    def update_legacy(self, channel: str, x: float) -> LegacyAnomalyResult:
        """Legacy update method that returns old-style result."""
        return self._legacy_update_and_detect(x)


class RangeGuard:
    """Simple range validator to catch saturation or impossible values."""

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val

    def check(self, x: float) -> bool:
        if self.min_val is not None and x < self.min_val:
            return True
        if self.max_val is not None and x > self.max_val:
            return True
        return False


# Register the Z-score detector with the factory
DetectorFactory.register("zscore", EWMAZScoreDetector)
DetectorFactory.register("ewma-zscore", EWMAZScoreDetector)  # Alternative name
