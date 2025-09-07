"""
IQR (Interquartile Range) Anomaly Detector

This module implements anomaly detection using the IQR method, which identifies
outliers based on the interquartile range of the data distribution. The IQR method
is robust against extreme values and doesn't assume normal distribution.

IQR Method:
- Q1 (25th percentile) and Q3 (75th percentile) define the interquartile range
- IQR = Q3 - Q1
- Lower fence = Q1 - (multiplier * IQR)
- Upper fence = Q3 + (multiplier * IQR)
- Values outside these fences are considered anomalies

Typical multiplier values:
- 1.5: Standard outlier detection (moderate sensitivity)
- 2.0: Conservative outlier detection (lower sensitivity)
- 3.0: Very conservative detection (very low sensitivity)
"""

from __future__ import annotations

import time
import statistics
from collections import deque
from typing import Dict, Optional

from .detector_base import AnomalyDetector, AnomalyResult, DetectorFactory


class IQRAnomalyDetector(AnomalyDetector):
    """
    IQR-based anomaly detector using sliding window of recent values.
    
    This detector maintains a sliding window of recent observations and uses
    the interquartile range (IQR) to identify statistical outliers.
    """
    
    def __init__(self, iqr_multiplier: float = 1.5, window_size: int = 50, 
                 min_samples: int = 10, channel: str = "default"):
        """
        Initialize IQR anomaly detector.
        
        Args:
            iqr_multiplier: Multiplier for IQR-based fence calculation (default: 1.5)
            window_size: Number of recent values to maintain for IQR calculation
            min_samples: Minimum samples needed before detection becomes active
            channel: Channel name for identification
        """
        super().__init__(f"IQR-{channel}")
        self.iqr_multiplier = iqr_multiplier
        self.window_size = window_size
        self.min_samples = min_samples
        self.channel = channel
        
        # Sliding window of recent values
        self.values = deque(maxlen=window_size)
        
        # Statistics cache (updated only when needed)
        self._stats_cache = {
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_fence": 0.0,
            "upper_fence": 0.0,
            "median": 0.0,
            "cache_valid": False
        }
    
    def detect(self, value: float, timestamp: Optional[float] = None) -> AnomalyResult:
        """
        Detect anomalies using IQR method.
        
        Args:
            value: The sensor value to analyze
            timestamp: Optional timestamp for temporal analysis
            
        Returns:
            AnomalyResult with detection decision and explanation
        """
        start_time = time.time()
        self.detection_count += 1
        
        # Need sufficient samples for reliable IQR calculation
        if len(self.values) < self.min_samples:
            # During learning period - never anomalous
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                explanation=None,
                detector_method=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                raw_data={
                    "value": value,
                    "samples_collected": len(self.values),
                    "min_samples_needed": self.min_samples,
                    "learning_mode": True
                }
            )
        
        # Calculate IQR statistics
        stats = self._calculate_iqr_stats()
        
        # Determine if value is anomalous
        is_anomaly = (value < stats["lower_fence"] or value > stats["upper_fence"])
        
        if is_anomaly:
            self.anomaly_count += 1
        
        # Calculate anomaly score (0.0 to 1.0)
        anomaly_score = self._calculate_anomaly_score(value, stats)
        
        # Create explanation if anomaly detected
        explanation = None
        if is_anomaly:
            explanation = self._create_iqr_explanation(value, stats, anomaly_score)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            explanation=explanation,
            detector_method=self.name,
            processing_time_ms=processing_time,
            raw_data={
                "value": value,
                "q1": stats["q1"],
                "q3": stats["q3"],
                "iqr": stats["iqr"],
                "lower_fence": stats["lower_fence"],
                "upper_fence": stats["upper_fence"],
                "median": stats["median"],
                "iqr_multiplier": self.iqr_multiplier,
                "window_size": len(self.values)
            }
        )
    
    def update(self, value: float, timestamp: Optional[float] = None) -> None:
        """
        Update detector's sliding window with new value.
        
        Args:
            value: The sensor value to learn from
            timestamp: Optional timestamp for temporal learning
        """
        self.values.append(value)
        # Invalidate statistics cache since window changed
        self._stats_cache["cache_valid"] = False
    
    def reset(self) -> None:
        """Reset detector to initial state."""
        self.detection_count = 0
        self.anomaly_count = 0
        self.values.clear()
        self._stats_cache["cache_valid"] = False
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current detector parameters."""
        return {
            "iqr_multiplier": self.iqr_multiplier,
            "window_size": float(self.window_size),
            "min_samples": float(self.min_samples)
        }
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Update detector parameters."""
        if "iqr_multiplier" in params:
            self.iqr_multiplier = params["iqr_multiplier"]
            self._stats_cache["cache_valid"] = False  # Invalidate cache
        
        if "window_size" in params:
            new_size = int(params["window_size"])
            if new_size != self.window_size:
                self.window_size = new_size
                # Recreate deque with new size
                old_values = list(self.values)
                self.values = deque(old_values[-new_size:], maxlen=new_size)
                self._stats_cache["cache_valid"] = False
        
        if "min_samples" in params:
            self.min_samples = int(params["min_samples"])
    
    def _calculate_iqr_stats(self) -> Dict[str, float]:
        """Calculate IQR statistics with caching."""
        if self._stats_cache["cache_valid"] and len(self.values) == self.window_size:
            return self._stats_cache
        
        # Sort values for percentile calculation
        sorted_values = sorted(self.values)
        
        # Calculate quartiles
        q1 = statistics.quantiles(sorted_values, n=4)[0]  # 25th percentile
        q3 = statistics.quantiles(sorted_values, n=4)[2]  # 75th percentile
        median = statistics.median(sorted_values)
        
        # Calculate IQR and fences
        iqr = q3 - q1
        lower_fence = q1 - (self.iqr_multiplier * iqr)
        upper_fence = q3 + (self.iqr_multiplier * iqr)
        
        # Update cache
        self._stats_cache.update({
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "median": median,
            "cache_valid": True
        })
        
        return self._stats_cache
    
    def _calculate_anomaly_score(self, value: float, stats: Dict[str, float]) -> float:
        """
        Calculate normalized anomaly score (0.0 to 1.0).
        
        Score represents how far outside the normal range the value is:
        - 0.0: Within normal range
        - 1.0: Extremely anomalous
        """
        lower_fence = stats["lower_fence"]
        upper_fence = stats["upper_fence"]
        iqr = stats["iqr"]
        
        if lower_fence <= value <= upper_fence:
            # Within normal range
            return 0.0
        
        if iqr == 0:
            # No variation in data - any deviation is significant
            return 1.0
        
        # Calculate distance beyond fence as multiple of IQR
        if value < lower_fence:
            distance = lower_fence - value
        else:  # value > upper_fence
            distance = value - upper_fence
        
        # Normalize to 0-1 scale (distance of 1 IQR = score of 0.5)
        score = min(distance / (2 * iqr), 1.0)
        return score
    
    def _create_iqr_explanation(self, value: float, stats: Dict[str, float], score: float) -> object:
        """Create detailed explanation for IQR anomaly detection."""
        q1, q3 = stats["q1"], stats["q3"]
        lower_fence, upper_fence = stats["lower_fence"], stats["upper_fence"]
        median = stats["median"]
        
        if value < lower_fence:
            direction = "below"
            fence_value = lower_fence
            deviation_mag = lower_fence - value
        else:
            direction = "above"
            fence_value = upper_fence
            deviation_mag = value - upper_fence
        
        # Determine severity based on how far beyond the fence
        severity = self._determine_iqr_severity(score)
        
        # Create summary
        summary = (f"Value {value:.3f} is {direction} IQR fence "
                  f"({fence_value:.3f}, multiplier: {self.iqr_multiplier})")
        
        # Generate suggested actions
        suggested_actions = self._get_iqr_suggested_actions(severity, direction, value, median)
        
        # Method-specific technical information
        method_info = {
            "q1_25th_percentile": f"{q1:.3f}",
            "q3_75th_percentile": f"{q3:.3f}",
            "iqr_range": f"{stats['iqr']:.3f}",
            "median": f"{median:.3f}",
            "fence_multiplier": f"{self.iqr_multiplier}",
            "samples_in_window": str(len(self.values))
        }
        
        return self._create_explanation(
            summary=summary,
            confidence=min(score * 1.2, 1.0),  # Slightly boost confidence for clear outliers
            deviation_mag=deviation_mag,
            deviation_dir=direction,
            baseline=median,
            suggested_actions=suggested_actions,
            severity=severity,
            method_info=method_info
        )
    
    def _determine_iqr_severity(self, score: float) -> str:
        """Determine severity based on anomaly score."""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_iqr_suggested_actions(self, severity: str, direction: str, value: float, median: float) -> list[str]:
        """Get IQR-specific suggested actions."""
        actions = []
        
        if severity in ["critical", "high"]:
            actions.extend([
                "IMMEDIATE: Statistical outlier detected - investigate cause",
                "Check for sensor malfunction or calibration drift",
                "Verify no external interference affecting readings"
            ])
        else:
            actions.extend([
                "Statistical anomaly detected - monitor for pattern",
                "Consider if reading represents valid operational state",
                "Check if environmental conditions have changed"
            ])
        
        # Add directional guidance
        if direction == "above":
            actions.append(f"Reading {value:.3f} significantly above typical range")
        else:
            actions.append(f"Reading {value:.3f} significantly below typical range")
        
        # Add channel-specific advice
        if "temp" in self.channel.lower():
            if direction == "above":
                actions.append("Check cooling system performance")
            else:
                actions.append("Verify heating system operation")
        elif "voltage" in self.channel.lower() or "current" in self.channel.lower():
            actions.append("Inspect electrical connections and power supply stability")
        
        return actions


# Register IQR detector with the factory
DetectorFactory.register("iqr", IQRAnomalyDetector)
DetectorFactory.register("interquartile", IQRAnomalyDetector)  # Alternative name
