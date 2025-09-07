"""
Modified Z-Score Anomaly Detector using Median Absolute Deviation (MAD)

This detector uses the Median Absolute Deviation instead of standard deviation,
making it more robust to outliers and extreme values compared to traditional Z-score.
Particularly effective for data with non-normal distributions or contaminated datasets.

Author: GitHub Copilot
"""

import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List
from collections import deque

from .detector_base import AnomalyDetector, AnomalyResult as BaseAnomalyResult, DetectorFactory

logger = logging.getLogger(__name__)

class ModifiedZScoreDetector(AnomalyDetector):
    """
    Modified Z-Score anomaly detector using Median Absolute Deviation (MAD).
    
    Uses median and MAD instead of mean and standard deviation, making it
    more robust to outliers and suitable for non-normal distributions.
    
    Formula: Modified Z-Score = 0.6745 * (X - median) / MAD
    where MAD = median(|X - median|)
    """
    
    def __init__(self, mad_threshold: float = 3.5, window_size: int = 100, 
                 min_samples: int = 10, channel: str = "default"):
        super().__init__(f"ModifiedZScore-{channel}")
        
        # Store channel for logging
        self.channel = channel
        
        # Configuration parameters
        self.mad_threshold = mad_threshold  # Threshold for modified Z-score
        self.window_size = window_size      # Size of rolling window
        self.min_samples = min_samples      # Minimum samples before detection
        
        # Data storage - using deque for efficient sliding window
        self.data_window: deque = deque(maxlen=window_size)
        
        # Statistics tracking
        self.current_median = 0.0
        self.current_mad = 0.0
        self.sample_count = 0
        
        # Constants
        self.mad_scaling_factor = 0.6745  # Makes MAD comparable to std dev for normal data
        
        logger.info(f"Initialized Modified Z-Score detector {self.name}")
        
    def update(self, value: float, timestamp: Optional[float] = None) -> None:
        """Update detector state with new value (called before detect)"""
        if timestamp is None:
            timestamp = time.time()
            
        # Add to sliding window
        self.data_window.append(value)
        self.sample_count += 1
        
        # Update statistics if we have enough samples
        if len(self.data_window) >= self.min_samples:
            self._update_statistics()
            
    def detect(self, value: float, timestamp: Optional[float] = None) -> BaseAnomalyResult:
        """
        Detect anomalies using Modified Z-Score method.
        
        Returns:
            BaseAnomalyResult with detection decision and explanation
        """
        if timestamp is None:
            timestamp = time.time()
            
        start_time = time.time()
        self.detection_count += 1
        
        # Need minimum samples for reliable detection
        if len(self.data_window) < self.min_samples:
            return BaseAnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                explanation=None,
                detector_method=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                raw_data={
                    "samples": float(len(self.data_window)),
                    "min_required": float(self.min_samples)
                }
            )
            
        try:
            # Calculate modified Z-score
            if self.current_mad == 0:
                # Handle case where MAD is zero (all values identical)
                modified_z_score = 0.0
            else:
                modified_z_score = self.mad_scaling_factor * (value - self.current_median) / self.current_mad
                
            # Check if anomaly
            is_anomaly = abs(modified_z_score) > self.mad_threshold
            
            if is_anomaly:
                self.anomaly_count += 1
                
            # Calculate confidence (how far beyond threshold)
            confidence = min(1.0, abs(modified_z_score) / self.mad_threshold) if self.mad_threshold > 0 else 0.0
            
            # Determine severity based on modified Z-score magnitude
            abs_score = abs(modified_z_score)
            if abs_score > 5.0:
                severity = "critical"
            elif abs_score > 4.0:
                severity = "high"
            elif abs_score > 3.5:
                severity = "medium"
            else:
                severity = "low"
                
            # Create explanation if anomaly detected
            explanation = None
            if is_anomaly:
                explanation = self._create_explanation(
                    summary=f"Modified Z-Score {modified_z_score:.2f} exceeds threshold ±{self.mad_threshold}",
                    confidence=confidence,
                    deviation_mag=abs(modified_z_score) - self.mad_threshold,
                    deviation_dir="above" if modified_z_score > 0 else "below",
                    baseline=self.current_median,
                    suggested_actions=self._get_suggested_actions(severity, value),
                    severity=severity,
                    method_info={
                        "modified_z_score": f"{modified_z_score:.3f}",
                        "threshold": f"±{self.mad_threshold}",
                        "median": f"{self.current_median:.3f}",
                        "mad": f"{self.current_mad:.3f}",
                        "data_points": str(len(self.data_window))
                    }
                )
                
            processing_time = (time.time() - start_time) * 1000
            
            return BaseAnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=confidence if is_anomaly else 0.0,
                explanation=explanation,
                detector_method=self.name,
                processing_time_ms=processing_time,
                raw_data={
                    "modified_z_score": modified_z_score,
                    "median": self.current_median,
                    "mad": self.current_mad,
                    "threshold": self.mad_threshold,
                    "value": value,
                    "samples": float(len(self.data_window))
                }
            )
                
        except Exception as e:
            logger.error(f"Error in modified Z-score detection for {self.name}: {e}")
            processing_time = (time.time() - start_time) * 1000
            return BaseAnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                explanation=None,
                detector_method=self.name,
                processing_time_ms=processing_time,
                raw_data={}
            )
            
    def reset(self) -> None:
        """Reset detector state"""
        self.data_window.clear()
        self.current_median = 0.0
        self.current_mad = 0.0
        self.sample_count = 0
        logger.info(f"Reset modified Z-score detector {self.name}")
        
    def get_parameters(self) -> Dict[str, float]:
        """Get current detector parameters for configuration."""
        return {
            "mad_threshold": self.mad_threshold,
            "window_size": float(self.window_size),
            "min_samples": float(self.min_samples)
        }
        
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Update detector parameters dynamically."""
        for key, value in params.items():
            if key == "mad_threshold":
                self.mad_threshold = value
                logger.info(f"Updated MAD threshold to {value} for {self.name}")
            elif key == "window_size":
                new_size = int(value)
                if new_size != self.window_size:
                    self.window_size = new_size
                    # Create new deque with new size
                    old_data = list(self.data_window)
                    self.data_window = deque(old_data[-new_size:], maxlen=new_size)
                    logger.info(f"Updated window size to {new_size} for {self.name}")
            elif key == "min_samples":
                self.min_samples = int(value)
                logger.info(f"Updated min samples to {int(value)} for {self.name}")
            else:
                logger.warning(f"Unknown parameter: {key}")
                
    def _update_statistics(self) -> None:
        """Update median and MAD statistics from current window"""
        if len(self.data_window) == 0:
            return
            
        data_array = np.array(self.data_window)
        
        # Calculate median
        self.current_median = float(np.median(data_array))
        
        # Calculate MAD (Median Absolute Deviation)
        absolute_deviations = np.abs(data_array - self.current_median)
        self.current_mad = float(np.median(absolute_deviations))
        
        # Handle edge case where MAD is 0 (all values are identical)
        if self.current_mad == 0:
            # Use a small epsilon to avoid division by zero
            self.current_mad = 1e-10
            
    def _get_suggested_actions(self, severity: str, value: float) -> List[str]:
        """Get suggested actions based on severity"""
        base_actions = [
            f"Check {self.channel} sensor readings",
            "Verify system normal operation",
            "Compare with historical data patterns"
        ]
        
        if severity == "critical":
            return [
                "IMMEDIATE ACTION REQUIRED",
                "Stop operations if safety critical",
                "Contact technical support immediately",
                f"Value {value:.3f} significantly deviates from median {self.current_median:.3f}",
            ] + base_actions
        elif severity == "high":
            return [
                "Investigation recommended within 1 hour",
                "Review recent system changes",
                f"Significant deviation detected (median: {self.current_median:.3f})",
            ] + base_actions
        elif severity == "medium":
            return [
                "Monitor closely for pattern confirmation",
                "Schedule maintenance check if pattern persists",
                f"Notable deviation from median baseline",
            ] + base_actions
        else:
            return [
                "Continue monitoring",
                "Log for trend analysis",
                "Consider if this represents new normal operating range",
            ] + base_actions
            
    def get_detector_info(self) -> Dict[str, Any]:
        """Get current detector information and statistics"""
        return {
            "type": "modified-zscore",
            "channel": self.channel,
            "config": {
                "mad_threshold": self.mad_threshold,
                "window_size": self.window_size,
                "min_samples": self.min_samples
            },
            "current_stats": {
                "median": self.current_median,
                "mad": self.current_mad,
                "data_points": len(self.data_window),
                "total_samples": self.sample_count
            },
            "advantages": [
                "Robust to outliers",
                "Works with non-normal distributions",
                "Less sensitive to extreme values than traditional Z-score",
                "Stable median-based baseline"
            ]
        }

# Register the detector with the factory
DetectorFactory.register("modified-zscore", ModifiedZScoreDetector)
DetectorFactory.register("mad-zscore", ModifiedZScoreDetector)  # Alternative name
DetectorFactory.register("robust-zscore", ModifiedZScoreDetector)  # Descriptive name
