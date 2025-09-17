"""
Isolation Forest Anomaly Detector

Machine learning-based anomaly detection using the Isolation Forest algorithm.
Particularly effective for multivariate anomalies and complex patterns that
traditional statistical methods might miss.
"""

import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

from .detector_base import AnomalyDetector, AnomalyResult as BaseAnomalyResult, DetectorFactory

logger = logging.getLogger(__name__)

class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest anomaly detector for multivariate anomaly detection.
    
    Uses ensemble of isolation trees to identify anomalies by measuring
    the path length required to isolate each point. Anomalies require
    fewer splits to isolate, resulting in shorter paths.
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, 
                 window_size: int = 100, min_samples: int = 50, 
                 retrain_interval: int = 20, anomaly_threshold: float = -0.5,
                 channel: str = "default"):
        super().__init__(f"IsolationForest-{channel}")
        
        # Store channel for logging
        self.channel = channel
        
        # Configuration
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval
        self.anomaly_threshold = anomaly_threshold
        
        # Core components
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        
        # Data management
        self.data_window: List[Tuple[float, float]] = []  # (timestamp, value) pairs
        self.feature_window: List[List[float]] = []  # Feature vectors for training
        self.sample_count = 0
        self.last_retrain = 0
        
        # Model state
        self.is_trained = False
        self.training_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 0.0,
            "samples_used": 0
        }
        
        logger.info(f"Initialized Isolation Forest detector {self.name}")
        
    def update(self, value: float, timestamp: Optional[float] = None) -> None:
        """Update detector state with new value (called before detect)"""
        if timestamp is None:
            timestamp = time.time()
            
        # Add to data window
        self.data_window.append((timestamp, value))
        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)
            
        # Extract features
        features = self._extract_features(timestamp, value)
        self.feature_window.append(features)
        if len(self.feature_window) > self.window_size:
            self.feature_window.pop(0)
            
        self.sample_count += 1
        
        # Check if we need to retrain
        if (not self.is_trained or 
            (self.sample_count - self.last_retrain) >= self.retrain_interval):
            self._retrain_model()
            
    def detect(self, value: float, timestamp: Optional[float] = None) -> BaseAnomalyResult:
        """
        Detect anomalies using the Isolation Forest model.
        
        Returns:
            BaseAnomalyResult with detection decision and explanation
        """
        if timestamp is None:
            timestamp = time.time()
            
        start_time = time.time()
        self.detection_count += 1
        
        # Can't detect without trained model
        if not self.is_trained or self.model is None:
            return BaseAnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                explanation=None,
                detector_method=self.name,
                processing_time_ms=(time.time() - start_time) * 1000,
                raw_data={}
            )
            
        try:
            # Get current features (should already be extracted in update)
            if not self.feature_window:
                return BaseAnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    explanation=None,
                    detector_method=self.name,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    raw_data={}
                )
                
            features = self.feature_window[-1]  # Most recent features
            
            # Prepare current features for prediction
            X_current = np.array([features])
            X_scaled = self.scaler.transform(X_current)
            
            # Get anomaly score and prediction
            anomaly_score = self.model.decision_function(X_scaled)[0]
            is_anomaly = self.model.predict(X_scaled)[0] == -1
            
            # Apply threshold check
            is_final_anomaly = anomaly_score <= self.anomaly_threshold or is_anomaly
            
            if is_final_anomaly:
                self.anomaly_count += 1
                
            # Calculate confidence based on how far below threshold
            confidence = min(1.0, abs(anomaly_score - self.anomaly_threshold) / 0.5)
            
            # Determine severity based on anomaly score
            if anomaly_score <= -0.8:
                severity = "critical"
            elif anomaly_score <= -0.6:
                severity = "high"
            elif anomaly_score <= -0.4:
                severity = "medium"
            else:
                severity = "low"
                
            # Create explanation
            explanation = None
            if is_final_anomaly:
                explanation = self._create_explanation(
                    summary=f"Isolation Forest detected {severity} anomaly",
                    confidence=confidence,
                    deviation_mag=abs(anomaly_score - self.anomaly_threshold),
                    deviation_dir="isolated",
                    baseline=self.anomaly_threshold,
                    suggested_actions=self._get_suggested_actions(severity, value),
                    severity=severity,
                    method_info={
                        "anomaly_score": f"{anomaly_score:.4f}",
                        "threshold": f"{self.anomaly_threshold}",
                        "features_used": str(len(features)),
                        "training_samples": str(self.training_stats["samples_used"])
                    }
                )
                
            processing_time = (time.time() - start_time) * 1000
            
            return BaseAnomalyResult(
                is_anomaly=is_final_anomaly,
                anomaly_score=abs(anomaly_score) if is_final_anomaly else 0.0,
                explanation=explanation,
                detector_method=self.name,
                processing_time_ms=processing_time,
                raw_data={
                    "raw_score": anomaly_score,
                    "threshold": self.anomaly_threshold,
                    "features_used": float(len(features)),
                    "training_samples": float(self.training_stats["samples_used"]),
                    "is_trained": float(self.is_trained)
                }
            )
                
        except Exception as e:
            logger.error(f"Error in isolation forest detection for {self.name}: {e}")
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
        self.feature_window.clear()
        self.model = None
        self.is_trained = False
        self.sample_count = 0
        self.last_retrain = 0
        self.training_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 0.0,
            "samples_used": 0
        }
        logger.info(f"Reset isolation forest detector {self.name}")
        
    def get_parameters(self) -> Dict[str, float]:
        """Get current detector parameters for configuration."""
        return {
            "contamination": self.contamination,
            "n_estimators": float(self.n_estimators),
            "window_size": float(self.window_size),
            "min_samples": float(self.min_samples),
            "retrain_interval": float(self.retrain_interval),
            "anomaly_threshold": self.anomaly_threshold
        }
        
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Update detector parameters dynamically."""
        for key, value in params.items():
            if key == "contamination":
                self.contamination = value
                self._mark_for_retrain()
            elif key == "n_estimators":
                self.n_estimators = int(value)
                self._mark_for_retrain()
            elif key == "window_size":
                self.window_size = int(value)
            elif key == "min_samples":
                self.min_samples = int(value)
            elif key == "retrain_interval":
                self.retrain_interval = int(value)
            elif key == "anomaly_threshold":
                self.anomaly_threshold = value
            else:
                logger.warning(f"Unknown parameter: {key}")
                
    def _mark_for_retrain(self) -> None:
        """Mark model for retraining on next detection"""
        self.is_trained = False
        logger.debug(f"Marked isolation forest model for retraining: {self.name}")
        
    def _extract_features(self, timestamp: float, value: float) -> List[float]:
        """
        Extract feature vector from current and historical data.
        
        Features include:
        - Current value
        - Recent trend (slope)
        - Local variance
        - Time-based features
        - Moving averages
        """
        features = [value]  # Base feature
        
        if len(self.data_window) < 2:
            # Not enough history, use basic features
            return [value, 0.0, 0.0, 0.0, 0.0]
            
        # Get recent values for trend calculation
        recent_data = self.data_window[-10:]  # Last 10 points
        values = [v for _, v in recent_data]
        timestamps = [t for t, _ in recent_data]
        
        # Trend feature (slope of recent points)
        if len(values) >= 2:
            time_diffs = np.diff(timestamps)
            value_diffs = np.diff(values)
            if np.sum(time_diffs) > 0:
                trend = float(np.sum(value_diffs) / np.sum(time_diffs))
            else:
                trend = 0.0
        else:
            trend = 0.0
        features.append(trend)
        
        # Local variance feature
        if len(values) >= 3:
            local_variance = float(np.var(values))
        else:
            local_variance = 0.0
        features.append(local_variance)
        
        # Moving average deviation
        if len(values) >= 5:
            moving_avg = float(np.mean(values[-5:]))
            ma_deviation = float(value - moving_avg)
        else:
            ma_deviation = 0.0
        features.append(ma_deviation)
        
        # Time-based feature (hour of day as sine wave)
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        hour_radians = 2 * np.pi * dt.hour / 24
        time_feature = float(np.sin(hour_radians))
        features.append(time_feature)
        
        return features
        
    def _retrain_model(self) -> None:
        """Retrain the Isolation Forest model with current data"""
        if len(self.feature_window) < self.min_samples:
            return
            
        try:
            # Prepare training data
            X = np.array(self.feature_window)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and train model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress sklearn warnings
                
                self.model = IsolationForest(
                    contamination=self.contamination,
                    n_estimators=self.n_estimators,
                    max_samples="auto",
                    max_features=1.0,
                    bootstrap=False,
                    random_state=42,  # For reproducibility
                    n_jobs=1  # Single thread for stability
                )
                
                self.model.fit(X_scaled)
                
            # Update training statistics
            values = [v for _, v in self.data_window]
            self.training_stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "samples_used": len(X)
            }
            
            self.is_trained = True
            self.last_retrain = self.sample_count
            logger.info(f"Retrained isolation forest model for {self.name} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train isolation forest model for {self.name}: {e}")
            self.is_trained = False
            
    def _get_suggested_actions(self, severity: str, value: float) -> List[str]:
        """Get suggested actions based on severity"""
        base_actions = [
            f"Check {self.name.lower()} sensor readings",
            "Verify system normal operation"
        ]
        
        if severity == "critical":
            return [
                "IMMEDIATE ACTION REQUIRED",
                "Stop operations if safety critical",
                "Contact technical support",
            ] + base_actions
        elif severity == "high":
            return [
                "Investigation recommended within 1 hour",
                "Review recent system changes",
            ] + base_actions
        elif severity == "medium":
            return [
                "Monitor closely for pattern confirmation",
                "Schedule maintenance check",
            ] + base_actions
        else:
            return [
                "Continue monitoring",
                "Log for trend analysis",
            ] + base_actions

# Register the detector with the factory
DetectorFactory.register("isolation-forest", IsolationForestDetector)
DetectorFactory.register("iforest", IsolationForestDetector)  # Alternative name
