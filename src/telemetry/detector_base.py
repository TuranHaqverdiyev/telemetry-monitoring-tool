"""
Abstract base classes and data structures for anomaly detection.

This module defines the common interface that all anomaly detectors must implement,
along with standardized result structures for consistent handling across the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AnomalyExplanation:
    """Structured explanation of why an anomaly was detected."""
    
    # Primary explanation
    summary: str  # Human-readable summary of the anomaly
    confidence: float  # Detection confidence (0.0 to 1.0)
    
    # Detailed breakdown
    deviation_magnitude: float  # How far from normal (method-specific units)
    deviation_direction: str  # "above", "below", or "pattern"
    baseline_value: Optional[float]  # Expected/normal value
    
    # Actionable insights
    suggested_actions: List[str]  # What operator should check
    severity: str  # "low", "medium", "high", "critical"
    
    # Technical details
    method_specific_info: Dict[str, str]  # Additional method-specific details


@dataclass 
class AnomalyResult:
    """Complete result of an anomaly detection operation."""
    
    # Core detection result
    is_anomaly: bool
    anomaly_score: float  # Normalized score (0.0 to 1.0, higher = more anomalous)
    
    # Explanation and context
    explanation: Optional[AnomalyExplanation] = None
    
    # Metadata
    detector_method: str = "unknown"
    processing_time_ms: float = 0.0
    
    # Raw method-specific data (for debugging/logging)
    raw_data: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}


class AnomalyDetector(ABC):
    """Abstract base class for all anomaly detection methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.detection_count = 0
        self.anomaly_count = 0
    
    @abstractmethod
    def detect(self, value: float, timestamp: Optional[float] = None) -> AnomalyResult:
        """
        Detect if a value is anomalous.
        
        Args:
            value: The sensor value to analyze
            timestamp: Optional timestamp for temporal analysis
            
        Returns:
            AnomalyResult with detection decision and explanation
        """
        pass
    
    @abstractmethod
    def update(self, value: float, timestamp: Optional[float] = None) -> None:
        """
        Update the detector's internal state with a new value.
        
        Args:
            value: The sensor value to learn from
            timestamp: Optional timestamp for temporal learning
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the detector to initial state."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Get current detector parameters for configuration."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, float]) -> None:
        """Update detector parameters dynamically."""
        pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get detector performance statistics."""
        detection_rate = (self.anomaly_count / max(self.detection_count, 1)) * 100
        return {
            "total_detections": self.detection_count,
            "anomaly_count": self.anomaly_count,
            "detection_rate_percent": detection_rate
        }
    
    def _create_explanation(self, summary: str, confidence: float, 
                          deviation_mag: float, deviation_dir: str,
                          baseline: Optional[float] = None,
                          suggested_actions: Optional[List[str]] = None,
                          severity: str = "medium",
                          method_info: Optional[Dict[str, str]] = None) -> AnomalyExplanation:
        """Helper to create standardized explanations."""
        
        if suggested_actions is None:
            suggested_actions = [
                f"Check {self.name.lower()} sensor readings",
                "Verify system normal operation",
                "Review recent maintenance logs"
            ]
        
        if method_info is None:
            method_info = {}
            
        return AnomalyExplanation(
            summary=summary,
            confidence=confidence,
            deviation_magnitude=deviation_mag,
            deviation_direction=deviation_dir,
            baseline_value=baseline,
            suggested_actions=suggested_actions,
            severity=severity,
            method_specific_info=method_info
        )


class DetectorFactory:
    """Factory for creating anomaly detectors."""
    
    _detectors = {}
    
    @classmethod
    def register(cls, name: str, detector_class):
        """Register a detector class."""
        cls._detectors[name] = detector_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> AnomalyDetector:
        """Create a detector instance by name."""
        if name not in cls._detectors:
            available = list(cls._detectors.keys())
            raise ValueError(f"Unknown detector: {name}. Available: {available}")
        
        return cls._detectors[name](**kwargs)
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available detection methods."""
        return list(cls._detectors.keys())
