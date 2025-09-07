"""
Anomaly model and agent utilities.

Concepts:
- An anomaly is an observation that deviates from expected behavior. We quantify deviation via a Z-score from an EWMA model.
- Estimation includes a tail probability (p-value) under a Gaussian assumption and an ETA to breach using forecasting.
- Severity policy mixes current deviation (|z|) and forecasted time-to-breach.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Optional


def gaussian_tail_pvalue(abs_z: float) -> float:
    """Two-sided tail probability for a standard normal at |z|.

    Uses 1 - erf for numerical stability.
    """
    # Two-tailed p-value: p = 2 * (1 - Phi(|z|)) ; Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    return max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(abs_z / sqrt(2.0))))))


@dataclass
class AnomalyAssessment:
    severity: str  # info, warn, critical
    pvalue: float
    eta_sec: Optional[float]
    threshold: Optional[float]
    direction: Optional[str]


def severity_from(abs_z: float, z_threshold: float, eta_sec: Optional[float]) -> str:
    """Rule-based severity classifier.

    - critical: |z| >= 2*z_threshold or ETA <= 10s
    - warn:     |z| >= z_threshold or ETA <= 30s
    - info:     otherwise
    """
    if eta_sec is not None and eta_sec <= 10:
        return "critical"
    if abs_z >= max(1.0, 2.0 * z_threshold):
        return "critical"
    if eta_sec is not None and eta_sec <= 30:
        return "warn"
    if abs_z >= z_threshold:
        return "warn"
    return "info"


def assess(abs_z: float, z_threshold: float, eta_sec: Optional[float], threshold: Optional[float], direction: Optional[str]) -> AnomalyAssessment:
    return AnomalyAssessment(
        severity=severity_from(abs_z, z_threshold, eta_sec),
        pvalue=gaussian_tail_pvalue(abs_z),
        eta_sec=eta_sec,
        threshold=threshold,
        direction=direction,
    )
