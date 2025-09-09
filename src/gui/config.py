from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import alert configuration
try:
    from alerts.alert_config import AlertConfig
except ImportError:
    # Fallback if alerts module not available
    AlertConfig = None


@dataclass
class DetectorConfig:
    """Configuration for a specific anomaly detector method."""
    method: str  # "zscore", "iqr", "isolation_forest", etc.
    parameters: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ChannelConfig:
    name: str
    unit: str
    min: Optional[float]
    max: Optional[float]
    breach_direction: Optional[str] = None
    breach_threshold: Optional[float] = None
    group: Optional[str] = None  # Plot group name; same group plots together
    detectors: List[DetectorConfig] = field(default_factory=list)  # Multiple detector methods


@dataclass
class AppConfig:
    rate_hz: float
    alpha: float  # Legacy Z-score parameter
    z_threshold: float  # Legacy Z-score parameter
    channels: List[ChannelConfig]
    window_sec: int
    group_by_source: bool
    sources: List[Dict[str, Any]]
    groups: Dict[str, List[str]]  # group -> channel names
    
    # New multi-method detector configuration
    default_detectors: List[DetectorConfig] = field(default_factory=list)
    detector_selection_mode: str = "first"  # "first", "majority", "any", "all"
    
    # Alert system configuration
    alerts: Optional[Any] = None  # Will be AlertConfig if available


def load_config(path: str) -> AppConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rate_hz = float(data.get("rate_hz", 5.0))
    
    # Handle both old and new config structures
    det = data.get("detector", data.get("anomaly_detection", {}))
    alpha = float(det.get("alpha", 0.05))
    zt = float(det.get("z_threshold", 3.0))
    
    # Load default detectors configuration
    default_detectors = _load_detectors_config(data.get("detectors", []))
    if not default_detectors:
        # If no detectors specified, use legacy Z-score as default
        default_detectors = [DetectorConfig(
            method="zscore",
            parameters={
                "alpha": alpha,
                "z_threshold": zt,
                "eps": 1e-6,
                "learning_period": 20.0,
                "drift_update_rate": 100.0
            },
            enabled=True
        )]
    
    detector_selection_mode = data.get("detector_selection_mode", "first")
    
    chans: List[ChannelConfig] = []
    groups: Dict[str, List[str]] = {}
    
    # Handle config structure - check both data_sources and top-level channels
    if "data_sources" in data:
        for source in data["data_sources"]:
            for c in source.get("channels", []):
                grp = c.get("group")
                
                # Load channel-specific detectors or use defaults
                channel_detectors = _load_detectors_config(c.get("detectors", []))
                if not channel_detectors:
                    channel_detectors = default_detectors.copy()
                
                chans.append(
                    ChannelConfig(
                        name=c["name"],
                        unit=c.get("units", c.get("unit", "")),
                        min=c.get("min"),
                        max=c.get("max"),
                        breach_direction=c.get("direction"),
                        breach_threshold=c.get("threshold"),
                        group=grp,
                        detectors=channel_detectors
                    )
                )
                if grp:
                    groups.setdefault(grp, []).append(c["name"])
    
    # Also check for top-level channels (new hybrid format)
    if "channels" in data:
        for c in data.get("channels", []):
            # Check if this channel was already added from data_sources
            if any(ch.name == c.get("name") for ch in chans):
                continue
                
            b = c.get("breach", c.get("forecast_threshold", {}))
            grp = c.get("group")
            
            # Load channel-specific detectors or use defaults
            channel_detectors = _load_detectors_config(c.get("detectors", []))
            if not channel_detectors:
                channel_detectors = default_detectors.copy()
            
            chans.append(
                ChannelConfig(
                    name=c["name"],
                    unit=c.get("unit", ""),
                    min=c.get("min"),
                    max=c.get("max"),
                    breach_direction=b.get("direction"),
                    breach_threshold=b.get("threshold", b.get("value")),
                    group=grp,
                    detectors=channel_detectors
                )
            )
            if grp:
                groups.setdefault(grp, []).append(c["name"])
    
    # Fallback to old config structure if no channels found yet
    if not chans and "channels" not in data:
        # Handle old config structure
        for c in data.get("channels", []):
            b = c.get("breach") or {}
            grp = c.get("group")
            
            # Load channel-specific detectors or use defaults
            channel_detectors = _load_detectors_config(c.get("detectors", []))
            if not channel_detectors:
                channel_detectors = default_detectors.copy()
            
            chans.append(
                ChannelConfig(
                    name=c["name"],
                    unit=c.get("unit", ""),
                    min=c.get("min"),
                    max=c.get("max"),
                    breach_direction=b.get("direction"),
                    breach_threshold=b.get("threshold"),
                    group=grp,
                    detectors=channel_detectors
                )
            )
            if grp:
                groups.setdefault(grp, []).append(c["name"])
    
    # Handle window and sources for both old and new formats
    window_sec = int(data.get("window_sec", data.get("time_window", 60)))
    group_by_source = bool(data.get("group_by_source", False))
    sources = list(data.get("sources", data.get("data_sources", [])))
    
    # Load alert configuration
    alerts = None
    if AlertConfig and "alerts" in data:
        try:
            alerts = AlertConfig.from_dict(data["alerts"])
        except Exception as e:
            print(f"Warning: Failed to load alert configuration: {e}")
    
    return AppConfig(
        rate_hz=rate_hz,
        alpha=alpha,
        z_threshold=zt,
        channels=chans,
        window_sec=window_sec,
        group_by_source=group_by_source,
        sources=sources,
        groups=groups,
        default_detectors=default_detectors,
        detector_selection_mode=detector_selection_mode,
        alerts=alerts
    )


def _load_detectors_config(detectors_data: List[Dict[str, Any]]) -> List[DetectorConfig]:
    """Load detector configurations from JSON data."""
    detectors = []
    
    for det_data in detectors_data:
        if isinstance(det_data, dict):
            method = det_data.get("method", "zscore")
            parameters = det_data.get("parameters", {})
            enabled = det_data.get("enabled", True)
            
            # Convert parameter values to appropriate types
            typed_params = {}
            for key, value in parameters.items():
                try:
                    # Integer parameters for IQR detector
                    if key in ["window_size", "min_samples", "learning_period", "drift_update_rate"]:
                        typed_params[key] = int(value)
                    else:
                        # Default to float for other numeric parameters
                        typed_params[key] = float(value)
                except (ValueError, TypeError):
                    typed_params[key] = value  # Keep non-numeric values as-is
            
            detectors.append(DetectorConfig(
                method=method,
                parameters=typed_params,
                enabled=enabled
            ))
        elif isinstance(det_data, str):
            # Simple string format - just method name with defaults
            detectors.append(DetectorConfig(method=det_data, enabled=True))
    
    return detectors


def save_config(config: AppConfig, path: str) -> None:
    """Save configuration to JSON file."""
    config_data = {
        "rate_hz": config.rate_hz,
        "detector": {
            "alpha": config.alpha,
            "z_threshold": config.z_threshold
        },
        "detectors": [
            {
                "method": det.method,
                "parameters": det.parameters,
                "enabled": det.enabled
            }
            for det in config.default_detectors
        ],
        "detector_selection_mode": config.detector_selection_mode,
        "channels": [
            {
                "name": ch.name,
                "unit": ch.unit,
                "min": ch.min,
                "max": ch.max,
                "breach": {
                    "threshold": ch.breach_threshold,
                    "direction": ch.breach_direction
                } if ch.breach_threshold is not None else None,
                "group": ch.group,
                "detectors": [
                    {
                        "method": det.method,
                        "parameters": det.parameters,
                        "enabled": det.enabled
                    }
                    for det in ch.detectors
                ] if ch.detectors else None
            }
            for ch in config.channels
        ],
        "window_sec": config.window_sec,
        "group_by_source": config.group_by_source,
        "sources": config.sources
    }
    
    # Remove None values for cleaner JSON
    config_data = _remove_none_values(config_data)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


def _remove_none_values(obj):
    """Recursively remove None values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: _remove_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_remove_none_values(item) for item in obj if item is not None]
    else:
        return obj
