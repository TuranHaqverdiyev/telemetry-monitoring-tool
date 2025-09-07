from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ChannelConfig:
    name: str
    unit: str
    min: Optional[float]
    max: Optional[float]
    breach_direction: Optional[str] = None
    breach_threshold: Optional[float] = None
    group: Optional[str] = None  # Plot group name; same group plots together


@dataclass
class AppConfig:
    rate_hz: float
    alpha: float
    z_threshold: float
    channels: List[ChannelConfig]
    window_sec: int
    group_by_source: bool
    sources: List[Dict[str, Any]]
    groups: Dict[str, List[str]]  # group -> channel names


def load_config(path: str) -> AppConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rate_hz = float(data.get("rate_hz", 5.0))
    
    # Handle both old and new config structures
    det = data.get("detector", data.get("anomaly_detection", {}))
    alpha = float(det.get("alpha", 0.05))
    zt = float(det.get("z_threshold", 3.0))
    
    chans: List[ChannelConfig] = []
    groups: Dict[str, List[str]] = {}
    
    # Handle new config structure with data_sources
    if "data_sources" in data:
        for source in data["data_sources"]:
            for c in source.get("channels", []):
                grp = c.get("group")
                chans.append(
                    ChannelConfig(
                        name=c["name"],
                        unit=c.get("units", c.get("unit", "")),
                        min=c.get("min"),
                        max=c.get("max"),
                        breach_direction=c.get("direction"),
                        breach_threshold=c.get("threshold"),
                        group=grp
                    )
                )
                if grp:
                    groups.setdefault(grp, []).append(c["name"])
    else:
        # Handle old config structure
        for c in data.get("channels", []):
            b = c.get("breach") or {}
            grp = c.get("group")
            chans.append(
                ChannelConfig(
                    name=c["name"],
                    unit=c.get("unit", ""),
                    min=c.get("min"),
                    max=c.get("max"),
                    breach_direction=b.get("direction"),
                    breach_threshold=b.get("threshold"),
                    group=grp,
                )
            )
            if grp:
                groups.setdefault(grp, []).append(c["name"])
    
    # Handle window and sources for both old and new formats
    window_sec = int(data.get("window_sec", data.get("time_window", 60)))
    group_by_source = bool(data.get("group_by_source", False))
    sources = list(data.get("sources", data.get("data_sources", [])))
    
    return AppConfig(
        rate_hz=rate_hz,
        alpha=alpha,
        z_threshold=zt,
        channels=chans,
        window_sec=window_sec,
        group_by_source=group_by_source,
        sources=sources,
        groups=groups,
    )
