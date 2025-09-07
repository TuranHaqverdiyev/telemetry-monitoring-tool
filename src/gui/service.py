from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from PySide6.QtCore import QObject, QThread, Signal, Slot  # type: ignore

from src.telemetry.simulator import TelemetrySimulator
from src.telemetry.anomaly import EWMAZScoreDetector, RangeGuard
from src.telemetry.iqr_detector import DetectorFactory
from src.telemetry.detector_base import AnomalyDetector, AnomalyResult
from src.telemetry.forecast import make_forecast_summary
from src.telemetry.model import assess
from .config import AppConfig, DetectorConfig
from pathlib import Path
import csv
from src.ingest.sources import file_tail, udp_json_stream


@dataclass
class AnomalyEvent:
    ts: float
    channel: str
    value: float
    z: float
    mean: float
    std: float
    # New fields for multi-method support
    detector_method: str = "zscore"
    anomaly_score: float = 0.0
    explanation_summary: str = ""
    confidence: float = 0.0
    severity: str = "medium"


@dataclass
class MultiMethodDetectionResult:
    """Result from running multiple detection methods on a value."""
    channel: str
    value: float
    timestamp: float
    results: List[AnomalyResult]
    final_decision: bool
    combined_score: float
    primary_explanation: Optional[str] = None
    detection_summary: str = ""


class StreamWorker(QObject):
    reading = Signal(float, dict)  # ts, values
    anomaly = Signal(object)       # AnomalyEvent or dict with assessment
    forecast = Signal(str, float, float)  # channel, threshold, eta_sec
    channel_stats = Signal(str, float, float)  # channel, mean, std (for Z-score visualization)
    multi_method_result = Signal(object)  # MultiMethodDetectionResult

    def __init__(self, cfg: AppConfig, log_csv_path: Optional[str] = "anomalies.csv"):
        super().__init__()
        self.cfg = cfg
        self._stop = False
        self._paused = False
        self._last_eta = {}

        # Multi-method detector setup
        self.channel_detectors: Dict[str, List[AnomalyDetector]] = {}
        self.detector_selection_mode = getattr(cfg, "detector_selection_mode", "first")
        
        # Initialize detectors for each channel
        self._initialize_detectors()
        
        # Legacy single detector for backward compatibility
        self.det = EWMAZScoreDetector(alpha=cfg.alpha, z_threshold=cfg.z_threshold)
        self.guards = {ch.name: RangeGuard(ch.min, ch.max) for ch in cfg.channels}

        # histories and windowing/logging config
        self.hist = {ch.name: [] for ch in cfg.channels}
        self.ts_hist = []
        self.window_sec = int(getattr(cfg, "window_sec", 60))
        self.group_by_source = bool(getattr(cfg, "group_by_source", False))
        self._csv_path = log_csv_path
        self._csv_file = None
        self._csv_writer = None
        
        # Store recent readings for new plot initialization
        self._recent_readings = []  # List of (timestamp, values) tuples
        self._max_recent_readings = 1000  # Keep last 1000 readings

    def _initialize_detectors(self):
        """Initialize detectors for each channel based on configuration."""
        for channel_config in self.cfg.channels:
            channel_name = channel_config.name
            detectors = []
            
            # Use channel-specific detectors if available, otherwise use defaults
            detector_configs = channel_config.detectors if channel_config.detectors else self.cfg.default_detectors
            
            for detector_config in detector_configs:
                if detector_config.enabled:
                    try:
                        detector = DetectorFactory.create(
                            detector_config.method,
                            channel=channel_name,
                            **detector_config.parameters
                        )
                        detectors.append(detector)
                        print(f"Initialized {detector_config.method} detector for channel {channel_name}")
                    except Exception as e:
                        print(f"Warning: Failed to initialize {detector_config.method} detector for {channel_name}: {e}")
                        # Fallback to Z-score if factory creation fails
                        if detector_config.method != "zscore":
                            try:
                                fallback_detector = EWMAZScoreDetector(
                                    alpha=self.cfg.alpha,
                                    z_threshold=self.cfg.z_threshold,
                                    channel=channel_name
                                )
                                detectors.append(fallback_detector)
                                print(f"Using Z-score fallback for channel {channel_name}")
                            except Exception:
                                pass
            
            # If no detectors were successfully created, use legacy Z-score
            if not detectors:
                try:
                    fallback_detector = EWMAZScoreDetector(
                        alpha=self.cfg.alpha,
                        z_threshold=self.cfg.z_threshold,
                        channel=channel_name
                    )
                    detectors.append(fallback_detector)
                    print(f"Using legacy Z-score detector for channel {channel_name}")
                except Exception as e:
                    print(f"Error: Could not create any detector for channel {channel_name}: {e}")
            
            self.channel_detectors[channel_name] = detectors

    def _run_multi_method_detection(self, channel: str, value: float, timestamp: float) -> MultiMethodDetectionResult:
        """Run multiple detection methods on a value and combine results."""
        detectors = self.channel_detectors.get(channel, [])
        results = []
        
        for detector in detectors:
            try:
                # Update detector state
                detector.update(value, timestamp)
                # Run detection
                result = detector.detect(value, timestamp)
                results.append(result)
            except Exception as e:
                print(f"Warning: Detector {detector.name} failed on {channel}: {e}")
                continue
        
        # Combine results based on selection mode
        final_decision, combined_score, primary_explanation, summary = self._combine_detection_results(results)
        
        return MultiMethodDetectionResult(
            channel=channel,
            value=value,
            timestamp=timestamp,
            results=results,
            final_decision=final_decision,
            combined_score=combined_score,
            primary_explanation=primary_explanation,
            detection_summary=summary
        )

    def _combine_detection_results(self, results: List[AnomalyResult]) -> tuple[bool, float, Optional[str], str]:
        """
        Combine multiple detection results based on the configured selection mode.
        
        Returns:
            (final_decision, combined_score, primary_explanation, summary)
        """
        if not results:
            return False, 0.0, None, "No detectors available"
        
        anomaly_results = [r for r in results if r.is_anomaly]
        normal_results = [r for r in results if not r.is_anomaly]
        
        if self.detector_selection_mode == "first":
            # Use first detector's result
            first_result = results[0]
            explanation = first_result.explanation.summary if first_result.explanation else None
            summary = f"{first_result.detector_method}: {'ANOMALY' if first_result.is_anomaly else 'Normal'}"
            return first_result.is_anomaly, first_result.anomaly_score, explanation, summary
        
        elif self.detector_selection_mode == "majority":
            # Majority vote
            decision = len(anomaly_results) > len(normal_results)
            avg_score = sum(r.anomaly_score for r in results) / len(results)
            if anomaly_results:
                # Use explanation from highest-confidence anomaly detector
                best_anomaly = max(anomaly_results, key=lambda r: r.explanation.confidence if r.explanation else 0)
                explanation = best_anomaly.explanation.summary if best_anomaly.explanation else None
            else:
                explanation = None
            summary = f"Majority vote: {len(anomaly_results)}/{len(results)} detected anomaly"
            return decision, avg_score, explanation, summary
        
        elif self.detector_selection_mode == "any":
            # Any detector triggers anomaly
            decision = len(anomaly_results) > 0
            max_score = max(r.anomaly_score for r in results)
            if anomaly_results:
                # Use explanation from highest-scoring detector
                best_anomaly = max(anomaly_results, key=lambda r: r.anomaly_score)
                explanation = best_anomaly.explanation.summary if best_anomaly.explanation else None
            else:
                explanation = None
            summary = f"Any mode: {len(anomaly_results)}/{len(results)} detected anomaly"
            return decision, max_score, explanation, summary
        
        elif self.detector_selection_mode == "all":
            # All detectors must agree
            decision = len(anomaly_results) == len(results) and len(results) > 0
            min_score = min(r.anomaly_score for r in results) if decision else 0.0
            if decision and anomaly_results:
                # All agree it's anomaly - use explanation from most confident
                best_anomaly = max(anomaly_results, key=lambda r: r.explanation.confidence if r.explanation else 0)
                explanation = best_anomaly.explanation.summary if best_anomaly.explanation else None
            else:
                explanation = None
            summary = f"All mode: {len(anomaly_results)}/{len(results)} detected anomaly"
            return decision, min_score, explanation, summary
        
        else:
            # Default to first detector
            first_result = results[0]
            explanation = first_result.explanation.summary if first_result.explanation else None
            summary = f"Default: {first_result.detector_method}"
            return first_result.is_anomaly, first_result.anomaly_score, explanation, summary

    def stop(self):
        self._stop = True

    @Slot(bool)
    def pause(self, value: bool):
        try:
            self._paused = bool(value)
        except Exception:
            self._paused = False

    # Live-update slots
    @Slot(float)
    def set_alpha(self, alpha: float):
        try:
            self.det.alpha = float(alpha)
        except Exception:
            pass

    @Slot(float)
    def set_z_threshold(self, z: float):
        try:
            new_threshold = float(z)
            # Validate threshold range
            if not (0.1 <= new_threshold <= 50.0):
                print(f"Warning: Z threshold {new_threshold} outside recommended range [0.1, 50.0]")
                
            # Check for significant threshold change that might warrant detector reset
            old_threshold = self.cfg.z_threshold
            threshold_change_ratio = abs(new_threshold - old_threshold) / max(old_threshold, 0.1)
            
            # Update both config and detector atomically
            self.cfg.z_threshold = new_threshold
            self.det.z_threshold = new_threshold
            
            # If threshold changed significantly (>50%), consider partial state reset
            if threshold_change_ratio > 0.5:
                print(f"Significant Z threshold change detected ({old_threshold} -> {new_threshold}), "
                      f"detector may recalibrate over next few readings")
                # Note: We don't fully reset detector state to maintain continuity,
                # but significant changes will naturally recalibrate through EWMA
            
            print(f"Z threshold updated: {old_threshold} -> {new_threshold}")
            
        except Exception as e:
            print(f"Error updating Z threshold: {e}")
            pass

    @Slot(int)
    def set_window_sec(self, sec: int):
        try:
            self.window_sec = int(sec)
        except Exception:
            pass

    @Slot(bool)
    def set_group_by_source(self, enabled: bool):
        try:
            self.group_by_source = bool(enabled)
        except Exception:
            pass

    @Slot(str)
    def set_detector_selection_mode(self, mode: str):
        """Update the detector selection mode at runtime."""
        try:
            if mode in ["first", "majority", "any", "all"]:
                self.detector_selection_mode = mode
                print(f"Detector selection mode updated to: {mode}")
        except Exception:
            pass

    def get_detector_info(self) -> Dict[str, List[str]]:
        """Get information about active detectors for each channel."""
        info = {}
        for channel, detectors in self.channel_detectors.items():
            info[channel] = [detector.name for detector in detectors]
        return info

    def get_detector_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all detectors."""
        stats = {}
        for channel, detectors in self.channel_detectors.items():
            stats[channel] = {}
            for detector in detectors:
                stats[channel][detector.name] = detector.get_stats()
        return stats

    def _open_csv(self):
        if self._csv_path:
            p = Path(self._csv_path)
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = p.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            # Enhanced header for multi-method support
            self._csv_writer.writerow([
                "ts", "channel", "value", "z", "mean", "std", 
                "detector_method", "anomaly_score", "explanation_summary"
            ])

    def _close_csv(self):
        try:
            if self._csv_file:
                self._csv_file.close()
        except Exception:
            pass

    def run(self):
        import time
        # Select inputs: simulator (default) or configured sources
        inputs = []
        if getattr(self.cfg, "sources", None):
            print(f"🔧 Found {len(self.cfg.sources)} configured sources")
            for src in self.cfg.sources:
                print(f"🔍 Processing source: {src}")
                t = src.get("type")
                print(f"   Type: {t}")
                if t == "file" and src.get("path"):
                    print(f"📁 Adding file source: {src.get('path')}")
                    inputs.append((src.get("name", "file"), file_tail(src["path"])) )
                elif t == "udp":
                    host = src.get("host", "0.0.0.0")
                    port = int(src.get("port", 9999))
                    print(f"📡 Adding UDP source: {host}:{port}")
                    try:
                        udp_stream = udp_json_stream(host, port)
                        inputs.append((src.get("name", "udp"), udp_stream))
                        print(f"✅ UDP source added successfully")
                    except Exception as e:
                        print(f"❌ Failed to create UDP stream: {e}")
                else:
                    print(f"⚠️ Unknown or incomplete source type: {t}")
        else:
            print("🎭 No sources configured, using simulator")
            sim = TelemetrySimulator(channels=[c.name for c in self.cfg.channels], rate_hz=self.cfg.rate_hz)
            inputs.append(("sim", sim.stream()))

        print(f"✅ Initialized {len(inputs)} data sources")
        self._open_csv()
        last_forecast = 0.0
        message_count = 0
        try:
            while not self._stop:
                for name, it in inputs:
                    try:
                        msg = next(it)
                        message_count += 1
                        if message_count % 50 == 0:
                            print(f"📊 Processed {message_count} messages from {name}")
                    except StopIteration:
                        continue
                    except Exception as e:
                        if message_count % 100 == 0:  # Only print occasional errors to avoid spam
                            print(f"⚠️ Error reading from {name}: {e}")
                        continue

                    if hasattr(msg, "ts") and hasattr(msg, "values"):
                        ts = float(msg.ts)
                        values = dict(msg.values)
                    elif isinstance(msg, dict):
                        # Robustly parse dict message; skip bad values
                        try:
                            ts_val = msg.get("ts", None)
                            ts = float(ts_val) if ts_val is not None else time.time()
                        except Exception:
                            ts = time.time()
                        values = {}
                        for k, v in msg.items():
                            if k == "ts":
                                continue
                            try:
                                values[k] = float(v)
                            except Exception:
                                # skip non-numeric or malformed values
                                continue
                    else:
                        continue

                    self.ts_hist.append(ts)
                    # Trim by time window
                    cutoff = ts - max(10, int(self.window_sec))
                    while len(self.ts_hist) > 1 and self.ts_hist[0] < cutoff:
                        self.ts_hist.pop(0)
                    # Ensure per-channel histories don't exceed ts history length
                    max_len = len(self.ts_hist)
                    for _ch, _hist in self.hist.items():
                        while len(_hist) > max_len:
                            _hist.pop(0)
                    # Optionally prefix channel with source for grouping
                    if self.group_by_source:
                        prefixed = {f"{name}:{k}": float(v) for k, v in values.items()}
                        emit_values = prefixed
                    else:
                        emit_values = values
                    if not self._paused:
                        if message_count % 100 == 0:  # Debug every 100th message
                            print(f"📤 Emitting data: {emit_values}")
                        self.reading.emit(ts, emit_values)
                        
                    # Store recent readings for new plot initialization
                    self._recent_readings.append((ts, emit_values.copy()))
                    if len(self._recent_readings) > self._max_recent_readings:
                        self._recent_readings.pop(0)

                    for c, v in emit_values.items():
                        self.hist.setdefault(c, []).append(v)
                        # keep same length as ts_hist
                        while len(self.hist[c]) > len(self.ts_hist):
                            self.hist[c].pop(0)
                        if not self._paused:
                            # Get base channel name (remove source prefix if present)
                            base_name = c.split(":", 1)[1] if ":" in c else c
                            
                            # Run multi-method detection if detectors are available for this channel
                            if base_name in self.channel_detectors:
                                multi_result = self._run_multi_method_detection(base_name, float(v), ts)
                                
                                # Emit multi-method result
                                self.multi_method_result.emit(multi_result)
                                
                                # Extract stats for visualization (prefer Z-score detector if available)
                                z_score_result = None
                                for result in multi_result.results:
                                    if "zscore" in result.detector_method.lower():
                                        z_score_result = result
                                        break
                                
                                # If we have Z-score stats, emit them for visualization
                                if z_score_result and z_score_result.raw_data:
                                    raw_data = z_score_result.raw_data
                                    mean = raw_data.get("mean", 0.0)
                                    std = raw_data.get("std", 0.0)
                                    self.channel_stats.emit(c, mean, std)
                                
                                # Process anomaly if detected
                                if multi_result.final_decision:
                                    ch_cfg = next((cc for cc in self.cfg.channels if cc.name == base_name), None)
                                    eta_for_ch = self._last_eta.get(c)
                                    thr = ch_cfg.breach_threshold if ch_cfg else None
                                    direction = (ch_cfg.breach_direction or "above") if ch_cfg else None
                                    
                                    # Use combined anomaly score for assessment
                                    a = assess(multi_result.combined_score * 10, 3.0, eta_for_ch, thr, direction)
                                    
                                    # Create enhanced anomaly event
                                    ev = {
                                        "ts": ts,
                                        "channel": c,
                                        "value": v,
                                        "z": multi_result.combined_score * 10,  # Scale for compatibility
                                        "mean": z_score_result.raw_data.get("mean", 0.0) if z_score_result and z_score_result.raw_data else 0.0,
                                        "std": z_score_result.raw_data.get("std", 0.0) if z_score_result and z_score_result.raw_data else 0.0,
                                        "severity": a.severity,
                                        "pvalue": a.pvalue,
                                        "eta_sec": eta_for_ch,
                                        # Enhanced fields
                                        "detector_method": multi_result.detection_summary,
                                        "anomaly_score": multi_result.combined_score,
                                        "explanation_summary": multi_result.primary_explanation or "",
                                        "num_detectors": len(multi_result.results),
                                        "num_anomalies": sum(1 for r in multi_result.results if r.is_anomaly)
                                    }
                                    self.anomaly.emit(ev)
                                    
                                    # Log to CSV
                                    if self._csv_writer:
                                        try:
                                            self._csv_writer.writerow([
                                                ev.get("ts"), ev.get("channel"), ev.get("value"), 
                                                ev.get("z"), ev.get("mean"), ev.get("std"),
                                                ev.get("detector_method"), ev.get("anomaly_score"),
                                                ev.get("explanation_summary")
                                            ])
                                        except Exception:
                                            pass
                            else:
                                # Fallback to legacy single detector for channels without multi-method setup
                                try:
                                    res = self.det.update_legacy(c, float(v))
                                    
                                    # Emit channel statistics for Z-score threshold visualization
                                    self.channel_stats.emit(c, res.mean, res.std)
                                    
                                    # Use detector's current threshold, not config (for dynamic updates)
                                    current_threshold = self.det.z_threshold
                                    if abs(res.zscore) >= current_threshold:
                                        ch_cfg = next((cc for cc in self.cfg.channels if cc.name == base_name), None)
                                        eta_for_ch = self._last_eta.get(c)
                                        thr = ch_cfg.breach_threshold if ch_cfg else None
                                        direction = (ch_cfg.breach_direction or "above") if ch_cfg else None
                                        a = assess(abs(res.zscore), float(current_threshold), eta_for_ch, thr, direction)
                                        ev = {
                                            "ts": ts,
                                            "channel": c,
                                            "value": v,
                                            "z": res.zscore,
                                            "mean": res.mean,
                                            "std": res.std,
                                            "severity": a.severity,
                                            "pvalue": a.pvalue,
                                            "eta_sec": eta_for_ch,
                                            "detector_method": "legacy-zscore",
                                            "anomaly_score": min(abs(res.zscore) / current_threshold, 1.0),
                                            "explanation_summary": f"Z-score {res.zscore:.2f} exceeds threshold ±{current_threshold}"
                                        }
                                        self.anomaly.emit(ev)
                                        if self._csv_writer:
                                            try:
                                                self._csv_writer.writerow([
                                                    ev.get("ts"), ev.get("channel"), ev.get("value"), 
                                                    ev.get("z"), ev.get("mean"), ev.get("std")
                                                ])
                                            except Exception:
                                                pass
                                except Exception as e:
                                    print(f"Warning: Legacy detection failed for {c}: {e}")
                                    continue

                    if not self._paused and ts - last_forecast >= 5.0:
                        last_forecast = ts
                        # Choose channels for forecasting
                        if self.group_by_source:
                            # forecast per prefixed key; map thresholds by base channel name
                            forecast_keys = list(self.hist.keys())
                        else:
                            forecast_keys = [ch.name for ch in self.cfg.channels]

                        for key in forecast_keys:
                            try:
                                series = self.hist.get(key, [])
                                if len(series) < 30:
                                    print(f"DEBUG: Skipping forecast for {key} - only {len(series)} points (need 30+)")
                                    continue
                                base_name = key.split(":", 1)[1] if ":" in key else key
                                # lookup channel config by base name
                                ch_cfg = next((c for c in self.cfg.channels if c.name == base_name), None)
                                seasonal_periods = 50 if base_name == "temp_c" else None
                                thr = ch_cfg.breach_threshold if ch_cfg else None
                                direction = (ch_cfg.breach_direction or "above") if ch_cfg else "above"
                                
                                print(f"DEBUG: Forecasting {key} - {len(series)} points, threshold={thr} ({direction})")
                                
                                summ = make_forecast_summary(
                                    series,
                                    self.ts_hist[-len(series):],
                                    horizon=min(120, int(self.window_sec)),
                                    seasonal_periods=seasonal_periods,
                                    threshold=thr,
                                    direction=direction,
                                )
                                
                                print(f"DEBUG: Forecast result for {key} - ETA: {summ.est_time_to_breach_sec}s")
                                
                                if summ.est_time_to_breach_sec is not None and summ.breach_threshold is not None:
                                    eta = float(summ.est_time_to_breach_sec)
                                    self._last_eta[key] = eta
                                    self.forecast.emit(key, float(summ.breach_threshold), eta)
                            except Exception:
                                # Forecasting should never crash the worker
                                continue
                # Yield a tiny bit to avoid tight-spin if inputs are idle
                time.sleep(0.001)
        finally:
            # Close CSV on exit
            self._close_csv()

    def get_historical_data(self, channels: Optional[List[str]] = None) -> List[tuple[float, Dict[str, float]]]:
        """
        Get historical data for specified channels to populate new plots.
        
        Args:
            channels: List of channel names to filter by. If None, returns all channels.
            
        Returns:
            List of (timestamp, values_dict) tuples containing recent readings.
        """
        if not self._recent_readings:
            return []
            
        if channels is None:
            return self._recent_readings.copy()
        
        # Filter readings to only include requested channels
        filtered_readings = []
        for ts, values in self._recent_readings:
            filtered_values = {}
            for key, value in values.items():
                # Match both direct channel names and source-prefixed names
                base_channel = key.split(":", 1)[1] if ":" in key else key
                if base_channel in channels or key in channels:
                    filtered_values[key] = value
            
            if filtered_values:  # Only include readings that have relevant data
                filtered_readings.append((ts, filtered_values))
        
        return filtered_readings

    def set_detector_enabled(self, channel: str, method: str, enabled: bool) -> None:
        """Enable or disable a specific detector for a channel."""
        if channel not in self.channel_detectors:
            return
        
        # Find or create detector for this method
        existing_detector = None
        for detector in self.channel_detectors[channel]:
            if method.lower() in detector.name.lower():
                existing_detector = detector
                break
        
        if enabled and not existing_detector:
            # Create new detector
            try:
                # Get default parameters for this method
                default_params = self._get_default_detector_params(method, channel)
                detector = DetectorFactory.create(method, channel=channel, **default_params)
                self.channel_detectors[channel].append(detector)
                print(f"Added {method} detector to {channel}")
            except Exception as e:
                print(f"Failed to create {method} detector for {channel}: {e}")
        elif not enabled and existing_detector:
            # Remove detector
            try:
                self.channel_detectors[channel].remove(existing_detector)
                print(f"Removed {method} detector from {channel}")
            except Exception as e:
                print(f"Failed to remove {method} detector from {channel}: {e}")

    def set_detector_parameter(self, channel: str, method: str, param: str, value) -> None:
        """Update a parameter for a specific detector."""
        if channel not in self.channel_detectors:
            return
        
        # Find the detector
        for detector in self.channel_detectors[channel]:
            if method.lower() in detector.name.lower():
                try:
                    # Update the parameter
                    current_params = detector.get_parameters()
                    current_params[param] = value
                    detector.set_parameters(current_params)
                    print(f"Updated {method} detector {param} = {value} for {channel}")
                except Exception as e:
                    print(f"Failed to update {method} detector parameter: {e}")
                break

    def set_channel_selection_mode(self, channel: str, mode: str) -> None:
        """Set selection mode for a specific channel (currently global only)."""
        # For now, update global mode - future enhancement could be per-channel
        self.set_detector_selection_mode(mode)

    def _get_default_detector_params(self, method: str, channel: str) -> dict:
        """Get default parameters for a detector method."""
        defaults = {
            "zscore": {
                "alpha": self.cfg.alpha,
                "z_threshold": self.cfg.z_threshold,
                "eps": 1e-6,
                "learning_period": 10,
                "drift_update_rate": 100
            },
            "iqr": {
                "iqr_multiplier": 1.5,
                "window_size": 20,
                "min_samples": 5
            }
        }
        return defaults.get(method, {})
