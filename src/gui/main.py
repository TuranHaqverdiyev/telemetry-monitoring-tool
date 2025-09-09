from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QInputDialog,
    QCheckBox,
    QTabWidget,
    QSplitter,
)
from PySide6.QtCore import QThread, QSettings, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QSystemTrayIcon

from gui.config import load_config
from gui.service import StreamWorker, AnomalyEvent
from gui.widgets.plots import PlotGrid
from gui.widgets.anomalies import AnomalyTable
from gui.widgets.forecasts import ForecastTable
from gui.widgets.detector_control import DetectorControlPanel
from gui.widgets.alert_config import AlertConfigWidget

# Optional dark theme
try:
    import qdarkstyle  # type: ignore
except Exception:  # pragma: no cover
    qdarkstyle = None  # type: ignore


class MainWindow(QMainWindow):
    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Telemetry Monitoring Tool")
        
        # Use provided config path or fall back to defaults
        if config_path:
            self.cfg_path = Path(config_path)
            print(f"🔧 Using config file: {config_path}")
        else:
            self.cfg_path = Path("config.json") if Path("config.json").exists() else Path("config.example.json")
            print(f"🔧 Using default config file: {self.cfg_path}")
            
        self.cfg = load_config(str(self.cfg_path))

        # Initialize alert manager
        self.alert_manager = None
        if hasattr(self.cfg, 'alerts') and self.cfg.alerts:
            try:
                from alerts.alert_manager import AlertManager
                self.alert_manager = AlertManager(self.cfg.alerts)
                print(f"📧 Alert system initialized (enabled: {self.cfg.alerts.enabled})")
            except ImportError as e:
                print(f"Warning: Could not initialize alert system: {e}")

        # UI widgets
        # No initial plots - user creates them on demand with "Add Plot" button
        self.plot = None
        self.plot_container = QWidget()  # Container for plots when created
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.addWidget(QLabel("📊 Click '+ Add Plot' to create charts"))
        
        self.table = AnomalyTable()  # Enhanced anomaly detection results table
        # Removed forecasts table - information is included in main results table
        self.status = QLabel("Idle")
        
        # Detector control panel
        channels_config = {ch.name: ch.detectors for ch in self.cfg.channels}
        # Detector panel now properly configured with channel detection methods
        self.detector_panel = DetectorControlPanel(
            channels_config, 
            getattr(self.cfg, 'detector_selection_mode', 'first')
        )
        
        # Alert configuration panel
        self.alert_config_widget = AlertConfigWidget(config_path=str(self.cfg_path))
        if hasattr(self.cfg, 'alerts') and self.cfg.alerts:
            self.alert_config_widget.set_config(self.cfg.alerts)
        
        # Connect alert manager to alert config widget
        if self.alert_manager:
            self.alert_config_widget.set_alert_manager(self.alert_manager)

        # Buttons and controls
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_pause = QPushButton("Pause")
        self.btn_resume = QPushButton("Resume")
        self.btn_clear = QPushButton("Clear")
        self.btn_load = QPushButton("Load Config…")
        self.btn_add_plot = QPushButton("+ Add Plot")
        self.btn_merge = QPushButton("Merge Channels")
        self.btn_set_log = QPushButton("Set Log CSV…")
        self.btn_snapshot = QPushButton("Save Snapshot…")
        self.chk_group_by_source = QCheckBox("Group by Source")
        self.chk_group_by_source.setChecked(bool(getattr(self.cfg, "group_by_source", False)))
        self.chk_notify = QCheckBox("Notifications")
        self.chk_notify.setChecked(True)

        # System tray for desktop notifications
        self.tray = QSystemTrayIcon(self)
        # Create a simple icon from application style
        if QSystemTrayIcon.isSystemTrayAvailable():
            # Use a simple application icon or create one
            icon = self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)
            self.tray.setIcon(icon)
            self.tray.setVisible(True)
        else:
            # System tray not available, disable notifications
            self.chk_notify.setEnabled(False)
            self.chk_notify.setChecked(False)

        # Left side layout - larger space for charts
        self._left_layout = QVBoxLayout()
        self._left_layout.addWidget(self.plot_container)
        self._left_layout.addWidget(self.status)

        # Controls panel
        ctrl_form = QFormLayout()
        self.spin_window = QSpinBox()
        self.spin_window.setRange(5, 3600)
        self.spin_window.setValue(int(self.cfg.window_sec))
        ctrl_form.addRow("Time window (s)", self.spin_window)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.setRange(0.001, 1.0)
        self.spin_alpha.setSingleStep(0.005)
        self.spin_alpha.setValue(float(self.cfg.alpha))
        ctrl_form.addRow("EWMA alpha", self.spin_alpha)

        self.spin_z = QDoubleSpinBox()
        self.spin_z.setDecimals(2)
        self.spin_z.setRange(0.5, 10.0)
        self.spin_z.setSingleStep(0.1)
        # Use a more reasonable default Z threshold (2.5) to allow more anomaly detection
        z_threshold = float(self.cfg.z_threshold) if hasattr(self.cfg, 'z_threshold') else 2.5
        if z_threshold > 5.0:  # If config has an unusually high threshold, use 2.5 instead
            z_threshold = 2.5
        self.spin_z.setValue(z_threshold)
        ctrl_form.addRow("Z threshold", self.spin_z)

        # Right side layout with tabs
        right_tabs = QTabWidget()
        
        # Controls tab
        controls_widget = QWidget()
        right = QVBoxLayout(controls_widget)
        right.addLayout(ctrl_form)
        right.addWidget(self.chk_group_by_source)
        right.addWidget(self.table)
        # Removed forecasts table - redundant information
        right.addWidget(self.btn_add_plot)
        right.addWidget(self.btn_merge)
        right.addWidget(self.btn_set_log)
        right.addWidget(self.btn_snapshot)
        right.addWidget(self.chk_notify)
        right.addWidget(self.btn_clear)
        right.addWidget(self.btn_start)
        right.addWidget(self.btn_pause)
        right.addWidget(self.btn_resume)
        right.addWidget(self.btn_stop)
        right.addWidget(self.btn_load)
        right.addStretch(1)
        
        right_tabs.addTab(controls_widget, "📊 Controls")
        right_tabs.addTab(self.detector_panel, "🔧 Detectors")
        right_tabs.addTab(self.alert_config_widget, "📧 Alerts")

        # Create a horizontal splitter for resizable layout
        central_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create left widget for charts
        left_widget = QWidget()
        left_widget.setLayout(self._left_layout)
        
        # Add widgets to splitter
        central_splitter.addWidget(left_widget)
        central_splitter.addWidget(right_tabs)
        
        # Set initial sizes (80% chart, 20% controls) but allow user to resize
        central_splitter.setSizes([800, 200])  # Initial proportions
        central_splitter.setStretchFactor(0, 1)  # Chart area can stretch
        central_splitter.setStretchFactor(1, 0)  # Control panel has fixed minimum
        
        self.setCentralWidget(central_splitter)

        self._thread = None
        self._worker = None
        self._log_path = "anomalies.csv"
        self._settings = QSettings("TelemetryTool", "App")

        # Wire controls
        self.spin_window.valueChanged.connect(self.on_window_changed)
        self.spin_alpha.valueChanged.connect(self.on_alpha_changed)
        self.spin_z.valueChanged.connect(self.on_z_changed)
        # Wire buttons
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_load.clicked.connect(self.load_config_dialog)
        self.btn_add_plot.clicked.connect(self.add_plot_dialog)
        self.btn_merge.clicked.connect(self.merge_channels_dialog)
        self.btn_set_log.clicked.connect(self.set_log_csv)
        self.chk_group_by_source.toggled.connect(self.on_group_by_source)
        self.btn_snapshot.clicked.connect(self.on_snapshot)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_resume.clicked.connect(self.on_resume)
        
        # Wire detector panel signals
        self.detector_panel.detector_enabled.connect(self.on_detector_enabled)
        self.detector_panel.detector_parameter_changed.connect(self.on_detector_parameter_changed)
        self.detector_panel.selection_mode_changed.connect(self.on_selection_mode_changed)
        self.detector_panel.apply_to_all_channels.connect(self.on_apply_to_all_channels)

    def _create_initial_plot(self) -> None:
        """Create the initial empty plot grid when user first adds a plot."""
        from .widgets.plots import PlotGrid  # Import here to avoid circular imports
        
        # Clear the placeholder
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)
            
        # Create EMPTY plot grid - no default panels
        self.plot = PlotGrid(None, [], time_window_sec=self.cfg.window_sec)
        
        # Set Z threshold for visual display
        self.plot.set_z_threshold(float(self.cfg.z_threshold))
        
        # Add plot to layout
        self.plot_layout.addWidget(self.plot)

    def start_stream(self) -> None:
        if self._worker:
            return
        self._thread = QThread()
        self._worker = StreamWorker(self.cfg, log_csv_path=self._log_path)
        # Apply current control values
        self._worker.set_window_sec(int(self.spin_window.value()))
        self._worker.set_alpha(float(self.spin_alpha.value()))
        self._worker.set_z_threshold(float(self.spin_z.value()))
        self._worker.set_group_by_source(bool(self.chk_group_by_source.isChecked()))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.reading.connect(self.on_reading)
        self._worker.anomaly.connect(self.on_anomaly)
        self._worker.forecast.connect(self.on_forecast)
        self._worker.channel_stats.connect(self.on_channel_stats)  # Connect stats for Z-score lines
        self._thread.start()
        self.status.setText("Streaming…")
        # save UI prefs
        self._save_settings()

    def stop_stream(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._thread:
            self._thread.quit()
            self._thread.wait(1000)
            self._thread = None
        self.status.setText("Stopped")
        # keep last prefs
        self._save_settings()

    def load_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open config", str(self.cfg_path.parent), "JSON Files (*.json)")
        if path:
            self.cfg_path = Path(path)
            self.cfg = load_config(path)
            self.stop_stream()
            # Rebuild plot panels using new groups/channels
            # Always start with a single panel after loading config; user can add more panels manually
            new_plot = PlotGrid(None, [c.name for c in self.cfg.channels], time_window_sec=self.cfg.window_sec)
            thr_map = {c.name: c.breach_threshold for c in self.cfg.channels if getattr(c, 'breach_threshold', None) is not None}
            if thr_map:
                new_plot.apply_thresholds(thr_map)
            # Replace widget in the left column
            self._left_layout.replaceWidget(self.plot, new_plot)
            self.plot.setParent(None)
            self.plot = new_plot
            # Update control defaults
            self.spin_window.setValue(int(self.cfg.window_sec))
            self.spin_alpha.setValue(float(self.cfg.alpha))
            self.spin_z.setValue(float(self.cfg.z_threshold))

    def on_reading(self, ts: float, values: dict) -> None:
        if self.plot:  # Only append if plot exists
            self.plot.append(ts, values)

    def on_channel_stats(self, channel: str, mean: float, std: float) -> None:
        """Update plot Z-score threshold lines based on channel statistics."""
        if self.plot:  # Only update if plot exists
            self.plot.update_channel_stats(channel, mean, std)

    def on_anomaly(self, ev: AnomalyEvent) -> None:
        # ev may be a dataclass or a dict with extra fields
        try:
            if isinstance(ev, dict):
                # Legacy table for backward compatibility
                pvalue = ev.get("pvalue")
                eta_sec = ev.get("eta_sec")
                self.table.add_row(
                    float(ev.get("ts", 0.0)),
                    str(ev.get("channel", "")),
                    float(ev.get("value", 0.0)),
                    float(ev.get("z", 0.0)),
                    float(ev.get("std", 0.0)),
                    str(ev.get("severity", "")),
                    float(pvalue) if pvalue is not None else 0.0,
                    float(eta_sec) if eta_sec is not None else 0.0,
                )
                
                # Send email alert if alert manager is enabled
                if self.alert_manager and self.alert_manager.config.enabled:
                    try:
                        self.alert_manager.send_alert(
                            channel=str(ev.get("channel", "")),
                            value=float(ev.get("value", 0.0)),
                            timestamp=float(ev.get("ts", 0.0)),
                            severity=str(ev.get("severity", "medium")),
                            detector_method=str(ev.get("detector_method", "unknown")),
                            anomaly_score=float(ev.get("anomaly_score", 0.0)),
                            explanation=ev.get("explanation_summary"),
                            raw_data=ev.get("raw_data", {})
                        )
                    except Exception as e:
                        print(f"Warning: Failed to send email alert: {e}")
                
                # Desktop notification for critical severity
                try:
                    if self.chk_notify.isChecked() and str(ev.get("severity", "")).lower() == "critical":
                        ch = str(ev.get("channel", ""))
                        z = float(ev.get("z", 0.0))
                        val = float(ev.get("value", 0.0))
                        self.tray.showMessage(
                            "Critical anomaly",
                            f"{ch}: z={z:.2f}, value={val:.3f}",
                            QSystemTrayIcon.MessageIcon.Critical,
                            5000,
                        )
                except Exception:
                    pass
            else:
                # Legacy anomaly event format
                self.table.add_row(ev.ts, ev.channel, ev.value, ev.z, ev.std)
                
                # Convert to dict format for enhanced panel
                legacy_dict = {
                    "ts": ev.ts,
                    "channel": ev.channel,
                    "value": ev.value,
                    "z": ev.z,
                    "std": ev.std,
                    "severity": "medium",  # Default severity for legacy events
                    "detector_method": "legacy-zscore",
                    "anomaly_score": min(abs(ev.z) / 3.0, 1.0),  # Normalize Z-score
                    "explanation_summary": f"Z-score anomaly detected: {ev.z:.2f}"
                }
        except Exception:
            # fallback minimal
            try:
                self.table.add_row(ev.ts, ev.channel, ev.value, ev.z, ev.std)
            except Exception:
                pass

    def on_forecast(self, channel: str, threshold: float, eta_sec: float) -> None:
        if eta_sec is not None:
            msg = f"Forecast: {channel} reaches {threshold} in ~{eta_sec:.0f}s"
            if eta_sec <= 15:
                self.status.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.status.setStyleSheet("")
            self.status.setText(msg)
            # Forecast data is now included in the main results table via ETA column
        else:
            msg = f"Forecast: {channel} - no trend toward {threshold}"
            self.status.setStyleSheet("")
            self.status.setText(msg)
            # Still update the table to show no forecast available
            self.forecasts.upsert(channel, threshold, float('inf'))

    # Control handlers
    def on_window_changed(self, v: int) -> None:
        if self.plot:
            self.plot.set_time_window(int(v))
        if self._worker:
            self._worker.set_window_sec(int(v))

    def on_alpha_changed(self, v: float) -> None:
        if self._worker:
            self._worker.set_alpha(float(v))

    def on_z_changed(self, v: float) -> None:
        if self._worker:
            self._worker.set_z_threshold(float(v))
            # Update visual Z-score threshold lines on all plots
            if self.plot:
                self.plot.set_z_threshold(float(v))
            # Provide visual feedback that threshold was updated
            self.status.setText(f"Z threshold updated to {v:.2f}")
            # Reset status back to normal after 3 seconds
            import threading
            def reset_status():
                import time
                time.sleep(3)
                if hasattr(self, 'status'):
                    self.status.setText("Streaming…" if self._worker else "Idle")
            threading.Thread(target=reset_status, daemon=True).start()

    def on_group_by_source(self, enabled: bool) -> None:
        if self._worker:
            self._worker.set_group_by_source(bool(enabled))

    # Plot management
    def add_plot_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "+ Add Plot", "Panel name:")
        if not ok or not name:
            return
        chans_csv, ok = QInputDialog.getText(self, "+ Add Plot", "Channels (comma-separated):")
        if not ok:
            return
        chans = [c.strip() for c in chans_csv.split(',') if c.strip()]
        if not chans:
            return
        self._create_plot_panel_with_data(name, chans)

    def _create_plot_panel_with_data(self, name: str, channels: list[str]) -> None:
        """Factory method that creates plots with proper data binding and historical data"""
        print(f"Creating new plot panel '{name}' with channels: {channels}")
        
        # Create initial plot if it doesn't exist
        if not self.plot:
            self._create_initial_plot()
        
        # Add the panel to the grid
        if self.plot:
            self.plot.add_panel(name, channels)
        
        # If we have an active worker with historical data, populate the new panel
        if self._worker and hasattr(self._worker, 'get_historical_data'):
            try:
                historical_data = self._worker.get_historical_data(channels)
                print(f"Retrieved {len(historical_data)} historical data points for new panel")
                
                panel = self.plot._panel_by_name.get(name) if self.plot else None
                if panel and historical_data:
                    for ts, values in historical_data:
                        filtered_values = {k: v for k, v in values.items() 
                                         if any(k.endswith(ch) or k == ch for ch in channels)}
                        if filtered_values:
                            panel.append(ts, filtered_values)
                    print(f"Populated new panel with historical data")
                else:
                    print(f"Warning: Panel not found or no historical data available")
            except Exception as e:
                print(f"Warning: Could not load historical data for new panel: {e}")
        else:
            print("Warning: No active worker or historical data method not available")
        
        # Apply any existing thresholds for the channels
        thr_map = {c.name: c.breach_threshold for c in self.cfg.channels 
                  if getattr(c, 'breach_threshold', None) is not None and c.name in channels}
        if thr_map and self.plot:
            panel = self.plot._panel_by_name.get(name)
            if panel:
                for ch, threshold in thr_map.items():
                    if threshold is not None:
                        panel.set_threshold(ch, float(threshold))
                print(f"Applied thresholds to new panel: {thr_map}")
        
        print(f"Successfully created plot panel '{name}'")

    def merge_channels_dialog(self) -> None:
        target, ok = QInputDialog.getText(self, "Merge Channels", "Target panel name:")
        if not ok or not target:
            return
        chans_csv, ok = QInputDialog.getText(self, "Merge Channels", "Channels to add (comma-separated):")
        if not ok:
            return
        chans = [c.strip() for c in chans_csv.split(',') if c.strip()]
        if not chans:
            return
        if self.plot:
            self.plot.merge_channels(target, chans)

    def closeEvent(self, event):  # type: ignore[override]
        try:
            self.stop_stream()
        finally:
            super().closeEvent(event)

    def set_log_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Set anomalies log CSV", str(self.cfg_path.parent), "CSV Files (*.csv)")
        if path:
            self._log_path = path
            self._save_settings()

    # Settings
    def _save_settings(self) -> None:
        self._settings.setValue("window_sec", int(self.spin_window.value()))
        self._settings.setValue("alpha", float(self.spin_alpha.value()))
        self._settings.setValue("z", float(self.spin_z.value()))
        self._settings.setValue("group_by_source", bool(self.chk_group_by_source.isChecked()))
        self._settings.setValue("log_csv", self._log_path)
        self._settings.setValue("notify", bool(self.chk_notify.isChecked()))

    def _restore_settings(self) -> None:
        w = self._settings.value("window_sec")
        if w is not None:
            try:
                self.spin_window.setValue(int(w))
            except Exception:
                pass
        a = self._settings.value("alpha")
        if a is not None:
            try:
                self.spin_alpha.setValue(float(a))
            except Exception:
                pass
        z = self._settings.value("z")
        if z is not None:
            try:
                z_val = float(z)
                # If saved Z threshold is too high, use a reasonable default instead
                if z_val > 5.0:
                    z_val = 2.5
                self.spin_z.setValue(z_val)
            except Exception:
                pass
        g = self._settings.value("group_by_source")
        if g is not None:
            try:
                self.chk_group_by_source.setChecked(bool(g in (True, 'true', '1', 1)))
            except Exception:
                pass
        p = self._settings.value("log_csv")
        if p:
            self._log_path = str(p)
        n = self._settings.value("notify")
        if n is not None:
            try:
                self.chk_notify.setChecked(bool(n in (True, 'true', '1', 1)))
            except Exception:
                pass

    # Extra controls
    def on_clear(self) -> None:
        try:
            self.table.table.setRowCount(0)
            # Forecast data is now included in main table
            if self.plot:
                self.plot.clear()
        except Exception:
            pass

    def on_pause(self) -> None:
        if self._worker:
            try:
                self._worker.pause(True)
                self.status.setText("Paused")
            except Exception:
                pass

    def on_resume(self) -> None:
        if self._worker:
            try:
                self._worker.pause(False)
                self.status.setText("Streaming…")
            except Exception:
                pass

    def on_detector_enabled(self, channel: str, method: str, enabled: bool) -> None:
        """Handle detector enable/disable from GUI."""
        if self._worker:
            try:
                # Update the worker's detector configuration
                self._worker.set_detector_enabled(channel, method, enabled)
                self.status.setText(f"{'Enabled' if enabled else 'Disabled'} {method} detector for {channel}")
            except Exception as e:
                self.status.setText(f"Error updating detector: {e}")

    def on_detector_parameter_changed(self, channel: str, method: str, param: str, value) -> None:
        """Handle detector parameter changes from GUI."""
        if self._worker:
            try:
                # Update the worker's detector parameters
                self._worker.set_detector_parameter(channel, method, param, value)
                self.status.setText(f"Updated {method} {param} = {value} for {channel}")
            except Exception as e:
                self.status.setText(f"Error updating parameter: {e}")

    def on_selection_mode_changed(self, channel: str, mode: str) -> None:
        """Handle selection mode changes from GUI."""
        if self._worker:
            try:
                # Update the worker's selection mode
                self._worker.set_channel_selection_mode(channel, mode)
                self.status.setText(f"Set {channel} selection mode to: {mode}")
            except Exception as e:
                self.status.setText(f"Error updating selection mode: {e}")

    def on_apply_to_all_channels(self, method: str, enabled: bool) -> None:
        """Handle apply to all channels from GUI."""
        if self._worker:
            try:
                # Apply detector setting to all channels
                for channel_config in self.cfg.channels:
                    channel_name = channel_config.name
                    self._worker.set_detector_enabled(channel_name, method, enabled)
                action = "Enabled" if enabled else "Disabled"
                self.status.setText(f"{action} {method} detector for all channels")
            except Exception as e:
                self.status.setText(f"Error applying to all channels: {e}")

    def on_snapshot(self) -> None:
        try:
            from datetime import datetime
            from pathlib import Path
            if self.plot:
                pix = self.plot.grab()
            else:
                # If no plot exists, take screenshot of the container
                pix = self.plot_container.grab()
            out_dir = Path("snapshots")
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pix.save(str(fname))
            self.status.setText(f"Saved snapshot: {fname.name}")
        except Exception:
            pass


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description='Telemetry Monitoring Tool')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    try:
        if qdarkstyle:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
    except Exception:
        pass
        
    # Pass config path to MainWindow
    win = MainWindow(config_path=args.config)
    # restore persisted settings after constructing widgets
    try:
        win._restore_settings()
    except Exception:
        pass
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
