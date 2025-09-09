"""
Detector Method Selector Widget

This widget provides a GUI interface for selecting and configuring
anomaly detection methods in real-time.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QPushButton, QTabWidget, QScrollArea, QFrame
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from ..config import DetectorConfig
from ...telemetry.detector_base import DetectorFactory


class DetectorParameterWidget(QWidget):
    """Widget for editing detector parameters."""
    
    parameter_changed = Signal(str, str, object)  # detector_method, param_name, value
    
    def __init__(self, method: str, parameters: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.method = method
        self.parameters = parameters.copy()
        self.param_widgets = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QFormLayout(self)
        
        # Create parameter controls based on parameter types
        for param_name, value in self.parameters.items():
            widget = self._create_parameter_widget(param_name, value)
            if widget:
                self.param_widgets[param_name] = widget
                layout.addRow(self._format_param_name(param_name), widget)
    
    def _create_parameter_widget(self, param_name: str, value: Any) -> Optional[QWidget]:
        """Create appropriate widget for parameter type."""
        
        if param_name in ["window_size", "min_samples", "learning_period", "drift_update_rate", "n_estimators"]:
            # Integer parameters
            spinbox = QSpinBox()
            if param_name == "n_estimators":
                spinbox.setRange(10, 1000)
                spinbox.setValue(int(value))
            else:
                spinbox.setRange(1, 10000)
                spinbox.setValue(int(value))
            spinbox.valueChanged.connect(
                lambda v, name=param_name: self.parameter_changed.emit(self.method, name, v)
            )
            return spinbox
            
        elif isinstance(value, float):
            # Float parameters
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(3)
            
            # Set appropriate ranges based on parameter name
            if param_name in ["alpha"]:
                spinbox.setRange(0.001, 1.0)
                spinbox.setSingleStep(0.005)
            elif param_name in ["z_threshold", "iqr_multiplier", "mad_threshold"]:
                spinbox.setRange(0.1, 10.0)
                spinbox.setSingleStep(0.1)
            elif param_name in ["contamination", "anomaly_threshold"]:
                spinbox.setRange(0.001, 0.5)
                spinbox.setSingleStep(0.01)
                spinbox.setDecimals(3)
            elif param_name in ["eps"]:
                spinbox.setRange(1e-10, 1e-3)
                spinbox.setSingleStep(1e-7)
                spinbox.setDecimals(10)
            else:
                spinbox.setRange(-1000.0, 1000.0)
                spinbox.setSingleStep(0.1)
            
            spinbox.setValue(float(value))
            spinbox.valueChanged.connect(
                lambda v, name=param_name: self.parameter_changed.emit(self.method, name, v)
            )
            return spinbox
            
        elif isinstance(value, bool):
            # Boolean parameters
            checkbox = QCheckBox()
            checkbox.setChecked(bool(value))
            checkbox.toggled.connect(
                lambda v, name=param_name: self.parameter_changed.emit(self.method, name, v)
            )
            return checkbox
            
        return None
    
    def _format_param_name(self, param_name: str) -> str:
        """Format parameter name for display."""
        # Convert snake_case to Title Case
        return param_name.replace("_", " ").title()
    
    def update_parameter(self, param_name: str, value: Any):
        """Update parameter value programmatically."""
        if param_name in self.param_widgets:
            widget = self.param_widgets[param_name]
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            self.parameters[param_name] = value


class ChannelDetectorWidget(QWidget):
    """Widget for managing detectors on a specific channel."""
    
    detector_enabled = Signal(str, str, bool)  # channel, method, enabled
    detector_parameter_changed = Signal(str, str, str, object)  # channel, method, param, value
    selection_mode_changed = Signal(str)  # mode
    
    def __init__(self, channel_name: str, detector_configs: List[DetectorConfig], 
                 selection_mode: str = "first", parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.detector_configs = {cfg.method: cfg for cfg in detector_configs}
        self.selection_mode = selection_mode
        self.detector_widgets = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Channel header
        header = QLabel(f"📊 {self.channel_name}")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(10)
        header.setFont(header_font)
        layout.addWidget(header)
        
        # Selection mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Combination Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["first", "majority", "any", "all"])
        self.mode_combo.setCurrentText(self.selection_mode)
        self.mode_combo.currentTextChanged.connect(self.selection_mode_changed.emit)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # Available detection methods (filter to primary names only)
        all_methods = DetectorFactory.get_available_methods()
        # Use only primary method names to avoid duplicates from alternative names
        primary_methods = ["zscore", "iqr", "isolation-forest", "modified-zscore"]
        available_methods = [m for m in primary_methods if m in all_methods]
        
        for method in available_methods:
            # Create detector group
            group = QGroupBox(f"{method.upper()} Detector")
            group_layout = QVBoxLayout(group)
            
            # Enable checkbox
            enable_cb = QCheckBox("Enable")
            is_enabled = method in self.detector_configs and self.detector_configs[method].enabled
            enable_cb.setChecked(is_enabled)
            enable_cb.toggled.connect(
                lambda checked, m=method: self.detector_enabled.emit(self.channel_name, m, checked)
            )
            group_layout.addWidget(enable_cb)
            
            # Parameter controls
            if method in self.detector_configs:
                param_widget = DetectorParameterWidget(
                    method, 
                    self.detector_configs[method].parameters
                )
                param_widget.parameter_changed.connect(
                    lambda m, p, v, ch=self.channel_name: 
                    self.detector_parameter_changed.emit(ch, m, p, v)
                )
                param_widget.setEnabled(is_enabled)
                group_layout.addWidget(param_widget)
                self.detector_widgets[method] = (enable_cb, param_widget)
            else:
                # Default parameters for new detectors
                default_params = self._get_default_parameters(method)
                param_widget = DetectorParameterWidget(method, default_params)
                param_widget.parameter_changed.connect(
                    lambda m, p, v, ch=self.channel_name: 
                    self.detector_parameter_changed.emit(ch, m, p, v)
                )
                param_widget.setEnabled(False)  # Disabled until enabled
                group_layout.addWidget(param_widget)
                self.detector_widgets[method] = (enable_cb, param_widget)
            
            # Connect enable/disable to parameter widget
            enable_cb.toggled.connect(
                lambda checked, pw=param_widget: pw.setEnabled(checked)
            )
            
            layout.addWidget(group)
    
    def _get_default_parameters(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a detection method."""
        defaults = {
            "zscore": {
                "alpha": 0.01,
                "z_threshold": 2.0,
                "eps": 1e-6,
                "learning_period": 10,
                "drift_update_rate": 100
            },
            "iqr": {
                "iqr_multiplier": 2.0,
                "window_size": 50,
                "min_samples": 10
            },
            "isolation-forest": {
                "contamination": 0.1,
                "n_estimators": 100,
                "anomaly_threshold": 0.5
            },
            "modified-zscore": {
                "mad_threshold": 3.5,
                "window_size": 50
            }
        }
        return defaults.get(method, {})
    
    def update_selection_mode(self, mode: str):
        """Update selection mode."""
        self.selection_mode = mode
        self.mode_combo.setCurrentText(mode)
    
    def update_detector_config(self, method: str, config: DetectorConfig):
        """Update detector configuration."""
        self.detector_configs[method] = config
        if method in self.detector_widgets:
            enable_cb, param_widget = self.detector_widgets[method]
            enable_cb.setChecked(config.enabled)
            for param_name, value in config.parameters.items():
                param_widget.update_parameter(param_name, value)


class GlobalDetectorWidget(QWidget):
    """Widget for managing detectors globally across all channels."""
    
    detector_enabled = Signal(str, str, bool)  # channel, method, enabled (will emit for all channels)
    detector_parameter_changed = Signal(str, str, str, object)  # channel, method, param, value (will emit for all channels)
    selection_mode_changed = Signal(str, str)  # channel, mode (will emit for all channels)
    
    def __init__(self, channels_list: List[str], detector_configs: List[DetectorConfig], 
                 default_selection_mode: str = "first", parent=None):
        super().__init__(parent)
        self.channels_list = channels_list
        self.detector_configs = {config.method: config for config in detector_configs}
        self.default_selection_mode = default_selection_mode
        self.detector_widgets = {}
        self.param_widgets = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🌐 Global Detector Configuration (All Sensors)")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        header.setFont(header_font)
        header.setStyleSheet("color: #2E86C1; margin: 10px 0px;")
        layout.addWidget(header)
        
        # Selection mode
        mode_group = QGroupBox("Combination Mode")
        mode_layout = QHBoxLayout(mode_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["first", "majority", "any"])
        self.mode_combo.setCurrentText(self.default_selection_mode)
        self.mode_combo.currentTextChanged.connect(self._on_selection_mode_changed)
        mode_layout.addWidget(QLabel("Selection mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        
        layout.addWidget(mode_group)
        
        # Detector methods
        detector_group = QGroupBox("Detection Methods")
        detector_layout = QVBoxLayout(detector_group)
        
        # Get available methods (filter to primary names only)
        available_methods = DetectorFactory.get_available_methods()
        primary_methods = [m for m in available_methods if m in ["zscore", "iqr", "isolation-forest", "modified-zscore"]]
        
        for method in primary_methods:
            # Method section
            method_frame = QFrame()
            method_frame.setFrameStyle(QFrame.Shape.Box)
            method_frame.setStyleSheet("QFrame { border: 1px solid #BDC3C7; border-radius: 5px; margin: 2px; }")
            method_layout = QVBoxLayout(method_frame)
            
            # Method header with enable checkbox
            header_layout = QHBoxLayout()
            
            enable_checkbox = QCheckBox(f"Enable {method.upper()}")
            enable_checkbox.setStyleSheet("font-weight: bold;")
            config = self.detector_configs.get(method)
            if config:
                enable_checkbox.setChecked(config.enabled)
            
            enable_checkbox.toggled.connect(
                lambda checked, m=method: self._on_detector_enabled(m, checked)
            )
            
            self.detector_widgets[method] = enable_checkbox
            header_layout.addWidget(enable_checkbox)
            header_layout.addStretch()
            
            method_layout.addLayout(header_layout)
            
            # Parameters
            if config and config.parameters:
                param_widget = DetectorParameterWidget(method, config.parameters)
                param_widget.parameter_changed.connect(
                    lambda m, p, v: self._on_parameter_changed(m, p, v)
                )
                method_layout.addWidget(param_widget)
                self.param_widgets[method] = param_widget
            else:
                # Create default parameters
                defaults = self._get_default_parameters(method)
                if defaults:
                    param_widget = DetectorParameterWidget(method, defaults)
                    param_widget.parameter_changed.connect(
                        lambda m, p, v: self._on_parameter_changed(m, p, v)
                    )
                    method_layout.addWidget(param_widget)
                    self.param_widgets[method] = param_widget
            
            detector_layout.addWidget(method_frame)
        
        layout.addWidget(detector_group)
        layout.addStretch()
        
    def _on_detector_enabled(self, method: str, enabled: bool):
        """Handle detector enable/disable - emit for all channels."""
        for channel in self.channels_list:
            self.detector_enabled.emit(channel, method, enabled)
    
    def _on_parameter_changed(self, method: str, param: str, value):
        """Handle parameter change - emit for all channels.""" 
        for channel in self.channels_list:
            self.detector_parameter_changed.emit(channel, method, param, value)
    
    def _on_selection_mode_changed(self, mode: str):
        """Handle selection mode change - emit for all channels."""
        for channel in self.channels_list:
            self.selection_mode_changed.emit(channel, mode)
    
    def _get_default_parameters(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a detector method."""
        defaults = {
            "zscore": {"z_threshold": 2.5},
            "iqr": {"iqr_multiplier": 1.5}, 
            "isolation-forest": {"contamination": 0.1, "n_estimators": 100, "window_size": 100, "min_samples": 20},
            "modified-zscore": {"mad_threshold": 3.5, "window_size": 100}
        }
        return defaults.get(method, {})
    
    def update_detector_config(self, method: str, config: DetectorConfig):
        """Update configuration for a detector method."""
        self.detector_configs[method] = config
        
        if method in self.detector_widgets:
            self.detector_widgets[method].setChecked(config.enabled)
        
        if method in self.param_widgets and config.parameters:
            for param_name, value in config.parameters.items():
                self.param_widgets[method].update_parameter(param_name, value)


class DetectorControlPanel(QWidget):
    """Main control panel for detector method selection."""
    
    detector_enabled = Signal(str, str, bool)  # channel, method, enabled
    detector_parameter_changed = Signal(str, str, str, object)  # channel, method, param, value
    selection_mode_changed = Signal(str, str)  # channel, mode
    apply_to_all_channels = Signal(str, bool)  # method, enabled
    
    def __init__(self, channels_config: Dict[str, List[DetectorConfig]], 
                 default_selection_mode: str = "first", parent=None):
        super().__init__(parent)
        self.channels_config = channels_config
        self.default_selection_mode = default_selection_mode
        self.channels_list = list(channels_config.keys())
        
        # Get the first channel's config as the base for global config
        first_channel_configs = next(iter(channels_config.values())) if channels_config else []
        self.global_detector_widget = None
        
        self.setup_ui(first_channel_configs)
        
    def setup_ui(self, detector_configs: List[DetectorConfig]):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔧 Anomaly Detection Methods")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Quick action buttons
        quick_actions = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(quick_actions)
        
        self.btn_enable_zscore = QPushButton("Enable Z-Score")
        self.btn_enable_zscore.clicked.connect(
            lambda: self._enable_method_for_all("zscore")
        )
        actions_layout.addWidget(self.btn_enable_zscore)
        
        self.btn_enable_iqr = QPushButton("Enable IQR")
        self.btn_enable_iqr.clicked.connect(
            lambda: self._enable_method_for_all("iqr")
        )
        actions_layout.addWidget(self.btn_enable_iqr)
        
        self.btn_enable_isolation = QPushButton("Enable Isolation Forest")
        self.btn_enable_isolation.clicked.connect(
            lambda: self._enable_method_for_all("isolation-forest")
        )
        actions_layout.addWidget(self.btn_enable_isolation)
        
        self.btn_disable_all = QPushButton("Disable All")
        self.btn_disable_all.clicked.connect(self._disable_all_detectors)
        actions_layout.addWidget(self.btn_disable_all)
        
        actions_layout.addStretch()
        layout.addWidget(quick_actions)
        
        # Global detector configuration
        self.global_detector_widget = GlobalDetectorWidget(
            self.channels_list,
            detector_configs,
            self.default_selection_mode
        )
        
        # Connect signals from global widget
        self.global_detector_widget.detector_enabled.connect(self.detector_enabled.emit)
        self.global_detector_widget.detector_parameter_changed.connect(self.detector_parameter_changed.emit)
        self.global_detector_widget.selection_mode_changed.connect(self.selection_mode_changed.emit)
        
        # Wrap in scroll area for long parameter lists
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.global_detector_widget)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        
        layout.addWidget(scroll)
    
    def _enable_method_for_all(self, method: str):
        """Enable a specific method for all channels."""
        if self.global_detector_widget:
            # Update the checkbox in the UI
            if method in self.global_detector_widget.detector_widgets:
                self.global_detector_widget.detector_widgets[method].setChecked(True)
            # This will trigger the signal emission to all channels
            self.global_detector_widget._on_detector_enabled(method, True)
    
    def _disable_all_detectors(self):
        """Disable all detectors on all channels."""
        if self.global_detector_widget:
            for method in self.global_detector_widget.detector_widgets:
                self.global_detector_widget.detector_widgets[method].setChecked(False)
                self.global_detector_widget._on_detector_enabled(method, False)
    
    def update_channel_config(self, channel_name: str, detector_configs: List[DetectorConfig]):
        """Update global configuration (applies to all channels)."""
        if self.global_detector_widget:
            for config in detector_configs:
                self.global_detector_widget.update_detector_config(config.method, config)
    
    def update_selection_mode(self, channel_name: str, mode: str):
        """Update selection mode globally (applies to all channels)."""
        if self.global_detector_widget:
            self.global_detector_widget.mode_combo.setCurrentText(mode)
