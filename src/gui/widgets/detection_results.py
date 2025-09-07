"""
Enhanced Detection Results Panel

This widget provides detailed anomaly information including explanations,
confidence scores, detection method details, and operator guidance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QTextEdit, QLabel, QSplitter, QGroupBox, QProgressBar, QPushButton,
    QTabWidget, QScrollArea, QFrame, QGridLayout, QHeaderView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap, QPainter

from ...telemetry.detector_base import AnomalyResult, AnomalyExplanation


class ConfidenceIndicator(QWidget):
    """Visual confidence score indicator."""
    
    def __init__(self, confidence: float, parent=None):
        super().__init__(parent)
        self.confidence = max(0.0, min(1.0, confidence))
        self.setFixedSize(100, 20)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # Confidence bar
        width = int(self.confidence * self.width())
        if self.confidence >= 0.8:
            color = QColor(76, 175, 80)  # Green - high confidence
        elif self.confidence >= 0.6:
            color = QColor(255, 193, 7)  # Yellow - medium confidence  
        else:
            color = QColor(244, 67, 54)  # Red - low confidence
        
        painter.fillRect(0, 0, width, self.height(), color)
        
        # Border
        painter.setPen(QColor(200, 200, 200))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        # Text
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.confidence:.1%}")


class SeverityBadge(QLabel):
    """Visual severity indicator badge."""
    
    def __init__(self, severity: str, parent=None):
        super().__init__(parent)
        self.severity = severity.lower()
        self.setText(severity.upper())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(80, 24)
        
        # Style based on severity
        if self.severity == "critical":
            self.setStyleSheet("""
                QLabel {
                    background-color: #f44336;
                    color: white;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 9px;
                }
            """)
        elif self.severity == "high":
            self.setStyleSheet("""
                QLabel {
                    background-color: #ff9800;
                    color: white;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 9px;
                }
            """)
        elif self.severity == "medium":
            self.setStyleSheet("""
                QLabel {
                    background-color: #ffc107;
                    color: black;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 9px;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: #4caf50;
                    color: white;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 9px;
                }
            """)


class DetectionMethodBadge(QLabel):
    """Badge showing detection method(s) used."""
    
    def __init__(self, methods: List[str], parent=None):
        super().__init__(parent)
        if len(methods) == 1:
            text = methods[0].upper()
            color = "#2196f3"  # Blue for single method
        else:
            text = f"MULTI ({len(methods)})"
            color = "#9c27b0"  # Purple for multiple methods
        
        self.setText(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(80, 20)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 8px;
            }}
        """)
        
        # Tooltip with method details
        if len(methods) > 1:
            self.setToolTip(f"Methods: {', '.join(methods)}")
        else:
            self.setToolTip(f"Detection method: {methods[0]}")


class AnomalyDetailWidget(QWidget):
    """Detailed view of a single anomaly."""
    
    def __init__(self, anomaly_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.anomaly_data = anomaly_data
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with timestamp and channel
        header_layout = QHBoxLayout()
        
        timestamp = datetime.fromtimestamp(self.anomaly_data.get("ts", 0))
        time_label = QLabel(f"🕐 {timestamp.strftime('%H:%M:%S')}")
        time_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        channel_label = QLabel(f"📊 {self.anomaly_data.get('channel', 'Unknown')}")
        channel_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #1976d2;")
        
        value_label = QLabel(f"📈 {self.anomaly_data.get('value', 0):.3f}")
        value_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        header_layout.addWidget(time_label)
        header_layout.addWidget(channel_label)
        header_layout.addWidget(value_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Badges and indicators row
        badges_layout = QHBoxLayout()
        
        # Severity badge
        severity = self.anomaly_data.get("severity", "medium")
        badges_layout.addWidget(SeverityBadge(severity))
        
        # Detection method badge
        detector_method = self.anomaly_data.get("detector_method", "unknown")
        if ":" in detector_method:
            methods = [method.strip() for method in detector_method.split(":")]
        else:
            methods = [detector_method]
        badges_layout.addWidget(DetectionMethodBadge(methods))
        
        # Confidence indicator
        anomaly_score = self.anomaly_data.get("anomaly_score", 0.0)
        confidence = min(anomaly_score, 1.0)  # Normalize to 0-1
        badges_layout.addWidget(QLabel("Confidence:"))
        badges_layout.addWidget(ConfidenceIndicator(confidence))
        
        badges_layout.addStretch()
        layout.addLayout(badges_layout)
        
        # Statistics grid
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QGridLayout(stats_group)
        
        # Z-score or anomaly score
        z_score = self.anomaly_data.get("z", 0.0)
        stats_layout.addWidget(QLabel("Anomaly Score:"), 0, 0)
        stats_layout.addWidget(QLabel(f"{anomaly_score:.3f}"), 0, 1)
        
        if abs(z_score) > 0.01:  # Show Z-score if available
            stats_layout.addWidget(QLabel("Z-Score:"), 1, 0)
            stats_layout.addWidget(QLabel(f"{z_score:.3f}"), 1, 1)
        
        # Statistical measures
        mean_val = self.anomaly_data.get("mean", 0.0)
        std_val = self.anomaly_data.get("std", 0.0)
        if abs(mean_val) > 0.01 or abs(std_val) > 0.01:
            stats_layout.addWidget(QLabel("Baseline Mean:"), 2, 0)
            stats_layout.addWidget(QLabel(f"{mean_val:.3f}"), 2, 1)
            stats_layout.addWidget(QLabel("Std Deviation:"), 3, 0)
            stats_layout.addWidget(QLabel(f"{std_val:.3f}"), 3, 1)
        
        # P-value if available
        pvalue = self.anomaly_data.get("pvalue")
        if pvalue is not None:
            stats_layout.addWidget(QLabel("P-Value:"), 4, 0)
            stats_layout.addWidget(QLabel(f"{pvalue:.3e}"), 4, 1)
        
        layout.addWidget(stats_group)
        
        # Explanation text
        explanation = self.anomaly_data.get("explanation_summary", "")
        if explanation:
            explanation_group = QGroupBox("Explanation")
            explanation_layout = QVBoxLayout(explanation_group)
            
            explanation_text = QTextEdit()
            explanation_text.setPlainText(explanation)
            explanation_text.setMaximumHeight(80)
            explanation_text.setReadOnly(True)
            explanation_layout.addWidget(explanation_text)
            
            layout.addWidget(explanation_group)
        
        # Multi-method details if available
        num_detectors = self.anomaly_data.get("num_detectors", 0)
        num_anomalies = self.anomaly_data.get("num_anomalies", 0)
        if num_detectors > 1:
            multi_group = QGroupBox("Multi-Method Analysis")
            multi_layout = QGridLayout(multi_group)
            
            multi_layout.addWidget(QLabel("Total Detectors:"), 0, 0)
            multi_layout.addWidget(QLabel(str(num_detectors)), 0, 1)
            
            multi_layout.addWidget(QLabel("Anomaly Detectors:"), 1, 0)
            multi_layout.addWidget(QLabel(str(num_anomalies)), 1, 1)
            
            consensus = f"{num_anomalies}/{num_detectors}"
            multi_layout.addWidget(QLabel("Consensus:"), 2, 0)
            multi_layout.addWidget(QLabel(consensus), 2, 1)
            
            layout.addWidget(multi_group)


class EnhancedAnomalyTable(QTableWidget):
    """Enhanced anomaly table with detailed information."""
    
    anomaly_selected = Signal(dict)  # Emitted when an anomaly is selected
    
    def __init__(self, parent=None):
        super().__init__(0, 9, parent)
        
        # Enhanced headers for multi-method support
        self.setHorizontalHeaderLabels([
            "Time", "Channel", "Value", "Method", "Score", 
            "Severity", "Confidence", "Explanation", "Details"
        ])
        
        # Configure table
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        
        # Resize columns
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Time
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Channel
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Value
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Method
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Score
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Severity
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # Confidence
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)         # Explanation
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)  # Details
        
        # Connect selection signal
        self.itemSelectionChanged.connect(self._on_selection_changed)
        
    def add_anomaly(self, anomaly_data: Dict[str, Any]):
        """Add an anomaly to the table with enhanced display."""
        row = self.rowCount()
        self.insertRow(row)
        
        # Timestamp
        ts = anomaly_data.get("ts", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        self.setItem(row, 0, QTableWidgetItem(time_str))
        
        # Channel
        channel = anomaly_data.get("channel", "")
        self.setItem(row, 1, QTableWidgetItem(channel))
        
        # Value
        value = anomaly_data.get("value", 0.0)
        self.setItem(row, 2, QTableWidgetItem(f"{value:.3f}"))
        
        # Detection method
        detector_method = anomaly_data.get("detector_method", "unknown")
        method_display = detector_method.split(":")[0] if ":" in detector_method else detector_method
        self.setItem(row, 3, QTableWidgetItem(method_display))
        
        # Anomaly score
        anomaly_score = anomaly_data.get("anomaly_score", 0.0)
        self.setItem(row, 4, QTableWidgetItem(f"{anomaly_score:.3f}"))
        
        # Severity with color coding
        severity = anomaly_data.get("severity", "medium")
        severity_item = QTableWidgetItem(severity.upper())
        if severity.lower() == "critical":
            severity_item.setBackground(QColor(255, 224, 224))
        elif severity.lower() == "high":
            severity_item.setBackground(QColor(255, 245, 204))
        elif severity.lower() == "medium":
            severity_item.setBackground(QColor(255, 255, 224))
        self.setItem(row, 5, severity_item)
        
        # Confidence as percentage
        confidence = min(anomaly_score, 1.0)
        confidence_item = QTableWidgetItem(f"{confidence:.1%}")
        if confidence >= 0.8:
            confidence_item.setBackground(QColor(200, 255, 200))
        elif confidence >= 0.6:
            confidence_item.setBackground(QColor(255, 255, 200))
        else:
            confidence_item.setBackground(QColor(255, 200, 200))
        self.setItem(row, 6, confidence_item)
        
        # Explanation summary
        explanation = anomaly_data.get("explanation_summary", "")
        explanation_item = QTableWidgetItem(explanation[:100] + "..." if len(explanation) > 100 else explanation)
        explanation_item.setToolTip(explanation)  # Full text in tooltip
        self.setItem(row, 7, explanation_item)
        
        # Details button
        details_btn = QPushButton("View")
        details_btn.setFixedSize(50, 25)
        details_btn.clicked.connect(lambda: self._show_details(anomaly_data))
        self.setCellWidget(row, 8, details_btn)
        
        # Store anomaly data for later retrieval
        first_item = self.item(row, 0)
        if first_item:
            first_item.setData(Qt.ItemDataRole.UserRole, anomaly_data)
        
        # Auto-scroll to bottom
        self.scrollToBottom()
    
    def _on_selection_changed(self):
        """Handle row selection to show detailed information."""
        selected_rows = set()
        for item in self.selectedItems():
            selected_rows.add(item.row())
        
        if selected_rows:
            row = list(selected_rows)[0]  # Get first selected row
            first_item = self.item(row, 0)
            if first_item:
                anomaly_data = first_item.data(Qt.ItemDataRole.UserRole)
                if anomaly_data:
                    self.anomaly_selected.emit(anomaly_data)
    
    def _show_details(self, anomaly_data: Dict[str, Any]):
        """Show detailed anomaly information in a popup."""
        # For now, just emit the selection signal
        self.anomaly_selected.emit(anomaly_data)


class DetectionResultsPanel(QWidget):
    """Enhanced detection results panel with detailed anomaly information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🔍 Detection Results & Analysis")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Create splitter for table and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Enhanced anomaly table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        table_label = QLabel("📋 Anomaly Detection Log")
        table_label.setStyleSheet("font-weight: bold; color: #1976d2;")
        table_layout.addWidget(table_label)
        
        self.anomaly_table = EnhancedAnomalyTable()
        table_layout.addWidget(self.anomaly_table)
        
        # Statistics summary
        self.stats_label = QLabel("📊 Total Anomalies: 0 | Critical: 0 | High: 0 | Medium: 0")
        self.stats_label.setStyleSheet("padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
        table_layout.addWidget(self.stats_label)
        
        splitter.addWidget(table_widget)
        
        # Right side: Detailed anomaly information
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        details_label = QLabel("🔬 Anomaly Details")
        details_label.setStyleSheet("font-weight: bold; color: #1976d2;")
        details_layout.addWidget(details_label)
        
        # Scroll area for anomaly details
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setMinimumWidth(300)
        
        # Placeholder for when no anomaly is selected
        self.no_selection_widget = QLabel("Select an anomaly from the table to view detailed information")
        self.no_selection_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_selection_widget.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
        self.details_scroll.setWidget(self.no_selection_widget)
        
        details_layout.addWidget(self.details_scroll)
        
        splitter.addWidget(details_widget)
        
        # Set splitter proportions (table gets more space)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
        
        # Connect signals
        self.anomaly_table.anomaly_selected.connect(self.show_anomaly_details)
        
        # Statistics tracking
        self.anomaly_stats = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    
    def add_anomaly(self, anomaly_data: Dict[str, Any]):
        """Add an anomaly to the results panel."""
        self.anomaly_table.add_anomaly(anomaly_data)
        
        # Update statistics
        self.anomaly_stats["total"] += 1
        severity = anomaly_data.get("severity", "medium").lower()
        if severity in self.anomaly_stats:
            self.anomaly_stats[severity] += 1
        
        # Update stats display
        self.update_stats_display()
    
    def show_anomaly_details(self, anomaly_data: Dict[str, Any]):
        """Show detailed information for the selected anomaly."""
        detail_widget = AnomalyDetailWidget(anomaly_data)
        self.details_scroll.setWidget(detail_widget)
    
    def update_stats_display(self):
        """Update the statistics display."""
        stats = self.anomaly_stats
        stats_text = (f"📊 Total: {stats['total']} | "
                     f"🔴 Critical: {stats['critical']} | "
                     f"🟠 High: {stats['high']} | "
                     f"🟡 Medium: {stats['medium']} | "
                     f"🟢 Low: {stats['low']}")
        self.stats_label.setText(stats_text)
    
    def clear_results(self):
        """Clear all anomaly results."""
        self.anomaly_table.clearContents()
        self.anomaly_table.setRowCount(0)
        self.anomaly_stats = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        self.update_stats_display()
        self.details_scroll.setWidget(self.no_selection_widget)
