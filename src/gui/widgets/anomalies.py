from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from PySide6.QtGui import QColor


class AnomalyTable(QWidget):
    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Time", "Channel", "Value", "Z", "Std", "Severity", "p-value", "ETA (s)"])
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_row(self, ts: float, channel: str, value: float, z: float, std: float, severity: str = "", pvalue: float = None, eta_sec: float = None):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f"{ts:.2f}"))
        self.table.setItem(row, 1, QTableWidgetItem(channel))
        self.table.setItem(row, 2, QTableWidgetItem(f"{value:.3f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{z:.2f}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{std:.3f}"))
        if severity:
            self.table.setItem(row, 5, QTableWidgetItem(severity))
        if pvalue is not None:
            self.table.setItem(row, 6, QTableWidgetItem(f"{pvalue:.3g}"))
        if eta_sec is not None:
            self.table.setItem(row, 7, QTableWidgetItem(f"{eta_sec:.0f}"))

        # Row coloring by severity
        color = None
        if severity.lower() == "critical":
            color = QColor(255, 224, 224)  # light red
        elif severity.lower() == "warn":
            color = QColor(255, 245, 204)  # light yellow
        if color:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(color)
        self.table.scrollToBottom()
