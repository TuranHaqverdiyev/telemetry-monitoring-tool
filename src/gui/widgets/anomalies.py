from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from PySide6.QtGui import QColor


class AnomalyTable(QWidget):
    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Time", "Channel", "Value", "Z", "Std", "Severity", "p-value", "ETA (s)"])
        
        # Make the table fully extendable and scrollable
        from PySide6.QtWidgets import QSizePolicy, QHeaderView
        
        # Set size policies to allow unlimited expansion
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Configure table for better display
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)  # Hide row numbers for cleaner look
        
        # Set minimum size but allow unlimited growth
        self.table.setMinimumHeight(200)  # Minimum height for at least 6-8 rows
        
        # Configure column headers
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)  # Last column takes remaining space
        
        # Layout setup
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        layout.addWidget(self.table)
        self.setLayout(layout)

    def add_row(self, ts: float, channel: str, value: float, z: float, std: float, severity: str = "", pvalue: float | None = None, eta_sec: float | None = None):
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

        # Enhanced row coloring by severity with better contrast
        color = None
        text_color = None
        if severity.lower() == "critical":
            color = QColor(220, 53, 69)  # Bootstrap danger red
            text_color = QColor(255, 255, 255)  # white text
        elif severity.lower() == "high":
            color = QColor(253, 126, 20)  # Bootstrap warning orange  
            text_color = QColor(255, 255, 255)  # white text
        elif severity.lower() == "warn" or severity.lower() == "medium":
            color = QColor(255, 193, 7)  # Bootstrap warning yellow
            text_color = QColor(33, 37, 41)  # dark text
        elif severity.lower() == "low":
            color = QColor(25, 135, 84)  # Bootstrap success green
            text_color = QColor(255, 255, 255)  # white text
        else:
            # Default for info/unknown
            color = QColor(13, 202, 240)  # Bootstrap info blue
            text_color = QColor(33, 37, 41)  # dark text
            
        if color:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(color)
                    if text_color:
                        item.setForeground(text_color)
        
        # Add alternating row colors for better readability
        if row % 2 == 0 and not color:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setBackground(QColor(248, 249, 250))  # Light gray
        
        # Optional: Limit table size for performance (keep last 1000 rows)
        max_rows = 1000
        if self.table.rowCount() > max_rows:
            self.table.removeRow(0)  # Remove oldest row
        
        self.table.scrollToBottom()

    def clear_table(self):
        """Clear all rows from the table"""
        self.table.setRowCount(0)
    
    def get_row_count(self):
        """Get current number of rows"""
        return self.table.rowCount()
    
    def export_to_csv(self, filename: str):
        """Export table data to CSV file"""
        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write headers
            headers = []
            for col in range(self.table.columnCount()):
                headers.append(self.table.horizontalHeaderItem(col).text())
            writer.writerow(headers)
            
            # Write data
            for row in range(self.table.rowCount()):
                row_data = []
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)
