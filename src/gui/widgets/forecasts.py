from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class ForecastTable(QTableWidget):
    def __init__(self) -> None:
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["channel", "threshold", "eta (s)"])
        self._row_for_channel: dict[str, int] = {}
        self.setSortingEnabled(True)

    def upsert(self, channel: str, threshold: float, eta_sec: float) -> None:
        if channel in self._row_for_channel:
            row = self._row_for_channel[channel]
        else:
            row = self.rowCount()
            self.insertRow(row)
            self._row_for_channel[channel] = row
            self.setItem(row, 0, QTableWidgetItem(channel))
        self.setItem(row, 1, QTableWidgetItem(f"{threshold:.3f}"))
        
        # Handle infinity case for no forecast
        if eta_sec == float('inf'):
            self.setItem(row, 2, QTableWidgetItem("No trend"))
        else:
            self.setItem(row, 2, QTableWidgetItem(f"{eta_sec:.0f}"))

    def clear_rows(self) -> None:
        self.setRowCount(0)
        self._row_for_channel.clear()
