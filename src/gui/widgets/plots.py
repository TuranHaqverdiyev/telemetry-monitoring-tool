from typing import Dict, List, Sequence
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout
from PySide6.QtCore import Qt


class MultiChannelPlot(QWidget):
    def __init__(self, channels: List[str], time_window_sec: int = 60):
        super().__init__()
        self._plot = pg.PlotWidget()
        self._plot.addLegend()
        self._curves = {}
        self._threshold_lines = {}  # Static channel thresholds
        self._zscore_lines = {}     # Dynamic Z-score thresholds (mean ± Z*std)
        self._x = []
        self._y = {c: [] for c in channels}
        self._time_window_sec = int(time_window_sec)
        self._current_z_threshold = 3.0  # Default Z threshold
        self._channel_stats = {}  # Store mean/std for each channel

        layout = QVBoxLayout()
        layout.addWidget(self._plot)
        self.setLayout(layout)

        self._colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ]
        self._color_idx = 0
        for i, c in enumerate(channels):
            curve = self._plot.plot(name=c, pen=pg.mkPen(self._colors[i % len(self._colors)], width=2))
            self._curves[c] = curve

    def append(self, ts: float, values: Dict[str, float]):
        self._x.append(ts)
        cutoff = ts - max(5, self._time_window_sec)
        while len(self._x) > 0 and self._x[0] < cutoff:
            self._x.pop(0)
        for c, v in values.items():
            if c not in self._y:
                self.add_channel(c)
            self._y[c].append(v)
            while len(self._y[c]) > len(self._x):
                self._y[c].pop(0)
            # Align X to the length of this channel's Y to avoid length mismatch
            ylen = len(self._y[c])
            x_for_c = self._x[-ylen:] if ylen > 0 else []
            self._curves[c].setData(x_for_c, self._y[c])
        if len(self._x) > 1:
            self._plot.setXRange(self._x[-1] - self._time_window_sec, self._x[-1])

    def set_time_window(self, seconds: int):
        self._time_window_sec = max(5, int(seconds))
        if len(self._x) > 1:
            self._plot.setXRange(self._x[-1] - self._time_window_sec, self._x[-1])

    def add_channel(self, name: str):
        if name in self._curves:
            return
        self._y[name] = []
        color = self._colors[self._color_idx % len(self._colors)]
        self._color_idx += 1
        curve = self._plot.plot(name=name, pen=pg.mkPen(color, width=2))
        self._curves[name] = curve

    def set_threshold(self, channel_base: str, y: float) -> None:
        # One threshold line per base channel in this panel (static breach thresholds)
        if channel_base in self._threshold_lines:
            self._threshold_lines[channel_base].setPos(y)
            return
        line = pg.InfiniteLine(angle=0, pos=y, pen=pg.mkPen('#888888', style=Qt.DashLine))
        self._plot.addItem(line)
        self._threshold_lines[channel_base] = line

    def set_z_threshold(self, z_threshold: float) -> None:
        """Update the Z threshold value and refresh Z-score threshold lines."""
        self._current_z_threshold = z_threshold
        self._update_zscore_lines()

    def update_channel_stats(self, channel: str, mean: float, std: float) -> None:
        """Update channel statistics and refresh Z-score threshold lines."""
        self._channel_stats[channel] = {'mean': mean, 'std': std}
        self._update_zscore_lines()

    def _update_zscore_lines(self) -> None:
        """Update Z-score threshold lines based on current statistics and Z threshold."""
        for channel, stats in self._channel_stats.items():
            if stats['std'] > 0.001:  # Only show if there's meaningful variance
                mean = stats['mean']
                std = stats['std']
                z_thresh = self._current_z_threshold
                
                upper_bound = mean + (z_thresh * std)
                lower_bound = mean - (z_thresh * std)
                
                # Update or create upper bound line
                upper_key = f"{channel}_z_upper"
                if upper_key in self._zscore_lines:
                    self._zscore_lines[upper_key].setPos(upper_bound)
                else:
                    line = pg.InfiniteLine(angle=0, pos=upper_bound, 
                                         pen=pg.mkPen('#ff6b6b', width=2))
                    self._plot.addItem(line)
                    self._zscore_lines[upper_key] = line
                
                # Update or create lower bound line  
                lower_key = f"{channel}_z_lower"
                if lower_key in self._zscore_lines:
                    self._zscore_lines[lower_key].setPos(lower_bound)
                else:
                    line = pg.InfiniteLine(angle=0, pos=lower_bound,
                                         pen=pg.mkPen('#ff6b6b', width=2))
                    self._plot.addItem(line)
                    self._zscore_lines[lower_key] = line

    def clear_data(self) -> None:
        self._x = []
        for ch in list(self._y.keys()):
            self._y[ch] = []
        for curve in self._curves.values():
            curve.setData([], [])
        # Clear channel statistics (Z-score lines will be updated when new data comes)
        self._channel_stats = {}


class PlotGrid(QWidget):
    """Grid of MultiChannelPlot panels, one per group.

    If groups is empty, creates one panel per channel.
    """

    def __init__(self, groups: Dict[str, Sequence[str]] | None, all_channels: Sequence[str], time_window_sec: int = 60):
        super().__init__()
        self._layout = QGridLayout()
        self.setLayout(self._layout)

        if groups:
            items = list(groups.items())
        elif all_channels:
            # Single panel with all channels by default
            items = [("All", list(all_channels))]
        else:
            # No channels provided, create empty grid
            items = []

        self._panels = []
        self._panel_by_name: Dict[str, MultiChannelPlot] = {}
        # Mapping of panel name -> allowed base channel names (for routing)
        self._panel_channels: Dict[str, set] = {}
        self._thresholds = {}
        for idx, (name, chs) in enumerate(items):
            panel = MultiChannelPlot(list(chs), time_window_sec=time_window_sec)
            self._panels.append(panel)
            self._panel_by_name[name] = panel
            self._panel_channels[name] = set(chs)
            r, c = divmod(idx, 2)
            self._layout.addWidget(panel, r, c)

        self._index = items

    def append(self, ts: float, values: Dict[str, float]):
        # Route values to panels based on base channel grouping only (no per-source panel splitting).
        for name, panel in self._panel_by_name.items():
            allowed = self._panel_channels.get(name, set())
            if not allowed:
                continue
            subset: Dict[str, float] = {}
            for k, v in values.items():
                base = k.split(":", 1)[1] if ":" in k else k
                if base in allowed:
                    subset[k] = v
            if subset:
                panel.append(ts, subset)

    def set_time_window(self, seconds: int):
        for panel in self._panels:
            panel.set_time_window(seconds)

    def set_z_threshold(self, z_threshold: float):
        """Update Z threshold for all panels."""
        for panel in self._panels:
            panel.set_z_threshold(z_threshold)

    def update_channel_stats(self, channel: str, mean: float, std: float):
        """Update channel statistics for Z-score threshold calculation."""
        # Route to panels that contain this channel
        base_channel = channel.split(":", 1)[1] if ":" in channel else channel
        for name, panel in self._panel_by_name.items():
            allowed = self._panel_channels.get(name, set())
            if base_channel in allowed:
                panel.update_channel_stats(channel, mean, std)

    def clear(self) -> None:
        for panel in self._panels:
            panel.clear_data()

    def add_panel(self, name: str, channels: Sequence[str]) -> None:
        if name in self._panel_by_name:
            return
        panel = MultiChannelPlot(list(channels))
        self._panels.append(panel)
        self._panel_by_name[name] = panel
        self._panel_channels[name] = set(channels)
        idx = len(self._panels) - 1
        r, c = divmod(idx, 2)
        self._layout.addWidget(panel, r, c)
        self._index.append((name, list(channels)))
        # apply thresholds for any known channels
        for ch in self._panel_channels[name]:
            if ch in self._thresholds:
                panel.set_threshold(ch, float(self._thresholds[ch]))

    def get_panel_names(self) -> List[str]:
        """Return the list of current panel names."""
        return list(self._panel_by_name.keys())

    def remove_panel(self, name: str) -> None:
        """Remove a panel by name and reflow the grid layout."""
        panel = self._panel_by_name.pop(name, None)
        if not panel:
            return
        # Remove bookkeeping
        if name in self._panel_channels:
            del self._panel_channels[name]
        # Remove index entry
        self._index = [(n, chs) for (n, chs) in self._index if n != name]
        # Remove widget from layout and delete it
        try:
            panel.setParent(None)
            panel.deleteLater()
        except Exception:
            pass
        try:
            # Remove from internal list
            self._panels = [p for p in self._panels if p is not panel]
        except Exception:
            pass
        # Reflow remaining widgets
        self._reflow_layout()

    def _reflow_layout(self) -> None:
        """Clear and re-add remaining panels to maintain a compact grid."""
        # Remove all items from layout (widgets remain alive)
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            w = item.widget()
            if w is not None:
                self._layout.removeWidget(w)
        # Re-add in 2-column grid
        for idx, (name, _) in enumerate(self._index):
            panel = self._panel_by_name.get(name)
            if not panel:
                continue
            r, c = divmod(idx, 2)
            self._layout.addWidget(panel, r, c)

    def apply_thresholds(self, thr_map: Dict[str, float]) -> None:
        """Apply thresholds by base-channel across all panels."""
        self._thresholds = dict(thr_map)
        for name, panel in self._panel_by_name.items():
            allowed = self._panel_channels.get(name, set())
            for ch in allowed:
                if ch in self._thresholds:
                    panel.set_threshold(ch, float(self._thresholds[ch]))
