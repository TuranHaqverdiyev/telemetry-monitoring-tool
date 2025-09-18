# Telemetry Monitoring Tool

Real-time telemetry monitoring with an intuitive GUI and multiple anomaly detection methods: Z-Score, IQR, Isolation Forest, and Modified Z-Score. Supports simulator, UDP, and file tail data sources, plus optional email alerts.

## Highlights

- Multi-method anomaly detection per channel (mix and match detectors)
- Detection combination modes: first, majority, any, all
- GUI controls for window size, Z-Score parameters, detector settings, and alerts
- Email alerting with rate limiting and severity-based recipients
- Works out of the box with a built-in simulator; optional UDP and file inputs
- CSV logging (anomalies.csv) and GUI snapshots (snapshots/)

---

## Quick Start (Windows PowerShell)

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Launch the GUI (uses config.json by default)
python -m src.gui.main

# Optional: specify a config file
python -m src.gui.main --config config.json
```

---

## Project Structure

```
telemetry-monitoring-tool/
├── config.json                # Main configuration (data sources, detectors, alerts)
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Package metadata
├── snapshots/                 # Saved GUI snapshots (output)
└── src/
    ├── gui/                   # GUI application
    │   ├── main.py            # Entry point (python -m src.gui.main)
    │   ├── config.py          # Config loader (detectors, alerts, channels)
    │   ├── service.py         # Streaming/processing worker    │   └── widgets/          # UI components (plots, anomalies, detectors, alerts)
    ├── telemetry/             # Detection algorithms and models
    │   ├── anomaly.py                 # EWMA Z-Score (+ RangeGuard)
    │   ├── iqr_detector.py            # IQR-based detector (sliding window)
    │   ├── isolation_forest_detector.py # Isolation Forest (scikit-learn)
    │   ├── modified_zscore_detector.py  # MAD-based detector
    │   └── detector_base.py            # Base classes + DetectorFactory
    ├── alerts/                # Email alerting
    │   ├── alert_config.py
    │   ├── alert_manager.py
    │   └── email_alerter.py
    └── ingest/                # Data source utilities
        └── sources.py         # udp_json_stream(), file_tail()
```

---

## Configuration Overview (config.json)

The app reads `config.json` by default. Key sections:

- data_sources: one or more input streams
  - type: "simulator" | "udp" | "file"
  - simulator example uses `simulator_config` (freq_hz, channels, noise, etc.)
  - udp example fields: host, port
  - file example fields: path (expects JSON lines)
- default_detectors: fallback detector list applied to channels that don’t override
- channels: per-channel overrides (method list and parameters)
- detector_selection_mode: "first" | "majority" | "any" | "all"
- alerts: email settings, recipients, rate limiting, and templates

Example snippets:

Simulator input (default):

```json
{
  "data_sources": [
    {
      "type": "simulator",
      "name": "internal_sim",
      "simulator_config": {
        "freq_hz": 1.0,
        "channels": ["sim:temp_c", "sim:voltage", "sim:current"]
      }
    }
  ]
}
```

UDP input:

```json
{
  "data_sources": [
    { "type": "udp", "name": "udp1", "host": "0.0.0.0", "port": 9999 }
  ]
}
```

File tail input (JSON lines):

```json
{
  "data_sources": [
    { "type": "file", "name": "log", "path": "C:/path/to/telemetry.jsonl" }
  ]
}
```

Detectors (defaults):

```json
{
  "default_detectors": [
    { "method": "zscore", "enabled": true, "parameters": { "z_threshold": 3.0, "alpha": 0.1 } },
    { "method": "iqr", "enabled": true, "parameters": { "iqr_multiplier": 2.0, "window_size": 50 } },
    { "method": "isolation-forest", "enabled": true, "parameters": { "contamination": 0.1, "window_size": 100, "min_samples": 50 } },
    { "method": "modified-zscore", "enabled": true, "parameters": { "mad_threshold": 3.5, "window_size": 100, "min_samples": 10 } }
  ],
  "detector_selection_mode": "majority"
}
```

Email alerts (placeholders):

```json
{
  "alerts": {
    "enabled": true,
    "email": {
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "use_tls": true,
      "username": "your-email@gmail.com",
      "password": "your-app-password",
      "from_email": "your-email@gmail.com"
    },
    "recipients": {
      "critical": ["ops@example.com"],
      "high": ["team@example.com"],
      "medium": [],
      "low": []
    }
  }
}
```

Notes:

- Channel names may be prefixed by the source when "Group by Source" is enabled in the GUI (e.g., "udp1:temp_c").
- The GUI exposes an Alerts tab where you can view/edit these settings at runtime.

---

## Using the GUI

- Controls tab: set time window, EWMA alpha, Z threshold, notifications, and start/stop
- Detectors tab: enable/disable methods per channel and tune parameters; set combination mode
- Alerts tab: configure SMTP and recipients; toggle alerts on/off
- Add Plot: create charts for selected channels; Save Snapshot writes to `snapshots/`
- Log CSV: anomalies are written to `anomalies.csv` in the project root

---

## Troubleshooting

- Virtualenv/pip issues: prefer `python -m pip ...` and ensure your venv is activated
- Import/module errors: launch from the repo root with `python -m src.gui.main` (the app also shims `src/` on sys.path)
- PySide6 not found: reinstall dependencies and verify the venv is active
- Email errors: use an app password, correct SMTP/port, and valid recipients; check firewall
- UDP input: ensure the port is open and packets are valid JSON

