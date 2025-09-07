# Multi-Method Telemetry Monitoring Tool

A comprehensive, real-time telemetry monitoring system featuring **4 advanced anomaly detection methods** with machine learning capabilities and an intuitive GUI interface.

## 🎯 **Key Features**

### **Multi-Method Anomaly Detection**

- **Z-Score Detection**: Traditional statistical method using exponentially weighted moving averages
- **IQR Detection**: Quartile-based robust method for non-normal data
- **Isolation Forest**: Machine learning approach for multivariate pattern detection
- **Modified Z-Score**: Median Absolute Deviation (MAD) based robust detection

### **Advanced Capabilities**

- **Real-time Processing**: Live telemetry monitoring with low-latency detection
- **Multi-Channel Support**: Independent detection per channel with method mixing
- **GUI Control Panel**: Intuitive interface for parameter tuning and method switching
- **Configurable Detection Modes**: First/Majority/Any/All selection strategies
- **Rich Explanations**: Detailed anomaly context with confidence scoring

### **Production Ready**

- **Factory Pattern**: Unified detector creation and management
- **Configuration System**: JSON-based setup with method-specific parameters
- **Data Source Flexibility**: UDP, file tail, or built-in simulator
- **Comprehensive Testing**: Validated detection algorithms with performance benchmarks

---

## 🚀 **Quick Start**

### **Installation**

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### **Launch GUI Application**

```powershell
# Start the complete monitoring interface
python -m src.gui.main
```

### **Run with Configurations**

```powershell
# Basic Z-Score detection
python -m src.gui.main --config config.json

# Full multi-method detection
python -m src.gui.main --config config.production.json

# UDP data source
python -m src.gui.main --config config.udp.json

# File tail monitoring
python -m src.gui.main --config config.tail.json
```

---

## 📁 **Project Structure**

```
telemetry-monitoring-tool/
├── src/                          # Core application code
│   ├── telemetry/               # Detection algorithms
│   │   ├── anomaly.py          # Z-Score detector
│   │   ├── iqr_detector.py     # IQR detector  
│   │   ├── isolation_forest_detector.py  # ML detector
│   │   ├── modified_zscore_detector.py   # MAD detector
│   │   ├── detector_base.py    # Factory and base classes
│   │   └── ...
│   ├── gui/                    # User interface
│   │   ├── main.py            # Main application
│   │   ├── widgets/           # GUI components
│   │   └── ...
│   └── ingest/                # Data sources
├── tests/                     # Unit tests
├── examples/                  # Sample data files
├── config.json               # Basic configuration
├── config.production.json    # Multi-method configuration
├── config.example.json       # Template configuration
└── requirements.txt          # Dependencies
```

---

## 🔧 **Configuration Guide**

### **Basic Configuration** (`config.json`)

Simple Z-Score detection setup:

```json
{
  "rate_hz": 5.0,
  "detector": {
    "alpha": 0.001,
    "z_threshold": 3.3
  },
  "channels": [
    {
      "name": "temp_c",
      "unit": "°C",
      "min": 0,
      "max": 100
    }
  ]
}
```

### **Production Configuration** (`config.production.json`)

Complete multi-method setup with all 4 detection algorithms:

```json
{
  "default_detectors": [
    {
      "method": "zscore",
      "enabled": true,
      "parameters": {
        "z_threshold": 3.0,
        "alpha": 0.1
      }
    },
    {
      "method": "iqr",
      "enabled": true,
      "parameters": {
        "iqr_multiplier": 2.0,
        "window_size": 50
      }
    },
    {
      "method": "isolation-forest",
      "enabled": true,
      "parameters": {
        "contamination": 0.1,
        "min_samples": 25,
        "anomaly_threshold": -0.4
      }
    },
    {
      "method": "modified-zscore",
      "enabled": true,
      "parameters": {
        "mad_threshold": 3.0,
        "window_size": 100
      }
    }
  ],
  "channels": {
    "temp_c": {
      "detectors": [
        {"method": "isolation-forest"},
        {"method": "modified-zscore"}
      ]
    }
  }
}
```

---

## 🎛️ **Detection Methods**

### **1. Z-Score Detection** (`zscore`)

- **Best For**: Normal distributed data, fast detection
- **Algorithm**: Exponentially weighted moving averages with configurable threshold
- **Parameters**: `z_threshold`, `alpha`, `eps`

### **2. IQR Detection** (`iqr`)

- **Best For**: Non-normal data, robust to outliers
- **Algorithm**: Interquartile range based detection with sliding window
- **Parameters**: `iqr_multiplier`, `window_size`, `min_samples`

### **3. Isolation Forest** (`isolation-forest`)

- **Best For**: Complex multivariate patterns, unknown anomaly types
- **Algorithm**: Machine learning ensemble with feature engineering
- **Parameters**: `contamination`, `min_samples`, `anomaly_threshold`

### **4. Modified Z-Score** (`modified-zscore`)

- **Best For**: Heavy outlier contamination, robust baselines
- **Algorithm**: Median Absolute Deviation (MAD) based detection
- **Parameters**: `mad_threshold`, `window_size`, `min_samples`

---

## 🖥️ **GUI Interface**

### **Controls Tab**

- **Data Source Selection**: Choose between simulator, UDP, or file sources
- **Rate Control**: Adjust data processing frequency
- **Detection Mode**: Select First/Majority/Any/All combination logic

### **Detectors Tab**

- **Method Selection**: Enable/disable individual detection methods
- **Parameter Tuning**: Real-time adjustment of detection parameters
- **Channel Configuration**: Method-specific settings per channel

### **Results Tab**

- **Live Anomaly Display**: Real-time detection results with confidence scores
- **Method Attribution**: Clear identification of which method detected anomalies
- **Severity Classification**: Color-coded severity levels (Low/Medium/High/Critical)

---

## 📊 **Performance Benchmarks**


| Method           | Avg Processing Time | Detection Sensitivity    | Best Use Case            |
| ---------------- | ------------------- | ------------------------ | ------------------------ |
| Z-Score          | 0.00ms              | High (normal data)       | Real-time statistical    |
| IQR              | 0.02ms              | Medium (robust)          | Non-normal distributions |
| Isolation Forest | 6.25ms              | High (multivariate)      | Complex patterns         |
| Modified Z-Score | 0.00ms              | High (outlier-resistant) | Contaminated data        |

---

## 🔌 **Data Sources**

### **Built-in Simulator**

```powershell
# Start with synthetic telemetry
python -m src.gui.main --config config.production.json
```

### **UDP Data Source**

```powershell
# Listen for UDP telemetry packets
python -m src.gui.main --config config.udp.json
```

### **File Tail Monitoring**

```powershell
# Monitor log files in real-time
python -m src.gui.main --config config.tail.json
```

---

## 🧪 **Testing**

Run the test suite to validate all detection methods:

```powershell
# Run unit tests
python -m pytest tests/

# Test individual methods
python -m pytest tests/test_anomaly.py
python -m pytest tests/test_forecast.py
```

---

## 🚀 **Production Deployment**

### **System Requirements**

- Python 3.8+
- 4GB RAM minimum (8GB recommended for ML methods)
- Windows/Linux/macOS support

### **Performance Tuning**

- **High Frequency**: Use Z-Score or Modified Z-Score for >100Hz data
- **Complex Patterns**: Enable Isolation Forest for multivariate analysis
- **Robust Detection**: Use IQR or Modified Z-Score for noisy environments

### **Packaging as Executable**

```powershell
pip install pyinstaller
pyinstaller --noconfirm --onefile --name telemetry-monitor src\gui\main.py
```

---

## 📈 **Advanced Features**

### **Multi-Method Ensemble**

Combine multiple detection methods for enhanced accuracy:

- **Majority Voting**: Anomaly declared when most methods agree
- **Any Detection**: Alert on first method detection
- **All Detection**: Conservative approach requiring all methods

### **Adaptive Thresholds**

- **Time-based**: Adjust sensitivity based on time of day
- **Performance-based**: Auto-tune based on false positive rates
- **Channel-specific**: Independent tuning per telemetry channel

---

## 🤝 **Contributing**

This project implements a complete multi-method anomaly detection pipeline. Key areas for enhancement:

1. **Additional ML Methods**: LSTM, Autoencoders, One-Class SVM
2. **Advanced Ensemble**: Meta-learning and weighted voting
3. **Real-time Adaptation**: Online learning and drift detection
4. **API Integration**: REST API for external system integration
