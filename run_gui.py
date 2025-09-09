#!/usr/bin/env python3
"""
Runner script for the Telemetry Monitoring GUI application.
"""
import sys
from pathlib import Path

# Add src to Python path so we can import modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from gui.main import main

if __name__ == "__main__":
    main()
