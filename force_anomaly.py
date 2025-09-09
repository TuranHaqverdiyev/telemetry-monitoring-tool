#!/usr/bin/env python3
"""
Force trigger anomaly alerts by creating extreme values
"""
import sys
from pathlib import Path
import time

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from alerts.alert_config import AlertConfig, EmailSettings, AlertRecipients, RateLimiting, EmailTemplateConfig
from alerts.alert_manager import AlertManager

def force_anomaly_alert():
    """Force send an anomaly alert."""
    print("🚨 Force Triggering Anomaly Alert...")
    
    # Create exact same config as application
    email_settings = EmailSettings(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        use_tls=True,
        username="demodata93@gmail.com",
        password="vhylwagyfynchbgc",
        from_email="demodata93@gmail.com"
    )
    
    recipients = AlertRecipients()
    recipients.critical = ["demo.turanrh@gmail.com"]
    recipients.high = ["demo.turanrh@gmail.com"] 
    recipients.medium = ["demo.turanrh@gmail.com"]
    recipients.low = ["demo.turanrh@gmail.com"]
    
    rate_limiting = RateLimiting(
        max_alerts_per_hour=10,
        cooldown_minutes=5,
        batch_window_minutes=15
    )
    
    templates = EmailTemplateConfig(
        subject_prefix="[TELEMETRY ALERT]",
        company_name="Telemetry Monitoring",
        include_raw_data=True,
        include_charts=False
    )
    
    config = AlertConfig(
        enabled=True,
        email=email_settings,
        recipients=recipients,
        rate_limiting=rate_limiting,
        templates=templates
    )
    
    # Create alert manager
    alert_manager = AlertManager(config)
    
    # Send forced anomaly alert
    print("📧 Sending critical anomaly alert...")
    success = alert_manager.send_alert(
        channel="test:temperature",
        value=99.9,  # Extreme temperature
        timestamp=time.time(),
        severity="critical",
        detector_method="manual-test",
        anomaly_score=1.0,  # Maximum anomaly score
        explanation="FORCED TEST: Extreme temperature anomaly detected for testing email alerts",
        raw_data={
            "detector": "manual-test",
            "z_score": 8.5,
            "threshold": 3.0,
            "baseline_mean": 22.0,
            "baseline_std": 1.2
        }
    )
    
    if success:
        print("✅ Critical anomaly alert sent successfully!")
        print("📧 Check demo.turanrh@gmail.com for the anomaly alert email")
    else:
        print("❌ Failed to send anomaly alert")
    
    return success

if __name__ == "__main__":
    force_anomaly_alert()
