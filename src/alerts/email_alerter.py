"""
Email Alerter

Handles SMTP email sending for anomaly alerts with retry logic and templating.
"""

import logging
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass

from .alert_config import AlertConfig, EmailSettings

logger = logging.getLogger(__name__)

@dataclass
class AlertData:
    """Alert data structure for email generation."""
    channel: str
    value: float
    timestamp: float
    severity: str
    detector_method: str
    anomaly_score: float
    explanation: Optional[str] = None
    raw_data: Dict[str, Any] = None

class EmailAlerter:
    """SMTP email alerter with retry logic and templating."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self._smtp_connection = None
        self._connection_lock = threading.Lock()
        self._last_connection_attempt = 0
        self._connection_retry_delay = 60  # seconds
        
    def send_alert(self, alert_data: AlertData, recipients: List[str]) -> bool:
        """
        Send email alert for anomaly detection.
        
        Args:
            alert_data: Alert information
            recipients: List of email addresses
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not recipients:
            logger.warning("No recipients specified for alert")
            return False
            
        try:
            # Generate email content
            subject = self._generate_subject(alert_data)
            body_html = self._generate_html_body(alert_data)
            body_text = self._generate_text_body(alert_data)
            
            # Send email
            return self._send_email(
                recipients=recipients,
                subject=subject,
                body_html=body_html,
                body_text=body_text
            )
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
    
    def send_test_email(self, recipient: str) -> bool:
        """Send a test email to verify configuration."""
        try:
            test_data = AlertData(
                channel="test_channel",
                value=42.0,
                timestamp=time.time(),
                severity="medium",
                detector_method="test",
                anomaly_score=0.8,
                explanation="This is a test email from the telemetry monitoring system."
            )
            
            subject = f"{self.config.templates.subject_prefix} Test Email"
            body_html = self._generate_test_html_body()
            body_text = self._generate_test_text_body()
            
            return self._send_email([recipient], subject, body_html, body_text)
            
        except Exception as e:
            logger.error(f"Failed to send test email: {e}")
            return False
    
    def _send_email(self, recipients: List[str], subject: str, 
                   body_html: str, body_text: str, retries: int = 3) -> bool:
        """Send email with retry logic."""
        for attempt in range(retries):
            try:
                # Create message
                msg = MIMEMultipart('alternative')
                msg['From'] = self.config.email.from_email
                msg['To'] = ', '.join(recipients)
                msg['Subject'] = subject
                
                # Add both plain text and HTML versions
                text_part = MIMEText(body_text, 'plain')
                html_part = MIMEText(body_html, 'html')
                
                msg.attach(text_part)
                msg.attach(html_part)
                
                # Send email
                if self._send_via_smtp(msg, recipients):
                    logger.info(f"Alert email sent successfully to {len(recipients)} recipients")
                    return True
                else:
                    logger.warning(f"Email send attempt {attempt + 1} failed")
                    
            except Exception as e:
                logger.error(f"Email send attempt {attempt + 1} failed: {e}")
                
            # Wait before retry (except on last attempt)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return False
    
    def _send_via_smtp(self, message: MIMEMultipart, recipients: List[str]) -> bool:
        """Send message via SMTP connection."""
        try:
            with self._connection_lock:
                # Get SMTP connection
                smtp = self._get_smtp_connection()
                if not smtp:
                    return False
                
                # Send message
                smtp.send_message(message, to_addrs=recipients)
                logger.debug(f"Email sent via SMTP to {recipients}")
                return True
                
        except Exception as e:
            logger.error(f"SMTP send failed: {e}")
            self._disconnect()
            return False
    
    def _get_smtp_connection(self) -> Optional[smtplib.SMTP]:
        """Get or create SMTP connection."""
        try:
            # Check if we should retry connection
            now = time.time()
            if (self._last_connection_attempt > 0 and 
                now - self._last_connection_attempt < self._connection_retry_delay):
                return None
            
            if not self._smtp_connection:
                logger.debug(f"Connecting to SMTP server {self.config.email.smtp_server}:{self.config.email.smtp_port}")
                
                # Create connection
                smtp = smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port)
                
                # Enable TLS if configured
                if self.config.email.use_tls:
                    smtp.starttls()
                
                # Login
                smtp.login(self.config.email.username, self.config.email.password)
                
                self._smtp_connection = smtp
                logger.info("SMTP connection established successfully")
            
            # Test connection
            self._smtp_connection.noop()
            return self._smtp_connection
            
        except Exception as e:
            logger.error(f"SMTP connection failed: {e}")
            self._last_connection_attempt = time.time()
            self._smtp_connection = None
            return None
    
    def _disconnect(self):
        """Disconnect SMTP connection."""
        try:
            if self._smtp_connection:
                self._smtp_connection.quit()
        except:
            pass
        finally:
            self._smtp_connection = None
    
    def _generate_subject(self, alert_data: AlertData) -> str:
        """Generate email subject line."""
        severity_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '⚡',
            'low': 'ℹ️'
        }
        
        emoji = severity_emoji.get(alert_data.severity.lower(), '📊')
        
        return (f"{self.config.templates.subject_prefix} "
                f"{emoji} {alert_data.severity.upper()} - "
                f"{alert_data.channel} Anomaly Detected")
    
    def _generate_html_body(self, alert_data: AlertData) -> str:
        """Generate HTML email body."""
        # Format timestamp
        dt = datetime.fromtimestamp(alert_data.timestamp)
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Severity colors
        severity_colors = {
            'critical': '#dc3545',
            'high': '#fd7e14', 
            'medium': '#ffc107',
            'low': '#17a2b8'
        }
        color = severity_colors.get(alert_data.severity.lower(), '#6c757d')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .alert-body {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 10px 0; }}
                .metric {{ margin: 10px 0; }}
                .metric strong {{ color: #495057; }}
                .actions {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .footer {{ color: #6c757d; font-size: 0.9em; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🚨 Anomaly Detection Alert</h2>
                <p>Severity: {alert_data.severity.upper()}</p>
            </div>
            
            <div class="alert-body">
                <div class="metric">
                    <strong>Channel:</strong> {alert_data.channel}
                </div>
                <div class="metric">
                    <strong>Value:</strong> {alert_data.value:.4f}
                </div>
                <div class="metric">
                    <strong>Time:</strong> {timestamp_str}
                </div>
                <div class="metric">
                    <strong>Detector:</strong> {alert_data.detector_method}
                </div>
                <div class="metric">
                    <strong>Anomaly Score:</strong> {alert_data.anomaly_score:.4f}
                </div>
            </div>
        """
        
        # Add explanation if available
        if alert_data.explanation:
            html += f"""
            <div class="actions">
                <h4>Details:</h4>
                <p>{alert_data.explanation}</p>
            </div>
            """
        
        # Add raw data if configured
        if self.config.templates.include_raw_data and alert_data.raw_data:
            html += """
            <div class="actions">
                <h4>Raw Data:</h4>
                <pre style="background-color: #ffffff; padding: 10px; border-radius: 3px; font-size: 0.9em;">
            """
            for key, value in alert_data.raw_data.items():
                html += f"{key}: {value}\\n"
            html += "</pre></div>"
        
        html += f"""
            <div class="footer">
                <p>Alert generated by {self.config.templates.company_name} Telemetry Monitoring System</p>
                <p>Timestamp: {timestamp_str}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_text_body(self, alert_data: AlertData) -> str:
        """Generate plain text email body."""
        dt = datetime.fromtimestamp(alert_data.timestamp)
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        text = f"""
ANOMALY DETECTION ALERT

Severity: {alert_data.severity.upper()}
Channel: {alert_data.channel}
Value: {alert_data.value:.4f}
Time: {timestamp_str}
Detector: {alert_data.detector_method}
Anomaly Score: {alert_data.anomaly_score:.4f}
        """
        
        if alert_data.explanation:
            text += f"\nDetails:\n{alert_data.explanation}\n"
        
        if self.config.templates.include_raw_data and alert_data.raw_data:
            text += "\nRaw Data:\n"
            for key, value in alert_data.raw_data.items():
                text += f"{key}: {value}\n"
        
        text += f"\n---\nGenerated by {self.config.templates.company_name} Telemetry Monitoring System\n"
        
        return text
    
    def _generate_test_html_body(self) -> str:
        """Generate test email HTML body."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .test-header { background-color: #17a2b8; color: white; padding: 15px; border-radius: 5px; }
                .test-body { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="test-header">
                <h2>✅ Test Email</h2>
                <p>Email configuration is working correctly!</p>
            </div>
            <div class="test-body">
                <p>This is a test email from your Telemetry Monitoring System.</p>
                <p>If you received this email, your alert configuration is set up correctly.</p>
            </div>
        </body>
        </html>
        """
    
    def _generate_test_text_body(self) -> str:
        """Generate test email plain text body."""
        return """
TEST EMAIL

✅ Email configuration is working correctly!

This is a test email from your Telemetry Monitoring System.
If you received this email, your alert configuration is set up correctly.

---
Generated by Telemetry Monitoring System
        """
    
    def __del__(self):
        """Cleanup SMTP connection on destruction."""
        self._disconnect()
