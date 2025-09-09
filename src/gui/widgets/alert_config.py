"""
Alert Configuration Widget

GUI panel for configuring email alerts in the telemetry monitoring system.
"""

import logging
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QSpinBox, QCheckBox, QPushButton, QLabel, 
    QTextEdit, QTabWidget, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

try:
    from alerts.alert_config import AlertConfig
    from alerts.alert_manager import AlertManager
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    AlertConfig = None
    AlertManager = None

logger = logging.getLogger(__name__)

class TestEmailThread(QThread):
    """Thread for sending test emails without blocking UI."""
    
    finished = Signal(bool, str)  # success, message
    
    def __init__(self, alert_manager: 'AlertManager', recipient: str):
        super().__init__()
        self.alert_manager = alert_manager
        self.recipient = recipient
    
    def run(self):
        try:
            success = self.alert_manager.send_test_alert(self.recipient)
            if success:
                self.finished.emit(True, "Test email sent successfully!")
            else:
                self.finished.emit(False, "Failed to send test email. Check configuration.")
        except Exception as e:
            self.finished.emit(False, f"Error sending test email: {str(e)}")

class AlertConfigWidget(QWidget):
    """Widget for configuring email alerts."""
    
    def __init__(self, config_path=None, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.alert_config: Optional[AlertConfig] = None
        self.alert_manager: Optional[AlertManager] = None
        self.test_thread: Optional[TestEmailThread] = None
        
        if not ALERTS_AVAILABLE:
            self.setup_unavailable_ui()
        else:
            self.setup_ui()
    
    def setup_unavailable_ui(self):
        """Setup UI when alert system is not available."""
        layout = QVBoxLayout(self)
        
        error_label = QLabel("⚠️ Alert system not available")
        error_label.setStyleSheet("color: orange; font-weight: bold; font-size: 14px;")
        layout.addWidget(error_label)
        
        info_label = QLabel("The alert system dependencies are not installed or configured.")
        layout.addWidget(info_label)
    
    def setup_ui(self):
        """Setup the main UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📧 Email Alert Configuration")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Tab widget for different config sections
        tabs = QTabWidget()
        
        # General settings tab
        general_tab = QWidget()
        self.setup_general_tab(general_tab)
        tabs.addTab(general_tab, "General")
        
        # Email settings tab
        email_tab = QWidget()
        self.setup_email_tab(email_tab)
        tabs.addTab(email_tab, "Email Server")
        
        # Recipients tab
        recipients_tab = QWidget()
        self.setup_recipients_tab(recipients_tab)
        tabs.addTab(recipients_tab, "Recipients")
        
        # Test tab
        test_tab = QWidget()
        self.setup_test_tab(test_tab)
        tabs.addTab(test_tab, "Test")
        
        layout.addWidget(tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_save = QPushButton("💾 Save Configuration")
        self.btn_save.clicked.connect(self.save_configuration)
        
        self.btn_load_defaults = QPushButton("🔄 Load Defaults")
        self.btn_load_defaults.clicked.connect(self.load_defaults)
        
        button_layout.addWidget(self.btn_save)
        button_layout.addWidget(self.btn_load_defaults)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Load default configuration
        self.load_defaults()
    
    def setup_general_tab(self, tab: QWidget):
        """Setup general alert settings."""
        layout = QVBoxLayout(tab)
        
        # Enable/disable alerts
        self.chk_enabled = QCheckBox("Enable Email Alerts")
        self.chk_enabled.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.chk_enabled)
        
        # Rate limiting group
        rate_group = QGroupBox("Rate Limiting")
        rate_layout = QFormLayout(rate_group)
        
        self.spin_max_per_hour = QSpinBox()
        self.spin_max_per_hour.setRange(1, 100)
        self.spin_max_per_hour.setValue(10)
        rate_layout.addRow("Max alerts per hour:", self.spin_max_per_hour)
        
        self.spin_cooldown = QSpinBox()
        self.spin_cooldown.setRange(1, 60)
        self.spin_cooldown.setValue(5)
        rate_layout.addRow("Cooldown (minutes):", self.spin_cooldown)
        
        self.spin_batch_window = QSpinBox()
        self.spin_batch_window.setRange(1, 60)
        self.spin_batch_window.setValue(15)
        rate_layout.addRow("Batch window (minutes):", self.spin_batch_window)
        
        layout.addWidget(rate_group)
        
        # Template settings
        template_group = QGroupBox("Email Templates")
        template_layout = QFormLayout(template_group)
        
        self.txt_subject_prefix = QLineEdit("[TELEMETRY ALERT]")
        template_layout.addRow("Subject prefix:", self.txt_subject_prefix)
        
        self.txt_company_name = QLineEdit("Telemetry Monitoring")
        template_layout.addRow("Company name:", self.txt_company_name)
        
        self.chk_include_charts = QCheckBox("Include charts (future)")
        self.chk_include_charts.setEnabled(False)
        template_layout.addRow("", self.chk_include_charts)
        
        self.chk_include_raw_data = QCheckBox("Include raw data")
        template_layout.addRow("", self.chk_include_raw_data)
        
        layout.addWidget(template_group)
        layout.addStretch()
    
    def setup_email_tab(self, tab: QWidget):
        """Setup email server settings."""
        layout = QVBoxLayout(tab)
        
        # SMTP settings
        smtp_group = QGroupBox("SMTP Server Configuration")
        smtp_layout = QFormLayout(smtp_group)
        
        self.txt_smtp_server = QLineEdit("smtp.gmail.com")
        smtp_layout.addRow("SMTP Server:", self.txt_smtp_server)
        
        self.spin_smtp_port = QSpinBox()
        self.spin_smtp_port.setRange(1, 65535)
        self.spin_smtp_port.setValue(587)
        smtp_layout.addRow("SMTP Port:", self.spin_smtp_port)
        
        self.chk_use_tls = QCheckBox("Use TLS/SSL")
        self.chk_use_tls.setChecked(True)
        smtp_layout.addRow("", self.chk_use_tls)
        
        layout.addWidget(smtp_group)
        
        # Authentication
        auth_group = QGroupBox("Authentication")
        auth_layout = QFormLayout(auth_group)
        
        self.txt_username = QLineEdit()
        self.txt_username.setPlaceholderText("your-email@example.com")
        auth_layout.addRow("Username:", self.txt_username)
        
        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_password.setPlaceholderText("App password or email password")
        auth_layout.addRow("Password:", self.txt_password)
        
        self.txt_from_email = QLineEdit()
        self.txt_from_email.setPlaceholderText("alerts@example.com")
        auth_layout.addRow("From Email:", self.txt_from_email)
        
        layout.addWidget(auth_group)
        
        # Help text
        help_text = QLabel("""
        💡 <b>Gmail Setup:</b><br>
        • Server: smtp.gmail.com, Port: 587, TLS: Enabled<br>
        • Use an App Password (not your regular password)<br>
        • Enable 2-factor authentication and generate app password<br><br>
        
        💡 <b>Outlook Setup:</b><br>
        • Server: smtp-mail.outlook.com, Port: 587, TLS: Enabled<br>
        • Use your regular email and password
        """)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(help_text)
        
        layout.addStretch()
    
    def setup_recipients_tab(self, tab: QWidget):
        """Setup recipient configuration."""
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("Configure email recipients for different severity levels:")
        layout.addWidget(info_label)
        
        # Recipients for each severity level
        for severity in ["Critical", "High", "Medium", "Low"]:
            group = QGroupBox(f"{severity} Severity Recipients")
            group_layout = QVBoxLayout(group)
            
            # Input for adding recipients
            input_layout = QHBoxLayout()
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("email@example.com")
            add_btn = QPushButton("Add")
            
            # Store references for later use
            setattr(self, f"txt_{severity.lower()}_email", line_edit)
            setattr(self, f"btn_add_{severity.lower()}", add_btn)
            
            input_layout.addWidget(line_edit)
            input_layout.addWidget(add_btn)
            group_layout.addLayout(input_layout)
            
            # List of current recipients
            recipient_list = QListWidget()
            recipient_list.setMaximumHeight(80)
            setattr(self, f"list_{severity.lower()}_recipients", recipient_list)
            group_layout.addWidget(recipient_list)
            
            # Connect add button
            add_btn.clicked.connect(lambda checked, s=severity.lower(): self.add_recipient(s))
            
            layout.addWidget(group)
        
        layout.addStretch()
    
    def setup_test_tab(self, tab: QWidget):
        """Setup test email functionality."""
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("Send a test email to verify your configuration:")
        layout.addWidget(info_label)
        
        # Test email input
        test_layout = QFormLayout()
        
        self.txt_test_email = QLineEdit()
        self.txt_test_email.setPlaceholderText("test@example.com")
        test_layout.addRow("Test Email:", self.txt_test_email)
        
        layout.addLayout(test_layout)
        
        # Test button
        self.btn_test_email = QPushButton("📧 Send Test Email")
        self.btn_test_email.clicked.connect(self.send_test_email)
        layout.addWidget(self.btn_test_email)
        
        # Progress bar
        self.progress_test = QProgressBar()
        self.progress_test.setVisible(False)
        layout.addWidget(self.progress_test)
        
        # Test results
        self.txt_test_results = QTextEdit()
        self.txt_test_results.setMaximumHeight(100)
        self.txt_test_results.setReadOnly(True)
        # Set background and text colors for better visibility
        self.txt_test_results.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        layout.addWidget(self.txt_test_results)
        
        layout.addStretch()
    
    def add_recipient(self, severity: str):
        """Add a recipient email for the specified severity level."""
        line_edit = getattr(self, f"txt_{severity}_email")
        recipient_list = getattr(self, f"list_{severity}_recipients")
        
        email = line_edit.text().strip()
        if email and "@" in email:
            recipient_list.addItem(email)
            line_edit.clear()
        else:
            QMessageBox.warning(self, "Invalid Email", "Please enter a valid email address.")
    
    def send_test_email(self):
        """Send a test email."""
        # Check if alerts are available
        if not ALERTS_AVAILABLE:
            QMessageBox.critical(self, "Alerts Unavailable", "Alert system is not available.")
            return
            
        # Always create a fresh alert manager from current settings for testing
        config = self.get_current_config()
        
        # Force enable for testing purposes
        config.enabled = True
        
        try:
            temp_alert_manager = AlertManager(config)
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to initialize alert manager: {str(e)}")
            return
        
        recipient = self.txt_test_email.text().strip()
        if not recipient or "@" not in recipient:
            QMessageBox.warning(self, "Invalid Email", "Please enter a valid test email address.")
            return
        
        # Show progress
        self.btn_test_email.setEnabled(False)
        self.progress_test.setVisible(True)
        self.progress_test.setRange(0, 0)  # Indeterminate progress
        self.txt_test_results.append(f"Sending test email to {recipient}...")
        
        # Start test thread with temporary alert manager
        self.test_thread = TestEmailThread(temp_alert_manager, recipient)
        self.test_thread.finished.connect(self.on_test_finished)
        self.test_thread.start()
    
    def on_test_finished(self, success: bool, message: str):
        """Handle test email completion."""
        self.btn_test_email.setEnabled(True)
        self.progress_test.setVisible(False)
        
        if success:
            self.txt_test_results.append(f"✅ {message}")
        else:
            self.txt_test_results.append(f"❌ {message}")
        
        # Show message box
        if success:
            QMessageBox.information(self, "Test Successful", message)
        else:
            QMessageBox.critical(self, "Test Failed", message)
    
    def get_current_config(self) -> AlertConfig:
        """Get current configuration from UI controls."""
        if not AlertConfig:
            return None
        
        # Collect recipients
        recipients = {}
        for severity in ["critical", "high", "medium", "low"]:
            recipient_list = getattr(self, f"list_{severity}_recipients")
            emails = [recipient_list.item(i).text() for i in range(recipient_list.count())]
            recipients[severity] = emails
        
        config_dict = {
            "enabled": self.chk_enabled.isChecked(),
            "email": {
                "smtp_server": self.txt_smtp_server.text(),
                "smtp_port": self.spin_smtp_port.value(),
                "use_tls": self.chk_use_tls.isChecked(),
                "username": self.txt_username.text(),
                "password": self.txt_password.text(),
                "from_email": self.txt_from_email.text()
            },
            "recipients": recipients,
            "rate_limiting": {
                "max_alerts_per_hour": self.spin_max_per_hour.value(),
                "cooldown_minutes": self.spin_cooldown.value(),
                "batch_window_minutes": self.spin_batch_window.value()
            },
            "templates": {
                "subject_prefix": self.txt_subject_prefix.text(),
                "include_charts": self.chk_include_charts.isChecked(),
                "include_raw_data": self.chk_include_raw_data.isChecked(),
                "company_name": self.txt_company_name.text()
            }
        }
        
        return AlertConfig.from_dict(config_dict)
    
    def set_config(self, config: AlertConfig):
        """Set UI controls from configuration."""
        if not config:
            return
        
        self.alert_config = config
        
        # General settings
        self.chk_enabled.setChecked(config.enabled)
        self.spin_max_per_hour.setValue(config.rate_limiting.max_alerts_per_hour)
        self.spin_cooldown.setValue(config.rate_limiting.cooldown_minutes)
        self.spin_batch_window.setValue(config.rate_limiting.batch_window_minutes)
        
        # Template settings
        self.txt_subject_prefix.setText(config.templates.subject_prefix)
        self.txt_company_name.setText(config.templates.company_name)
        self.chk_include_charts.setChecked(config.templates.include_charts)
        self.chk_include_raw_data.setChecked(config.templates.include_raw_data)
        
        # Email settings
        self.txt_smtp_server.setText(config.email.smtp_server)
        self.spin_smtp_port.setValue(config.email.smtp_port)
        self.chk_use_tls.setChecked(config.email.use_tls)
        self.txt_username.setText(config.email.username)
        self.txt_password.setText(config.email.password)
        self.txt_from_email.setText(config.email.from_email)
        
        # Recipients
        for severity in ["critical", "high", "medium", "low"]:
            recipient_list = getattr(self, f"list_{severity}_recipients")
            recipient_list.clear()
            emails = getattr(config.recipients, severity)
            for email in emails:
                recipient_list.addItem(email)
    
    def load_defaults(self):
        """Load default configuration."""
        if AlertConfig:
            default_config = AlertConfig()
            self.set_config(default_config)
    
    def save_configuration(self):
        """Save current configuration to file and update alert manager."""
        config = self.get_current_config()
        if not config:
            return
        
        # Validate configuration
        errors = config.validate()
        if errors:
            error_msg = "Configuration errors:\\n" + "\\n".join(errors)
            QMessageBox.warning(self, "Configuration Errors", error_msg)
            return
        
        # Save to config.json file
        if self.config_path:
            try:
                import json
                from pathlib import Path
                
                config_file = Path(self.config_path)
                if config_file.exists():
                    # Load existing config
                    with open(config_file, 'r') as f:
                        full_config = json.load(f)
                    
                    # Update alerts section
                    full_config['alerts'] = config.to_dict()
                    
                    # Write back to file
                    with open(config_file, 'w') as f:
                        json.dump(full_config, f, indent=2)
                    
                    logger.info(f"Alert configuration saved to {config_file}")
                else:
                    logger.warning(f"Config file not found: {config_file}")
                    
            except Exception as e:
                logger.error(f"Failed to save configuration to file: {e}")
                QMessageBox.warning(self, "Save Error", 
                                   f"Failed to save configuration to file: {e}")
                return
        
        # Update alert manager if it exists
        if self.alert_manager:
            self.alert_manager.update_config(config)
            logger.info("Alert manager configuration updated")
        else:
            # Try to find alert manager via parent (for backward compatibility)
            parent = self.parent()
            if parent and hasattr(parent, 'alert_manager') and parent.alert_manager:
                parent.alert_manager.update_config(config)  # type: ignore
                logger.info("Alert manager configuration updated via parent")
        
        QMessageBox.information(self, "Configuration Saved", 
                               "Alert configuration has been saved successfully!")
    
    def set_alert_manager(self, alert_manager):
        """Set the alert manager instance."""
        self.alert_manager = alert_manager
