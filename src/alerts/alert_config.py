"""
Alert Configuration Management

Handles email alert configuration settings and validation.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class EmailSettings:
    """Email server configuration."""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    from_email: str = ""

@dataclass
class AlertRecipients:
    """Alert recipient configuration by severity."""
    critical: List[str] = field(default_factory=list)
    high: List[str] = field(default_factory=list) 
    medium: List[str] = field(default_factory=list)
    low: List[str] = field(default_factory=list)

@dataclass
class RateLimiting:
    """Rate limiting configuration."""
    max_alerts_per_hour: int = 10
    cooldown_minutes: int = 5
    batch_window_minutes: int = 15
    
@dataclass
class EmailTemplateConfig:
    """Email template configuration."""
    subject_prefix: str = "[TELEMETRY ALERT]"
    include_charts: bool = False
    include_raw_data: bool = False
    company_name: str = "Telemetry Monitoring"

@dataclass
class AlertConfig:
    """Complete alert system configuration."""
    enabled: bool = False
    email: EmailSettings = field(default_factory=EmailSettings)
    recipients: AlertRecipients = field(default_factory=AlertRecipients)
    rate_limiting: RateLimiting = field(default_factory=RateLimiting)
    templates: EmailTemplateConfig = field(default_factory=EmailTemplateConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlertConfig':
        """Create AlertConfig from dictionary."""
        try:
            email_dict = config_dict.get('email', {})
            email = EmailSettings(
                smtp_server=email_dict.get('smtp_server', 'smtp.gmail.com'),
                smtp_port=email_dict.get('smtp_port', 587),
                use_tls=email_dict.get('use_tls', True),
                username=email_dict.get('username', ''),
                password=email_dict.get('password', ''),
                from_email=email_dict.get('from_email', '')
            )
            
            recipients_dict = config_dict.get('recipients', {})
            recipients = AlertRecipients(
                critical=recipients_dict.get('critical', []),
                high=recipients_dict.get('high', []),
                medium=recipients_dict.get('medium', []),
                low=recipients_dict.get('low', [])
            )
            
            rate_dict = config_dict.get('rate_limiting', {})
            rate_limiting = RateLimiting(
                max_alerts_per_hour=rate_dict.get('max_alerts_per_hour', 10),
                cooldown_minutes=rate_dict.get('cooldown_minutes', 5),
                batch_window_minutes=rate_dict.get('batch_window_minutes', 15)
            )
            
            template_dict = config_dict.get('templates', {})
            templates = EmailTemplateConfig(
                subject_prefix=template_dict.get('subject_prefix', '[TELEMETRY ALERT]'),
                include_charts=template_dict.get('include_charts', False),
                include_raw_data=template_dict.get('include_raw_data', False),
                company_name=template_dict.get('company_name', 'Telemetry Monitoring')
            )
            
            return cls(
                enabled=config_dict.get('enabled', False),
                email=email,
                recipients=recipients,
                rate_limiting=rate_limiting,
                templates=templates
            )
        except Exception as e:
            logger.error(f"Error parsing alert config: {e}")
            return cls()  # Return default config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AlertConfig to dictionary."""
        return {
            'enabled': self.enabled,
            'email': {
                'smtp_server': self.email.smtp_server,
                'smtp_port': self.email.smtp_port,
                'use_tls': self.email.use_tls,
                'username': self.email.username,
                'password': self.email.password,
                'from_email': self.email.from_email
            },
            'recipients': {
                'critical': self.recipients.critical,
                'high': self.recipients.high,
                'medium': self.recipients.medium,
                'low': self.recipients.low
            },
            'rate_limiting': {
                'max_alerts_per_hour': self.rate_limiting.max_alerts_per_hour,
                'cooldown_minutes': self.rate_limiting.cooldown_minutes,
                'batch_window_minutes': self.rate_limiting.batch_window_minutes
            },
            'templates': {
                'subject_prefix': self.templates.subject_prefix,
                'include_charts': self.templates.include_charts,
                'include_raw_data': self.templates.include_raw_data,
                'company_name': self.templates.company_name
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.enabled:
            # Email validation
            if not self.email.smtp_server:
                errors.append("SMTP server is required")
            if not self.email.username:
                errors.append("Email username is required")
            if not self.email.password:
                errors.append("Email password is required")
            if not self.email.from_email:
                errors.append("From email address is required")
            
            # Recipients validation
            all_recipients = (
                self.recipients.critical + 
                self.recipients.high + 
                self.recipients.medium + 
                self.recipients.low
            )
            if not all_recipients:
                errors.append("At least one recipient must be configured")
                
            # Validate email formats
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            for email in all_recipients + [self.email.from_email]:
                if email and not re.match(email_pattern, email):
                    errors.append(f"Invalid email format: {email}")
        
        return errors
    
    def get_recipients_for_severity(self, severity: str) -> List[str]:
        """Get recipient list for specific severity level."""
        severity_map = {
            'critical': self.recipients.critical,
            'high': self.recipients.high,
            'medium': self.recipients.medium,
            'low': self.recipients.low
        }
        return severity_map.get(severity.lower(), [])
