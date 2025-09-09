"""
Alert Manager

Central coordinator for the alert system. Handles rate limiting, batching,
and orchestrates email alerts for anomaly detection events.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .alert_config import AlertConfig
from .email_alerter import EmailAlerter, AlertData

logger = logging.getLogger(__name__)

@dataclass
class AlertBatch:
    """Batch of related alerts for combined sending."""
    alerts: List[AlertData] = field(default_factory=list)
    first_alert_time: float = 0.0
    last_alert_time: float = 0.0
    channel: str = ""
    severity: str = ""
    
    def add_alert(self, alert: AlertData):
        """Add alert to batch."""
        self.alerts.append(alert)
        if not self.first_alert_time:
            self.first_alert_time = alert.timestamp
        self.last_alert_time = alert.timestamp
        if not self.channel:
            self.channel = alert.channel
        if not self.severity or alert.severity == 'critical':
            self.severity = alert.severity

@dataclass 
class RateLimitInfo:
    """Rate limiting information for a channel/severity combination."""
    alert_count: int = 0
    last_alert_time: float = 0.0
    next_allowed_time: float = 0.0

class AlertManager:
    """Central alert management system with rate limiting and batching."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.email_alerter = EmailAlerter(config) if config.enabled else None
        
        # Rate limiting tracking
        self._rate_limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._hourly_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Batching
        self._batch_buffer: Dict[str, AlertBatch] = {}
        self._batch_timer: Optional[threading.Timer] = None
        self._batch_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'alerts_received': 0,
            'alerts_sent': 0,
            'alerts_suppressed': 0,
            'emails_sent': 0,
            'emails_failed': 0,
            'last_alert_time': 0.0
        }
        
        logger.info(f"Alert manager initialized (enabled: {config.enabled})")
    
    def send_alert(self, channel: str, value: float, timestamp: float,
                  severity: str, detector_method: str, anomaly_score: float,
                  explanation: Optional[str] = None, raw_data: Optional[Dict] = None) -> bool:
        """
        Send an alert for an anomaly detection event.
        
        Args:
            channel: Data channel name
            value: Anomaly value
            timestamp: Event timestamp
            severity: Alert severity (critical, high, medium, low)
            detector_method: Detection method used
            anomaly_score: Anomaly confidence score
            explanation: Optional explanation text
            raw_data: Optional raw detection data
            
        Returns:
            True if alert was sent or queued, False if suppressed
        """
        if not self.config.enabled:
            return False
        
        self.stats['alerts_received'] += 1
        self.stats['last_alert_time'] = timestamp
        
        # Create alert data
        alert_data = AlertData(
            channel=channel,
            value=value,
            timestamp=timestamp,
            severity=severity,
            detector_method=detector_method,
            anomaly_score=anomaly_score,
            explanation=explanation,
            raw_data=raw_data or {}
        )
        
        # Check rate limiting
        if self._is_rate_limited(alert_data):
            logger.debug(f"Alert suppressed due to rate limiting: {channel}/{severity}")
            self.stats['alerts_suppressed'] += 1
            return False
        
        # Handle immediate vs batched alerts
        if severity.lower() == 'critical':
            # Send critical alerts immediately
            return self._send_immediate_alert(alert_data)
        else:
            # Batch non-critical alerts
            return self._add_to_batch(alert_data)
    
    def send_test_alert(self, recipient: str) -> bool:
        """Send a test alert to verify configuration."""
        if not self.email_alerter:
            logger.error("Email alerter not initialized")
            return False
        
        return self.email_alerter.send_test_email(recipient)
    
    def get_statistics(self) -> Dict:
        """Get alert system statistics."""
        stats = self.stats.copy()
        
        # Add rate limiting stats
        active_limits = sum(1 for limit in self._rate_limits.values() 
                          if time.time() < limit.next_allowed_time)
        stats['active_rate_limits'] = active_limits
        
        # Add batch stats
        stats['pending_batches'] = len(self._batch_buffer)
        
        return stats
    
    def update_config(self, new_config: AlertConfig):
        """Update alert configuration."""
        self.config = new_config
        
        if new_config.enabled and not self.email_alerter:
            self.email_alerter = EmailAlerter(new_config)
        elif self.email_alerter:
            self.email_alerter.config = new_config
        
        logger.info(f"Alert configuration updated (enabled: {new_config.enabled})")
    
    def _is_rate_limited(self, alert_data: AlertData) -> bool:
        """Check if alert should be rate limited."""
        now = time.time()
        key = f"{alert_data.channel}:{alert_data.severity}"
        
        # Check cooldown period
        rate_info = self._rate_limits[key]
        if now < rate_info.next_allowed_time:
            return True
        
        # Check hourly limit
        hourly_key = f"{alert_data.severity}:hourly"
        hourly_times = self._hourly_counts[hourly_key]
        
        # Remove old entries (older than 1 hour)
        cutoff_time = now - 3600  # 1 hour ago
        while hourly_times and hourly_times[0] < cutoff_time:
            hourly_times.popleft()
        
        # Check if we're over the hourly limit
        if len(hourly_times) >= self.config.rate_limiting.max_alerts_per_hour:
            return True
        
        # Update rate limiting info
        rate_info.alert_count += 1
        rate_info.last_alert_time = now
        rate_info.next_allowed_time = now + (self.config.rate_limiting.cooldown_minutes * 60)
        
        # Add to hourly count
        hourly_times.append(now)
        
        return False
    
    def _send_immediate_alert(self, alert_data: AlertData) -> bool:
        """Send an alert immediately (for critical alerts)."""
        if not self.email_alerter:
            return False
        
        recipients = self.config.get_recipients_for_severity(alert_data.severity)
        if not recipients:
            logger.warning(f"No recipients configured for severity: {alert_data.severity}")
            return False
        
        success = self.email_alerter.send_alert(alert_data, recipients)
        if success:
            self.stats['alerts_sent'] += 1
            self.stats['emails_sent'] += 1
            logger.info(f"Immediate alert sent for {alert_data.channel} ({alert_data.severity})")
        else:
            self.stats['emails_failed'] += 1
            logger.error(f"Failed to send immediate alert for {alert_data.channel}")
        
        return success
    
    def _add_to_batch(self, alert_data: AlertData) -> bool:
        """Add alert to batch for later sending."""
        with self._batch_lock:
            batch_key = f"{alert_data.channel}:{alert_data.severity}"
            
            if batch_key not in self._batch_buffer:
                self._batch_buffer[batch_key] = AlertBatch()
            
            self._batch_buffer[batch_key].add_alert(alert_data)
            
            # Start batch timer if not already running
            if not self._batch_timer or not self._batch_timer.is_alive():
                delay = self.config.rate_limiting.batch_window_minutes * 60
                self._batch_timer = threading.Timer(delay, self._send_batched_alerts)
                self._batch_timer.start()
                logger.debug(f"Batch timer started for {delay} seconds")
        
        return True
    
    def _send_batched_alerts(self):
        """Send all batched alerts."""
        with self._batch_lock:
            if not self._batch_buffer:
                return
            
            logger.info(f"Sending {len(self._batch_buffer)} batched alerts")
            
            for batch_key, batch in self._batch_buffer.items():
                try:
                    self._send_batch(batch)
                except Exception as e:
                    logger.error(f"Failed to send batch {batch_key}: {e}")
            
            # Clear batch buffer
            self._batch_buffer.clear()
    
    def _send_batch(self, batch: AlertBatch):
        """Send a batch of alerts as a single email."""
        if not self.email_alerter or not batch.alerts:
            return
        
        recipients = self.config.get_recipients_for_severity(batch.severity)
        if not recipients:
            logger.warning(f"No recipients for batch severity: {batch.severity}")
            return
        
        # Create combined alert data
        combined_alert = self._create_batch_alert(batch)
        
        success = self.email_alerter.send_alert(combined_alert, recipients)
        if success:
            self.stats['alerts_sent'] += len(batch.alerts)
            self.stats['emails_sent'] += 1
            logger.info(f"Batch alert sent with {len(batch.alerts)} alerts to {len(recipients)} recipients")
        else:
            self.stats['emails_failed'] += 1
            logger.error(f"Failed to send batch alert with {len(batch.alerts)} alerts")
    
    def _create_batch_alert(self, batch: AlertBatch) -> AlertData:
        """Create a combined alert from a batch."""
        first_alert = batch.alerts[0]
        
        # Create summary explanation
        channel_counts = defaultdict(int)
        method_counts = defaultdict(int)
        
        for alert in batch.alerts:
            channel_counts[alert.channel] += 1
            method_counts[alert.detector_method] += 1
        
        summary_parts = []
        summary_parts.append(f"Batch of {len(batch.alerts)} alerts")
        
        if len(channel_counts) == 1:
            summary_parts.append(f"Channel: {first_alert.channel}")
        else:
            channels_str = ", ".join(f"{ch}({cnt})" for ch, cnt in channel_counts.items())
            summary_parts.append(f"Channels: {channels_str}")
        
        methods_str = ", ".join(f"{method}({cnt})" for method, cnt in method_counts.items())
        summary_parts.append(f"Methods: {methods_str}")
        
        time_range = datetime.fromtimestamp(batch.first_alert_time).strftime("%H:%M:%S") + " - " + \
                     datetime.fromtimestamp(batch.last_alert_time).strftime("%H:%M:%S")
        summary_parts.append(f"Time range: {time_range}")
        
        explanation = "\n".join(summary_parts)
        
        # Add individual alert details
        explanation += "\n\nIndividual Alerts:"
        for i, alert in enumerate(batch.alerts, 1):
            alert_time = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
            explanation += f"\n{i}. [{alert_time}] {alert.channel}: {alert.value:.4f} ({alert.detector_method})"
        
        return AlertData(
            channel=batch.channel,
            value=first_alert.value,  # Use first alert's value
            timestamp=batch.last_alert_time,  # Use latest timestamp
            severity=batch.severity,
            detector_method="batch",
            anomaly_score=max(alert.anomaly_score for alert in batch.alerts),
            explanation=explanation,
            raw_data={
                "batch_size": len(batch.alerts),
                "channels": list(channel_counts.keys()),
                "time_span_seconds": batch.last_alert_time - batch.first_alert_time
            }
        )
    
    def force_send_batches(self):
        """Force sending of all pending batched alerts."""
        if self._batch_timer and self._batch_timer.is_alive():
            self._batch_timer.cancel()
        self._send_batched_alerts()
    
    def __del__(self):
        """Cleanup on destruction."""
        if self._batch_timer and self._batch_timer.is_alive():
            self._batch_timer.cancel()
        self.force_send_batches()
