"""
Comprehensive memory management and resource monitoring system.

This module provides:
- Real-time memory and CPU monitoring
- Automatic data pruning and cleanup
- Resource-aware data archiving
- Performance profiling and optimization
"""

import gc
import os
import time
import sqlite3
import threading
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, NamedTuple, Callable
from dataclasses import dataclass, field
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import pickle
import zlib
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceAlert(Enum):
    """Types of resource alerts."""
    MEMORY_HIGH = "memory_high"
    MEMORY_CRITICAL = "memory_critical"
    CPU_HIGH = "cpu_high"
    DISK_SPACE_LOW = "disk_space_low"
    DETECTOR_OVERLOAD = "detector_overload"

@dataclass
class ResourceStats(NamedTuple):
    """System resource statistics."""
    timestamp: float
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    disk_usage_percent: float
    detector_count: int
    total_data_points: int
    active_channels: int

@dataclass
class MemoryProfile:
    """Memory usage profile for components."""
    component_name: str
    memory_mb: float
    data_points: int
    last_updated: float
    growth_rate_mb_per_hour: float = 0.0
    predicted_memory_mb: Dict[str, float] = field(default_factory=dict)  # time_horizon -> predicted_memory

@dataclass
class CleanupAction:
    """Represents a cleanup action that was performed."""
    action_type: str
    component: str
    data_removed: int
    memory_freed_mb: float
    timestamp: float
    reason: str

class ResourceMonitor:
    """Advanced resource monitoring with predictive analysis."""
    
    def __init__(self, 
                 memory_limit_mb: int = 2048,
                 cpu_limit_percent: float = 80.0,
                 disk_limit_percent: float = 90.0,
                 check_interval_seconds: int = 30,
                 history_size: int = 288):  # 24 hours at 5-min intervals
        
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.disk_limit_percent = disk_limit_percent
        self.check_interval = check_interval_seconds
        
        # Monitoring data
        self.stats_history: deque = deque(maxlen=history_size)
        self.component_profiles: Dict[str, MemoryProfile] = {}
        self.alert_callbacks: List[Callable[[ResourceAlert, Dict[str, Any]], None]] = []
        self.cleanup_history: List[CleanupAction] = []
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Process reference
        self.process = psutil.Process()
        
        logger.info(f"🔍 ResourceMonitor initialized with limits: {memory_limit_mb}MB memory, {cpu_limit_percent}% CPU")
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert, Dict[str, Any]], None]):
        """Add callback function for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("⚠️ Resource monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("🚀 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if self._monitor_thread:
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        
        logger.info("⏹️ Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.check_interval):
            try:
                stats = self._collect_resource_stats()
                self._process_stats(stats)
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
    
    def _collect_resource_stats(self) -> ResourceStats:
        """Collect current resource statistics."""
        try:
            # Memory info
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            
            # CPU info
            cpu_percent = self.process.cpu_percent()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Component stats (would be provided by detector manager)
            detector_count = sum(len(profile.component_name.split('_')) 
                               for profile in self.component_profiles.values())
            total_data_points = sum(profile.data_points for profile in self.component_profiles.values())
            active_channels = len(self.component_profiles)
            
            return ResourceStats(
                timestamp=time.time(),
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_usage_percent=disk_percent,
                detector_count=detector_count,
                total_data_points=total_data_points,
                active_channels=active_channels
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to collect resource stats: {e}")
            return ResourceStats(
                timestamp=time.time(),
                memory_mb=0, memory_percent=0, cpu_percent=0, disk_usage_percent=0,
                detector_count=0, total_data_points=0, active_channels=0
            )
    
    def _process_stats(self, stats: ResourceStats):
        """Process collected statistics and trigger alerts if needed."""
        with self._lock:
            self.stats_history.append(stats)
            
            # Check for alerts
            self._check_alerts(stats)
            
            # Update component profiles with growth predictions
            self._update_growth_predictions()
    
    def _check_alerts(self, stats: ResourceStats):
        """Check for resource alerts and trigger callbacks."""
        alerts = []
        
        # Memory alerts
        if stats.memory_mb > self.memory_limit_mb * 0.9:
            if stats.memory_mb > self.memory_limit_mb:
                alerts.append((ResourceAlert.MEMORY_CRITICAL, {
                    "current_mb": stats.memory_mb,
                    "limit_mb": self.memory_limit_mb,
                    "percent_over": ((stats.memory_mb / self.memory_limit_mb) - 1) * 100
                }))
            else:
                alerts.append((ResourceAlert.MEMORY_HIGH, {
                    "current_mb": stats.memory_mb,
                    "limit_mb": self.memory_limit_mb,
                    "percent_used": (stats.memory_mb / self.memory_limit_mb) * 100
                }))
        
        # CPU alerts
        if stats.cpu_percent > self.cpu_limit_percent:
            alerts.append((ResourceAlert.CPU_HIGH, {
                "current_percent": stats.cpu_percent,
                "limit_percent": self.cpu_limit_percent
            }))
        
        # Disk space alerts
        if stats.disk_usage_percent > self.disk_limit_percent:
            alerts.append((ResourceAlert.DISK_SPACE_LOW, {
                "current_percent": stats.disk_usage_percent,
                "limit_percent": self.disk_limit_percent
            }))
        
        # Detector overload alerts
        if stats.detector_count > 100 or stats.total_data_points > 1000000:
            alerts.append((ResourceAlert.DETECTOR_OVERLOAD, {
                "detector_count": stats.detector_count,
                "data_points": stats.total_data_points
            }))
        
        # Trigger alert callbacks
        for alert_type, alert_data in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, alert_data)
                except Exception as e:
                    logger.error(f"❌ Error in alert callback: {e}")
    
    def _update_growth_predictions(self):
        """Update memory growth predictions for components."""
        current_time = time.time()
        
        for component_name, profile in self.component_profiles.items():
            # Calculate growth rate from recent history
            recent_stats = [s for s in self.stats_history 
                          if current_time - s.timestamp < 3600]  # Last hour
            
            if len(recent_stats) >= 2:
                time_diff = recent_stats[-1].timestamp - recent_stats[0].timestamp
                if time_diff > 0:
                    # This is simplified - in practice, you'd track per-component memory
                    memory_diff = recent_stats[-1].memory_mb - recent_stats[0].memory_mb
                    growth_rate = (memory_diff / time_diff) * 3600  # MB per hour
                    profile.growth_rate_mb_per_hour = growth_rate
                    
                    # Predict memory usage at different time horizons
                    profile.predicted_memory_mb = {
                        "1_hour": profile.memory_mb + growth_rate,
                        "4_hours": profile.memory_mb + growth_rate * 4,
                        "24_hours": profile.memory_mb + growth_rate * 24
                    }
    
    def update_component_profile(self, component_name: str, memory_mb: float, 
                               data_points: int):
        """Update memory profile for a component."""
        with self._lock:
            if component_name in self.component_profiles:
                profile = self.component_profiles[component_name]
                profile.memory_mb = memory_mb
                profile.data_points = data_points
                profile.last_updated = time.time()
            else:
                self.component_profiles[component_name] = MemoryProfile(
                    component_name=component_name,
                    memory_mb=memory_mb,
                    data_points=data_points,
                    last_updated=time.time()
                )
    
    def get_current_stats(self) -> Optional[ResourceStats]:
        """Get the most recent resource statistics."""
        with self._lock:
            return self.stats_history[-1] if self.stats_history else None
    
    def get_stats_history(self, hours: int = 1) -> List[ResourceStats]:
        """Get resource statistics for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        with self._lock:
            return [stats for stats in self.stats_history 
                   if stats.timestamp >= cutoff_time]
    
    def get_memory_predictions(self) -> Dict[str, Dict[str, float]]:
        """Get memory usage predictions for all components."""
        with self._lock:
            return {
                name: profile.predicted_memory_mb.copy()
                for name, profile in self.component_profiles.items()
                if profile.predicted_memory_mb
            }
    
    def trigger_cleanup(self, cleanup_callback: Callable[[], List[CleanupAction]]):
        """Trigger manual cleanup and record actions."""
        try:
            logger.info("🧹 Triggering manual cleanup...")
            actions = cleanup_callback()
            
            with self._lock:
                self.cleanup_history.extend(actions)
                # Keep only recent cleanup history
                cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
                self.cleanup_history = [
                    action for action in self.cleanup_history 
                    if action.timestamp >= cutoff_time
                ]
            
            total_freed = sum(action.memory_freed_mb for action in actions)
            logger.info(f"✅ Cleanup completed: {len(actions)} actions, {total_freed:.2f}MB freed")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

class DataArchiver:
    """Manages data archiving and compression to reduce memory usage."""
    
    def __init__(self, archive_dir: str = "data_archive", 
                 compression_level: int = 6):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
        self.compression_level = compression_level
        
        # Initialize SQLite database for archived data
        self.db_path = self.archive_dir / "archived_data.db"
        self._init_database()
        
        logger.info(f"📦 DataArchiver initialized: {self.archive_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for archived data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS archived_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL NOT NULL,
                    data_points INTEGER NOT NULL,
                    compressed_data BLOB NOT NULL,
                    compression_ratio REAL NOT NULL,
                    archived_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_channel_time 
                ON archived_telemetry(channel, start_timestamp, end_timestamp)
            ''')
    
    def archive_channel_data(self, channel: str, timestamps: List[float], 
                           values: List[float], metadata: Optional[Dict] = None) -> bool:
        """Archive channel data with compression."""
        if not timestamps or not values:
            return False
        
        try:
            # Prepare data for compression
            data_dict = {
                'timestamps': timestamps,
                'values': values,
                'metadata': metadata or {}
            }
            
            # Serialize and compress
            serialized_data = pickle.dumps(data_dict)
            compressed_data = zlib.compress(serialized_data, self.compression_level)
            
            compression_ratio = len(serialized_data) / len(compressed_data)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO archived_telemetry 
                    (channel, start_timestamp, end_timestamp, data_points, 
                     compressed_data, compression_ratio, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    channel,
                    min(timestamps),
                    max(timestamps),
                    len(values),
                    compressed_data,
                    compression_ratio,
                    json.dumps(metadata) if metadata else None
                ))
            
            logger.info(f"📦 Archived {len(values)} points for {channel} "
                       f"(compression: {compression_ratio:.1f}x)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to archive data for {channel}: {e}")
            return False
    
    def retrieve_archived_data(self, channel: str, start_time: float, 
                             end_time: float) -> Optional[Tuple[List[float], List[float]]]:
        """Retrieve archived data for a channel within time range."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT compressed_data FROM archived_telemetry
                    WHERE channel = ? AND start_timestamp <= ? AND end_timestamp >= ?
                    ORDER BY start_timestamp
                ''', (channel, end_time, start_time))
                
                all_timestamps = []
                all_values = []
                
                for (compressed_data,) in cursor:
                    # Decompress and deserialize
                    serialized_data = zlib.decompress(compressed_data)
                    data_dict = pickle.loads(serialized_data)
                    
                    timestamps = data_dict['timestamps']
                    values = data_dict['values']
                    
                    # Filter by exact time range
                    for ts, val in zip(timestamps, values):
                        if start_time <= ts <= end_time:
                            all_timestamps.append(ts)
                            all_values.append(val)
                
                if all_timestamps:
                    # Sort by timestamp
                    sorted_data = sorted(zip(all_timestamps, all_values))
                    timestamps, values = zip(*sorted_data)
                    return list(timestamps), list(values)
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve archived data for {channel}: {e}")
        
        return None
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about archived data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_archives,
                        COUNT(DISTINCT channel) as unique_channels,
                        SUM(data_points) as total_data_points,
                        AVG(compression_ratio) as avg_compression_ratio,
                        MIN(start_timestamp) as earliest_data,
                        MAX(end_timestamp) as latest_data
                    FROM archived_telemetry
                ''')
                
                row = cursor.fetchone()
                if row:
                    return {
                        'total_archives': row[0],
                        'unique_channels': row[1], 
                        'total_data_points': row[2],
                        'avg_compression_ratio': row[3],
                        'earliest_data': datetime.fromtimestamp(row[4]) if row[4] else None,
                        'latest_data': datetime.fromtimestamp(row[5]) if row[5] else None,
                        'database_size_mb': self.db_path.stat().st_size / (1024 * 1024)
                    }
        except Exception as e:
            logger.error(f"❌ Failed to get archive stats: {e}")
        
        return {}
    
    def cleanup_old_archives(self, max_age_days: int = 30) -> int:
        """Remove old archived data beyond retention period."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM archived_telemetry 
                    WHERE end_timestamp < ?
                ''', (cutoff_time,))
                
                deleted_count = cursor.rowcount
                
                # Vacuum database to reclaim space
                conn.execute('VACUUM')
                
                logger.info(f"🗑️ Cleaned up {deleted_count} old archives (>{max_age_days} days)")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old archives: {e}")
            return 0

class MemoryEfficientDataManager:
    """Manages telemetry data with automatic memory optimization."""
    
    def __init__(self,
                 max_memory_mb: int = 1024,
                 archive_threshold_hours: int = 24,
                 cleanup_interval_minutes: int = 30):
        
        self.max_memory_mb = max_memory_mb
        self.archive_threshold = archive_threshold_hours * 3600
        self.cleanup_interval = cleanup_interval_minutes * 60
        
        # Data storage
        self.channel_data: Dict[str, deque] = {}
        self.channel_timestamps: Dict[str, deque] = {}
        self.max_points_per_channel = 10000  # Configurable limit
        
        # Components
        self.resource_monitor = ResourceMonitor(memory_limit_mb=max_memory_mb)
        self.archiver = DataArchiver()
        
        # Threading
        self._cleanup_timer: Optional[threading.Timer] = None
        self._lock = threading.RLock()
        
        # Setup resource monitoring callbacks
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        
        logger.info(f"💾 MemoryEfficientDataManager initialized: {max_memory_mb}MB limit")
    
    def start(self):
        """Start memory management services."""
        self.resource_monitor.start_monitoring()
        self._schedule_cleanup()
        logger.info("🚀 Memory management started")
    
    def stop(self):
        """Stop memory management services."""
        self.resource_monitor.stop_monitoring()
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        logger.info("⏹️ Memory management stopped")
    
    def add_data_point(self, channel: str, timestamp: float, value: float):
        """Add a data point with automatic memory management."""
        with self._lock:
            # Initialize channel storage if needed
            if channel not in self.channel_data:
                self.channel_data[channel] = deque(maxlen=self.max_points_per_channel)
                self.channel_timestamps[channel] = deque(maxlen=self.max_points_per_channel)
            
            # Add data point
            self.channel_data[channel].append(value)
            self.channel_timestamps[channel].append(timestamp)
            
            # Update resource monitor
            memory_estimate = len(self.channel_data[channel]) * 16  # 8 bytes each for timestamp and value
            self.resource_monitor.update_component_profile(
                f"channel_{channel}",
                memory_estimate / (1024 * 1024),  # Convert to MB
                len(self.channel_data[channel])
            )
    
    def get_recent_data(self, channel: str, max_points: int = 1000) -> Tuple[List[float], List[float]]:
        """Get recent data points for a channel."""
        with self._lock:
            if channel not in self.channel_data:
                return [], []
            
            timestamps = list(self.channel_timestamps[channel])[-max_points:]
            values = list(self.channel_data[channel])[-max_points:]
            
            return timestamps, values
    
    def get_historical_data(self, channel: str, start_time: float, 
                          end_time: float) -> Tuple[List[float], List[float]]:
        """Get historical data, combining in-memory and archived data."""
        # Get in-memory data
        recent_timestamps, recent_values = self.get_recent_data(channel, 100000)
        
        # Filter recent data by time range
        in_memory_timestamps = []
        in_memory_values = []
        
        for ts, val in zip(recent_timestamps, recent_values):
            if start_time <= ts <= end_time:
                in_memory_timestamps.append(ts)
                in_memory_values.append(val)
        
        # Get archived data for older time ranges
        archived_data = self.archiver.retrieve_archived_data(channel, start_time, end_time)
        
        if archived_data:
            archived_timestamps, archived_values = archived_data
            
            # Combine and sort
            all_timestamps = archived_timestamps + in_memory_timestamps
            all_values = archived_values + in_memory_values
            
            if all_timestamps:
                sorted_data = sorted(zip(all_timestamps, all_values))
                timestamps, values = zip(*sorted_data)
                return list(timestamps), list(values)
        
        return in_memory_timestamps, in_memory_values
    
    def _handle_resource_alert(self, alert_type: ResourceAlert, alert_data: Dict[str, Any]):
        """Handle resource alerts by triggering cleanup."""
        logger.warning(f"⚠️ Resource alert: {alert_type.value} - {alert_data}")
        
        if alert_type in [ResourceAlert.MEMORY_HIGH, ResourceAlert.MEMORY_CRITICAL]:
            self._perform_emergency_cleanup()
    
    def _perform_emergency_cleanup(self) -> List[CleanupAction]:
        """Perform emergency cleanup to free memory."""
        actions = []
        current_time = time.time()
        
        with self._lock:
            for channel in list(self.channel_data.keys()):
                timestamps = list(self.channel_timestamps[channel])
                values = list(self.channel_data[channel])
                
                if not timestamps:
                    continue
                
                # Archive old data (older than threshold)
                cutoff_time = current_time - self.archive_threshold
                old_timestamps = []
                old_values = []
                new_timestamps = []
                new_values = []
                
                for ts, val in zip(timestamps, values):
                    if ts < cutoff_time:
                        old_timestamps.append(ts)
                        old_values.append(val)
                    else:
                        new_timestamps.append(ts)
                        new_values.append(val)
                
                if old_timestamps:
                    # Archive old data
                    success = self.archiver.archive_channel_data(
                        channel, old_timestamps, old_values,
                        {'reason': 'emergency_cleanup', 'original_size': len(timestamps)}
                    )
                    
                    if success:
                        # Keep only recent data in memory
                        self.channel_data[channel] = deque(new_values, maxlen=self.max_points_per_channel)
                        self.channel_timestamps[channel] = deque(new_timestamps, maxlen=self.max_points_per_channel)
                        
                        memory_freed = len(old_timestamps) * 16 / (1024 * 1024)  # MB
                        
                        actions.append(CleanupAction(
                            action_type="archive_old_data",
                            component=channel,
                            data_removed=len(old_timestamps),
                            memory_freed_mb=memory_freed,
                            timestamp=current_time,
                            reason="emergency_cleanup"
                        ))
        
        return actions
    
    def _schedule_cleanup(self):
        """Schedule periodic cleanup."""
        def cleanup():
            try:
                self.resource_monitor.trigger_cleanup(self._perform_scheduled_cleanup)
            except Exception as e:
                logger.error(f"❌ Scheduled cleanup failed: {e}")
            finally:
                # Schedule next cleanup
                self._cleanup_timer = threading.Timer(self.cleanup_interval, cleanup)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
        
        # Start first cleanup timer
        self._cleanup_timer = threading.Timer(self.cleanup_interval, cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _perform_scheduled_cleanup(self) -> List[CleanupAction]:
        """Perform scheduled maintenance cleanup."""
        actions = []
        current_time = time.time()
        
        # Archive old data
        archive_actions = self._perform_emergency_cleanup()
        actions.extend(archive_actions)
        
        # Clean up old archives
        deleted_count = self.archiver.cleanup_old_archives(max_age_days=30)
        if deleted_count > 0:
            actions.append(CleanupAction(
                action_type="cleanup_old_archives",
                component="archiver",
                data_removed=deleted_count,
                memory_freed_mb=0.0,  # Disk space, not memory
                timestamp=current_time,
                reason="scheduled_cleanup"
            ))
        
        return actions
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = self.resource_monitor.get_current_stats()
        archive_stats = self.archiver.get_archive_stats()
        
        with self._lock:
            channel_stats = {}
            total_in_memory_points = 0
            
            for channel in self.channel_data:
                points = len(self.channel_data[channel])
                memory_mb = points * 16 / (1024 * 1024)
                
                channel_stats[channel] = {
                    'in_memory_points': points,
                    'memory_mb': memory_mb
                }
                total_in_memory_points += points
        
        return {
            'system_stats': stats._asdict() if stats else {},
            'archive_stats': archive_stats,
            'channel_stats': channel_stats,
            'total_in_memory_points': total_in_memory_points,
            'total_channels': len(self.channel_data),
            'memory_predictions': self.resource_monitor.get_memory_predictions()
        }

# Example usage
def example_memory_management():
    """Example of memory management system usage."""
    
    def alert_handler(alert_type: ResourceAlert, alert_data: Dict[str, Any]):
        print(f"🚨 ALERT: {alert_type.value} - {alert_data}")
    
    # Create memory manager
    memory_manager = MemoryEfficientDataManager(
        max_memory_mb=512,  # 512MB limit for demo
        archive_threshold_hours=1,  # Archive data older than 1 hour
        cleanup_interval_minutes=5  # Cleanup every 5 minutes
    )
    
    # Add alert handler
    memory_manager.resource_monitor.add_alert_callback(alert_handler)
    
    # Start memory management
    memory_manager.start()
    
    try:
        # Simulate data ingestion
        print("📊 Simulating data ingestion...")
        import random
        
        channels = ['temp_c', 'voltage', 'current', 'pressure']
        current_time = time.time()
        
        for i in range(10000):  # Simulate 10k data points
            for channel in channels:
                timestamp = current_time + i
                value = random.uniform(0, 100)
                memory_manager.add_data_point(channel, timestamp, value)
            
            # Print stats every 1000 points
            if i % 1000 == 0:
                stats = memory_manager.get_memory_stats()
                system_stats = stats['system_stats']
                print(f"  Progress: {i}/10000, Memory: {system_stats.get('memory_mb', 0):.1f}MB")
        
        # Get final statistics
        print("\n📈 Final Statistics:")
        final_stats = memory_manager.get_memory_stats()
        
        print(f"System Memory: {final_stats['system_stats'].get('memory_mb', 0):.1f}MB")
        print(f"Total Channels: {final_stats['total_channels']}")
        print(f"In-Memory Points: {final_stats['total_in_memory_points']}")
        
        print("\nArchive Statistics:")
        archive_stats = final_stats['archive_stats']
        for key, value in archive_stats.items():
            print(f"  {key}: {value}")
        
        print("\nChannel Statistics:")
        for channel, stats in final_stats['channel_stats'].items():
            print(f"  {channel}: {stats['in_memory_points']} points, {stats['memory_mb']:.2f}MB")
        
        # Test historical data retrieval
        print("\n🔍 Testing historical data retrieval...")
        start_time = current_time
        end_time = current_time + 5000
        
        timestamps, values = memory_manager.get_historical_data('temp_c', start_time, end_time)
        print(f"Retrieved {len(values)} historical points for temp_c")
        
        # Wait a bit to see monitoring in action
        print("\n⏱️ Monitoring for 30 seconds...")
        time.sleep(30)
        
    finally:
        # Stop memory management
        memory_manager.stop()
        print("✅ Memory management example completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_memory_management()
