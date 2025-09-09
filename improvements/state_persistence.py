"""
State persistence system for anomaly detectors.

This module provides:
- Serialization/deserialization for all detector types
- Automatic state saving and restoration
- State validation and migration
- Backup and recovery mechanisms
"""

import json
import pickle
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Type, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class DetectorState:
    """Standardized detector state container."""
    detector_type: str
    detector_id: str  # Unique identifier for the detector instance
    channel: str
    parameters: Dict[str, Any]
    internal_state: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectorState':
        return cls(**data)

class StateValidationError(Exception):
    """Raised when detector state validation fails."""
    pass

class StateMigrationError(Exception):
    """Raised when state migration fails."""
    pass

class PersistentDetectorMixin:
    """Mixin class to add persistence capabilities to detectors."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_version = "1.0"
        self._detector_id = f"{self.__class__.__name__}_{id(self)}"
    
    @abstractmethod
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get detector's internal state for serialization."""
        pass
    
    @abstractmethod
    def _set_internal_state(self, state: Dict[str, Any]) -> None:
        """Restore detector's internal state from serialization."""
        pass
    
    def save_state(self) -> DetectorState:
        """Save complete detector state."""
        return DetectorState(
            detector_type=self.__class__.__name__,
            detector_id=self._detector_id,
            channel=getattr(self, 'channel', 'unknown'),
            parameters=self.get_parameters(),
            internal_state=self._get_internal_state(),
            metadata={
                'creation_time': getattr(self, '_creation_time', time.time()),
                'detection_count': getattr(self, 'detection_count', 0),
                'anomaly_count': getattr(self, 'anomaly_count', 0),
                'last_update': time.time()
            },
            timestamp=time.time(),
            version=self._state_version
        )
    
    def load_state(self, state: DetectorState) -> None:
        """Load detector state."""
        # Validate state compatibility
        if state.detector_type != self.__class__.__name__:
            raise StateValidationError(
                f"State type mismatch: expected {self.__class__.__name__}, got {state.detector_type}"
            )
        
        # Migrate state if needed
        if state.version != self._state_version:
            state = self._migrate_state(state)
        
        # Restore state
        self.set_parameters(state.parameters)
        self._set_internal_state(state.internal_state)
        
        # Restore metadata
        self._detector_id = state.detector_id
        if 'detection_count' in state.metadata:
            self.detection_count = state.metadata['detection_count']
        if 'anomaly_count' in state.metadata:
            self.anomaly_count = state.metadata['anomaly_count']
        
        logger.info(f"✅ Loaded state for {self._detector_id} from {datetime.fromtimestamp(state.timestamp)}")
    
    def _migrate_state(self, state: DetectorState) -> DetectorState:
        """Migrate state between versions."""
        logger.info(f"🔄 Migrating state from v{state.version} to v{self._state_version}")
        
        # Example migration logic
        if state.version == "1.0" and self._state_version == "1.1":
            # Add new fields with defaults
            state.internal_state.setdefault('new_field', 0.0)
        
        state.version = self._state_version
        return state

class StateManager:
    """Manages saving and loading of detector states."""
    
    def __init__(self, 
                 state_dir: str = "detector_states",
                 backup_count: int = 5,
                 auto_save_interval: int = 300):  # 5 minutes
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.backup_count = backup_count
        self.auto_save_interval = auto_save_interval
        
        # Thread safety
        self._lock = threading.RLock()
        self._auto_save_timer: Optional[threading.Timer] = None
        
        # State tracking
        self._last_save = 0.0
        self._pending_changes = False
        
        logger.info(f"📁 StateManager initialized with directory: {self.state_dir}")
    
    def start_auto_save(self, detectors_getter):
        """Start automatic state saving."""
        def auto_save():
            try:
                if self._pending_changes:
                    detectors = detectors_getter()
                    self.save_all_detectors(detectors)
                    self._pending_changes = False
            except Exception as e:
                logger.error(f"❌ Auto-save failed: {e}")
            finally:
                # Schedule next save
                self._auto_save_timer = threading.Timer(self.auto_save_interval, auto_save)
                self._auto_save_timer.daemon = True
                self._auto_save_timer.start()
        
        # Start first timer
        self._auto_save_timer = threading.Timer(self.auto_save_interval, auto_save)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()
        
        logger.info(f"⏰ Auto-save started with {self.auto_save_interval}s interval")
    
    def stop_auto_save(self):
        """Stop automatic state saving."""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None
        logger.info("⏹️ Auto-save stopped")
    
    def mark_changes_pending(self):
        """Mark that detector states have changed."""
        self._pending_changes = True
    
    @contextmanager
    def _file_lock(self, file_path: Path):
        """Simple file locking mechanism."""
        lock_file = file_path.with_suffix(file_path.suffix + '.lock')
        
        try:
            # Wait for lock to be available
            attempts = 0
            while lock_file.exists() and attempts < 10:
                time.sleep(0.1)
                attempts += 1
            
            # Create lock
            lock_file.write_text(str(time.time()))
            yield
        finally:
            # Remove lock
            if lock_file.exists():
                lock_file.unlink()
    
    def save_all_detectors(self, detectors: Dict[str, List[Any]]) -> bool:
        """Save states of all detectors."""
        with self._lock:
            try:
                timestamp = time.time()
                state_file = self.state_dir / f"detector_states_{int(timestamp)}.json"
                
                # Collect all states
                all_states = {}
                for channel, channel_detectors in detectors.items():
                    channel_states = []
                    for detector in channel_detectors:
                        if hasattr(detector, 'save_state'):
                            state = detector.save_state()
                            channel_states.append(state.to_dict())
                        else:
                            logger.warning(f"⚠️ Detector {detector} doesn't support state saving")
                    all_states[channel] = channel_states
                
                # Save to file with locking
                with self._file_lock(state_file):
                    with state_file.open('w') as f:
                        json.dump({
                            'version': '1.0',
                            'timestamp': timestamp,
                            'detector_states': all_states,
                            'metadata': {
                                'total_detectors': sum(len(ch_det) for ch_det in detectors.values()),
                                'channels': list(detectors.keys()),
                                'save_time': datetime.now().isoformat()
                            }
                        }, f, indent=2)
                
                # Create symlink to latest
                latest_link = self.state_dir / "latest.json"
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(state_file.name)
                
                # Cleanup old backups
                self._cleanup_old_backups()
                
                self._last_save = timestamp
                logger.info(f"💾 Saved states for {sum(len(ch_det) for ch_det in detectors.values())} detectors")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to save detector states: {e}")
                return False
    
    def load_latest_states(self) -> Optional[Dict[str, List[DetectorState]]]:
        """Load the most recent detector states."""
        latest_file = self.state_dir / "latest.json"
        
        if not latest_file.exists():
            logger.info("📂 No saved states found")
            return None
        
        return self.load_states_from_file(latest_file)
    
    def load_states_from_file(self, file_path: Path) -> Optional[Dict[str, List[DetectorState]]]:
        """Load detector states from specific file."""
        try:
            with file_path.open('r') as f:
                data = json.load(f)
            
            states = {}
            for channel, channel_states in data['detector_states'].items():
                states[channel] = [
                    DetectorState.from_dict(state_dict) 
                    for state_dict in channel_states
                ]
            
            logger.info(f"📄 Loaded states from {file_path} (saved: {data['metadata']['save_time']})")
            return states
            
        except Exception as e:
            logger.error(f"❌ Failed to load states from {file_path}: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        state_files = sorted(
            self.state_dir.glob("detector_states_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove files beyond backup count
        for old_file in state_files[self.backup_count:]:
            try:
                old_file.unlink()
                logger.debug(f"🗑️ Removed old backup: {old_file.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove old backup {old_file}: {e}")
    
    def create_manual_backup(self, detectors: Dict[str, List[Any]], name: str = None) -> bool:
        """Create a manual backup with optional custom name."""
        if name is None:
            name = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.state_dir / f"backup_{name}.json"
        
        # Temporarily redirect save to backup file
        original_save = self.save_all_detectors
        try:
            # Save directly to backup file
            with self._lock:
                timestamp = time.time()
                all_states = {}
                for channel, channel_detectors in detectors.items():
                    channel_states = []
                    for detector in channel_detectors:
                        if hasattr(detector, 'save_state'):
                            state = detector.save_state()
                            channel_states.append(state.to_dict())
                    all_states[channel] = channel_states
                
                with backup_file.open('w') as f:
                    json.dump({
                        'version': '1.0',
                        'timestamp': timestamp,
                        'detector_states': all_states,
                        'backup_name': name,
                        'metadata': {
                            'total_detectors': sum(len(ch_det) for ch_det in detectors.values()),
                            'channels': list(detectors.keys()),
                            'backup_time': datetime.now().isoformat()
                        }
                    }, f, indent=2)
                
                logger.info(f"💾 Created manual backup: {backup_file.name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to create manual backup: {e}")
            return False
    
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backup files with metadata."""
        backups = []
        
        for backup_file in self.state_dir.glob("*.json"):
            if backup_file.name == "latest.json":
                continue
                
            try:
                with backup_file.open('r') as f:
                    data = json.load(f)
                
                backups.append({
                    'filename': backup_file.name,
                    'timestamp': data.get('timestamp', 0),
                    'save_time': data.get('metadata', {}).get('save_time', 'unknown'),
                    'total_detectors': data.get('metadata', {}).get('total_detectors', 0),
                    'channels': data.get('metadata', {}).get('channels', []),
                    'size_bytes': backup_file.stat().st_size
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to read backup metadata from {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)

# Example detector implementation with persistence
class PersistentEWMAZScoreDetector(PersistentDetectorMixin):
    """Example of EWMA Z-Score detector with persistence."""
    
    def __init__(self, alpha: float = 0.05, z_threshold: float = 3.0, channel: str = "default"):
        super().__init__()
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.channel = channel
        
        # Internal state
        self._state = {"mean": 0.0, "var": 0.0, "count": 0}
        self.detection_count = 0
        self.anomaly_count = 0
        self._creation_time = time.time()
    
    def get_parameters(self) -> Dict[str, float]:
        return {
            'alpha': self.alpha,
            'z_threshold': self.z_threshold
        }
    
    def set_parameters(self, params: Dict[str, float]) -> None:
        self.alpha = params.get('alpha', self.alpha)
        self.z_threshold = params.get('z_threshold', self.z_threshold)
    
    def _get_internal_state(self) -> Dict[str, Any]:
        return {
            'mean': self._state['mean'],
            'var': self._state['var'], 
            'count': self._state['count']
        }
    
    def _set_internal_state(self, state: Dict[str, Any]) -> None:
        self._state = {
            'mean': state.get('mean', 0.0),
            'var': state.get('var', 0.0),
            'count': state.get('count', 0)
        }

# Example usage
def example_usage():
    """Example of how to use the state persistence system."""
    
    # Create state manager
    state_manager = StateManager(
        state_dir="./detector_states",
        auto_save_interval=60  # Save every minute
    )
    
    # Create detectors
    detectors = {
        'temp_c': [PersistentEWMAZScoreDetector(channel='temp_c')],
        'voltage': [PersistentEWMAZScoreDetector(channel='voltage')]
    }
    
    # Load previous states if available
    saved_states = state_manager.load_latest_states()
    if saved_states:
        for channel, channel_states in saved_states.items():
            if channel in detectors:
                for i, state in enumerate(channel_states):
                    if i < len(detectors[channel]):
                        try:
                            detectors[channel][i].load_state(state)
                        except Exception as e:
                            logger.error(f"Failed to load state for {channel}[{i}]: {e}")
    
    # Start auto-save
    state_manager.start_auto_save(lambda: detectors)
    
    # Simulate some processing
    print("Processing data...")
    for detector_list in detectors.values():
        for detector in detector_list:
            detector.detection_count += 10
            detector.anomaly_count += 1
            state_manager.mark_changes_pending()
    
    # Manual save
    success = state_manager.save_all_detectors(detectors)
    print(f"Manual save: {'✅ Success' if success else '❌ Failed'}")
    
    # Create backup
    backup_success = state_manager.create_manual_backup(detectors, "example_backup")
    print(f"Backup: {'✅ Success' if backup_success else '❌ Failed'}")
    
    # List backups
    backups = state_manager.list_available_backups()
    print(f"📄 Available backups: {len(backups)}")
    for backup in backups:
        print(f"  - {backup['filename']}: {backup['total_detectors']} detectors, {backup['save_time']}")
    
    # Stop auto-save
    state_manager.stop_auto_save()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()
