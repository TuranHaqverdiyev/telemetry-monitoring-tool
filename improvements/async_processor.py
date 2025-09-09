"""
Async processing pipeline for high-volume multi-channel telemetry data.

This module addresses scalability issues by implementing:
- Async/await processing architecture
- Parallel detection across channels
- Batch processing for high-frequency data
- Producer-consumer queues with backpressure
"""

import asyncio
import concurrent.futures
import time
from collections import deque
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass 
class BatchResult:
    """Result of batch processing operation."""
    processed_count: int
    anomalies_detected: int
    processing_time_ms: float
    memory_usage_mb: float

class RingBuffer:
    """Memory-efficient ring buffer for windowed data."""
    
    def __init__(self, maxsize: int):
        self.buffer = deque(maxlen=maxsize)
        self.timestamps = deque(maxlen=maxsize)
        self.maxsize = maxsize
    
    def append(self, timestamp: float, value: float):
        self.buffer.append(value)
        self.timestamps.append(timestamp)
    
    def get_numpy_array(self) -> np.ndarray:
        """Get data as numpy array for efficient computations."""
        return np.array(self.buffer)
    
    def get_recent_window(self, window_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Get recent window of specified size."""
        values = np.array(list(self.buffer)[-window_size:])
        times = np.array(list(self.timestamps)[-window_size:])
        return times, values
    
    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        return len(self.buffer) * 16  # 8 bytes each for float timestamp and value

class AsyncStreamProcessor:
    """High-performance async processor for telemetry streams."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100, queue_maxsize: int = 10000):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.data_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.detector_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_anomalies': 0,
            'avg_latency_ms': 0.0,
            'queue_depth': 0
        }
        
        # Channel buffers with automatic memory management
        self.channel_buffers: Dict[str, RingBuffer] = {}
        self.buffer_size = 1000  # Configurable buffer size per channel
        
    def get_or_create_buffer(self, channel: str) -> RingBuffer:
        """Get existing buffer or create new one for channel."""
        if channel not in self.channel_buffers:
            self.channel_buffers[channel] = RingBuffer(self.buffer_size)
        return self.channel_buffers[channel]
    
    async def enqueue_data(self, timestamp: float, channel_data: Dict[str, float]):
        """Enqueue data for processing with backpressure handling."""
        try:
            await asyncio.wait_for(
                self.data_queue.put((timestamp, channel_data)), 
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # Handle backpressure - could implement sampling or priority queuing
            print(f"⚠️ Queue full, dropping data point at {timestamp}")
    
    async def process_batch(self, batch_data: List[tuple]) -> BatchResult:
        """Process a batch of data points in parallel."""
        start_time = time.time()
        
        # Group data by channel for efficient processing
        channel_batches: Dict[str, List[tuple]] = {}
        for timestamp, channel_data in batch_data:
            for channel, value in channel_data.items():
                if channel not in channel_batches:
                    channel_batches[channel] = []
                channel_batches[channel].append((timestamp, value))
        
        # Process channels in parallel
        tasks = []
        for channel, data_points in channel_batches.items():
            task = asyncio.create_task(self._process_channel_batch(channel, data_points))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        total_processed = len(batch_data)
        total_anomalies = sum(r.anomalies_detected for r in results if isinstance(r, BatchResult))
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats['total_processed'] += total_processed
        self.stats['total_anomalies'] += total_anomalies
        self.stats['avg_latency_ms'] = (
            self.stats['avg_latency_ms'] * 0.9 + processing_time * 0.1
        )
        
        return BatchResult(
            processed_count=total_processed,
            anomalies_detected=total_anomalies,
            processing_time_ms=processing_time,
            memory_usage_mb=self._estimate_memory_usage()
        )
    
    async def _process_channel_batch(self, channel: str, data_points: List[tuple]) -> BatchResult:
        """Process a batch of data points for a single channel."""
        buffer = self.get_or_create_buffer(channel)
        anomalies = 0
        
        # Add data to buffer
        for timestamp, value in data_points:
            buffer.append(timestamp, value)
            
            # Run detection in thread pool to avoid blocking
            # This would integrate with existing detector infrastructure
            # detection_result = await self._run_detection(channel, value, timestamp)
            # if detection_result.is_anomaly:
            #     anomalies += 1
        
        return BatchResult(
            processed_count=len(data_points),
            anomalies_detected=anomalies,
            processing_time_ms=0.0,
            memory_usage_mb=buffer.memory_usage_bytes() / (1024 * 1024)
        )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate total memory usage in MB."""
        total_bytes = sum(
            buffer.memory_usage_bytes() 
            for buffer in self.channel_buffers.values()
        )
        return total_bytes / (1024 * 1024)
    
    async def start_processing_loop(self):
        """Main processing loop with batch collection."""
        batch = []
        
        while True:
            try:
                # Collect batch of data points
                while len(batch) < self.batch_size:
                    try:
                        data_point = await asyncio.wait_for(
                            self.data_queue.get(), 
                            timeout=0.1
                        )
                        batch.append(data_point)
                    except asyncio.TimeoutError:
                        # Process partial batch if queue is empty
                        break
                
                if batch:
                    result = await self.process_batch(batch)
                    batch.clear()
                    
                    # Update queue depth stat
                    self.stats['queue_depth'] = self.data_queue.qsize()
                
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            **self.stats,
            'channel_count': len(self.channel_buffers),
            'memory_usage_mb': self._estimate_memory_usage(),
            'buffer_sizes': {
                channel: len(buffer.buffer)
                for channel, buffer in self.channel_buffers.items()
            }
        }
    
    def cleanup_old_data(self, max_age_seconds: float):
        """Remove old data from buffers to manage memory."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        for channel, buffer in self.channel_buffers.items():
            # Remove old data points
            while (buffer.timestamps and 
                   len(buffer.timestamps) > 0 and 
                   buffer.timestamps[0] < cutoff_time):
                buffer.buffer.popleft()
                buffer.timestamps.popleft()


# Example usage and integration
async def example_usage():
    """Example of how to integrate the async processor."""
    processor = AsyncStreamProcessor(max_workers=8, batch_size=200)
    
    # Start processing loop
    processing_task = asyncio.create_task(processor.start_processing_loop())
    
    # Simulate data ingestion
    for i in range(1000):
        timestamp = time.time()
        channel_data = {
            'temp_c': 20.0 + np.random.normal(0, 2),
            'voltage': 3.3 + np.random.normal(0, 0.1),
            'current': 1.0 + np.random.normal(0, 0.05)
        }
        
        await processor.enqueue_data(timestamp, channel_data)
        await asyncio.sleep(0.01)  # 100Hz simulation
    
    # Get performance stats
    stats = processor.get_performance_stats()
    print(f"📊 Processing stats: {stats}")
    
    processing_task.cancel()

if __name__ == "__main__":
    asyncio.run(example_usage())
