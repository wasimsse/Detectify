"""Metrics calculation utilities."""

import time
from typing import List, Dict, Any
from collections import deque
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculate and track performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize metrics calculator.
        
        Args:
            window_size: Number of samples for moving average
        """
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.processing_time_history = deque(maxlen=window_size)
        self.detection_count_history = deque(maxlen=window_size)
        
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
    
    def update(
        self,
        fps: float,
        processing_time: float,
        detection_count: int
    ):
        """
        Update metrics with new values.
        
        Args:
            fps: Current FPS
            processing_time: Processing time for last frame
            detection_count: Number of detections in last frame
        """
        self.fps_history.append(fps)
        self.processing_time_history.append(processing_time)
        self.detection_count_history.append(detection_count)
        
        self.total_frames += 1
        self.total_detections += detection_count
    
    @property
    def avg_fps(self) -> float:
        """Get average FPS."""
        if not self.fps_history:
            return 0.0
        return np.mean(self.fps_history)
    
    @property
    def current_fps(self) -> float:
        """Get current FPS."""
        if not self.fps_history:
            return 0.0
        return self.fps_history[-1]
    
    @property
    def avg_processing_time(self) -> float:
        """Get average processing time."""
        if not self.processing_time_history:
            return 0.0
        return np.mean(self.processing_time_history)
    
    @property
    def avg_detections(self) -> float:
        """Get average number of detections per frame."""
        if not self.detection_count_history:
            return 0.0
        return np.mean(self.detection_count_history)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'current_fps': round(self.current_fps, 2),
            'avg_fps': round(self.avg_fps, 2),
            'avg_processing_time': round(self.avg_processing_time * 1000, 2),  # ms
            'avg_detections': round(self.avg_detections, 2),
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'elapsed_time': round(self.elapsed_time, 2)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.fps_history.clear()
        self.processing_time_history.clear()
        self.detection_count_history.clear()
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        logger.info("Metrics reset")


class FPSCounter:
    """Simple FPS counter."""
    
    def __init__(self):
        """Initialize FPS counter."""
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.fps = 0.0
    
    def update(self) -> float:
        """
        Update FPS counter.
        
        Returns:
            Current FPS
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Update FPS every second
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        return self.fps
    
    @property
    def current_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def reset(self):
        """Reset counter."""
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = self.start_time
        self.fps = 0.0

