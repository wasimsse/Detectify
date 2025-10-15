"""Object tracking implementation using ByteTrack-like algorithm."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """Represents a tracked object."""
    
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    
    def update(self, bbox: List[float], confidence: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
    
    def predict(self):
        """Predict next position (simple constant velocity model)."""
        self.time_since_update += 1
        self.age += 1
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bbox center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get bbox area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class ObjectTracker:
    """Simple object tracker using IoU matching."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize object tracker.
        
        Args:
            max_age: Maximum frames to keep alive track without detections
            min_hits: Minimum hits to consider track as valid
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0
        
        logger.info("ObjectTracker initialized")
    
    def update(
        self,
        boxes: List[List[float]],
        classes: List[str],
        confidences: List[float]
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            classes: List of class names
            confidences: List of confidence scores
        
        Returns:
            List of current tracks
        """
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        if len(boxes) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_tracks = self._match_detections(
                boxes, classes, confidences
            )
            
            # Update matched tracks
            for det_idx, track_idx in matched:
                self.tracks[track_idx].update(boxes[det_idx], confidences[det_idx])
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track(boxes[det_idx], classes[det_idx], confidences[det_idx])
            
            # Mark unmatched tracks
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].time_since_update += 1
        
        elif len(boxes) > 0:
            # No existing tracks, create new ones
            for i in range(len(boxes)):
                self._create_track(boxes[i], classes[i], confidences[i])
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]
        
        # Return valid tracks
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]
    
    def _match_detections(
        self,
        boxes: List[List[float]],
        classes: List[str],
        confidences: List[float]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            Tuple of (matched pairs, unmatched detections, unmatched tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(boxes))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(boxes), len(self.tracks)))
        
        for d, det_box in enumerate(boxes):
            for t, track in enumerate(self.tracks):
                # Only match same class
                if classes[d] == track.class_name:
                    iou_matrix[d, t] = self._compute_iou(det_box, track.bbox)
        
        # Use Hungarian algorithm for matching
        matched_indices = []
        
        if min(iou_matrix.shape) > 0:
            # Get matches where IoU > threshold
            det_indices, track_indices = linear_sum_assignment(-iou_matrix)
            
            for d, t in zip(det_indices, track_indices):
                if iou_matrix[d, t] >= self.iou_threshold:
                    matched_indices.append((d, t))
        
        # Find unmatched detections and tracks
        matched_dets = [m[0] for m in matched_indices]
        matched_tracks = [m[1] for m in matched_indices]
        
        unmatched_dets = [i for i in range(len(boxes)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        
        return matched_indices, unmatched_dets, unmatched_tracks
    
    def _create_track(self, bbox: List[float], class_name: str, confidence: float):
        """Create a new track."""
        track = Track(
            track_id=self.next_id,
            bbox=bbox,
            class_name=class_name,
            confidence=confidence,
            hits=1,
            time_since_update=0
        )
        self.tracks.append(track)
        self.next_id += 1
    
    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
        
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
        logger.info("Tracker reset")
    
    @property
    def track_count(self) -> int:
        """Get number of active tracks."""
        return len([t for t in self.tracks if t.hits >= self.min_hits])

