"""Object detection engine with YOLO models."""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from ultralytics import YOLO
import torch

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DetectionResult:
    """Container for detection results."""
    
    def __init__(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        classes: List[str],
        confidences: List[float],
        processing_time: float,
        model_name: str
    ):
        self.frame = frame
        self.boxes = boxes
        self.classes = classes
        self.confidences = confidences
        self.processing_time = processing_time
        self.model_name = model_name
        self.timestamp = time.time()
    
    @property
    def object_counts(self) -> Dict[str, int]:
        """Get count of objects by class."""
        counts = {}
        for cls in self.classes:
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    @property
    def total_objects(self) -> int:
        """Get total number of detected objects."""
        return len(self.classes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        objects = []
        for i in range(len(self.classes)):
            objects.append({
                'class': self.classes[i],
                'confidence': self.confidences[i],
                'bbox': self.boxes[i]
            })
        
        return {
            'timestamp': self.timestamp,
            'model': self.model_name,
            'processing_time': self.processing_time,
            'total_objects': self.total_objects,
            'object_counts': self.object_counts,
            'objects': objects
        }


class ObjectDetector:
    """YOLO-based object detection engine."""
    
    def __init__(
        self,
        model_name: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        device: str = None
    ):
        """
        Initialize object detector.
        
        Args:
            model_name: Name of the YOLO model to use
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
        """
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or settings.IOU_THRESHOLD
        self.device = device or settings.device
        
        self.model = None
        self.class_names = []
        self._load_model()
        
        logger.info(f"Initialized ObjectDetector with model: {self.model_name} on {self.device}")
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            model_path = settings.BASE_DIR / self.model_name
            
            # Download model if it doesn't exist
            if not model_path.exists():
                logger.info(f"Model {self.model_name} not found, downloading...")
                self.model = YOLO(self.model_name)
            else:
                self.model = YOLO(str(model_path))
            
            # Move model to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("Model loaded on GPU")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.model.to("mps")
                logger.info("Model loaded on Apple Silicon GPU")
            else:
                self.model.to("cpu")
                logger.info("Model loaded on CPU")
            
            # Get class names
            self.class_names = self.model.names
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def change_model(self, model_name: str):
        """Change the detection model."""
        self.model_name = model_name
        self._load_model()
        logger.info(f"Model changed to: {model_name}")
    
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        draw_boxes: bool = True
    ) -> DetectionResult:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Override confidence threshold
            draw_boxes: Whether to draw bounding boxes on frame
        
        Returns:
            DetectionResult object
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")
        
        conf_thresh = confidence_threshold or self.confidence_threshold
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=conf_thresh,
                iou=self.iou_threshold,
                max_det=settings.MAX_DETECTIONS,
                verbose=False
            )
            
            # Extract results
            boxes = []
            classes = []
            confidences = []
            
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    classes.append(self.class_names[cls])
                    confidences.append(conf)
            
            processing_time = time.time() - start_time
            
            # Draw boxes if requested
            output_frame = frame.copy()
            if draw_boxes and len(boxes) > 0:
                output_frame = self._draw_detections(
                    output_frame, boxes, classes, confidences
                )
            
            return DetectionResult(
                frame=output_frame,
                boxes=boxes,
                classes=classes,
                confidences=confidences,
                processing_time=processing_time,
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        classes: List[str],
        confidences: List[float],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            boxes: List of bounding boxes [x1, y1, x2, y2]
            classes: List of class names
            confidences: List of confidence scores
            color: Box color in BGR
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label = f"{classes[i]} {confidences[i]:.2f}"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        confidence_threshold: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Perform batch detection on multiple frames.
        
        Args:
            frames: List of input frames
            confidence_threshold: Override confidence threshold
        
        Returns:
            List of DetectionResult objects
        """
        results = []
        for frame in frames:
            result = self.detect(frame, confidence_threshold)
            results.append(result)
        return results
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'num_classes': len(self.class_names),
            'class_names': list(self.class_names.values()) if isinstance(self.class_names, dict) else self.class_names
        }

