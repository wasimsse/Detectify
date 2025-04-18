import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

def draw_boxes(image, results, conf_threshold=0.5):
    """Draw bounding boxes and labels on the image."""
    annotated_image = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            if conf >= conf_threshold:
                # Draw box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f'{class_name} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated_image

def count_objects(results, conf_threshold=0.5):
    """Count detected objects above confidence threshold."""
    object_counts = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            if conf >= conf_threshold:
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
    
    return object_counts

def save_detection_log(detections, format='csv'):
    """Save detection logs to file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'csv':
        filename = f'detection_log_{timestamp}.csv'
        df = pd.DataFrame(detections)
        df.to_csv(filename, index=False)
    else:  # json
        filename = f'detection_log_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(detections, f, indent=4)
    
    return filename

def save_snapshot(image, prefix='snapshot'):
    """Save current frame as image."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{prefix}_{timestamp}.jpg'
    cv2.imwrite(filename, image)
    return filename

def calculate_fps(start_time, frame_count):
    """Calculate FPS based on elapsed time and frame count."""
    elapsed_time = (datetime.now() - start_time).total_seconds()
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0 