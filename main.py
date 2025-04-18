import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pandas as pd
from utils import draw_boxes, count_objects, save_detection_log, save_snapshot, calculate_fps

# Page config
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = YOLO('yolov8n.pt')
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# Sidebar controls
st.sidebar.title("Controls")

# Camera selection
camera_source = st.sidebar.radio(
    "Select Camera Source",
    ["Laptop Webcam", "IP Camera", "Both"]
)

# IP camera URL input
ip_camera_url = ""
if camera_source in ["IP Camera", "Both"]:
    ip_camera_url = st.sidebar.text_input(
        "IP Camera URL",
        "http://192.168.1.100:8080/video"
    )

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Export format
export_format = st.sidebar.radio(
    "Export Format",
    ["CSV", "JSON"]
)

# Main content
st.title("Real-Time Object Detection")

# Create columns for video display
if camera_source == "Both":
    col1, col2 = st.columns(2)
else:
    col1 = st.container()

# Function to process video stream
def process_video_stream(cap, placeholder, source_name):
    if not cap.isOpened():
        st.error(f"Error: Could not open {source_name}")
        return None
    
    ret, frame = cap.read()
    if not ret:
        st.error(f"Error: Could not read frame from {source_name}")
        return None
    
    # Run detection
    results = st.session_state.model(frame)
    
    # Draw boxes and count objects
    annotated_frame = draw_boxes(frame, results, conf_threshold)
    object_counts = count_objects(results, conf_threshold)
    
    # Calculate FPS
    st.session_state.frame_count += 1
    fps = calculate_fps(st.session_state.start_time, st.session_state.frame_count)
    
    # Add FPS and object count to frame
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Log detections
    for obj, count in object_counts.items():
        st.session_state.detection_log.append({
            'timestamp': datetime.now().isoformat(),
            'source': source_name,
            'object': obj,
            'count': count,
            'confidence': conf_threshold
        })
    
    # Display frame
    placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    
    # Display object counts
    st.write(f"Objects detected in {source_name}:")
    for obj, count in object_counts.items():
        st.write(f"- {obj}: {count}")
    
    return annotated_frame

# Main video processing loop
try:
    if camera_source in ["Laptop Webcam", "Both"]:
        with col1:
            st.subheader("Laptop Webcam")
            laptop_cap = cv2.VideoCapture(0)
            laptop_placeholder = st.empty()
            laptop_frame = process_video_stream(laptop_cap, laptop_placeholder, "Laptop Webcam")
    
    if camera_source in ["IP Camera", "Both"]:
        with col2 if camera_source == "Both" else col1:
            st.subheader("IP Camera")
            ip_cap = cv2.VideoCapture(ip_camera_url)
            ip_placeholder = st.empty()
            ip_frame = process_video_stream(ip_cap, ip_placeholder, "IP Camera")
    
    # Snapshot and export buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Take Snapshot"):
            if camera_source == "Laptop Webcam":
                filename = save_snapshot(laptop_frame)
            else:
                filename = save_snapshot(ip_frame)
            st.success(f"Snapshot saved as {filename}")
    
    with col2:
        if st.button("Export Detection Log"):
            filename = save_detection_log(
                st.session_state.detection_log,
                format=export_format.lower()
            )
            st.success(f"Detection log saved as {filename}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

finally:
    # Clean up
    if 'laptop_cap' in locals():
        laptop_cap.release()
    if 'ip_cap' in locals():
        ip_cap.release() 