"""Main Streamlit application."""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
from pathlib import Path
from datetime import datetime
import uuid

from ..config.settings import settings
from ..core.detector import ObjectDetector
from ..core.camera import CameraStream, DroidCamStream
from ..core.tracker import ObjectTracker
from ..database.models import DetectionLog
from ..utils.logger import get_logger, setup_logging
from ..utils.video_processor import VideoProcessor
from ..utils.image_processor import ImageProcessor
from ..utils.metrics import MetricsCalculator
from .components import (
    sidebar,
    analytics_dashboard,
    video_upload_section,
    image_upload_section,
    detection_history_table
)

# Setup logging
setup_logging()
logger = get_logger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    
    if 'tracker' not in st.session_state:
        st.session_state.tracker = None
    
    if 'db_log' not in st.session_state:
        st.session_state.db_log = DetectionLog() if settings.ENABLE_DB_LOGGING else None
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = MetricsCalculator()
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if 'current_source' not in st.session_state:
        st.session_state.current_source = None


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=settings.PAGE_TITLE,
        page_icon=settings.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .detection-frame {
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)


def process_webcam(config: dict):
    """
    Process webcam stream.
    
    Args:
        config: Configuration dictionary from sidebar
    """
    st.markdown("### 📹 Live Webcam Detection")
    
    # Initialize detector
    if st.session_state.detector is None or st.session_state.detector.model_name != config['model_name']:
        with st.spinner("Loading model..."):
            st.session_state.detector = ObjectDetector(
                model_name=config['model_name'],
                confidence_threshold=config['confidence_threshold'],
                iou_threshold=config['iou_threshold']
            )
    
    # Initialize tracker
    if config.get('enable_tracking', False):
        if st.session_state.tracker is None:
            st.session_state.tracker = ObjectTracker()
    else:
        st.session_state.tracker = None
    
    # Create placeholders
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Control buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        stop_button = st.button("⏹️ Stop", type="primary")
    
    # Initialize camera
    try:
        camera = CameraStream(camera_index=settings.DEFAULT_CAMERA_INDEX)
        st.session_state.camera_active = True
        st.session_state.current_source = "webcam"
        
        # Create session in database
        if st.session_state.db_log and config.get('save_detections', False):
            st.session_state.db_log.create_session(
                st.session_state.session_id,
                "webcam",
                config['model_name']
            )
        
        # Main processing loop
        while st.session_state.camera_active and not stop_button:
            ret, frame = camera.read()
            
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Detect objects
            result = st.session_state.detector.detect(
                frame,
                confidence_threshold=config['confidence_threshold'],
                draw_boxes=True
            )
            
            # Update tracker
            if st.session_state.tracker:
                tracks = st.session_state.tracker.update(
                    result.boxes,
                    result.classes,
                    result.confidences
                )
            
            # Update metrics
            st.session_state.metrics.update(
                camera.get_fps(),
                result.processing_time,
                result.total_objects
            )
            
            # Draw FPS if enabled
            if config.get('show_fps', True):
                cv2.putText(
                    result.frame,
                    f"FPS: {camera.get_fps():.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display frame
            video_placeholder.image(result.frame, channels="BGR", use_container_width=True)
            
            # Log to database
            if st.session_state.db_log and config.get('save_detections', False):
                st.session_state.db_log.log_detection(
                    source="webcam",
                    model=config['model_name'],
                    confidence_threshold=config['confidence_threshold'],
                    objects=result.to_dict()['objects'],
                    fps=camera.get_fps(),
                    processing_time=result.processing_time,
                    session_id=st.session_state.session_id
                )
            
            # Update stats
            with stats_placeholder.container():
                analytics_dashboard(
                    st.session_state.metrics.get_summary(),
                    result.object_counts
                )
            
            time.sleep(0.01)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Webcam error: {e}", exc_info=True)
    
    finally:
        if 'camera' in locals():
            camera.release()
        st.session_state.camera_active = False


def process_ip_camera(config: dict):
    """
    Process IP camera stream.
    
    Args:
        config: Configuration dictionary from sidebar
    """
    st.markdown("### 📱 IP Camera Detection")
    
    ip_address = config.get('ip_address', '')
    
    if not ip_address:
        st.warning("Please enter IP address in the sidebar")
        return
    
    # Test connection
    with st.spinner("Testing connection..."):
        if not DroidCamStream.test_connection(ip_address):
            st.error(
                f"❌ Cannot connect to {ip_address}\n\n"
                "Please check:\n"
                "- DroidCam app is running\n"
                "- Devices on same WiFi\n"
                "- IP address is correct\n"
                "- Port 4747 is not blocked"
            )
            return
    
    st.success(f"✅ Connected to {ip_address}")
    
    # Initialize detector
    if st.session_state.detector is None or st.session_state.detector.model_name != config['model_name']:
        with st.spinner("Loading model..."):
            st.session_state.detector = ObjectDetector(
                model_name=config['model_name'],
                confidence_threshold=config['confidence_threshold'],
                iou_threshold=config['iou_threshold']
            )
    
    # Initialize tracker
    if config.get('enable_tracking', False):
        if st.session_state.tracker is None:
            st.session_state.tracker = ObjectTracker()
    else:
        st.session_state.tracker = None
    
    # Create placeholders
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Control buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        stop_button = st.button("⏹️ Stop", type="primary")
    
    # Initialize camera
    try:
        camera = DroidCamStream(ip_address).start()
        time.sleep(2)  # Give camera time to start
        
        st.session_state.camera_active = True
        st.session_state.current_source = "ip_camera"
        
        # Create session in database
        if st.session_state.db_log and config.get('save_detections', False):
            st.session_state.db_log.create_session(
                st.session_state.session_id,
                "ip_camera",
                config['model_name']
            )
        
        # Main processing loop
        while st.session_state.camera_active and not stop_button:
            ret, frame = camera.read()
            
            if not ret:
                time.sleep(0.1)
                continue
            
            # Detect objects
            result = st.session_state.detector.detect(
                frame,
                confidence_threshold=config['confidence_threshold'],
                draw_boxes=True
            )
            
            # Update tracker
            if st.session_state.tracker:
                tracks = st.session_state.tracker.update(
                    result.boxes,
                    result.classes,
                    result.confidences
                )
            
            # Update metrics
            st.session_state.metrics.update(
                camera.get_fps(),
                result.processing_time,
                result.total_objects
            )
            
            # Draw FPS if enabled
            if config.get('show_fps', True):
                cv2.putText(
                    result.frame,
                    f"FPS: {camera.get_fps():.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display frame
            video_placeholder.image(result.frame, channels="BGR", use_container_width=True)
            
            # Log to database
            if st.session_state.db_log and config.get('save_detections', False):
                st.session_state.db_log.log_detection(
                    source="ip_camera",
                    model=config['model_name'],
                    confidence_threshold=config['confidence_threshold'],
                    objects=result.to_dict()['objects'],
                    fps=camera.get_fps(),
                    processing_time=result.processing_time,
                    session_id=st.session_state.session_id
                )
            
            # Update stats
            with stats_placeholder.container():
                analytics_dashboard(
                    st.session_state.metrics.get_summary(),
                    result.object_counts
                )
            
            time.sleep(0.01)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"IP camera error: {e}", exc_info=True)
    
    finally:
        if 'camera' in locals():
            camera.stop()
        st.session_state.camera_active = False


def process_video(config: dict):
    """
    Process uploaded video file.
    
    Args:
        config: Configuration dictionary from sidebar
    """
    st.markdown("### 🎬 Video Detection")
    
    video_bytes = video_upload_section()
    
    if not video_bytes:
        st.info("👆 Upload a video file to get started")
        return
    
    # Initialize detector
    if st.session_state.detector is None or st.session_state.detector.model_name != config['model_name']:
        with st.spinner("Loading model..."):
            st.session_state.detector = ObjectDetector(
                model_name=config['model_name'],
                confidence_threshold=config['confidence_threshold'],
                iou_threshold=config['iou_threshold']
            )
    
    # Save video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = Path(tmp_file.name)
    
    try:
        # Load video
        video_proc = VideoProcessor(tmp_path)
        st.info(f"📹 Video: {video_proc.width}x{video_proc.height}, {video_proc.fps} FPS, {video_proc.duration:.2f}s")
        
        # Process button
        if st.button("🚀 Process Video", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            video_placeholder = st.empty()
            
            # Output path
            output_path = settings.EXPORT_DIR / f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            frame_num = 0
            total_frames = video_proc.frame_count
            
            # Process each frame
            for frame in video_proc.frames():
                # Detect
                result = st.session_state.detector.detect(
                    frame,
                    confidence_threshold=config['confidence_threshold'],
                    draw_boxes=True
                )
                
                # Update progress
                frame_num += 1
                progress = frame_num / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_num}/{total_frames}")
                
                # Show every Nth frame
                if frame_num % 10 == 0:
                    video_placeholder.image(result.frame, channels="BGR", use_container_width=True)
            
            st.success(f"✅ Video processed! Saved to {output_path}")
    
    except Exception as e:
        st.error(f"Error processing video: {e}")
        logger.error(f"Video processing error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        tmp_path.unlink(missing_ok=True)


def process_image(config: dict):
    """
    Process uploaded image file.
    
    Args:
        config: Configuration dictionary from sidebar
    """
    st.markdown("### 🖼️ Image Detection")
    
    image_bytes = image_upload_section()
    
    if not image_bytes:
        st.info("👆 Upload an image to get started")
        return
    
    # Initialize detector
    if st.session_state.detector is None or st.session_state.detector.model_name != config['model_name']:
        with st.spinner("Loading model..."):
            st.session_state.detector = ObjectDetector(
                model_name=config['model_name'],
                confidence_threshold=config['confidence_threshold'],
                iou_threshold=config['iou_threshold']
            )
    
    # Load image
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        st.image(image, channels="BGR", use_container_width=True)
    
    # Detect objects
    with st.spinner("Detecting objects..."):
        result = st.session_state.detector.detect(
            image,
            confidence_threshold=config['confidence_threshold'],
            draw_boxes=True
        )
    
    with col2:
        st.markdown("#### Detected Objects")
        st.image(result.frame, channels="BGR", use_container_width=True)
    
    # Display results
    st.markdown("#### Detection Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Objects", result.total_objects)
    with col2:
        st.metric("Processing Time", f"{result.processing_time*1000:.1f} ms")
    with col3:
        st.metric("Classes Found", len(result.object_counts))
    
    # Object counts
    if result.object_counts:
        st.markdown("#### Object Counts")
        for cls, count in result.object_counts.items():
            st.write(f"- **{cls}**: {count}")
    
    # Save button
    if st.button("💾 Save Detected Image"):
        output_path = settings.SNAPSHOT_DIR / f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        ImageProcessor.save_image(result.frame, output_path)
        st.success(f"✅ Image saved to {output_path}")


def main():
    """Main application entry point."""
    configure_page()
    initialize_session_state()
    
    # Title
    st.title(f"{settings.PAGE_ICON} {settings.APP_NAME}")
    st.markdown(f"**Version {settings.APP_VERSION}** - Advanced Object Detection System")
    st.markdown("---")
    
    # Sidebar configuration
    config = sidebar()
    
    # Handle actions
    action = config.get('action')
    if action == 'snapshot':
        st.sidebar.success("📸 Snapshot saved!")
    elif action == 'export_csv':
        if st.session_state.db_log:
            output_path = settings.EXPORT_DIR / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.db_log.export_to_csv(output_path)
            st.sidebar.success(f"📊 Exported to {output_path.name}")
        else:
            st.sidebar.warning("Database logging not enabled")
    
    # Route to appropriate processing function
    source_type = config['source_type']
    
    if source_type == "Webcam":
        process_webcam(config)
    elif source_type == "IP Camera (DroidCam)":
        process_ip_camera(config)
    elif source_type == "Upload Video":
        process_video(config)
    elif source_type == "Upload Image":
        process_image(config)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Made with ❤️ using Streamlit and YOLOv8+ | "
        f"Running on {settings.device.upper()}"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

