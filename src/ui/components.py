"""UI components for Streamlit interface."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from ..config.settings import settings


def sidebar() -> Dict[str, Any]:
    """
    Render sidebar with controls.
    
    Returns:
        Dictionary with user selections
    """
    st.sidebar.title("⚙️ Controls")
    
    # Input source selection
    st.sidebar.markdown("### 📹 Input Source")
    source_type = st.sidebar.selectbox(
        "Select Source",
        ["Webcam", "IP Camera (DroidCam)", "Upload Video", "Upload Image"],
        help="Choose your input source for detection"
    )
    
    config = {"source_type": source_type}
    
    # IP Camera configuration
    if source_type == "IP Camera (DroidCam)":
        st.sidebar.markdown("#### DroidCam Setup")
        st.sidebar.info(
            "1. Install DroidCam on your phone\n"
            "2. Connect to same WiFi\n"
            "3. Enter IP address shown in app"
        )
        ip_address = st.sidebar.text_input(
            "IP Address",
            placeholder="192.168.1.100",
            help="IP address from DroidCam app"
        )
        config["ip_address"] = ip_address
    
    # Model selection
    st.sidebar.markdown("### 🤖 Model Settings")
    
    model_name = st.sidebar.selectbox(
        "YOLO Model",
        settings.AVAILABLE_MODELS,
        index=0,
        help="Select YOLO model variant"
    )
    config["model_name"] = model_name
    
    # Display model info
    model_info = get_model_info(model_name)
    st.sidebar.caption(f"**Size**: {model_info['size']} | **Speed**: {model_info['speed']}")
    
    # Detection parameters
    st.sidebar.markdown("### 🎯 Detection Parameters")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence for detections"
    )
    config["confidence_threshold"] = confidence_threshold
    
    iou_threshold = st.sidebar.slider(
        "IOU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.IOU_THRESHOLD,
        step=0.05,
        help="IOU threshold for Non-Maximum Suppression"
    )
    config["iou_threshold"] = iou_threshold
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        enable_tracking = st.checkbox(
            "Enable Object Tracking",
            value=True,
            help="Track objects across frames"
        )
        config["enable_tracking"] = enable_tracking
        
        save_detections = st.checkbox(
            "Save Detections to Database",
            value=settings.ENABLE_DB_LOGGING,
            help="Log detections to SQLite database"
        )
        config["save_detections"] = save_detections
        
        show_fps = st.checkbox(
            "Show FPS Counter",
            value=True,
            help="Display FPS on video"
        )
        config["show_fps"] = show_fps
        
        show_labels = st.checkbox(
            "Show Labels",
            value=True,
            help="Display class names and confidence"
        )
        config["show_labels"] = show_labels
    
    # Export options
    st.sidebar.markdown("### 💾 Export")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📸 Snapshot", use_container_width=True):
            config["action"] = "snapshot"
    
    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            config["action"] = "export_csv"
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        f"**{settings.APP_NAME}** v{settings.APP_VERSION}\n\n"
        f"Advanced Object Detection System\n\n"
        f"Device: **{settings.device.upper()}**"
    )
    
    return config


def analytics_dashboard(stats: Dict[str, Any], object_counts: Dict[str, int]):
    """
    Render analytics dashboard.
    
    Args:
        stats: Performance statistics
        object_counts: Object detection counts
    """
    st.markdown("### 📊 Analytics Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "FPS",
            f"{stats.get('current_fps', 0):.1f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Objects Detected",
            stats.get('total_detections', 0)
        )
    
    with col3:
        st.metric(
            "Processing Time",
            f"{stats.get('avg_processing_time', 0):.1f} ms"
        )
    
    with col4:
        st.metric(
            "Total Frames",
            stats.get('total_frames', 0)
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Object distribution pie chart
        if object_counts:
            fig = create_object_distribution_chart(object_counts)
            st.plotly_chart(fig, use_container_width=True, key="object_distribution_chart")
        else:
            st.info("No objects detected yet")
    
    with col2:
        # FPS timeline (placeholder for now)
        st.info("📈 FPS Timeline - Coming soon")


def detection_history_table(detections: List[Dict[str, Any]]):
    """
    Display detection history table.
    
    Args:
        detections: List of detection records
    """
    if not detections:
        st.info("No detection history available")
        return
    
    df = pd.DataFrame(detections)
    
    # Format columns
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def create_object_distribution_chart(object_counts: Dict[str, int]) -> go.Figure:
    """
    Create pie chart for object distribution.
    
    Args:
        object_counts: Dictionary of object counts
    
    Returns:
        Plotly figure
    """
    labels = list(object_counts.keys())
    values = list(object_counts.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Object Distribution",
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Get model information for display.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model info
    """
    # Extract model size (n, s, m, l, x)
    if 'n' in model_name.lower():
        size = "Nano"
        speed = "⚡⚡⚡ Fastest"
    elif 's' in model_name.lower():
        size = "Small"
        speed = "⚡⚡ Fast"
    elif 'm' in model_name.lower():
        size = "Medium"
        speed = "⚡ Balanced"
    elif 'l' in model_name.lower():
        size = "Large"
        speed = "🎯 Accurate"
    elif 'x' in model_name.lower():
        size = "Extra Large"
        speed = "🎯🎯 Most Accurate"
    else:
        size = "Unknown"
        speed = "N/A"
    
    return {
        "size": size,
        "speed": speed
    }


def video_upload_section() -> Optional[bytes]:
    """
    Render video upload section.
    
    Returns:
        Uploaded video file bytes or None
    """
    st.markdown("### 📹 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=settings.SUPPORTED_VIDEO_FORMATS,
        help=f"Supported formats: {', '.join(settings.SUPPORTED_VIDEO_FORMATS)}"
    )
    
    if uploaded_file:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
            st.error(f"File too large! Maximum size: {settings.MAX_VIDEO_SIZE_MB} MB")
            return None
        
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        return uploaded_file.read()
    
    return None


def image_upload_section() -> Optional[bytes]:
    """
    Render image upload section.
    
    Returns:
        Uploaded image file bytes or None
    """
    st.markdown("### 🖼️ Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=settings.SUPPORTED_IMAGE_FORMATS,
        help=f"Supported formats: {', '.join(settings.SUPPORTED_IMAGE_FORMATS)}"
    )
    
    if uploaded_file:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            st.error(f"File too large! Maximum size: {settings.MAX_IMAGE_SIZE_MB} MB")
            return None
        
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        return uploaded_file.read()
    
    return None


def detection_zone_selector(frame_shape: tuple) -> Optional[List[int]]:
    """
    Allow user to select detection zone.
    
    Args:
        frame_shape: Shape of the frame (height, width)
    
    Returns:
        List of zone coordinates [x1, y1, x2, y2] or None for full frame
    """
    st.markdown("### 🎯 Detection Zone")
    
    use_zone = st.checkbox("Use custom detection zone", value=False)
    
    if not use_zone:
        return None
    
    height, width = frame_shape[:2]
    
    col1, col2 = st.columns(2)
    
    with col1:
        x1 = st.slider("X1 (Left)", 0, width, 0)
        y1 = st.slider("Y1 (Top)", 0, height, 0)
    
    with col2:
        x2 = st.slider("X2 (Right)", 0, width, width)
        y2 = st.slider("Y2 (Bottom)", 0, height, height)
    
    return [x1, y1, x2, y2]

