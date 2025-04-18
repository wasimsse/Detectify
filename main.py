import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pandas as pd
from utils import draw_boxes, count_objects, save_detection_log, save_snapshot, calculate_fps
import requests
from threading import Thread
import queue
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
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
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=1)
if 'ip_camera_thread' not in st.session_state:
    st.session_state.ip_camera_thread = None

class DroidCam:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.base_url = f"http://{ip_address}:4747"
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.session = requests.Session()
        self.fps = 0
        self.frame_count = 0
        self.last_frame_time = time.time()

    def start(self):
        """Start the camera thread"""
        self.running = True
        self.thread = Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        return self

    def _capture_frames(self):
        """Continuously capture frames using HTTP streaming"""
        while self.running:
            try:
                # Try MJPEG stream first
                response = self.session.get(f"{self.base_url}/mjpegfeed", stream=True, timeout=5)
                if response.status_code == 200:
                    bytes_data = bytes()
                    for chunk in response.iter_content(chunk_size=1024):
                        if not self.running:
                            break
                        if chunk:
                            bytes_data += chunk
                            a = bytes_data.find(b'\xff\xd8')
                            b = bytes_data.find(b'\xff\xd9')
                            if a != -1 and b != -1:
                                jpg = bytes_data[a:b+2]
                                bytes_data = bytes_data[b+2:]
                                try:
                                    # Convert to PIL Image first
                                    img = Image.open(io.BytesIO(jpg))
                                    # Convert to numpy array
                                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                    
                                    # Update FPS
                                    current_time = time.time()
                                    self.frame_count += 1
                                    if current_time - self.last_frame_time >= 1.0:
                                        self.fps = self.frame_count
                                        self.frame_count = 0
                                        self.last_frame_time = current_time

                                    # Update frame queue
                                    if not self.frame_queue.empty():
                                        try:
                                            self.frame_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                    self.frame_queue.put((frame, self.fps))
                                except Exception as e:
                                    st.error(f"Error processing frame: {str(e)}")
                                    continue
                else:
                    st.error("Failed to connect to MJPEG stream")
                    time.sleep(1)
                    
            except Exception as e:
                st.error(f"Error in stream: {str(e)}")
                time.sleep(1)

    def read(self):
        """Read the most recent frame"""
        if not self.running:
            return False, None, 0
            
        try:
            frame, fps = self.frame_queue.get_nowait()
            return True, frame, fps
        except queue.Empty:
            return False, None, 0

    def stop(self):
        """Stop the camera thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.session.close()

def test_ip_camera(ip_address):
    """Test if IP camera is accessible"""
    url = f"http://{ip_address}:4747/mjpegfeed"
    try:
        response = requests.get(url, timeout=3, stream=True)
        return response.status_code == 200
    except:
        return False

def initialize_camera(source):
    """Initialize a camera with proper settings"""
    if isinstance(source, str):  # IP address for DroidCam
        # Remove any protocol prefix and port
        ip_address = source.replace("http://", "").replace("https://", "").split(":")[0]
        return DroidCam(ip_address).start()
    else:
        # Regular OpenCV VideoCapture for laptop camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

def process_frame(frame, model, conf_threshold):
    """Process a single frame"""
    if frame is None or frame.size == 0:
        return None, {}
    
    # Run detection
    results = model(frame, stream=True)
    
    # Draw boxes and count objects
    annotated_frame = frame.copy()
    object_counts = {}
    
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
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f'{class_name} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Update counts
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
    
    return annotated_frame, object_counts

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
    st.sidebar.markdown("""
    ### IP Camera Setup
    1. Install DroidCam on your phone
    2. Connect phone to the same WiFi as your computer
    3. Enter the IP address shown in DroidCam app
    """)
    
    ip_address = st.sidebar.text_input(
        "IP Address (e.g., 192.168.1.100)",
        ""
    )
    
    ip_camera_url = f"http://{ip_address}:4747"
    
    if ip_camera_url:
        st.sidebar.info(f"Using camera URL: {ip_camera_url}")

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Main content
st.title("Real-Time Object Detection")

# Create columns for video display
if camera_source == "Both":
    col1, col2 = st.columns(2)
else:
    col1 = st.container()

# Initialize cameras and create placeholders
laptop_cap = None
ip_cap = None

# Add a button to stop/restart the cameras
if st.sidebar.button("Reset Cameras"):
    st.session_state.camera_active = False
    if 'laptop_cap' in locals() and laptop_cap is not None:
        if isinstance(laptop_cap, DroidCam):
            laptop_cap.stop()
        else:
            laptop_cap.release()
    if 'ip_cap' in locals() and ip_cap is not None:
        if isinstance(ip_cap, DroidCam):
            ip_cap.stop()
        else:
            ip_cap.release()
    st.rerun()

try:
    # Initialize laptop webcam if selected
    if camera_source in ["Laptop Webcam", "Both"]:
        with col1:
            st.subheader("Laptop Webcam")
            laptop_placeholder = st.empty()
            laptop_stats = st.empty()
            laptop_cap = initialize_camera(0)
            if laptop_cap is None:
                st.error("Could not open laptop webcam. Please check if it's connected properly.")

    # Initialize IP camera if selected
    if camera_source in ["IP Camera", "Both"]:
        with col2 if camera_source == "Both" else col1:
            st.subheader("IP Camera")
            ip_placeholder = st.empty()
            ip_stats = st.empty()
            
            # Test connection with multiple retries
            connection_successful = False
            for attempt in range(3):  # Try 3 times
                if test_ip_camera(ip_address):
                    connection_successful = True
                    break
                time.sleep(1)  # Wait 1 second between attempts
                
            if connection_successful:
                ip_cap = initialize_camera(ip_address)
                if isinstance(ip_cap, DroidCam):
                    st.success(f"Successfully connected to DroidCam at {ip_address}")
                    st.info("If you don't see the video stream, try accessing http://" + ip_address + ":4747/mjpegfeed in your browser to verify the stream is working")
                else:
                    st.error("Failed to initialize IP camera stream")
            else:
                st.error(f"Could not connect to DroidCam at {ip_address}. Please check:\n" +
                        "1. DroidCam app is running on your phone\n" +
                        "2. Phone and computer are on the same WiFi network (IP: {ip_address})\n" +
                        "3. Try opening http://" + ip_address + ":4747 in your browser\n" +
                        "4. No firewall is blocking port 4747\n" +
                        "5. Try restarting the DroidCam app")

    # Start real-time detection
    st.session_state.camera_active = True
    
    while st.session_state.camera_active:
        try:
            # Process laptop webcam
            if laptop_cap is not None:
                if isinstance(laptop_cap, DroidCam):
                    ret, frame, fps = laptop_cap.read()
                else:
                    ret, frame = laptop_cap.read()
                    fps = laptop_cap.get(cv2.CAP_PROP_FPS)
                    
                if ret:
                    st.sidebar.success("Laptop camera frame captured successfully")
                    annotated_frame, object_counts = process_frame(frame, st.session_state.model, conf_threshold)
                    if annotated_frame is not None:
                        # Add FPS and frame info to frame
                        cv2.putText(annotated_frame, f"FPS: {fps if fps > 0 else 'N/A'}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Frame: {frame.shape}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        laptop_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                        laptop_stats.write("Objects detected:")
                        for obj, count in object_counts.items():
                            laptop_stats.write(f"- {obj}: {count}")
                    else:
                        st.sidebar.warning("Failed to process laptop camera frame")
                else:
                    st.sidebar.error("Failed to capture laptop camera frame")

            # Process IP camera
            if ip_cap is not None:
                if isinstance(ip_cap, DroidCam):
                    ret, frame, fps = ip_cap.read()
                else:
                    ret, frame = ip_cap.read()
                    fps = ip_cap.get(cv2.CAP_PROP_FPS)
                    
                if ret:
                    st.sidebar.success("IP camera frame captured successfully")
                    annotated_frame, object_counts = process_frame(frame, st.session_state.model, conf_threshold)
                    if annotated_frame is not None:
                        # Add FPS and frame info to frame
                        cv2.putText(annotated_frame, f"FPS: {fps if fps > 0 else 'N/A'}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Frame: {frame.shape}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        ip_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                        ip_stats.write("Objects detected:")
                        for obj, count in object_counts.items():
                            ip_stats.write(f"- {obj}: {count}")
                    else:
                        st.sidebar.warning("Failed to process IP camera frame")
                else:
                    st.sidebar.error("Failed to capture IP camera frame")

            # Small delay to prevent overwhelming the CPU
            time.sleep(0.01)

        except Exception as e:
            st.sidebar.error(f"Error in main loop: {str(e)}")
            import traceback
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            time.sleep(1)  # Wait a bit before retrying
            continue

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.error(f"Traceback: {traceback.format_exc()}")
    if laptop_cap is not None:
        if isinstance(laptop_cap, DroidCam):
            laptop_cap.stop()
        else:
            laptop_cap.release()
    if ip_cap is not None:
        if isinstance(ip_cap, DroidCam):
            ip_cap.stop()
        else:
            ip_cap.release()
    st.session_state.camera_active = False