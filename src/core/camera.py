"""Camera stream handlers for webcam and IP cameras."""

import cv2
import numpy as np
import time
import queue
import requests
from threading import Thread
from typing import Optional, Tuple
from PIL import Image
import io

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CameraStream:
    """Standard webcam stream handler."""
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = None,
        height: int = None,
        fps: int = None
    ):
        """
        Initialize camera stream.
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.camera_index = camera_index
        self.width = width or settings.CAMERA_WIDTH
        self.height = height or settings.CAMERA_HEIGHT
        self.target_fps = fps or settings.CAMERA_FPS
        
        self.cap = None
        self.is_running = False
        self.current_fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.FRAME_BUFFER_SIZE)
            
            self.is_running = True
            logger.info(f"Camera {self.camera_index} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            raise
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self._update_fps()
        
        return ret, frame
    
    def _update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def release(self):
        """Release camera resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Camera {self.camera_index} released")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()


class DroidCamStream:
    """IP camera stream handler for DroidCam and similar apps."""
    
    def __init__(self, ip_address: str, port: int = 4747):
        """
        Initialize DroidCam stream.
        
        Args:
            ip_address: IP address of the camera
            port: Port number (default 4747 for DroidCam)
        """
        self.ip_address = ip_address.replace("http://", "").replace("https://", "")
        self.port = port
        self.base_url = f"http://{self.ip_address}:{self.port}"
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.thread = None
        self.session = requests.Session()
        self.current_fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        logger.info(f"DroidCam stream initialized for {self.base_url}")
    
    def start(self):
        """Start the camera thread."""
        self.is_running = True
        self.thread = Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        logger.info("DroidCam stream started")
        return self
    
    def _capture_frames(self):
        """Continuously capture frames using HTTP streaming."""
        while self.is_running:
            try:
                # Connect to MJPEG stream
                response = self.session.get(
                    f"{self.base_url}/mjpegfeed",
                    stream=True,
                    timeout=5
                )
                
                if response.status_code == 200:
                    bytes_data = bytes()
                    
                    for chunk in response.iter_content(chunk_size=1024):
                        if not self.is_running:
                            break
                        
                        if chunk:
                            bytes_data += chunk
                            
                            # Find JPEG boundaries
                            a = bytes_data.find(b'\xff\xd8')  # JPEG start
                            b = bytes_data.find(b'\xff\xd9')  # JPEG end
                            
                            if a != -1 and b != -1:
                                jpg = bytes_data[a:b+2]
                                bytes_data = bytes_data[b+2:]
                                
                                try:
                                    # Decode image
                                    img = Image.open(io.BytesIO(jpg))
                                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                    
                                    # Update FPS
                                    self._update_fps()
                                    
                                    # Update queue
                                    if self.frame_queue.full():
                                        try:
                                            self.frame_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                    
                                    self.frame_queue.put(frame)
                                    
                                except Exception as e:
                                    logger.error(f"Error decoding frame: {e}")
                                    continue
                else:
                    logger.error(f"Failed to connect to stream: {response.status_code}")
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in camera stream: {e}")
                time.sleep(1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the most recent frame.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running:
            return False, None
        
        try:
            frame = self.frame_queue.get(timeout=1)
            return True, frame
        except queue.Empty:
            return False, None
    
    def _update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def stop(self):
        """Stop the camera thread."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.session.close()
        logger.info("DroidCam stream stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
    
    @staticmethod
    def test_connection(ip_address: str, port: int = 4747) -> bool:
        """
        Test if IP camera is accessible.
        
        Args:
            ip_address: IP address to test
            port: Port number
        
        Returns:
            True if accessible, False otherwise
        """
        ip_address = ip_address.replace("http://", "").replace("https://", "")
        url = f"http://{ip_address}:{port}/mjpegfeed"
        
        try:
            response = requests.get(url, timeout=3, stream=True)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

