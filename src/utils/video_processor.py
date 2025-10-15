"""Video processing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Generator
import tempfile

from ..config.settings import settings
from .logger import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Handles video file processing."""
    
    def __init__(self, video_path: Path):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {video_path.name} ({self.width}x{self.height}, {self.fps} FPS, {self.duration:.2f}s)")
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from the video.
        
        Yields:
            Video frames
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.
        
        Args:
            frame_number: Frame number (0-indexed)
        
        Returns:
            Frame or None if out of bounds
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def process_video(
        self,
        output_path: Path,
        frame_processor: Callable[[np.ndarray], np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Process video and save to output file.
        
        Args:
            output_path: Path to save processed video
            frame_processor: Function that processes each frame
            progress_callback: Optional callback for progress updates (current, total)
        """
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        try:
            frame_num = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = frame_processor(frame)
                
                # Write frame
                out.write(processed_frame)
                
                frame_num += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(frame_num, self.frame_count)
            
            logger.info(f"Video processing complete: {output_path}")
            
        finally:
            out.release()
    
    def extract_frames(
        self,
        output_dir: Path,
        interval: int = 1,
        max_frames: Optional[int] = None
    ) -> int:
        """
        Extract frames from video.
        
        Args:
            output_dir: Directory to save frames
            interval: Extract every N frames
            max_frames: Maximum number of frames to extract
        
        Returns:
            Number of frames extracted
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_num = 0
        extracted = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_num % interval == 0:
                output_path = output_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                extracted += 1
                
                if max_frames and extracted >= max_frames:
                    break
            
            frame_num += 1
        
        logger.info(f"Extracted {extracted} frames to {output_dir}")
        return extracted
    
    def release(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
    
    def __del__(self):
        """Cleanup."""
        self.release()
    
    @property
    def info(self) -> dict:
        """Get video information."""
        return {
            'path': str(self.video_path),
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration
        }
    
    @staticmethod
    def is_valid_video(file_path: Path) -> bool:
        """
        Check if file is a valid video.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if valid video, False otherwise
        """
        if not file_path.exists():
            return False
        
        suffix = file_path.suffix.lower().lstrip('.')
        if suffix not in settings.SUPPORTED_VIDEO_FORMATS:
            return False
        
        try:
            cap = cv2.VideoCapture(str(file_path))
            is_valid = cap.isOpened()
            cap.release()
            return is_valid
        except:
            return False

