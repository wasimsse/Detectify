"""Image processing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

from ..config.settings import settings
from .logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def load_image(image_path: Path) -> Optional[np.ndarray]:
        """
        Load an image from file.
        
        Args:
            image_path: Path to image
        
        Returns:
            Image as numpy array (BGR) or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Path) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image as numpy array (BGR)
            output_path: Path to save image
        
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            logger.info(f"Image saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image.
        
        Args:
            image: Input image
            width: Target width (None to auto-calculate)
            height: Target height (None to auto-calculate)
            maintain_aspect: Whether to maintain aspect ratio
        
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if maintain_aspect:
            if width is not None:
                aspect = width / w
                height = int(h * aspect)
            elif height is not None:
                aspect = height / h
                width = int(w * aspect)
        else:
            if width is None:
                width = w
            if height is None:
                height = h
        
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def crop_image(
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> np.ndarray:
        """
        Crop image to bounding box.
        
        Args:
            image: Input image
            x1, y1, x2, y2: Bounding box coordinates
        
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur to image."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (1.0 = no change, <1 = darker, >1 = brighter)
        
        Returns:
            Adjusted image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def draw_text(
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        background: bool = True
    ) -> np.ndarray:
        """
        Draw text on image with optional background.
        
        Args:
            image: Input image
            text: Text to draw
            position: (x, y) position
            font_scale: Font scale
            color: Text color in BGR
            thickness: Text thickness
            background: Whether to draw background rectangle
        
        Returns:
            Image with text
        """
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background
        if background:
            cv2.rectangle(
                result,
                (x, y - text_h - baseline - 5),
                (x + text_w, y + baseline),
                (0, 0, 0),
                -1
            )
        
        # Draw text
        cv2.putText(
            result,
            text,
            (x, y - baseline),
            font,
            font_scale,
            color,
            thickness
        )
        
        return result
    
    @staticmethod
    def is_valid_image(file_path: Path) -> bool:
        """
        Check if file is a valid image.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if valid image, False otherwise
        """
        if not file_path.exists():
            return False
        
        suffix = file_path.suffix.lower().lstrip('.')
        if suffix not in settings.SUPPORTED_IMAGE_FORMATS:
            return False
        
        try:
            img = Image.open(file_path)
            img.verify()
            return True
        except:
            return False

