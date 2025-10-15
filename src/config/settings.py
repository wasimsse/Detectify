"""Application settings and configuration management."""

import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "Detectify")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    SRC_DIR: Path = BASE_DIR / "src"
    DATA_DIR: Path = BASE_DIR / "data"
    EXPORT_DIR: Path = BASE_DIR / os.getenv("EXPORT_DIR", "exports")
    SNAPSHOT_DIR: Path = BASE_DIR / os.getenv("SNAPSHOT_DIR", "snapshots")
    LOG_DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")
    
    # Model Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", "300"))
    
    # Available YOLO models
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
    ])
    
    # Performance
    ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "True").lower() == "true"
    FRAME_BUFFER_SIZE: int = int(os.getenv("FRAME_BUFFER_SIZE", "1"))
    MAX_FPS: int = int(os.getenv("MAX_FPS", "30"))
    
    # Database
    DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "data/detections.db")
    ENABLE_DB_LOGGING: bool = os.getenv("ENABLE_DB_LOGGING", "True").lower() == "true"
    
    # Camera
    DEFAULT_CAMERA_INDEX: int = int(os.getenv("DEFAULT_CAMERA_INDEX", "0"))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "480"))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    
    # Video Processing
    MAX_VIDEO_SIZE_MB: int = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
    SUPPORTED_VIDEO_FORMATS: List[str] = field(default_factory=lambda: 
        os.getenv("SUPPORTED_VIDEO_FORMATS", "mp4,avi,mov,mkv").split(","))
    
    # Image Processing
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    SUPPORTED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: 
        os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,webp").split(","))
    
    # UI Settings
    PAGE_TITLE: str = os.getenv("PAGE_TITLE", "Detectify - Advanced Object Detection")
    PAGE_ICON: str = os.getenv("PAGE_ICON", "🎥")
    THEME: str = os.getenv("THEME", "dark")
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def device(self) -> str:
        """Get the device for model inference."""
        if self.ENABLE_GPU:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"


# Global settings instance
settings = Settings()

