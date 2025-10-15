"""Database models and operations for detection logging."""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from contextlib import contextmanager
import json

from ..config.settings import settings


class DetectionLog:
    """Manages detection logging to SQLite database."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize detection log with database path."""
        self.db_path = db_path or settings.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize database and create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create detections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL,
                    model TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    total_objects INTEGER DEFAULT 0,
                    fps REAL DEFAULT 0,
                    processing_time REAL DEFAULT 0,
                    session_id TEXT
                )
            """)
            
            # Create detected_objects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detected_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    FOREIGN KEY (detection_id) REFERENCES detections (id)
                )
            """)
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    source_type TEXT,
                    model_name TEXT,
                    total_frames INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    avg_fps REAL DEFAULT 0
                )
            """)
            
            # Create analytics table for object counts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS object_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    total_count INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def log_detection(
        self,
        source: str,
        model: str,
        confidence_threshold: float,
        objects: List[Dict[str, Any]],
        fps: float = 0.0,
        processing_time: float = 0.0,
        session_id: Optional[str] = None
    ) -> int:
        """
        Log a detection event.
        
        Args:
            source: Source of the detection (webcam, ip_camera, video, image)
            model: Model name used for detection
            confidence_threshold: Confidence threshold used
            objects: List of detected objects with class, confidence, and bbox
            fps: Frames per second
            processing_time: Time taken to process the frame
            session_id: Session identifier
        
        Returns:
            Detection ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert detection record
            cursor.execute("""
                INSERT INTO detections 
                (source, model, confidence_threshold, total_objects, fps, processing_time, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source, model, confidence_threshold, len(objects), fps, processing_time, session_id))
            
            detection_id = cursor.lastrowid
            
            # Insert detected objects
            for obj in objects:
                cursor.execute("""
                    INSERT INTO detected_objects 
                    (detection_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection_id,
                    obj['class'],
                    obj['confidence'],
                    obj['bbox'][0],
                    obj['bbox'][1],
                    obj['bbox'][2],
                    obj['bbox'][3]
                ))
            
            conn.commit()
            return detection_id
    
    def create_session(self, source_type: str, model_name: str, session_id: str):
        """Create a new detection session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, source_type, model_name)
                VALUES (?, ?, ?)
            """, (session_id, source_type, model_name))
            conn.commit()
    
    def end_session(self, session_id: str, total_frames: int, total_detections: int, avg_fps: float):
        """End a detection session and update statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET end_time = CURRENT_TIMESTAMP,
                    total_frames = ?,
                    total_detections = ?,
                    avg_fps = ?
                WHERE id = ?
            """, (total_frames, total_detections, avg_fps, session_id))
            conn.commit()
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """Get recent detections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_detection_stats(self, session_id: Optional[str] = None) -> Dict:
        """Get detection statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if session_id:
                where_clause = "WHERE session_id = ?"
                params = (session_id,)
            else:
                where_clause = ""
                params = ()
            
            # Get total detections
            cursor.execute(f"SELECT COUNT(*) as count FROM detections {where_clause}", params)
            total_detections = cursor.fetchone()['count']
            
            # Get object class distribution
            cursor.execute(f"""
                SELECT do.class_name, COUNT(*) as count, AVG(do.confidence) as avg_conf
                FROM detected_objects do
                JOIN detections d ON do.detection_id = d.id
                {where_clause}
                GROUP BY do.class_name
                ORDER BY count DESC
            """, params)
            class_distribution = [dict(row) for row in cursor.fetchall()]
            
            # Get average FPS
            cursor.execute(f"SELECT AVG(fps) as avg_fps FROM detections {where_clause}", params)
            avg_fps = cursor.fetchone()['avg_fps'] or 0
            
            return {
                'total_detections': total_detections,
                'class_distribution': class_distribution,
                'avg_fps': avg_fps
            }
    
    def get_sessions(self, limit: int = 50) -> List[Dict]:
        """Get recent sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def export_to_csv(self, output_path: Path, session_id: Optional[str] = None):
        """Export detections to CSV."""
        import pandas as pd
        
        with self._get_connection() as conn:
            if session_id:
                query = """
                    SELECT d.*, do.class_name, do.confidence, 
                           do.bbox_x1, do.bbox_y1, do.bbox_x2, do.bbox_y2
                    FROM detections d
                    LEFT JOIN detected_objects do ON d.id = do.detection_id
                    WHERE d.session_id = ?
                    ORDER BY d.timestamp DESC
                """
                df = pd.read_sql_query(query, conn, params=(session_id,))
            else:
                query = """
                    SELECT d.*, do.class_name, do.confidence, 
                           do.bbox_x1, do.bbox_y1, do.bbox_x2, do.bbox_y2
                    FROM detections d
                    LEFT JOIN detected_objects do ON d.id = do.detection_id
                    ORDER BY d.timestamp DESC
                """
                df = pd.read_sql_query(query, conn)
            
            df.to_csv(output_path, index=False)
            return output_path
    
    def clear_old_detections(self, days: int = 30):
        """Clear detections older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM detections 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.commit()
            return cursor.rowcount


def init_db() -> DetectionLog:
    """Initialize and return a DetectionLog instance."""
    return DetectionLog()

