# 🏗️ Detectify Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Detectify v2.0                          │
│                  Advanced Object Detection System                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          User Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  Web Browser  →  Streamlit UI  →  http://localhost:8501        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  src/ui/                                                         │
│  ├── app.py              ← Main Streamlit application           │
│  └── components.py       ← Reusable UI components               │
│      ├── sidebar()                                               │
│      ├── analytics_dashboard()                                   │
│      ├── video_upload_section()                                  │
│      └── image_upload_section()                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  src/core/                                                       │
│  ├── detector.py         ← ObjectDetector class                 │
│  │   ├── detect()        → Run YOLO inference                   │
│  │   ├── detect_batch()  → Process multiple frames              │
│  │   └── change_model()  → Switch YOLO models                   │
│  ├── camera.py           ← Stream handlers                      │
│  │   ├── CameraStream    → Webcam handling                      │
│  │   └── DroidCamStream  → IP camera handling                   │
│  └── tracker.py          ← ObjectTracker class                  │
│      └── update()        → Track objects across frames          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Service Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  src/utils/                                                      │
│  ├── video_processor.py  ← VideoProcessor                       │
│  ├── image_processor.py  ← ImageProcessor                       │
│  ├── metrics.py          ← MetricsCalculator                    │
│  └── logger.py           ← Logging system                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  src/database/                                                   │
│  └── models.py           ← DetectionLog class                   │
│      ├── log_detection() → Save detection to DB                 │
│      ├── get_stats()     → Retrieve statistics                  │
│      └── export_to_csv() → Export data                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Configuration Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  src/config/                                                     │
│  └── settings.py         ← Settings class                       │
│      └── Load from .env  → Environment variables                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Storage Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  ├── data/detections.db     ← SQLite database                   │
│  ├── exports/*.mp4          ← Processed videos                  │
│  ├── snapshots/*.jpg        ← Saved snapshots                   │
│  └── logs/*.log             ← Application logs                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      External Dependencies                       │
├─────────────────────────────────────────────────────────────────┤
│  ├── PyTorch              ← Deep learning framework             │
│  ├── Ultralytics YOLO     ← Object detection models             │
│  ├── OpenCV               ← Computer vision                     │
│  └── Streamlit            ← Web framework                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### 1. Webcam Detection Flow

```
┌──────────┐
│  User    │
│  Opens   │
│  Browser │
└────┬─────┘
     │
     ↓
┌────────────────┐
│  Streamlit UI  │
│  Loads         │
└────┬───────────┘
     │
     ↓
┌────────────────────┐
│  Initialize        │
│  - Detector        │
│  - Tracker         │
│  - Metrics         │
└────┬───────────────┘
     │
     ↓
┌────────────────────┐
│  CameraStream      │
│  Opens webcam      │
└────┬───────────────┘
     │
     ↓  Frame Loop
┌────────────────────┐
│  Capture Frame     │ ←──────┐
└────┬───────────────┘        │
     │                        │
     ↓                        │
┌────────────────────┐        │
│  ObjectDetector    │        │
│  Runs YOLO         │        │
└────┬───────────────┘        │
     │                        │
     ↓                        │
┌────────────────────┐        │
│  ObjectTracker     │        │
│  Updates tracks    │        │
└────┬───────────────┘        │
     │                        │
     ↓                        │
┌────────────────────┐        │
│  Draw Boxes        │        │
│  Add Labels        │        │
└────┬───────────────┘        │
     │                        │
     ↓                        │
┌────────────────────┐        │
│  Log to Database   │        │
│  (Optional)        │        │
└────┬───────────────┘        │
     │                        │
     ↓                        │
┌────────────────────┐        │
│  Update Metrics    │        │
│  Display UI        │        │
└────┬───────────────┘        │
     │                        │
     └────────────────────────┘
```

### 2. Video Processing Flow

```
┌──────────────┐
│  User        │
│  Uploads     │
│  Video       │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│  Save to         │
│  Temp File       │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  VideoProcessor  │
│  Loads video     │
└──────┬───────────┘
       │
       ↓  For each frame
┌──────────────────┐
│  Extract Frame   │ ←──────┐
└──────┬───────────┘        │
       │                    │
       ↓                    │
┌──────────────────┐        │
│  ObjectDetector  │        │
│  Detect objects  │        │
└──────┬───────────┘        │
       │                    │
       ↓                    │
┌──────────────────┐        │
│  Draw boxes      │        │
│  on frame        │        │
└──────┬───────────┘        │
       │                    │
       ↓                    │
┌──────────────────┐        │
│  Write to        │        │
│  Output video    │        │
└──────┬───────────┘        │
       │                    │
       ↓                    │
┌──────────────────┐        │
│  Update          │        │
│  Progress bar    │────────┘
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Save output     │
│  Provide         │
│  Download link   │
└──────────────────┘
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       app.py (Main)                          │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ↓                    ↓                    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  components  │    │   detector   │    │    camera    │
│              │    │              │    │              │
│ - sidebar    │    │ - detect()   │    │ - read()     │
│ - analytics  │    │ - track()    │    │ - release()  │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         │                    ↓                    │
         │           ┌──────────────┐              │
         │           │   tracker    │              │
         │           │              │              │
         │           │ - update()   │              │
         │           └──────────────┘              │
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ↓                    ↓                    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   database   │    │    utils     │    │   config     │
│              │    │              │    │              │
│ - log_det()  │    │ - logger     │    │ - settings   │
│ - get_stats()│    │ - metrics    │    │ - .env       │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## Class Hierarchy

```
ObjectDetector
├── __init__(model_name, conf_threshold, iou_threshold, device)
├── detect(frame) → DetectionResult
├── detect_batch(frames) → List[DetectionResult]
├── change_model(model_name)
└── info → Dict

DetectionResult
├── frame: np.ndarray
├── boxes: List[List[float]]
├── classes: List[str]
├── confidences: List[float]
├── processing_time: float
├── model_name: str
├── object_counts → Dict[str, int]
├── total_objects → int
└── to_dict() → Dict

CameraStream
├── __init__(camera_index, width, height, fps)
├── read() → Tuple[bool, np.ndarray]
├── get_fps() → float
├── release()
└── _update_fps()

DroidCamStream
├── __init__(ip_address, port)
├── start() → self
├── read() → Tuple[bool, np.ndarray]
├── stop()
├── get_fps() → float
└── test_connection(ip_address) → bool

ObjectTracker
├── __init__(max_age, min_hits, iou_threshold)
├── update(boxes, classes, confidences) → List[Track]
├── reset()
└── track_count → int

Track
├── track_id: int
├── bbox: List[float]
├── class_name: str
├── confidence: float
├── update(bbox, confidence)
├── predict()
└── center → Tuple[float, float]

DetectionLog
├── __init__(db_path)
├── log_detection(source, model, objects, ...)
├── create_session(source_type, model_name, session_id)
├── end_session(session_id, ...)
├── get_recent_detections(limit) → List[Dict]
├── get_detection_stats(session_id) → Dict
├── export_to_csv(output_path, session_id)
└── clear_old_detections(days)
```

---

## Module Dependencies

```
app.py
 └── src/ui/app.py
      ├── src/config/settings.py
      │    └── .env
      ├── src/core/detector.py
      │    ├── ultralytics.YOLO
      │    └── torch
      ├── src/core/camera.py
      │    ├── cv2
      │    └── requests
      ├── src/core/tracker.py
      │    └── scipy
      ├── src/database/models.py
      │    ├── sqlite3
      │    └── pandas
      ├── src/utils/logger.py
      │    └── logging
      ├── src/utils/metrics.py
      │    └── numpy
      ├── src/utils/video_processor.py
      │    └── cv2
      ├── src/utils/image_processor.py
      │    └── cv2, PIL
      └── src/ui/components.py
           ├── streamlit
           └── plotly
```

---

## Design Patterns Used

### 1. **Singleton Pattern**
- **Settings class**: Single global configuration instance

### 2. **Factory Pattern**
- **Camera initialization**: Creates appropriate camera type

### 3. **Strategy Pattern**
- **Detection algorithms**: Interchangeable YOLO models

### 4. **Observer Pattern**
- **Metrics updates**: Notifies UI of changes

### 5. **Repository Pattern**
- **DetectionLog**: Abstracts database operations

### 6. **Builder Pattern**
- **DetectionResult**: Builds complex result objects

---

## Threading & Concurrency

```
Main Thread (Streamlit)
│
├── UI Rendering
├── User Input Handling
└── Frame Display

DroidCam Thread (daemon)
│
└── Continuous frame capture
    ├── HTTP streaming
    ├── JPEG parsing
    └── Queue updates
```

---

## Error Handling Strategy

```
┌─────────────────┐
│   User Action   │
└────────┬────────┘
         │
         ↓
    ┌────────┐
    │  Try   │
    └───┬────┘
        │
        ├──→ Success → Continue
        │
        └──→ Exception
             ├──→ Log error (logger)
             ├──→ Show user message (Streamlit)
             └──→ Cleanup resources (finally)
```

---

## Performance Optimization

### 1. **Frame Buffering**
- Minimal buffer size (1) for low latency
- Queue-based frame management for IP cameras

### 2. **GPU Acceleration**
- Automatic CUDA detection
- Model loaded on GPU when available

### 3. **Async Operations**
- Threaded frame capture for IP cameras
- Non-blocking UI updates

### 4. **Resource Management**
- Context managers for cleanup
- Explicit resource release
- Garbage collection friendly

---

## Security Considerations

1. **Input Validation**
   - File size limits
   - Format verification
   - IP address validation

2. **Resource Limits**
   - Max video size
   - Max image size
   - Max detections per frame

3. **Error Handling**
   - No sensitive data in errors
   - Graceful degradation
   - Proper logging

---

## Scalability

### Current Architecture Supports
- ✅ Horizontal scaling (multiple instances)
- ✅ Stateless design (except session state)
- ✅ Database-backed persistence
- ✅ Containerization (Docker)

### Future Enhancements
- Load balancing
- Distributed processing
- Cloud storage
- Microservices architecture

---

## Deployment Architecture

```
┌────────────────────────────────────────┐
│           Load Balancer                │
└────────────────┬───────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ↓           ↓           ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Docker  │ │ Docker  │ │ Docker  │
│ Detectify│ │ Detectify│ │ Detectify│
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └───────────┼───────────┘
                 │
                 ↓
        ┌────────────────┐
        │  Shared Storage│
        │  (Database)    │
        └────────────────┘
```

---

This architecture provides:
- ✅ **Modularity**: Easy to extend and modify
- ✅ **Maintainability**: Clear structure and separation
- ✅ **Scalability**: Can grow with requirements
- ✅ **Testability**: Each component can be tested independently
- ✅ **Performance**: Optimized for speed and efficiency

