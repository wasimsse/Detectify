# 🎥 Detectify v2.0 - Advanced Object Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io)
[![YOLOv8+](https://img.shields.io/badge/YOLO-v8%2B-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, feature-rich real-time object detection application built with **YOLOv8+**, **Streamlit**, and **PyTorch**. Supports multiple input sources including webcam, IP cameras (DroidCam), video files, and images with advanced features like object tracking, analytics dashboard, and detection logging.

![Detectify Banner](https://via.placeholder.com/800x200/1E1E1E/FFFFFF?text=Detectify+v2.0)

## ✨ Features

### 🎯 Core Detection
- **Multiple YOLO Models**: Support for YOLOv8 (n/s/m/l/x), YOLOv10, and YOLOv11
- **Real-time Detection**: High-performance object detection with GPU acceleration
- **Multi-source Support**: Webcam, IP cameras, video files, and images
- **Adjustable Parameters**: Confidence threshold, IOU threshold, and more

### 🚀 Advanced Features
- **Object Tracking**: Track objects across frames using advanced algorithms
- **Analytics Dashboard**: Real-time metrics, FPS counter, and object distribution charts
- **Database Logging**: SQLite database for detection history and session management
- **Export Capabilities**: Save snapshots, export videos, and generate CSV reports
- **Modern UI**: Clean, intuitive Streamlit interface with dark/light themes

### 📊 Analytics & Monitoring
- Real-time FPS and processing time metrics
- Object count distribution with interactive charts
- Detection history and session tracking
- Performance statistics and averages

### 🛠️ Technical Highlights
- **Modular Architecture**: Clean separation of concerns (config, core, UI, utils)
- **Type Hints**: Full type annotation for better code quality
- **Logging System**: Comprehensive logging with file and console output
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Configuration Management**: Environment-based settings with `.env` support
- **Error Handling**: Robust error handling and recovery

## 📋 Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Optional**: NVIDIA GPU with CUDA for accelerated inference
- **Memory**: Minimum 4GB RAM (8GB+ recommended)

## 🚀 Quick Start

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/wasimsse/Detectify.git
cd Detectify
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create environment file**
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open in browser**
Navigate to `http://localhost:8501`

### Option 2: Docker

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **Access the application**
Navigate to `http://localhost:8501`

3. **Stop the application**
```bash
docker-compose down
```

## 📖 Usage Guide

### 1. Webcam Detection
- Select "Webcam" from the sidebar
- Adjust confidence threshold and other parameters
- Click start to begin real-time detection
- Objects will be detected and tracked automatically

### 2. IP Camera (DroidCam)
- Install DroidCam app on your phone
- Connect phone and computer to the same WiFi
- Select "IP Camera (DroidCam)" from the sidebar
- Enter the IP address shown in the DroidCam app
- Start detection

### 3. Video File Detection
- Select "Upload Video" from the sidebar
- Upload a video file (mp4, avi, mov, mkv)
- Click "Process Video" to analyze
- Download the processed video with detections

### 4. Image Detection
- Select "Upload Image" from the sidebar
- Upload an image file (jpg, png, webp)
- View detected objects instantly
- Save or export results

## 🎛️ Configuration

### Environment Variables
Edit `.env` file to customize settings:

```bash
# Model Settings
DEFAULT_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Performance
ENABLE_GPU=True
MAX_FPS=30

# Database
ENABLE_DB_LOGGING=True

# Camera Settings
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
```

### Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | Nano | ⚡⚡⚡ | ⭐⭐⭐ | Real-time, edge devices |
| YOLOv8s | Small | ⚡⚡ | ⭐⭐⭐⭐ | Balanced performance |
| YOLOv8m | Medium | ⚡ | ⭐⭐⭐⭐ | Higher accuracy |
| YOLOv8l | Large | 🐢 | ⭐⭐⭐⭐⭐ | Maximum accuracy |
| YOLOv8x | Extra Large | 🐢🐢 | ⭐⭐⭐⭐⭐ | Research, offline |
| YOLOv10/v11 | Various | Varies | Enhanced | Latest models |

## 📁 Project Structure

```
Detectify/
├── app.py                      # Main entry point
├── src/
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # Settings and environment variables
│   ├── core/                   # Core detection engine
│   │   ├── __init__.py
│   │   ├── detector.py         # YOLO detection engine
│   │   ├── camera.py           # Camera stream handlers
│   │   └── tracker.py          # Object tracking
│   ├── database/               # Database management
│   │   ├── __init__.py
│   │   └── models.py           # SQLite models and operations
│   ├── ui/                     # Streamlit UI
│   │   ├── __init__.py
│   │   ├── app.py              # Main UI application
│   │   └── components.py       # UI components
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logger.py           # Logging configuration
│       ├── video_processor.py  # Video processing
│       ├── image_processor.py  # Image processing
│       └── metrics.py          # Performance metrics
├── data/                       # Database and data files
├── exports/                    # Exported videos and CSVs
├── snapshots/                  # Saved snapshots
├── logs/                       # Application logs
├── assets/                     # Static assets
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── .env.example                # Example environment file
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🎨 Features in Detail

### Object Tracking
- Tracks objects across frames with unique IDs
- Uses IoU-based matching algorithm
- Maintains track history and statistics
- Configurable tracking parameters

### Analytics Dashboard
- Real-time FPS monitoring
- Object count distribution charts
- Processing time metrics
- Session-based analytics

### Database Logging
- SQLite database for persistence
- Stores detection history
- Session management
- Export to CSV functionality
- Query and analysis capabilities

### Export Options
- **Snapshots**: Save current frame with detections
- **Video Export**: Process and save entire videos
- **CSV Export**: Export detection logs for analysis
- **Image Export**: Save detected images

## 🔧 Troubleshooting

### Camera Issues
- **Webcam not detected**: Check camera permissions, try different camera index
- **IP Camera connection failed**: Verify WiFi connection, check IP address, ensure DroidCam is running

### Performance Issues
- **Low FPS**: Use smaller model (YOLOv8n), reduce resolution, enable GPU
- **High memory usage**: Reduce max detections, use smaller model

### GPU Issues
- **CUDA not available**: Install CUDA toolkit and compatible PyTorch
- **Out of memory**: Reduce batch size, use smaller model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[Streamlit](https://streamlit.io/)** - Beautiful web apps for ML
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

## 📧 Contact

- **Author**: Wasim
- **GitHub**: [@wasimsse](https://github.com/wasimsse)
- **Project**: [Detectify](https://github.com/wasimsse/Detectify)

## 🗺️ Roadmap

### v2.1 (Planned)
- [ ] Multi-object tracking with DeepSORT
- [ ] Custom model training interface
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Mobile app support
- [ ] REST API endpoints

### v2.2 (Future)
- [ ] Multiple camera synchronization
- [ ] Advanced analytics with ML insights
- [ ] Alert system for specific objects
- [ ] Integration with popular cameras (Nest, Ring, etc.)
- [ ] Web-based annotation tool

## 📊 Changelog

### v2.0.0 (Current)
- ✨ Complete rewrite with modular architecture
- 🚀 Support for YOLOv8, YOLOv10, and YOLOv11
- 📊 Analytics dashboard with real-time metrics
- 🗄️ SQLite database for detection logging
- 🎯 Object tracking implementation
- 🐳 Docker support
- 📹 Video and image file support
- ⚙️ Configuration management with .env
- 📝 Comprehensive logging system

### v1.0.0 (Original)
- Basic webcam and IP camera detection
- YOLOv8n support
- Simple Streamlit interface
- CSV export

---

<div align="center">
  <p>Made with ❤️ using Python, Streamlit, and YOLOv8+</p>
  <p>⭐ Star this repo if you find it useful!</p>
</div>
