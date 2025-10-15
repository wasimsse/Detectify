# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-10-15

### 🎉 Major Release - Complete Rewrite

#### Added
- **Modular Architecture**: Complete restructuring with separate modules for config, core, UI, and utils
- **Multiple YOLO Models**: Support for YOLOv8 (n/s/m/l/x), YOLOv10, and YOLOv11
- **Object Tracking**: IoU-based object tracking across frames
- **Video File Support**: Upload and process video files with detection
- **Image File Support**: Upload and process images with detection
- **Analytics Dashboard**: Real-time metrics, charts, and statistics
- **Database Logging**: SQLite database for detection history and session management
- **Docker Support**: Dockerfile and docker-compose.yml for easy deployment
- **Configuration Management**: Environment-based settings with .env support
- **Comprehensive Logging**: File and console logging with different log levels
- **Type Hints**: Full type annotation throughout the codebase
- **Export Features**: 
  - Snapshot saving
  - CSV export for detection logs
  - Processed video export
- **Advanced UI Components**:
  - Model selection dropdown
  - Parameter sliders
  - Analytics dashboard with charts
  - Session-based tracking
- **Performance Metrics**:
  - FPS counter
  - Processing time tracking
  - Object count statistics
  - Average metrics calculation
- **Setup Scripts**: Automated setup for Windows (setup.bat) and Unix (setup.sh)
- **Documentation**:
  - Comprehensive README.md
  - CONTRIBUTING.md guide
  - This CHANGELOG.md
  - Inline code documentation

#### Changed
- **UI Framework**: Enhanced Streamlit interface with modern design
- **Camera Handling**: Improved webcam and IP camera stream management
- **Error Handling**: Robust error handling and recovery mechanisms
- **Performance**: Optimized frame processing and model inference
- **Code Quality**: Improved code organization, naming, and structure

#### Improved
- Detection accuracy with latest YOLO models
- Frame processing speed with better buffering
- Memory management and resource cleanup
- User experience with intuitive controls
- Documentation and code comments

#### Technical Improvements
- Separation of concerns with modular design
- Better state management in Streamlit
- Asynchronous frame capture for IP cameras
- Context managers for resource cleanup
- Configuration validation and defaults

### [1.0.0] - 2025-04-18

#### Initial Release
- Basic webcam detection with YOLOv8n
- IP camera support via DroidCam
- Simple Streamlit interface
- Confidence threshold adjustment
- Object counting
- CSV export for detection logs
- FPS counter
- Basic error handling

---

## Upgrade Guide

### From v1.0.0 to v2.0.0

1. **Backup your data**:
   ```bash
   cp -r detection_log_*.csv backups/
   ```

2. **Update repository**:
   ```bash
   git pull origin main
   ```

3. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

4. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run new version**:
   ```bash
   streamlit run app.py
   ```

**Breaking Changes**:
- Old detection log CSVs need to be migrated manually if you want to import them to the new database
- Configuration is now managed through .env file instead of hardcoded values
- Main entry point changed from `main.py` to `app.py`

**Migration Script** (coming soon):
```bash
python scripts/migrate_v1_to_v2.py
```

---

## Future Releases

### [2.1.0] - Planned
- Multi-object tracking with DeepSORT
- Custom zone detection
- Alert notifications
- REST API endpoints
- Cloud storage integration
- Performance optimizations

### [2.2.0] - Future
- Multi-camera synchronization
- Advanced analytics with ML insights
- Mobile app support
- Integration with smart home devices
- Real-time collaboration features

---

[2.0.0]: https://github.com/wasimsse/Detectify/releases/tag/v2.0.0
[1.0.0]: https://github.com/wasimsse/Detectify/releases/tag/v1.0.0

