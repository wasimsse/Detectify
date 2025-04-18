# Real-Time Object Detection Web Application

A real-time object detection web application built with YOLOv8 and Streamlit. This application allows you to perform object detection using your laptop's webcam or an IP camera stream, with features like snapshot capture and detection logging.

## Features

- Real-time object detection using YOLOv8
- Support for multiple camera sources:
  - Laptop webcam
  - IP camera stream
  - Both cameras side-by-side
- Real-time display of:
  - Bounding boxes
  - Object labels
  - Confidence scores
  - FPS counter
  - Object counts
- Snapshot capture functionality
- Detection log export (CSV/JSON)
- Adjustable confidence threshold
- Mobile-friendly interface

## Requirements

- Python 3.8+
- Webcam (for laptop camera feature)
- IP camera (optional, for remote camera feature)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RealTimeObjectDetection-v01.git
cd RealTimeObjectDetection-v01
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar controls to:
   - Select camera source
   - Enter IP camera URL (if using IP camera)
   - Adjust confidence threshold
   - Choose export format

4. The main window will show:
   - Real-time video feed with object detection
   - Object counts
   - FPS counter

5. Use the buttons at the bottom to:
   - Take snapshots
   - Export detection logs

## IP Camera Setup

To use an IP camera:

1. Install an IP camera app on your mobile device (e.g., IP Webcam for Android)
2. Connect your mobile device to the same network as your computer
3. Start the IP camera app and note the URL
4. Enter the URL in the application's IP camera URL field

## Project Structure

```
RealTimeObjectDetection-v01/
├── main.py           # Streamlit application
├── utils.py          # Helper functions
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/) 