# Real-Time Object Detection

A real-time object detection application using YOLOv8 and Streamlit. This application supports both laptop webcam and IP camera (DroidCam) inputs for object detection.

## Features

- Real-time object detection using YOLOv8
- Support for multiple camera sources:
  - Laptop webcam
  - IP camera (DroidCam)
  - Simultaneous dual camera mode
- Adjustable confidence threshold
- FPS counter and performance metrics
- Object counting and tracking

## Requirements

- Python 3.8+
- OpenCV
- Streamlit
- Ultralytics YOLOv8
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RealTimeObjectDetection.git
cd RealTimeObjectDetection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Select your camera source in the sidebar:
   - Laptop Webcam
   - IP Camera (requires DroidCam app)
   - Both (dual camera mode)

3. If using DroidCam:
   - Install the DroidCam app on your phone
   - Connect your phone to the same WiFi network as your computer
   - Enter the IP address shown in the DroidCam app
   - Make sure port 4747 is not blocked by your firewall

4. Adjust the confidence threshold in the sidebar as needed

## Configuration

- Default confidence threshold: 0.5
- Default image resolution: 640x480
- YOLO model: YOLOv8n (can be changed to other YOLOv8 variants)

## Troubleshooting

If you encounter issues with the IP camera:
1. Verify that DroidCam is running and showing an IP address
2. Ensure both devices are on the same network
3. Try accessing the MJPEG stream directly in your browser
4. Check firewall settings for port 4747

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Structure

```
RealTimeObjectDetection/
├── main.py           # Streamlit application
├── utils.py          # Helper functions
├── requirements.txt  # Project dependencies
└── README.md        # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/) 