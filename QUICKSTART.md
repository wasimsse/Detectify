# 🚀 Quick Start Guide

Get Detectify up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- Webcam (optional, for webcam detection)
- Internet connection (for initial model download)

## Installation

### Option 1: Automated Setup (Recommended)

#### On macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

#### On Windows:
```bash
setup.bat
```

### Option 2: Manual Setup

1. **Clone the repository**
```bash
git clone https://github.com/wasimsse/Detectify.git
cd Detectify
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create .env file**
```bash
cp .env.example .env
```

## Running the Application

### Local Run
```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Docker Run
```bash
docker-compose up -d
```

Then open your browser to: **http://localhost:8501**

## First Steps

### 1. Webcam Detection
1. Select **"Webcam"** from the sidebar
2. Adjust **Confidence Threshold** (default: 0.5 works well)
3. The detection will start automatically
4. Wave an object in front of your webcam!

### 2. Upload an Image
1. Select **"Upload Image"** from the sidebar
2. Click **"Browse files"** and select an image
3. See instant results with detected objects
4. Click **"Save Detected Image"** to export

### 3. Process a Video
1. Select **"Upload Video"** from the sidebar
2. Upload a video file (mp4, avi, mov, mkv)
3. Click **"Process Video"**
4. Download the processed video

### 4. IP Camera (DroidCam)
1. Install **DroidCam** app on your phone
2. Connect phone and computer to **same WiFi**
3. Note the **IP address** shown in DroidCam
4. Select **"IP Camera (DroidCam)"** in Detectify
5. Enter the IP address
6. Start detection!

## Tips & Tricks

### Performance Optimization
- **For faster processing**: Use `yolov8n.pt` (nano model)
- **For better accuracy**: Use `yolov8l.pt` or `yolov8x.pt`
- **Enable GPU**: Set `ENABLE_GPU=True` in `.env` (requires CUDA)

### Model Selection
- **YOLOv8n**: Fastest, good for real-time on CPU
- **YOLOv8s**: Balanced speed and accuracy
- **YOLOv8m**: Medium accuracy, slower
- **YOLOv8l**: High accuracy
- **YOLOv8x**: Best accuracy, slowest

### Adjusting Confidence
- **Lower threshold (0.3-0.4)**: More detections, some false positives
- **Medium threshold (0.5-0.6)**: Balanced (recommended)
- **Higher threshold (0.7-0.8)**: Fewer but more confident detections

## Common Issues

### "Cannot open camera"
- Check camera permissions
- Try a different camera index in settings
- Restart the application

### "IP Camera connection failed"
- Verify both devices are on same WiFi
- Check the IP address is correct
- Ensure DroidCam app is running
- Try accessing `http://IP_ADDRESS:4747` in browser

### "Low FPS"
- Use a smaller model (yolov8n)
- Reduce camera resolution in `.env`
- Enable GPU if available
- Close other applications

### "CUDA not available"
- Install CUDA toolkit from NVIDIA
- Install compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## Next Steps

- **Explore Advanced Settings**: Check the expandable "Advanced Settings" section
- **Enable Object Tracking**: Turn on tracking to follow objects across frames
- **View Analytics**: Check the analytics dashboard for statistics
- **Export Data**: Use the export buttons to save snapshots and CSV logs
- **Read Full Documentation**: See [README.md](README.md) for detailed information

## Getting Help

- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/wasimsse/Detectify/issues)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Video Tutorial

*Coming soon! A video walkthrough of Detectify's features.*

---

**Enjoy using Detectify! 🎉**

If you find it useful, please ⭐ star the repository on GitHub!

