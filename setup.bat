@echo off
REM Detectify Setup Script for Windows
REM This script sets up the development environment for Detectify

echo.
echo 🎥 Detectify Setup Script
echo =========================
echo.

REM Check Python installation
echo 📋 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo ✅ Python found
echo.

REM Create virtual environment
echo 🔨 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo ✅ Virtual environment created
)
echo.

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip -q
echo ✅ Pip upgraded
echo.

REM Install dependencies
echo 📦 Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt -q
echo ✅ Dependencies installed
echo.

REM Create necessary directories
echo 📁 Creating directories...
if not exist data mkdir data
if not exist exports mkdir exports
if not exist snapshots mkdir snapshots
if not exist logs mkdir logs
if not exist assets mkdir assets
if not exist assets\icons mkdir assets\icons
if not exist assets\samples mkdir assets\samples
echo ✅ Directories created
echo.

REM Copy environment file
echo ⚙️  Setting up environment file...
if exist .env (
    echo ⚠️  .env file already exists. Skipping...
) else (
    copy .env.example .env >nul
    echo ✅ .env file created (please edit with your settings)
)
echo.

REM Download YOLO model (optional)
set /p download_model="📥 Download YOLOv8n model now? (y/N): "
if /i "%download_model%"=="y" (
    echo ⬇️  Downloading YOLOv8n model...
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    echo ✅ Model downloaded
) else (
    echo ⏭️  Skipping model download (will download on first run)
)
echo.

REM Setup complete
echo ✅ Setup Complete!
echo.
echo 🚀 To start the application:
echo    1. Activate virtual environment: venv\Scripts\activate.bat
echo    2. Run the app: streamlit run app.py
echo.
echo 🐳 Or use Docker:
echo    docker-compose up -d
echo.
echo 📖 For more information, see README.md
echo.
pause

