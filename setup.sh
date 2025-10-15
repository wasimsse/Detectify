#!/bin/bash

# Detectify Setup Script
# This script sets up the development environment for Detectify

set -e  # Exit on error

echo "🎥 Detectify Setup Script"
echo "========================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required (found: $python_version)"
    exit 1
fi
echo "✅ Python $python_version found"
echo ""

# Create virtual environment
echo "🔨 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
echo "✅ Pip upgraded"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data exports snapshots logs assets/icons assets/samples
echo "✅ Directories created"
echo ""

# Copy environment file
echo "⚙️  Setting up environment file..."
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists. Skipping..."
else
    cp .env.example .env
    echo "✅ .env file created (please edit with your settings)"
fi
echo ""

# Download YOLO model (optional)
read -p "📥 Download YOLOv8n model now? (y/N): " download_model
if [ "$download_model" = "y" ] || [ "$download_model" = "Y" ]; then
    echo "⬇️  Downloading YOLOv8n model..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || true
    echo "✅ Model downloaded"
else
    echo "⏭️  Skipping model download (will download on first run)"
fi
echo ""

# Setup complete
echo "✅ Setup Complete!"
echo ""
echo "🚀 To start the application:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run the app: streamlit run app.py"
echo ""
echo "🐳 Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "📖 For more information, see README.md"
echo ""

