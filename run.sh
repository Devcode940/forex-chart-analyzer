#!/bin/bash
# Quick launch script for Forex Chart Analyzer Pro
# Auto-activates venv to avoid pip/system Python conflicts

echo "🔍 Forex Chart Analyzer Pro"
echo "==========================="
echo ""

# Check if running in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Run this script from the forex-analyzer directory."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "📦 Installing dependencies into venv..."
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
else
    # Activate existing venv
    source venv/bin/activate
fi

echo "🚀 Starting Forex Chart Analyzer Pro..."
echo "📱 Open your browser to: http://localhost:8501"
echo ""

python -m streamlit run app.py --server.headless true
