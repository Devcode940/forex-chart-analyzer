#!/bin/bash
# Quick launch script for Forex Chart Analyzer Pro

echo "🔍 Forex Chart Analyzer Pro"
echo "==========================="
echo ""

# Check if running in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Run this script from the forex-analyzer directory."
    exit 1
fi

# Install dependencies if needed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo "🚀 Starting Forex Chart Analyzer Pro..."
echo "📱 Open your browser to: http://localhost:8501"
echo ""

python -m streamlit run app.py --server.headless true
