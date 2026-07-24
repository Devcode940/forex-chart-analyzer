#!/bin/bash
# Forex Chart Analyzer launch script
# Works on Termux, Ubuntu, macOS

echo "Forex Chart Analyzer"
echo "===================="

if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Run from forex-analyzer directory."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python -m venv venv
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Starting server on http://localhost:8501"
echo ""

python -m streamlit run app.py \
    --server.headless true \
    --server.port 8501 \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
