# 📱 Termux Installation & Deployment Guide

## Forex Chart Analyzer Pro v2 on Android via Termux

Complete guide to running the full app on an Android phone/tablet using Termux.

---

## 1. Install Termux

Download from **F-Droid** (NOT Play Store — Play Store version is outdated and broken):

```bash
# Option A: F-Droid (recommended)
# Download from https://f-droid.org/en/packages/com.termux/

# Option B: GitHub releases
# https://github.com/termux/termux-app/releases
```

> ⚠️ **Do NOT use the Google Play Store version.** It has a signing key conflict and cannot install packages.

---

## 2. Termux Base Setup

```bash
# Update all packages
pkg update && pkg upgrade -y

# Grant storage access
termux-setup-storage

# Install core tools
pkg install -y git python python-pip build-essential libffi openssl rust binutils
```

---

## 3. Install Python Dependencies

```bash
# Upgrade pip and core build tools
pip install --upgrade pip setuptools wheel

# Install numpy first (needs Rust compiler for some archs)
pip install numpy

# Install scientific stack
pip install scipy scikit-learn

# Install image processing
pip install opencv-python-headless Pillow

# Install remaining dependencies
pip install streamlit plotly xgboost pandas
```

> **Note on ARM64**: Some packages may need to compile from source on ARM. If a `pip install` fails, try:
> ```bash
> pip install <package> --no-binary :all:
> ```
> This forces source compilation using the Rust/C compilers installed above.

---

## 4. Clone the Repository

```bash
# Navigate to home directory
cd ~

# Clone the repo
git clone https://github.com/Devcode940/forex-chart-analyzer.git

cd forex-chart-analyzer
```

---

## 5. Run the App

```bash
# Method 1: Using the run script
chmod +x run.sh
./run.sh

# Method 2: Direct command
python -m streamlit run app.py --server.port 8501 --server.headless true
```

### Access the App

Open your phone's browser and navigate to:

```
http://localhost:8501
```

---

## 6. Alternative: Port Forwarding (Access from PC)

If you want to view the app on your computer while it runs on your phone:

```bash
# On Termux - note your phone's IP
ifconfig wlan0 | grep inet

# Start the app binding to all interfaces
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Then on your PC browser: `http://<PHONE_IP>:8501`

---

## 7. Taking Screenshots for Analysis

Since you're on Android, the easiest workflow:

1. **Take a screenshot** of your forex chart (TradingView, MT4/MT5, etc.)
2. **Open the Streamlit app** in Chrome/Firefox at `http://localhost:8501`
3. **Upload the screenshot** via the sidebar file uploader
4. **Review analysis** across all 13 tabs

### Pro Tip: Split Screen

Use Android's split-screen mode:
- **Top half**: Trading app (TradingView/MT4)
- **Bottom half**: Browser with Streamlit app

Take a screenshot of just the chart, then upload to the analyzer.

---

## 8. Performance Tuning for Mobile

The app runs ML models that can be CPU-intensive. Adjust these settings for smoother mobile performance:

```python
# In app.py, reduce ML pipeline intensity:
# Change n_simulations from 2000 to 500
# Change n_bootstrap from 2000 to 500
# Change n_windows from 5 to 3
```

Or set environment variables:

```bash
# Reduce thread count to prevent overheating
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Then run
python -m streamlit run app.py --server.port 8501 --server.headless true
```

---

## 9. Persistent Service (Keep Running)

### Option A: tmux (Recommended)

```bash
pkg install -y tmux

# Create a new session
tmux new -s forex

# Start the app
python -m streamlit run app.py --server.port 8501 --server.headless true

# Detach: Press Ctrl+B then D
# Reattach: tmux attach -t forex
```

### Option B: nohup

```bash
nohup python -m streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
```

---

## 10. Troubleshooting

### pip install fails with compilation error

```bash
pkg install -y cmake ndk-sysroot
pip install <package> --no-binary :all:
```

### OpenCV won't install

```bash
pip install opencv-python-headless
# NOT opencv-python (requires GUI libraries not available in Termux)
```

### Streamlit not found

```bash
python -m streamlit run app.py
# NOT: streamlit run app.py
```

### Port 8501 already in use

```bash
# Find and kill the process
lsof -ti:8501 | xargs kill -9

# Or use a different port
python -m streamlit run app.py --server.port 8502 --server.headless true
```

### Database not created

```bash
# The database auto-creates on first run in data/trade_database.db
# If permissions issue:
mkdir -p data
chmod 755 data
```

### App is slow / phone overheating

```bash
# Reduce thread count
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Or reduce ML parameters in the app
```

### Storage permission denied

```bash
termux-setup-storage
# Accept the permission prompt
```

---

## 11. Quick-Start Script (Copy-Paste)

One-shot install everything:

```bash
pkg update -y && pkg upgrade -y && \
pkg install -y git python python-pip build-essential libffi openssl rust binutils && \
termux-setup-storage && \
pip install --upgrade pip setuptools wheel && \
pip install numpy scipy scikit-learn opencv-python-headless Pillow streamlit plotly xgboost pandas && \
cd ~ && \
git clone https://github.com/Devcode940/forex-chart-analyzer.git && \
cd forex-chart-analyzer && \
echo "✅ Installation complete! Run with:" && \
echo "   cd ~/forex-chart-analyzer && python -m streamlit run app.py --server.port 8501 --server.headless true"
```

---

## 12. Updating

```bash
cd ~/forex-chart-analyzer
git pull origin main
pip install -r requirements.txt
```

---

## Architecture on Termux

```
/data/data/com.termux/files/home/
├── forex-chart-analyzer/
│   ├── app.py                          ← Streamlit entry point
│   ├── analyzers/                      ← 25 analysis modules
│   ├── utils/                          ← Visualizer
│   ├── data/
│   │   └── trade_database.db           ← Auto-created SQLite DB
│   ├── requirements.txt
│   └── run.sh
└── storage/                            ← Android storage access
    └── shared/                         ← Screenshots live here
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Android | 10+ | 12+ |
| RAM | 4 GB | 6 GB+ |
| Storage | 500 MB free | 1 GB free |
| CPU | ARM64, 4 cores | ARM64, 8 cores |
| Termux | 0.118+ | Latest from F-Droid |

---

## Security Notes

- The app runs entirely **locally** — no data leaves your device
- No internet connection required after installation
- The SQLite database is stored locally in `data/`
- Trade journal entries are private to your device
- GitHub token is NOT stored in the codebase (excluded via .gitignore)

---

## Repository

**GitHub**: https://github.com/Devcode940/forex-chart-analyzer

Report issues, request features, or contribute.
