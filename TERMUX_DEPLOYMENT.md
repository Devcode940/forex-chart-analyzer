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

# Install core tools (no python-pip — we use venv instead)
pkg install -y git python build-essential libffi openssl rust binutils
```

> **Why no `python-pip`?** Termux's system pip frequently crashes when installing scientific packages (scipy, scikit-learn, xgboost) because it conflicts with system-managed packages. Using a **virtual environment** completely isolates dependencies and eliminates these crashes.

---

## 3. Create Virtual Environment & Install Dependencies

```bash
# Navigate to home
cd ~

# Clone the repo first
git clone https://github.com/Devcode940/forex-chart-analyzer.git
cd forex-chart-analyzer

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip inside the venv (safe, no system conflicts)
pip install --upgrade pip setuptools wheel

# Install numpy first (needs Rust compiler for some archs)
pip install numpy

# Install scientific stack
pip install scipy scikit-learn

# Install image processing
pip install opencv-python-headless Pillow

# Install remaining dependencies
pip install streamlit plotly xgboost pandas

# Or install everything at once from requirements.txt
pip install -r requirements.txt
```

> **Note on ARM64**: Some packages may need to compile from source on ARM. If a `pip install` fails, try:
> ```bash
> pip install <package> --no-binary :all:
> ```
> This forces source compilation using the Rust/C compilers installed above.

### Why venv?

| Problem | System pip | venv pip |
|---------|-----------|----------|
| `externally-managed-environment` error | ✗ Common | ✗ Never |
| Pip crashes on scipy/scikit-learn | ✗ Frequent | ✗ Rare |
| Conflicts with Termux system packages | ✗ Yes | ✗ No |
| `pip install --break-system-packages` hack needed | ✗ Yes | ✗ No |
| Clean uninstall (just delete `venv/`) | ✗ No | ✗ Yes |
| Reproducible builds | ✗ No | ✗ Yes |

---

## 4. Run the App

**You MUST activate the venv every time** before running:

```bash
cd ~/forex-chart-analyzer
source venv/bin/activate
python -m streamlit run app.py --server.port 8501 --server.headless true
```

### Shortcut: Create an alias

```bash
# Add to ~/.bashrc
echo 'alias forex="cd ~/forex-chart-analyzer && source venv/bin/activate && python -m streamlit run app.py --server.port 8501 --server.headless true"' >> ~/.bashrc
source ~/.bashrc

# Now just type:
forex
```

### Shortcut: Use the run script

```bash
chmod +x run.sh
./run.sh
```

The `run.sh` script automatically activates the venv before starting.

### Access the App

Open your phone's browser and navigate to:

```
http://localhost:8501
```

---

## 5. Alternative: Port Forwarding (Access from PC)

If you want to view the app on your computer while it runs on your phone:

```bash
# On Termux - note your phone's IP
ifconfig wlan0 | grep inet

# Start the app binding to all interfaces
source venv/bin/activate
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Then on your PC browser: `http://<PHONE_IP>:8501`

---

## 6. Taking Screenshots for Analysis

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

## 7. Performance Tuning for Mobile

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
source venv/bin/activate
python -m streamlit run app.py --server.port 8501 --server.headless true
```

---

## 8. Persistent Service (Keep Running)

### Option A: tmux (Recommended)

```bash
pkg install -y tmux

# Create a new session
tmux new -s forex

# Start the app
cd ~/forex-chart-analyzer
source venv/bin/activate
python -m streamlit run app.py --server.port 8501 --server.headless true

# Detach: Press Ctrl+B then D
# Reattach: tmux attach -t forex
```

### Option B: nohup

```bash
cd ~/forex-chart-analyzer
source venv/bin/activate
nohup python -m streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
```

---

## 9. Troubleshooting

### `externally-managed-environment` error

This is the #1 reason to use venv. If you see:
```
error: externally-managed-environment
× This environment is externally managed
```
**Solution**: Always use the venv:
```bash
source venv/bin/activate
pip install <package>
```
Never run `pip install` outside the venv.

### pip install fails with compilation error

```bash
pkg install -y cmake ndk-sysroot
source venv/bin/activate
pip install <package> --no-binary :all:
```

### OpenCV won't install

```bash
source venv/bin/activate
pip install opencv-python-headless
# NOT opencv-python (requires GUI libraries not available in Termux)
```

### venv activation fails

```bash
# If venv/bin/activate doesn't exist, recreate it:
cd ~/forex-chart-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Streamlit not found

```bash
# Make sure venv is activated first!
source venv/bin/activate
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

### numpy/scipy wheel not found for aarch64

```bash
# Make sure you're on the latest pip
source venv/bin/activate
pip install --upgrade pip

# If still failing, build from source
pkg install -y rust
pip install numpy --no-binary :all:
```

### Clean reinstall (nuke and rebuild)

```bash
cd ~/forex-chart-analyzer
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 10. Quick-Start Script (Copy-Paste)

One-shot install everything with venv:

```bash
pkg update -y && pkg upgrade -y && \
pkg install -y git python build-essential libffi openssl rust binutils && \
termux-setup-storage && \
cd ~ && \
git clone https://github.com/Devcode940/forex-chart-analyzer.git && \
cd forex-chart-analyzer && \
python -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip setuptools wheel && \
pip install -r requirements.txt && \
echo "" && \
echo "✅ Installation complete!" && \
echo "Run anytime with:" && \
echo "  cd ~/forex-chart-analyzer && source venv/bin/activate && python -m streamlit run app.py --server.port 8501 --server.headless true"
```

---

## 11. Updating

```bash
cd ~/forex-chart-analyzer
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
```

---

## 12. Directory Structure on Termux

```
/data/data/com.termux/files/home/
├── forex-chart-analyzer/
│   ├── venv/                           ← Isolated Python environment
│   │   ├── bin/
│   │   │   ├── activate                ← Source this before running
│   │   │   ├── python                  ← venv's own Python
│   │   │   └── pip                     ← venv's own pip (no crashes)
│   │   └── lib/
│   │       └── python3.x/
│   │           └── site-packages/      ← All pip packages live here
│   ├── app.py                          ← Streamlit entry point
│   ├── analyzers/                      ← 25 analysis modules
│   ├── utils/                          ← Visualizer
│   ├── data/
│   │   └── trade_database.db           ← Auto-created SQLite DB
│   ├── requirements.txt
│   └── run.sh                          ← Auto-activates venv
└── storage/                            ← Android storage access
    └── shared/                         ← Screenshots live here
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Android | 10+ | 12+ |
| RAM | 4 GB | 6 GB+ |
| Storage | 1 GB free | 2 GB free (venv + packages) |
| CPU | ARM64, 4 cores | ARM64, 8 cores |
| Termux | 0.118+ | Latest from F-Droid |

---

## Security Notes

- The app runs entirely **locally** — no data leaves your device
- No internet connection required after installation
- The SQLite database is stored locally in `data/`
- Trade journal entries are private to your device
- venv is excluded from git via `.gitignore` (never committed)
- GitHub token is NOT stored in the codebase

---

## Repository

**GitHub**: https://github.com/Devcode940/forex-chart-analyzer

Report issues, request features, or contribute.
