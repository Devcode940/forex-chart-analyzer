# 🚀 Deployment Guide

## Forex Chart Analyzer Pro v2 — Install & Run on Any Platform

---

# 📱 Termux (Android)

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

## 9. Termux Troubleshooting

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

## 10. Termux Quick-Start (Copy-Paste)

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
---

# 🐧 Ubuntu (Desktop / Server / WSL)

Complete guide for Ubuntu 20.04, 22.04, 24.04, and WSL2.

---

## 1. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y

# Core build tools and Python headers
sudo apt install -y \
    git python3 python3-pip python3-venv \
    build-essential python3-dev \
    libffi-dev libssl-dev \
    libopencv-dev python3-opencv \
    pkg-config

# For scikit-learn / scipy compilation (if wheels unavailable)
sudo apt install -y \
    gfortran liblapack-dev libblas-dev \
    cython
```

### Minimum by Ubuntu version

| Ubuntu | Python 3 | Notes |
|--------|----------|-------|
| 20.04 LTS | 3.8 | Works, but Python 3.8 is EOL. Consider deadsnakes PPA for 3.11+ |
| 22.04 LTS | 3.10 | Stable, well-supported |
| 24.04 LTS | 3.12 | Best — all packages have prebuilt wheels |
| WSL2 | matches host | Same steps as native Ubuntu |

### Optional: Install newer Python (Ubuntu 20.04)

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Use python3.11 instead of python3 for the rest of this guide
```

---

## 2. Clone & Create Virtual Environment

```bash
cd ~
git clone https://github.com/Devcode940/forex-chart-analyzer.git
cd forex-chart-analyzer

# Create venv (ALWAYS use venv — Ubuntu 23.04+ blocks system pip)
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip inside venv
pip install --upgrade pip setuptools wheel
```

> **Why venv on Ubuntu?** Ubuntu 23.04+ enforces PEP 668 — system pip refuses to install packages without `--break-system-packages`. venv is the correct approach on ALL modern Ubuntu versions. No hacks needed.

---

## 3. Install Python Dependencies

```bash
# Activate venv first
source venv/bin/activate

# Option A: Install everything from requirements.txt
pip install -r requirements.txt

# Option B: Install in stages (safer if any step fails)
pip install numpy
pip install scipy scikit-learn
pip install opencv-python-headless Pillow
pip install streamlit plotly xgboost pandas
```

> **Ubuntu 24.04+**: All packages have prebuilt wheels. Installation is fast — no compilation needed.
>
> **Ubuntu 20.04**: Some packages may compile from source. This is slower but automatic.

### Headless vs full OpenCV

```bash
# For servers / WSL without display (recommended):
pip install opencv-python-headless

# For desktop Ubuntu with GUI (only if you need cv2.imshow):
pip install opencv-python
```

---

## 4. Run the App

```bash
cd ~/forex-chart-analyzer
source venv/bin/activate

# Development (auto-reload on file changes)
python -m streamlit run app.py

# Production (headless, specific port)
python -m streamlit run app.py --server.port 8501 --server.headless true

# Bind to all interfaces (LAN access)
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Open browser → `http://localhost:8501`

### Create a desktop shortcut

```bash
# Add alias to ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# Forex Chart Analyzer
alias forex="cd ~/forex-chart-analyzer && source venv/bin/activate && python -m streamlit run app.py --server.port 8501 --server.headless true"
EOF

source ~/.bashrc

# Now just type:
forex
```

---

## 5. WSL2 Specific Setup

If running Ubuntu inside WSL2 on Windows:

### Install WSL2 (if not already)

```powershell
# In PowerShell (Admin):
wsl --install -d Ubuntu-24.04
```

### Browser access

WSL2 automatically forwards ports to Windows. After starting the app:

```
# Access from Windows browser:
http://localhost:8501
```

### Clipboard / screenshot workflow

1. Take screenshot in Windows (Win+Shift+S)
2. Save to `\\wsl$\Ubuntu\home\<user>\forex-chart-analyzer\screenshots\`
3. Upload from the Streamlit app's file picker

Or use the shared filesystem:
```bash
# Screenshots saved to Windows Desktop are accessible at:
ls /mnt/c/Users/<YourName>/Desktop/*.png
```

### GUI apps in WSL2 (optional)

If you want the Streamlit app to auto-open in a browser:
```bash
# WSL2 supports WSLg — browsers open automatically
# If not working, set BROWSER:
export BROWSER="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
```

---

## 6. Production Deployment (systemd)

Run as a persistent system service on Ubuntu Server:

### Create systemd service

```bash
sudo tee /etc/systemd/system/forex-analyzer.service << 'EOF'
[Unit]
Description=Forex Chart Analyzer Pro
After=network.target

[Service]
Type=simple
User=<YOUR_USERNAME>
WorkingDirectory=/home/<YOUR_USERNAME>/forex-chart-analyzer
ExecStart=/home/<YOUR_USERNAME>/forex-chart-analyzer/venv/bin/python -m streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=on-failure
RestartSec=10
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=4

[Install]
WantedBy=multi-user.target
EOF
```

### Enable and start

```bash
# Replace <YOUR_USERNAME> with your actual username
sudo sed -i "s/<YOUR_USERNAME>/$USER/g" /etc/systemd/system/forex-analyzer.service

sudo systemctl daemon-reload
sudo systemctl enable forex-analyzer
sudo systemctl start forex-analyzer

# Check status
sudo systemctl status forex-analyzer

# View logs
sudo journalctl -u forex-analyzer -f
```

### Stop / restart

```bash
sudo systemctl stop forex-analyzer
sudo systemctl restart forex-analyzer
```

---

## 7. Nginx Reverse Proxy (Public Access)

Serve the app on port 80 with SSL:

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

### Nginx config

```bash
sudo tee /etc/nginx/sites-available/forex-analyzer << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # Streamlit WebSocket endpoint
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/forex-analyzer /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Add SSL (Let's Encrypt)

```bash
sudo certbot --nginx -d your-domain.com
```

---

## 8. Docker Deployment (Ubuntu)

```bash
# Install Docker
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Log out and back in for group change
```

### Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.12-slim

# System deps for OpenCV and scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
    "--server.port=8501", "--server.headless=true", \
    "--server.address=0.0.0.0"]
```

### Build and run

```bash
# Build
docker build -t forex-analyzer .

# Run
docker run -d \
    --name forex-analyzer \
    -p 8501:8501 \
    -v forex-data:/app/data \
    --restart unless-stopped \
    forex-analyzer

# View logs
docker logs -f forex-analyzer

# Stop
docker stop forex-analyzer

# Restart
docker start forex-analyzer
```

---

## 9. Ubuntu Troubleshooting

### `externally-managed-environment` error (Ubuntu 23.04+)

```
error: externally-managed-environment
× This environment is externally managed
```

**Fix**: Use venv. Never `--break-system-packages`:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'cv2'`

```bash
# Option A: Install headless OpenCV in venv
source venv/bin/activate
pip install opencv-python-headless

# Option B: Use system OpenCV (if python3-opencv is installed)
# This is NOT recommended — stick with venv's opencv-python-headless
```

### `error: command 'x86_64-linux-gnu-gcc' failed`

```bash
sudo apt install -y build-essential python3-dev
source venv/bin/activate
pip install -r requirements.txt
```

### `numpy.distutils` deprecation warning (Python 3.12)

This is a harmless warning from older numpy versions. Upgrade numpy:
```bash
source venv/bin/activate
pip install --upgrade numpy
```

### Port 8501 already in use

```bash
# Find the process
sudo lsof -i :8501

# Kill it
kill -9 <PID>

# Or change port
python -m streamlit run app.py --server.port 8502 --server.headless true
```

### WSL2: Cannot connect to localhost:8501

```bash
# Check WSL2 IP address
hostname -I

# Try accessing via the WSL2 IP instead of localhost
# Or add port forwarding in PowerShell:
# netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=$(wsl hostname -I)
```

### Clean reinstall

```bash
cd ~/forex-chart-analyzer
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 10. Ubuntu Quick-Start (Copy-Paste)

### Desktop / WSL2

```bash
sudo apt update && sudo apt upgrade -y && \
sudo apt install -y git python3 python3-venv python3-dev build-essential libffi-dev libssl-dev && \
cd ~ && \
git clone https://github.com/Devcode940/forex-chart-analyzer.git && \
cd forex-chart-analyzer && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip setuptools wheel && \
pip install -r requirements.txt && \
echo "" && \
echo "✅ Installation complete!" && \
echo "Run with:" && \
echo "  cd ~/forex-chart-analyzer && source venv/bin/activate && python -m streamlit run app.py"
```

### Server (headless + systemd)

```bash
sudo apt update && sudo apt upgrade -y && \
sudo apt install -y git python3 python3-venv python3-dev build-essential libffi-dev libssl-dev && \
cd ~ && \
git clone https://github.com/Devcode940/forex-chart-analyzer.git && \
cd forex-chart-analyzer && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip setuptools wheel && \
pip install -r requirements.txt && \
deactivate && \
sudo tee /etc/systemd/system/forex-analyzer.service << EOF
[Unit]
Description=Forex Chart Analyzer Pro
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/forex-chart-analyzer
ExecStart=$HOME/forex-chart-analyzer/venv/bin/python -m streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && \
sudo systemctl enable forex-analyzer && \
sudo systemctl start forex-analyzer && \
echo "" && \
echo "✅ Installation complete! Service is running." && \
echo "  Status: sudo systemctl status forex-analyzer" && \
echo "  Logs:   sudo journalctl -u forex-analyzer -f" && \
echo "  URL:    http://$(hostname -I | awk '{print $1}'):8501"
```

---

## 11. Updating

```bash
cd ~/forex-chart-analyzer
git pull origin main
source venv/bin/activate
pip install -r requirements.txt

# If running as systemd service:
sudo systemctl restart forex-analyzer
```

---

# 📋 Platform Comparison

| Feature | Termux | Ubuntu Desktop | Ubuntu Server | WSL2 | Docker |
|---------|--------|---------------|---------------|------|--------|
| venv required | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✘ (isolated) |
| System deps | Rust, binutils | build-essential, python3-dev | Same as desktop | Same as desktop | In Dockerfile |
| Browser auto-open | ✘ Manual | ✅ Auto | ✘ Manual | ✅ Auto | ✘ Manual |
| Systemd service | ✘ | Optional | ✅ Recommended | Optional | ✘ Use Docker |
| Nginx reverse proxy | ✘ | Optional | ✅ For public | ✘ | ✅ With compose |
| Port forwarding | Phone IP | localhost | Server IP | localhost (auto) | localhost |
| Storage access | `termux-setup-storage` | Native | Native | `/mnt/c/` | Volume mount |
| GPU acceleration | ✘ | ✅ CUDA optional | ✅ CUDA optional | ✅ CUDA optional | ✅ nvidia-docker |

---
---

# 🔒 Security Notes

- The app runs entirely **locally** — no data leaves your device by default
- No internet connection required after installation
- The SQLite database is stored locally in `data/`
- Trade journal entries are private to your device
- venv is excluded from git via `.gitignore` (never committed)
- For public deployments: always use Nginx + SSL, never expose port 8501 directly
- If using Nginx: add authentication (`htpasswd` or OAuth) for public access

---

# 📂 Directory Structure

```
forex-chart-analyzer/
├── venv/                               ← Isolated Python environment (gitignored)
│   ├── bin/
│   │   ├── activate                    ← Source this before running
│   │   ├── python                      ← venv's own Python
│   │   └── pip                         ← venv's own pip (no crashes)
│   └── lib/
│       └── python3.x/
│           └── site-packages/          ← All pip packages live here
├── app.py                              ← Streamlit entry point
├── analyzers/                          ← 25 analysis modules
├── utils/                              ← Visualizer
├── data/
│   └── trade_database.db               ← Auto-created SQLite DB
├── requirements.txt
├── run.sh                              ← Auto-activates venv
├── Dockerfile                          ← For Docker deployment
├── DEPLOYMENT.md                       ← This file
└── .gitignore                          ← Excludes venv/, data/, __pycache__/
```

---

# 📊 System Requirements

| Requirement | Termux | Ubuntu |
|-------------|--------|--------|
| OS | Android 10+ | Ubuntu 20.04+ |
| RAM | 4 GB min | 2 GB min, 4 GB recommended |
| Storage | 1 GB free | 500 MB free (1 GB with venv) |
| CPU | ARM64, 4 cores | x86_64 or ARM64, 2+ cores |
| Python | 3.10+ | 3.10+ (3.12 recommended) |

---

**Repository**: https://github.com/Devcode940/forex-chart-analyzer
