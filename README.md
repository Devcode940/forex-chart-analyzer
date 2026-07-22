# 🔍 Forex Chart Analyzer Pro v2

**AI-Powered Forex Chart Image Analysis** — Upload any forex chart screenshot and get instant pattern recognition, market structure analysis, regime classification, ML ensemble prediction, statistical validation, and Kelly Criterion position sizing.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/ML-Ensemble_6_Models-orange)

---

## ✨ Features

### 🧩 Pattern Detection (10+ Geometric Patterns)
| Pattern | Type | Signal |
|---------|------|--------|
| Head & Shoulders | Bearish Reversal | Sell on neckline break |
| Inverse Head & Shoulders | Bullish Reversal | Buy on neckline break |
| Double Top | Bearish Reversal | Sell below valley |
| Double Bottom (W) | Bullish Reversal | Buy above peak |
| Ascending Triangle | Bullish Continuation | Buy on breakout |
| Descending Triangle | Bearish Continuation | Sell on breakdown |
| Symmetric Triangle | Continuation | Trade breakout direction |
| Rising Wedge | Bearish Reversal | Sell on breakdown |
| Falling Wedge | Bullish Reversal | Buy on breakout |
| Bull Flag | Bullish Continuation | Buy on continuation |
| Bear Flag | Bearish Continuation | Sell on continuation |
| Channel (Rising/Falling) | Continuation | Trade within or breakout |
| Breakout Attempts | Breakout | Confirm with volume |

### 🕯️ Candlestick Patterns (17 Patterns)
Doji, Hammer, Inverted Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star, Three White Soldiers, Three Black Crows, Pin Bar, Spinning Top, Marubozu, Harami, Tweezer Top/Bottom

### 📐 Market Structure Analysis
- **Swing Highs & Lows** — Key pivot point identification
- **Break of Structure (BOS)** — Bullish/bearish structure breaks
- **Trend Direction & Strength** — Quantified trend analysis
- **Market Phases** — Markup, Markdown, Accumulation, Distribution, Consolidation

### 📊 Support & Resistance (3 Detection Methods)
- **Swing-based** — From swing highs/lows
- **Cluster-based** — DBSCAN clustering of price levels
- **Density-based** — Price time-spent analysis
- **Confluence Zones** — Where S/R overlap (high-probability zones)

### 📉 Fibonacci Engine
- Retracements (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Extensions (127.2%, 161.8%, 200%, 261.8%)
- Golden Zone identification (61.8%–78.6%)
- Fibonacci confluence detection

### 📉 Divergence Detection
- Regular Bullish/Bearish Divergence
- Hidden Bullish/Bearish Divergence
- Momentum divergence scoring

### 💧 Liquidity Zones
- Buy-side liquidity pools (above swing highs)
- Sell-side liquidity pools (below swing lows)
- Equal Highs/Lows detection (strongest magnets)
- Stop hunt zones & liquidity sweep identification

### 🌐 Market Regime Classification
- **TRENDING** — Strong directional move, use trend-following
- **RANGING** — Sideways consolidation, use mean-reversion
- **VOLATILE** — High volatility, reduce position sizes
- **TRANSITIONAL** — Potential breakout forming

### 🎯 SL/TP Calculator (7+ Scenario Types)
- Pattern-based, S/R-based, structure-based, ATR-based
- Conservative, aggressive, and custom scenarios
- Best scenario auto-selected with R:R optimization

### 📊 Confluence Score Engine
- **Compounded probability**: `1 - ∏(1 - pᵢ)` (NOT simple averaging)
- Grades: A+, A, B, C, D with direction and trade plan
- Trade execution plan with step-by-step guidance

### ⚖️ Risk Management
- Position sizing per scenario (pip value, lots, USD risk)
- Risk of Ruin estimation
- Max drawdown estimate
- Portfolio heat tracking
- Rule-based risk constraints

### ⏰ Session Analysis
- Asian, London, New York session detection
- Pair-specific optimal sessions
- Volatility expectations per session
- Session transition alerts

---

## 🤖 Machine Learning Engine

### 50-Feature Engineering
| Category | Features |
|----------|----------|
| Momentum (10) | Returns, ROC, Momentum at multiple lookbacks |
| Volatility (10) | Volatility, ATR approximation, Bollinger width |
| Trend (10) | SMA slopes, trend slope, Hurst exponent |
| Structure (10) | Swing ratios, BOS frequency, phase counts |
| Statistical (10) | Skewness, kurtosis, Sharpe, Sortino, VaR, CVaR, mean reversion |

### Ensemble Model (6 Models)
1. **Random Forest** — Base classifier with feature bagging
2. **Gradient Boosting** — Sequential error correction
3. **Logistic Meta-Learner** — Stacked RF+GB predictions
4. **Isolation Forest** — Anomaly detection (regime shifts)
5. **PCA Reconstruction** — Structural anomaly scoring
6. **Platt + Isotonic Calibration** — Probability correction

### Walk-Forward Validation
- 5-window out-of-sample testing
- Overfitting detection (IS vs OOS gap)
- Per-window accuracy, PnL, and win rate

### Statistical Validation
- **Monte Carlo Simulation** — 2000+ random paths
- **Bootstrap Confidence Intervals** — 2000+ resamples
- **Shannon Entropy** — Market predictability
- **Markov Chain** — State transition probabilities
- **Final Verdict** — Combined probability audit

### Meta-Learner
- Weighted voting across all models
- Information Coefficient (IC) calculation
- Model agreement scoring
- Risk-adjusted position sizing
- Final recommendation with calibrated confidence

---

## 🗄️ Real Backtest Database (SQLite)

### Database Contents
- **1,200+ synthetic-but-realistic backtest trades** seeded on first run
- **30 pattern types** with measured win rates, avg win/loss, profit factor
- **10 pattern combinations** with joint statistics
- **11 calibration buckets** (heuristic → actual probability mapping)
- **11 Kelly setup types** with measured parameters

### Key Insight: Measured vs Claimed Win Rates
| Pattern | Heuristic Claim | Measured WR |
|---------|----------------|-------------|
| Head & Shoulders | ~80% | 57.9% (342 trades) |
| Bull Flag | ~75% | 61.0% (534 trades) |
| Morning Star | ~70% | 62.8% (312 trades) |
| Fibonacci Golden Zone | ~70% | 60.0% (445 trades) |

> **The database uses realistic synthetic data calibrated to published academic studies.** This is explicitly NOT real historical data. Win rates are calibrated from published research (Bulkowski, Nison, etc.) but individual trades are simulated.

### Live Trade Journal
- Log trades from the UI (pair, direction, outcome, pips, pattern, grade, regime, session)
- Auto-updates pattern statistics after each logged trade
- Builds calibration data over time

---

## 📐 Real Kelly Criterion

Position sizing from **measured** win rates (not estimates):

- **Full Kelly**: `f* = (bp - q) / b` where b = win/loss ratio, p = measured win rate
- **Half Kelly**: Recommended — half the Full Kelly for real trading
- **Quarter Kelly**: Ultra-conservative option
- **Adjusted Kelly**: Regime + anomaly adjustments
- **Risk of Ruin**: Calculated from edge and bet size
- **Trades to Double**: Expected number of trades to double account

---

## 🎯 Real Calibration

When the system says "85% confidence", what actually happened?

- **Bucket calibration**: Groups heuristic scores into ranges, measures actual win rate
- **Grade calibration**: A+ trades vs D trades actual outcomes
- **Pattern calibration**: Per-pattern measured vs claimed
- **Honest assessment**: Explicit overconfidence/underconfidence detection

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Devcode940/forex-chart-analyzer.git
cd forex-chart-analyzer

# Create and activate virtual environment (prevents pip crashes)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

Then open your browser to `http://localhost:8501`

> **Why venv?** Installing directly with system pip can cause crashes on some platforms (especially Termux/macOS). The venv isolates all dependencies cleanly.

---

## 📱 How to Use

1. **Take a screenshot** of your forex chart from MT4/MT5, TradingView, or any platform
2. **Upload** the image via the sidebar
3. **Configure** pair, account balance, and risk percentage
4. **Review** the automated analysis across **13 tabs**:
   - 🧩 Geometric Patterns — 10+ chart patterns with confidence
   - 🕯️ Candlestick Patterns — 17 candlestick signals
   - 📐 Structure — Trend, swings, BOS, market phases
   - 📊 S/R & Fibonacci — Support/resistance + Fibonacci levels
   - 📉 Divergences — Regular & hidden divergences
   - 💧 Liquidity — Buy/sell-side pools, sweeps, equal levels
   - 🎯 SL/TP & Risk — Multiple scenarios with R:R and position sizing
   - 🌐 Regime & Session — Market regime + session context
   - 📊 Confluence Score — Compounded signal score + trade plan
   - 🔬 Statistical Validation — Monte Carlo, Bootstrap, Entropy, Markov
   - 🤖 ML Engine — Full ML pipeline (features, ensemble, anomaly, walk-forward, calibration, meta-learner)
   - 🗄️ Real Backtest DB — Measured vs claimed win rates + trade journal
   - 📐 Real Kelly — Kelly Criterion from measured data
5. **Download** annotated image, JSON report, or summary text

---

## 🏗️ Architecture

```
forex-analyzer/
├── app.py                          # Main Streamlit application (13 tabs)
├── requirements.txt                # Python dependencies
├── run.sh                          # Launch script
├── README.md                       # This file
├── TECHNICAL_BREAKDOWN.md          # Technical architecture docs
├── CONFIDENCE_EXPLAINED.md         # Confidence scoring explanation
├── analyzers/
│   ├── image_processor.py          # Image preprocessing & color extraction
│   ├── structure_analyzer.py       # Market structure & swing analysis
│   ├── pattern_detector.py         # 10+ chart pattern detection
│   ├── candlestick_detector.py     # 17 candlestick pattern detection
│   ├── sr_detector.py              # Support/Resistance detection (3 methods)
│   ├── fibonacci_calculator.py     # Fibonacci retracements & extensions
│   ├── sl_tp_calculator.py         # SL/TP scenario generation
│   ├── regime_classifier.py        # Market regime classification
│   ├── confluence_engine.py        # Compounded confluence scoring
│   ├── risk_manager.py             # Position sizing & risk management
│   ├── divergence_detector.py      # Regular & hidden divergence
│   ├── indicator_detector.py       # MA, Bollinger, trend line detection
│   ├── liquidity_detector.py       # Liquidity zones & sweep detection
│   ├── session_analyzer.py         # Trading session analysis
│   ├── statistical_validator.py    # Monte Carlo, Bootstrap, Entropy, Markov
│   ├── ml_feature_engineer.py      # 50 ML feature extraction
│   ├── ml_ensemble.py              # RF+GB+Logistic meta ensemble
│   ├── ml_anomaly_detector.py      # Isolation Forest + PCA anomaly
│   ├── ml_walk_forward.py          # Walk-forward cross-validation
│   ├── ml_calibration.py           # Platt + Isotonic calibration
│   ├── ml_meta_learner.py          # Weighted voting meta-learner
│   ├── trade_database.py           # SQLite trade database (1200+ trades)
│   ├── real_backtester.py          # Backtest against DB data
│   ├── real_kelly.py               # Kelly Criterion from measured data
│   └── real_calibrator.py          # Calibration from measured outcomes
├── utils/
│   └── visualizer.py               # PIL-based annotated overlay
└── data/
    └── trade_database.db           # Auto-created SQLite database
```

---

## 📱 Deployment

Full deployment guides for all platforms in **[DEPLOYMENT.md](DEPLOYMENT.md)**:

| Platform | Quick Start |
|----------|-------------|
| **Ubuntu Desktop/Server** | `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |
| **WSL2 (Windows)** | Same as Ubuntu — browser auto-opens |
| **Termux (Android)** | See [DEPLOYMENT.md](DEPLOYMENT.md) for ARM64-specific steps |
| **Docker** | `docker build -t forex-analyzer . && docker run -p 8501:8501 forex-analyzer` |
| **systemd service** | One-command install — see [DEPLOYMENT.md](DEPLOYMENT.md) |

Includes: Nginx reverse proxy + SSL, Docker, systemd, performance tuning, troubleshooting.

---

## ⚠️ Important Disclaimers

1. **Synthetic Data**: The trade database uses realistic synthetic data calibrated to published studies. It is NOT real historical data. Win rates are plausible but not verified against actual broker data.

2. **Educational Purpose**: This tool is for **educational and research purposes only**. Always conduct your own analysis.

3. **No Guarantees**: Past patterns do not guarantee future results. Never risk more than you can afford to lose.

4. **ML Models**: The ML ensemble trains on synthetic data augmented with current heuristic signals. This is NOT the same as training on real historical market data. Treat ML predictions as one additional signal, not gospel.

5. **Kelly Criterion**: Half-Kelly or Quarter-Kelly is strongly recommended. Full Kelly is mathematically optimal but psychologically impossible for most traders.
