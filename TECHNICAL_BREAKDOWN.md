# 🔬 Forex Chart Analyzer Pro v2 — Complete Technical Breakdown

## Every Technology, Algorithm & Method Used

---

## 📦 External Libraries (10)

| Library | Version | Role in the App |
|---------|---------|-----------------|
| **Streamlit** | ≥1.28 | Web UI framework — renders all tabs, charts, forms, file upload |
| **Pillow (PIL)** | ≥10.0 | Image loading, format conversion, overlay drawing with alpha blending |
| **NumPy** | ≥1.24 | Array operations, statistical calculations, matrix math everywhere |
| **OpenCV (cv2)** | ≥4.8 | **Core vision engine**: color segmentation, edge detection, Hough transforms, thresholding |
| **SciPy** | ≥1.11 | Signal processing: `argrelextrema` (swing detection), `gaussian_filter1d` (smoothing), `polyfit` (trend lines) |
| **scikit-learn** | ≥1.3 | **DBSCAN clustering** — groups nearby price levels into S/R zones |
| **Matplotlib** | ≥3.7 | (Available for static charts; Plotly used primarily) |
| **Plotly** | ≥5.15 | Interactive charts: price structure, S/R maps, gauges, radar, bar comparisons |
| **Pandas** | ≥2.0 | Data handling, CSV export support |
| **Requests** | ≥2.31 | HTTP client (future: news API integration) |

---

## 🧠 Algorithms & Methods by Module

### 1. Image Processor (`image_processor.py`)

| Method | Algorithm | What It Does |
|--------|-----------|-------------|
| `preprocess()` | **Canny Edge Detection** | Detects all edges in the chart image (candle boundaries, lines, grid) |
| `preprocess()` | **Gaussian Blur** (5×5 kernel) | Noise reduction before edge detection |
| `preprocess()` | **Adaptive Thresholding** (Gaussian method) | Binarizes image for better line detection in varying brightness |
| `extract_chart_colors()` | **HSV Color Segmentation** | Separates bullish candles (green), bearish (red), indicators (blue/yellow) using HSV ranges |
| `extract_chart_colors()` | **Bitwise Masking** | Creates binary masks per color channel for pixel counting |
| `extract_chart_colors()` | **Ratio Analysis** | `bullish_ratio = green_pixels / (green + red)` → sentiment score |
| `detect_grid_lines()` | **Probabilistic Hough Transform** (`HoughLinesP`) | Detects horizontal and vertical grid lines from edge map |
| `extract_price_series()` | **Column-wise Projection** | Slices image vertically, finds top/bottom of colored regions → approximates OHLC |

### 2. Structure Analyzer (`structure_analyzer.py`)

| Method | Algorithm | What It Does |
|--------|-----------|-------------|
| `_detect_swings()` | **Local Extrema Detection** (`scipy.signal.argrelextrema`) | Finds swing highs (local maxima) and swing lows (local minima) in smoothed price |
| `_determine_trend()` | **Linear Regression Slope** (`numpy.polyfit`) | Fits a line to swing points; slope direction = trend direction |
| `_determine_trend()` | **Normalized Slope** | `slope / mean(y)` — makes trend strength comparable across price levels |
| `_detect_structure_breaks()` | **Break of Structure (BOS) Detection** | Compares consecutive swing points: if current swing low > previous swing high = Bullish BOS |
| `_identify_phases()` | **Sliding Window Classification** | Divides price into segments, classifies each as Markup/Markdown/Accumulation/Distribution by slope & volatility |
| Smoothing | **Gaussian Filter** (`scipy.ndimage.gaussian_filter1d`, σ=3) | Removes noise from price series before swing detection |

### 3. Pattern Detector (`pattern_detector.py`)

| Pattern | Detection Method |
|---------|-----------------|
| **Head & Shoulders** | 3 consecutive swing highs where middle > left ≈ right. Validates shoulder equality (`diff / prominence < 2.0`) |
| **Inverse H&S** | Mirror of H&S: 3 swing lows where middle < left ≈ right |
| **Double Top** | 2 swing highs with `abs(diff) / price_range < 8%` |
| **Double Bottom** | 2 swing lows with `abs(diff) / price_range < 8%` |
| **Ascending Triangle** | Flat resistance (high range < 6% of range) + rising support (positive slope) |
| **Descending Triangle** | Flat support (low range < 6%) + descending highs (negative slope) |
| **Symmetric Triangle** | Converging: highs have negative slope, lows have positive slope |
| **Rising Wedge** | Both slopes positive, but low slope > high slope (converging upward) |
| **Falling Wedge** | Both slopes negative, but high slope < low slope (converging downward) |
| **Bull Flag** | Sharp rise (first ⅓ up) + slight decline (middle ⅓, slope ≈ -0.01 to 0) |
| **Bear Flag** | Sharp drop + slight rise |
| **Channel** | Parallel slopes: `abs(high_slope - low_slope) < abs(avg_slope) * 0.5 + 0.02` |
| **Breakout** | Current price near 85th/15th percentile + momentum in that direction |

### 4. Candlestick Detector (`candlestick_detector.py`)

| Pattern | Detection Logic |
|---------|----------------|
| **Doji** | `body / range < 0.1` (body is <10% of total candle range) |
| **Hammer** | `lower_wick > body * 2` AND `upper_wick < body * 0.5` AND `body_ratio < 0.4` |
| **Inverted Hammer** | `upper_wick > body * 2` AND `lower_wick < body * 0.5` AND bearish |
| **Shooting Star** | `upper_wick > body * 2` AND `lower_wick < body * 0.5` AND bearish + uptrend context |
| **Bullish Engulfing** | Current bullish candle body fully contains previous bearish body |
| **Bearish Engulfing** | Current bearish body fully contains previous bullish body |
| **Morning Star** | Bearish → small body (ratio < 0.3) → bullish closing above midpoint of first candle |
| **Evening Star** | Bullish → small body → bearish closing below midpoint |
| **Bullish Pin Bar** | `lower_wick / range > 0.65` AND `body_ratio < 0.25` AND bullish |
| **Bearish Pin Bar** | `upper_wick / range > 0.65` AND `body_ratio < 0.25` AND bearish |
| **Three White Soldiers** | 3 consecutive bullish candles with progressive higher closes and opens within prior body |
| **Three Black Crows** | 3 consecutive bearish candles with progressive lower closes |
| **Spinning Top** | `body_ratio < 0.2` AND `abs(upper_wick - lower_wick) / range < 0.2` |
| **Bullish Marubozu** | `body_ratio > 0.85` AND `(upper_wick + lower_wick) / range < 0.15` AND bullish |
| **Bearish Marubozu** | Same as above but bearish |
| **Tweezer Top** | 2 candles with matching highs (`abs(diff) / range < 3%`), first bullish, second bearish |
| **Tweezer Bottom** | 2 candles with matching lows, first bearish, second bullish |

### 5. S/R Detector (`sr_detector.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Swing-based** | Local extrema from `argrelextrema` | Each swing high = resistance candidate, each swing low = support candidate |
| **Cluster-based** | **DBSCAN Clustering** (`sklearn.cluster.DBSCAN`, eps=0.03, min_samples=adaptive) | Groups nearby price levels into zones. Levels that cluster together form stronger S/R |
| **Density-based** | **Histogram Peak Detection** | Bins price into 50 bins, finds bins above `mean + std` threshold — where price spent most time |
| **Consolidation** | **Tolerance-based Merging** | Merges levels within 3% of each other. Strength = base strength + 10% per merge (confluence bonus) |
| **Confluence** | **Cross-method validation** | Where support and resistance overlap within 2% → confluence zone |

### 6. Fibonacci Calculator (`fibonacci_calculator.py`)

| Method | Details |
|--------|---------|
| **Swing point identification** | Uses highest swing high and lowest swing low as Fib anchors |
| **Retracement calculation** | `level = high - range × ratio` for uptrend, `low + range × ratio` for downtrend |
| **Ratios used** | 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100% |
| **Extension calculation** | `level = high + range × ratio` (up) or `low - range × ratio` (down) |
| **Extension ratios** | 127.2%, 141.4%, 161.8%, 200%, 261.8% |
| **Golden Zone** | 61.8%–78.6% zone — highest-probability reversal area |
| **Confluence detection** | If 2+ key Fib levels within 5% of each other → confluence zone |
| **Fallback** | If no swing points detected, uses price series max/min directly |

### 7. Divergence Detector (`divergence_detector.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Momentum approximation** | **RSI-like Oscillator** (14-period) | Calculates smoothed average gains/losses → `100 - (100/(1+RS))` |
| **Rate of Change** | 5-period ROC | `(price[i] - price[i-5]) / abs(price[i-5]) × 100` |
| **Regular Bearish Divergence** | Price makes Higher High + Momentum makes Lower High | Bearish reversal signal |
| **Regular Bullish Divergence** | Price makes Lower Low + Momentum makes Higher Low | Bullish reversal signal |
| **Hidden Bullish Divergence** | Price makes Higher Low + Momentum makes Lower Low | Bullish continuation |
| **Hidden Bearish Divergence** | Price makes Lower High + Momentum makes Higher High | Bearish continuation |
| **Swing matching** | Index-based alignment | Matches price swing indices with momentum swing indices in same time window |

### 8. Indicator Detector (`indicator_detector.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **MA detection** | **HSV Color Segmentation** | Yellow → Fast MA (20 SMA), Blue → Medium (50 SMA), Red → Slow (200 SMA) |
| **Bollinger Band detection** | Light blue/gray HSV range | Paired parallel lines with middle line |
| **Line point extraction** | **Column-wise Projection** | Divides mask into 100 vertical slices, finds center of colored pixels per slice |
| **Trend line detection** | **Hough Transform** (`HoughLinesP`) | Detects lines with angle between 10°–80° (not grid lines) |
| **Horizontal line detection** | **Hough Transform** (horizontal) + **Color validation** | Distinguishes colored manual lines from gray grid by checking RGB values |
| **MA Crossover detection** | **Point-by-point intersection** | Compares fast MA y-values vs slow MA y-values across slices; direction change = crossover |
| **Golden Cross** | Fast MA crosses above Slow MA | Bullish signal |
| **Death Cross** | Fast MA crosses below Slow MA | Bearish signal |

### 9. Liquidity Detector (`liquidity_detector.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Buy-side liquidity** | Above all swing highs + resistance | Where buy stops cluster |
| **Sell-side liquidity** | Below all swing lows + support | Where sell stops cluster |
| **Equal Highs/Lows** | Pairwise comparison with 2% tolerance | Multiple touches at same level = strongest liquidity magnet |
| **Liquidity Sweep Detection** | Price goes above prior high then reverses below within 3 bars | Confirms stop hunt occurred |
| **Pending Stop Hunt Zones** | Liquidity levels NOT yet swept | Targets for smart money to hunt next |
| **Summary interpretation** | Rule-based logic | Compares buy-side vs sell-side count, checks for recent sweeps |

### 10. Session Analyzer (`session_analyzer.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Volatility calculation** | **Standard deviation of returns** | `std(diff(price) / price)` → LOW/MODERATE/HIGH/VERY_HIGH |
| **Range expansion** | **Half-range ratio** | `range_2nd_half / range_1st_half` → EXPANDING/CONTRACTING/STABLE |
| **Session inference** | Rule-based mapping | Low vol + contracting = Asian; High vol + expanding = London/NY Overlap |
| **Pair-specific advice** | Lookup table | Each pair has best/second-best session to trade |
| **Transition alerts** | Time-based rules | Upcoming session change warnings |

### 11. Regime Classifier (`regime_classifier.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Trend strength** | **R² (Coefficient of Determination)** | `1 - SS_res/SS_tot` from linear regression of price vs time |
| **Volatility** | **Normalized standard deviation of returns** | `std(returns)` where returns = `diff(price)/price` |
| **Efficiency Ratio** | **Kaufman ER** | `abs(net_change) / sum(abs(bar_changes))` — measures how efficiently price moved |
| **ADX Approximation** | **Directional Movement Index** | `abs(avg_up - avg_down) / (avg_up + avg_down)` over 14-period window |
| **Regime classification** | **Rule-based** using ER + ADX | ER > 0.4 + ADX > 0.5 = TRENDING; ER < 0.25 + ADX < 0.3 = RANGING |
| **Trading style mapping** | Lookup per regime | Trending → trend following; Ranging → mean reversion |

### 12. Confluence Engine (`confluence_engine.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Signal collection** | 6 source categories | Patterns (1.5×), Candlesticks (1.0×), S/R (1.2×), Fib (1.3-1.8×), Structure (2.0×), Regime (0.5-1.3×) |
| **Score compounding** | **`1 - ∏(1 - p_i)`** | **NOT averaging** — compounds probabilities. 5 weak signals at 20% each = 67% combined, not 20% |
| **Counter-signal penalty** | 10% cross-bleed | Bullish signal adds 10% of its strength to bear (represents uncertainty) |
| **Grade assignment** | Threshold mapping | >75% = A+, >65% = A, >55% = B, >45% = C, else D |
| **Trade plan generation** | Matched scenario + entry trigger | Finds best SL/TP matching confluence direction + identifies candlestick entry trigger |
| **Position advice** | Kelly-fraction-based | Grade A+ → 2% risk; Grade C → 0.5% risk |

### 13. Risk Manager (`risk_manager.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **Position sizing** | `lots = risk_amount / (SL_pips × pip_value)` | Standard forex lot formula |
| **Pip value lookup** | Pair-specific table | EURUSD=10, USDJPY=6.5, XAUUSD=1 per standard lot per pip |
| **Win rate estimation** | Base 45% + adjustments | +5% for clear bias, +5% for trend alignment, +5% × strength |
| **Risk of Ruin** | **Simplified RoR formula** | `(q/p)^(1/(risk% × avg_win/avg_loss))` where p=win_rate, q=1-p |
| **Kelly Criterion** | `kelly = p - q/avg_win` | Optimal bet size; we recommend half-Kelly |
| **Max Drawdown** | **Expected consecutive losses** × risk% | `consecutive = 1/(1-win_rate)`, max ≈ 3× average |
| **Recovery calculation** | `1/(1-DD%) - 1` | E.g., 20% DD needs 25% gain to recover |
| **Lot rounding** | Standard increments | ≥1 lot: 0.1 precision; ≥0.1: 0.01; micro: 0.001 |

### 14. Visualizer (`visualizer.py`)

| Method | Algorithm | Details |
|--------|-----------|---------|
| **S/R lines** | Dashed line drawing (8px dash, 4px gap) | Alpha-scaled by S/R strength |
| **Pattern zones** | Semi-transparent rectangles | Color-coded by bullish/bearish/neutral |
| **Swing markers** | Triangle polygons | ▲ for highs (red), ▽ for lows (green) |
| **BOS markers** | Circle outlines + labels | Bullish=BOS (aqua), Bearish=BOS (orange-red) |
| **Regime panel** | Semi-transparent overlay (RGBA) | Top-right corner with all regime info |
| **Summary panel** | RGBA drawing | Bottom-left with pattern count, S/R, bias |
| **SL/TP box** | Semi-transparent info box | Bottom of image with entry/SL/TP/R:R |
| **All text** | PIL `ImageDraw` with alpha channel | Anti-aliased text on dark background panels |

### 15. Main App (`app.py`)

| Feature | Technology |
|---------|-----------|
| **File upload** | Streamlit `file_uploader` with type validation |
| **9 analysis tabs** | Streamlit `st.tabs()` |
| **Interactive charts** | Plotly `go.Figure`, `go.Indicator` (gauges), `go.Scatterpolar` (radar) |
| **Metric cards** | Custom HTML/CSS with inline styles |
| **Confluence bars** | CSS flex layout with dynamic width fills |
| **Export** | PIL → PNG bytes, JSON serialization, plain text |
| **Session state** | Streamlit `st.session_state` for persistence across reruns |
| **Responsive layout** | `st.columns()` with proportional widths |

---

## 📊 Data Flow Pipeline

```
Upload Image (PNG/JPG)
        │
        ▼
┌─────────────────────────┐
│  1. IMAGE PROCESSOR     │ ← OpenCV: Canny, Gaussian Blur, Adaptive Threshold
│  - Canny Edge Detection │   HSV Segmentation: Green/Red/Blue/Yellow masks
│  - Color Segmentation   │   Column Projection: Extract price series
│  - Grid Detection       │   HoughLinesP: Detect grid lines
│  - Price Extraction     │
└────────┬────────────────┘
         │ price_series, color_info, edges
         ▼
┌─────────────────────────┐
│  2. STRUCTURE ANALYZER  │ ← SciPy: argrelextrema, gaussian_filter1d
│  - Swing H/L Detection  │   NumPy: polyfit (trend slope)
│  - Trend Direction      │   Custom: BOS detection, Phase classification
│  - Break of Structure   │
│  - Market Phases        │
└────────┬────────────────┘
         │ swing_highs, swing_lows, price_series
         ▼
┌───────────┬───────────┬───────────┬───────────┬──────────────┐
│ 3. PATTERN│ 4. CANDLE │ 5. S/R    │ 6. DIVERG │ 7. INDICATOR │
│ DETECTOR  │ DETECTOR  │ DETECTOR  │ DETECTOR  │ DETECTOR     │
│           │           │           │           │              │
│ Swing +   │ OHLC from │ 3 Methods:│ RSI-like  │ HSV Color +  │
│ Slope     │ price     │ Swing,    │ Oscillator│ Hough +      │
│ analysis  │ series    │ DBSCAN,   │ + Price   │ Column       │
│           │           │ Density   │ swings    │ Projection   │
│ 10 patterns│ 17 patterns│ Merged   │ 4 types   │ MAs, BB,    │
│           │           │ + Confl.  │           │ Trend lines  │
└─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴──────┬───────┘
      │           │           │           │            │
      └─────┬─────┴─────┬─────┘           │            │
            │           │                 │            │
            ▼           ▼                 ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│  8. FIBONACCI CALCULATOR                                    │
│  - Uses swing points as anchors                             │
│  - Calculates 7 retracement + 5 extension levels            │
│  - Identifies Golden Zone (61.8%-78.6%)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  9. REGIME CLASSIFIER                                       │
│  - Kaufman ER + ADX + R² + Volatility → Regime label        │
│  - Trending / Ranging / Volatile / Transitional             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  10. SESSION ANALYZER                                       │
│  - Infers session from volatility + range expansion          │
│  - Provides pair-specific session recommendations            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  11. SL/TP CALCULATOR                                       │
│  - 7 scenario types: Pattern, S/R, Structure, ATR,          │
│    Conservative, Aggressive                                  │
│  - Each with Entry, SL, TP, Risk:Reward                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  12. LIQUIDITY DETECTOR                                     │
│  - Buy/sell-side pools from swing structure                  │
│  - Equal highs/lows detection                                │
│  - Liquidity sweep confirmation                              │
│  - Pending stop hunt zones                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  13. CONFLUENCE ENGINE ← THE BRAIN                          │
│  - Collects ALL signals (6 categories, weighted)             │
│  - COMPOUNDS: 1 - ∏(1 - pᵢ) — not averaging!               │
│  - Assigns Grade A+ to D                                    │
│  - Generates full trade execution plan                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  14. RISK MANAGER                                           │
│  - Lot sizing: risk_amount / (SL_pips × pip_value)          │
│  - Risk of Ruin: (q/p)^(1/(risk×W/L))                      │
│  - Kelly Criterion: p - q/avg_win                           │
│  - Max Drawdown estimate                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  15. VISUALIZER                                             │
│  - PIL ImageDraw with RGBA alpha blending                    │
│  - S/R dashed lines, Pattern zones, Swing markers,          │
│    BOS circles, Regime panel, SL/TP box, Summary panel      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  STREAMLIT UI (app) │
              │  9 Interactive Tabs  │
              │  + Export (PNG/JSON) │
              └─────────────────────┘
```

---

## 🔑 Key Mathematical Formulas

| Formula | Usage |
|---------|-------|
| `1 - ∏(1 - pᵢ)` | Confluence score compounding |
| `1 - SS_res / SS_tot` | R² trend strength |
| `abs(Δnet) / Σ(abs(Δbar))` | Kaufman Efficiency Ratio |
| `abs(avg_up - avg_down) / (avg_up + avg_down)` | ADX approximation |
| `lots = risk$ / (SL_pips × pip_value)` | Position sizing |
| `(q/p)^(1/(risk% × W/L))` | Risk of Ruin |
| `kelly = p - q/avg_win` | Kelly Criterion |
| `100 - 100/(1 + RS)` where `RS = avg_gain/avg_loss` | RSI-like oscillator |
| `σ(returns)` | Volatility measurement |
| `Σ(hits) / Σ(total)` | Support/Resistance strength score |

---

## 📥 Input → Output Summary

| Input | Output |
|-------|--------|
| Chart image (PNG/JPG/WEBP) | Confluence Grade (A+ to D) |
| | 10+ geometric patterns with confidence |
| | 17 candlestick patterns with implications |
| | 7 Fibonacci retracement levels |
| | 5+ Fibonacci extension targets |
| | Support & Resistance zones (3 methods) |
| | 4 divergence types |
| | Liquidity pools + sweep alerts |
| | Market regime + session context |
| | 7+ SL/TP scenarios with R:R |
| | Position sizing (lots) + Risk of Ruin |
| | Full trade execution plan (10 steps) |
| | Annotated overlay image |
| | JSON analysis report |
