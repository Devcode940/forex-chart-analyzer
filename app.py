"""
Forex Chart Analyzer — Main Application
Upload forex chart images for pattern recognition, structure analysis,
regime classification, divergence detection, Fib levels, liquidity zones,
confluence scoring, risk management, and SL/TP recommendations.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from analyzers.image_processor import ImageProcessor
from analyzers.structure_analyzer import StructureAnalyzer
from analyzers.pattern_detector import PatternDetector
from analyzers.sr_detector import SRDetector
from analyzers.sl_tp_calculator import SLTPCalculator
from analyzers.regime_classifier import RegimeClassifier
from analyzers.candlestick_detector import CandlestickDetector
from analyzers.fibonacci_calculator import FibonacciCalculator
from analyzers.confluence_engine import ConfluenceEngine
from analyzers.risk_manager import RiskManager
from analyzers.divergence_detector import DivergenceDetector
from analyzers.indicator_detector import IndicatorDetector
from analyzers.liquidity_detector import LiquidityDetector
from analyzers.session_analyzer import SessionAnalyzer
from analyzers.statistical_validator import StatisticalValidator
from analyzers.ml_feature_engineer import FeatureEngineer
from analyzers.ml_ensemble import MLEnsemble
from analyzers.ml_anomaly_detector import MLAnomalyDetector
from analyzers.ml_walk_forward import WalkForwardValidator
from analyzers.ml_calibration import CalibrationEngine
from analyzers.ml_meta_learner import MetaLearner
from analyzers.trade_database import TradeDatabase
from analyzers.real_backtester import RealBacktester
from analyzers.real_kelly import RealKellyCalculator
from analyzers.real_calibrator import RealCalibrator
from utils.visualizer import Visualizer

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Chart Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.3rem;
    }
    .sub-header { text-align: center; color: #8899aa; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a3a5e; border-radius: 12px; padding: 1.1rem;
        text-align: center; transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #3a7bd5; }
    .pattern-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin: 3px;
    }
    .bullish { background: #1a3a2a; color: #00ff88; border: 1px solid #00ff88; }
    .bearish { background: #3a1a1a; color: #ff4444; border: 1px solid #ff4444; }
    .neutral { background: #3a3a1a; color: #ffdd44; border: 1px solid #ffdd44; }
    .scenario-card {
        background: #1a1a2e; border: 1px solid #2a3a5e;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .confidence-bar { height: 6px; border-radius: 3px; background: #2a2a3e; }
    .confidence-fill { height: 100%; border-radius: 3px; }
    .grade-a { color: #00ff88; font-size: 2.5rem; font-weight: 900; }
    .grade-b { color: #88ff44; font-size: 2.5rem; font-weight: 900; }
    .grade-c { color: #ffdd44; font-size: 2.5rem; font-weight: 900; }
    .grade-d { color: #ff4444; font-size: 2.5rem; font-weight: 900; }
    .fib-zone { border-left: 3px solid; padding-left: 10px; margin: 5px 0; }
    .liquidity-zone { padding: 8px; border-radius: 8px; margin: 4px 0; }
    .step-box { background: #1a1a2e; border-left: 3px solid #3a7bd5; padding: 8px 12px; margin: 4px 0; border-radius: 0 6px 6px 0; }
    div[data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg, #0d1117, #161b22); }
</style>
""", unsafe_allow_html=True)

# ─── Init State ────────────────────────────────────────────────────────────────
def init_state():
    keys = {
        "analysis_complete": False,
        "image_processor": ImageProcessor(),
        "structure_analyzer": StructureAnalyzer(),
        "pattern_detector": PatternDetector(),
        "sr_detector": SRDetector(),
        "sltp_calculator": SLTPCalculator(),
        "regime_classifier": RegimeClassifier(),
        "candlestick_detector": CandlestickDetector(),
        "fibonacci_calculator": FibonacciCalculator(),
        "confluence_engine": ConfluenceEngine(),
        "risk_manager": RiskManager(),
        "divergence_detector": DivergenceDetector(),
        "indicator_detector": IndicatorDetector(),
        "liquidity_detector": LiquidityDetector(),
        "session_analyzer": SessionAnalyzer(),
        "statistical_validator": StatisticalValidator(),
        "feature_engineer": FeatureEngineer(),
        "ml_ensemble": MLEnsemble(),
        "ml_anomaly_detector": MLAnomalyDetector(),
        "ml_walk_forward": WalkForwardValidator(),
        "ml_calibration": CalibrationEngine(),
        "ml_meta_learner": MetaLearner(),
        "trade_database": TradeDatabase(),
        "real_backtester": RealBacktester(),
        "real_kelly": RealKellyCalculator(),
        "real_calibrator": RealCalibrator(),
        "visualizer": Visualizer(),
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Analysis Pipeline ─────────────────────────────────────────────────────────
def run_analysis(image: np.ndarray, account_balance: float, risk_pct: float, pair: str):
    """Run the complete v2 analysis pipeline."""
    processor = st.session_state.image_processor

    with st.spinner("🔄 Processing image..."):
        preprocessed = processor.preprocess(image)
        st.session_state.preprocessed = preprocessed
        color_info = processor.extract_chart_colors(image)
        st.session_state.color_info = color_info
        grid_info = processor.detect_grid_lines(preprocessed["edges"])
        st.session_state.grid_info = grid_info
        green_points = processor.extract_price_series(image, color_info["green_mask"])
        red_points = processor.extract_price_series(image, color_info["red_mask"])
        all_points = green_points + red_points
        all_points.sort(key=lambda p: p["x"])
        st.session_state.price_points = all_points

    with st.spinner("📐 Analyzing market structure..."):
        structure = st.session_state.structure_analyzer.analyze(all_points, image.shape[0]) if all_points else {}
        st.session_state.structure_results = structure

    with st.spinner("🔍 Detecting geometric patterns..."):
        patterns = st.session_state.pattern_detector.detect_all(
            structure.get("price_series", {}), image.shape[0]
        ) if structure.get("price_series") else []
        st.session_state.pattern_results = patterns

    with st.spinner("🕯️ Detecting candlestick patterns..."):
        candlesticks = st.session_state.candlestick_detector.detect_all(
            structure.get("price_series", {})
        ) if structure.get("price_series") else []
        st.session_state.candlestick_results = candlesticks

    with st.spinner("📊 Identifying support & resistance..."):
        sr = st.session_state.sr_detector.detect(
            structure.get("price_series", {}), image.shape[0]
        ) if structure.get("price_series") else {"support": [], "resistance": []}
        st.session_state.sr_results = sr

    with st.spinner("🌐 Classifying market regime..."):
        regime = st.session_state.regime_classifier.classify(
            structure.get("price_series", {}), structure
        ) if structure.get("price_series") else {"regime": "UNKNOWN"}
        st.session_state.regime_results = regime

    with st.spinner("📈 Calculating Fibonacci levels..."):
        fib = st.session_state.fibonacci_calculator.calculate(
            structure, None
        ) if structure else {}
        st.session_state.fib_results = fib

    with st.spinner("📉 Detecting divergences..."):
        divergences = st.session_state.divergence_detector.detect_all(
            structure.get("price_series", {}), structure
        ) if structure.get("price_series") else []
        st.session_state.divergence_results = divergences

    with st.spinner("🎯 Detecting indicator lines..."):
        indicators = st.session_state.indicator_detector.detect_all(
            image, preprocessed.get("edges", np.zeros((1,1)))
        )
        st.session_state.indicator_results = indicators
        # Detect MA crossovers if MAs found
        st.session_state.ma_crossovers = st.session_state.indicator_detector.get_ma_crossovers(indicators)

    with st.spinner("💧 Detecting liquidity zones..."):
        liquidity = st.session_state.liquidity_detector.detect_all(
            structure.get("price_series", {}), structure, sr
        ) if structure.get("price_series") else {}
        st.session_state.liquidity_results = liquidity

    with st.spinner("⏰ Analyzing session context..."):
        session = st.session_state.session_analyzer.analyze(
            structure.get("price_series", {}), regime, pair
        ) if structure.get("price_series") else {}
        st.session_state.session_results = session

    with st.spinner("💰 Calculating SL/TP levels..."):
        sltp = st.session_state.sltp_calculator.calculate(
            patterns, sr, structure, structure.get("price_series", {})
        ) if structure.get("price_series") else {"scenarios": [], "bias": "NEUTRAL"}
        st.session_state.sltp_results = sltp

    with st.spinner("🎯 Computing confluence score..."):
        confluence = st.session_state.confluence_engine.analyze(
            patterns, candlesticks, sr, fib, structure, regime, sltp
        )
        st.session_state.confluence_results = confluence

    with st.spinner("⚖️ Calculating risk management..."):
        risk = st.session_state.risk_manager.calculate(
            sltp, sr, structure, image.shape[0],
            account_balance, risk_pct, pair
        )
        st.session_state.risk_results = risk

    with st.spinner("🎨 Creating visual overlay..."):
        annotated = st.session_state.visualizer.create_full_overlay(
            image, patterns, sr, structure, sltp, regime
        )
        st.session_state.annotated_image = annotated

    st.session_state.analysis_complete = True

# ─── MAIN UI ───────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🔍 Forex Chart Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pattern Recognition • Candlestick Analysis • Fibonacci • Divergence • Liquidity Zones • Confluence Scoring • Risk Management</p>', unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📤 Upload Chart")
    uploaded_file = st.file_uploader(
        "Upload a forex chart image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Upload a screenshot from MT4/MT5, TradingView, or any platform"
    )

    st.markdown("---")
    st.markdown("## ⚙️ Analysis Settings")
    confidence_threshold = st.slider("Min. Confidence", 0.3, 0.9, 0.4, 0.05)
    show_overlay = st.checkbox("Show Annotated Overlay", value=True)
    show_price_chart = st.checkbox("Show Extracted Price Chart", value=True)

    st.markdown("---")
    st.markdown("## 💰 Risk Settings")
    account_balance = st.number_input("Account Balance ($)", value=10000.0, min_value=100.0, step=500.0)
    risk_pct = st.selectbox("Risk Per Trade", [0.25, 0.5, 1.0, 1.5, 2.0], index=2)
    pair = st.selectbox("Currency Pair", [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD",
        "EURJPY", "GBPJPY", "AUDJPY", "XAUUSD"
    ])

    st.markdown("---")
    st.markdown("## 📖 Quick Guides")
    with st.expander("🧩 Pattern Guide"):
        for name, desc in [
            ("Head & Shoulders", "🐻 Bearish reversal. Neckline break = sell."),
            ("Inv. Head & Shoulders", "🐂 Bullish reversal. Neckline break = buy."),
            ("Double Top/Bottom", "🐻/🐂 Reversal at same price level."),
            ("Rising/Falling Wedge", "🐻/🐂 Reversal despite appearance."),
            ("Bull/Bear Flag", "🐂/🐻 Continuation after consolidation."),
            ("Doji", "⚖️ Indecision. Wait for confirmation."),
            ("Hammer", "🐂 Bullish reversal after downtrend."),
            ("Engulfing", "🐂/🐻 Strong momentum shift."),
            ("Pin Bar", "🐂/🐻 Long wick rejection signal."),
        ]:
            st.markdown(f"**{name}**: {desc}")

    with st.expander("📉 Divergence Guide"):
        st.markdown("""
        **Regular Bearish** 🐻: Price HH + Oscillator LH → Sell signal
        **Regular Bullish** 🐂: Price LL + Oscillator HL → Buy signal
        **Hidden Bullish** 🐂: Price HL + Oscillator LL → Buy continuation
        **Hidden Bearish** 🐻: Price LH + Oscillator HH → Sell continuation
        """)

    with st.expander("💧 Liquidity Guide"):
        st.markdown("""
        **Buy-side liquidity** = Above swing highs (buy stops cluster)
        **Sell-side liquidity** = Below swing lows (sell stops cluster)
        **Stop hunt** = Price sweeps a zone then reverses
        **Equal highs/lows** = Strongest liquidity magnets
        """)

    st.markdown("---")
    st.caption("⚠️ Educational only. Always manage risk properly.")

# ─── Main Content ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
    processor = st.session_state.image_processor
    image = processor.load_image(uploaded_file)
    run_analysis(image, account_balance, risk_pct, pair)

    if st.session_state.analysis_complete:
        regime = st.session_state.regime_results
        color_info = st.session_state.color_info
        structure = st.session_state.structure_results
        patterns = st.session_state.pattern_results
        candlesticks = st.session_state.candlestick_results
        sr = st.session_state.sr_results
        sltp = st.session_state.sltp_results
        fib = st.session_state.fib_results
        confluence = st.session_state.confluence_results
        risk = st.session_state.risk_results
        divergences = st.session_state.divergence_results
        indicators = st.session_state.indicator_results
        liquidity = st.session_state.liquidity_results
        session = st.session_state.session_results
        ma_crossovers = st.session_state.ma_crossovers

        # ── Confluence Master Score (TOP) ──
        master = confluence.get("master", {})
        grade = master.get("grade", "D")
        grade_css = {"A+": "grade-a", "A": "grade-a", "B": "grade-b", "C": "grade-c", "D": "grade-d"}

        col_conf1, col_conf2, col_conf3, col_conf4 = st.columns([1, 1, 1, 2])

        with col_conf1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#8899aa;">CONFLUENCE GRADE</div>
                <div class="{grade_css.get(grade, 'grade-d')}">{grade}</div>
                <div style="font-size:0.7rem;color:#6677aa;">{master.get('direction', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_conf2:
            bias = sltp.get("bias", "NEUTRAL")
            bias_color = "#00ff88" if bias == "BULLISH" else "#ff4444" if bias == "BEARISH" else "#ffdd44"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#8899aa;">BIAS</div>
                <div style="font-size:1.1rem;font-weight:700;color:{bias_color};">
                    {'🐂' if bias=='BULLISH' else '🐻' if bias=='BEARISH' else '⚖️'} {bias}
                </div>
                <div style="font-size:0.7rem;color:#6677aa;">{confluence.get('signal_count', 0)} signals</div>
            </div>
            """, unsafe_allow_html=True)

        with col_conf3:
            trend = structure.get("trend_direction", "N/A")
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#8899aa;">TREND</div>
                <div style="font-size:1.1rem;font-weight:700;color:{'#00ff88' if trend=='UPTREND' else '#ff4444' if trend=='DOWNTREND' else '#ffdd44'};">
                    {trend}
                </div>
                <div style="font-size:0.7rem;color:#6677aa;">{regime.get('regime', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_conf4:
            # Confluence bar
            bull = confluence.get("bull_score", 0)
            bear = confluence.get("bear_score", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#8899aa;">SIGNAL STRENGTH</div>
                <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                    <span style="color:#00ff88;font-weight:700;">🐂 {bull:.0%}</span>
                    <div style="flex:1;height:8px;background:#2a2a3e;border-radius:4px;overflow:hidden;">
                        <div style="width:{bull*100}%;background:#00ff88;float:left;height:100%;"></div>
                        <div style="width:{bear*100}%;background:#ff4444;float:right;height:100%;"></div>
                    </div>
                    <span style="color:#ff4444;font-weight:700;">🐻 {bear:.0%}</span>
                </div>
                <div style="font-size:0.7rem;color:#6677aa;">{master.get('strength_description', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Images ──
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("### 📷 Original Chart")
            st.image(image, use_container_width=True)
        with col_right:
            if show_overlay and st.session_state.annotated_image is not None:
                st.markdown("### 🎯 Annotated Analysis")
                st.image(st.session_state.annotated_image, use_container_width=True)

        # ── Trade Execution Plan (if actionable) ──
        trade_plan = confluence.get("trade_plan", {})
        if trade_plan.get("action") != "DO NOT TRADE":
            st.markdown("---")
            st.markdown("## Trade Execution Plan")

            plan_direction = trade_plan.get("action", "")
            plan_grade = trade_plan.get("confluence_grade", "")
            steps = trade_plan.get("execution_steps", [])
            confluence_factors = trade_plan.get("confluence_factors", [])
            risk_factors = trade_plan.get("risk_factors", [])
            scenario = trade_plan.get("scenario", {})

            col_plan1, col_plan2 = st.columns([2, 1])

            with col_plan1:
                # Execution steps
                for step in steps:
                    st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

            with col_plan2:
                # Confluence factors
                st.markdown("#### ✅ Aligned Signals")
                for f in confluence_factors[:8]:
                    st.markdown(f"- {f}")

                st.markdown("#### ⚠️ Conflicting Signals")
                for f in risk_factors[:4]:
                    st.markdown(f"- {f}")

                if scenario:
                    st.markdown("#### 📊 Key Numbers")
                    st.markdown(f"**Entry:** {scenario.get('entry', 'N/A'):.2f}")
                    st.markdown(f"**SL:** {scenario.get('sl', 'N/A'):.2f}")
                    st.markdown(f"**TP:** {scenario.get('tp', 'N/A'):.2f}")
                    st.markdown(f"**R:R:** 1:{scenario.get('risk_reward', 0):.2f}")

        # ── Detailed Tabs ──
        st.markdown("---")
        st.markdown("## 📊 Detailed Analysis")

        tabs = st.tabs([
            "🧩 Geometric Patterns", "🕯️ Candlestick Patterns", "📐 Structure",
            "📊 S/R & Fibonacci", "📉 Divergences", "💧 Liquidity",
            "🎯 SL/TP & Risk", "🌐 Regime & Session", "📊 Confluence Score",
            "🔬 Statistical Validation", "🤖 ML Engine",
            "🗄️ Real Backtest DB", "📐 Real Kelly"
        ])

        # ── TAB: Geometric Patterns ──
        with tabs[0]:
            st.markdown("### Geometric Chart Patterns")
            filtered = [p for p in patterns if p.get("confidence", 0) >= confidence_threshold]
            if filtered:
                badge_html = "<div style='margin-bottom:1rem;'>"
                for p in filtered:
                    pt = p.get("type", "")
                    css = "bullish" if "BULLISH" in pt else "bearish" if "BEARISH" in pt else "neutral"
                    badge_html += f"<span class='pattern-badge {css}'>{p['name']} ({p['confidence']:.0%})</span>"
                badge_html += "</div>"
                st.markdown(badge_html, unsafe_allow_html=True)

                for pattern in filtered:
                    icon = "🟢" if "BULLISH" in pattern.get("type", "") else "🔴" if "BEARISH" in pattern.get("type", "") else "🟡"
                    with st.expander(f"{icon} {pattern['name']} — {pattern['confidence']:.0%}"):
                        st.markdown(f"**Type:** {pattern.get('type', 'N/A')}")
                        st.markdown(f"**Direction:** {pattern.get('target_direction', 'PENDING')}")
                        st.markdown(f"**Description:** {pattern.get('description', 'N/A')}")
            else:
                st.info("No geometric patterns detected above threshold.")

        # ── TAB: Candlestick Patterns ──
        with tabs[1]:
            st.markdown("### Candlestick Patterns")
            if candlesticks:
                # Count by category
                bullish_cs = [c for c in candlesticks if c.get("signal") == "BUY"]
                bearish_cs = [c for c in candlesticks if c.get("signal") == "SELL"]
                neutral_cs = [c for c in candlesticks if c.get("signal") in ["REVERSAL_POSSIBLE", "WAIT"]]

                col_cs1, col_cs2, col_cs3 = st.columns(3)
                with col_cs1:
                    st.metric("🐂 Bullish Candles", len(bullish_cs))
                with col_cs2:
                    st.metric("🐻 Bearish Candles", len(bearish_cs))
                with col_cs3:
                    st.metric("⚖️ Indecision", len(neutral_cs))

                for c in candlesticks:
                    icon = "🟢" if c.get("signal") == "BUY" else "🔴" if c.get("signal") == "SELL" else "🟡"
                    with st.expander(f"{icon} {c['name']} ({c.get('category', '')}) — {c['confidence']:.0%}"):
                        st.markdown(f"**Signal:** {c.get('signal', 'N/A')}")
                        st.markdown(f"**Category:** {c.get('category', 'N/A')}")
                        st.markdown(f"**Description:** {c.get('description', 'N/A')}")
                        st.info(f"💡 **Implication:** {c.get('implication', 'N/A')}")
            else:
                st.info("No candlestick patterns detected.")

        # ── TAB: Structure ──
        with tabs[2]:
            st.markdown("### Market Structure")
            trend_dir = structure.get("trend_direction", "N/A")
            trend_str = structure.get("trend_strength", 0)

            col_st1, col_st2 = st.columns([1, 1])
            with col_st1:
                st.markdown(f"**Trend:** {trend_dir} (strength: {trend_str:.0%})")
                st.markdown(f"**Swing Highs:** {len(structure.get('swing_highs', []))}")
                st.markdown(f"**Swing Lows:** {len(structure.get('swing_lows', []))}")
                for bos in structure.get("structure_breaks", []):
                    icon = "🟢" if "BULLISH" in bos.get("type", "") else "🔴"
                    st.markdown(f"  {icon} **{bos['type']}** at index {bos.get('index', '?')}")

                st.markdown("#### Phases")
                for phase in structure.get("phases", []):
                    st.markdown(f"- **{phase['phase']}** (bars {phase['start']}-{phase['end']}, slope: {phase['slope']:.3f})")

            with col_st2:
                if show_price_chart and structure.get("price_series"):
                    ps = structure["price_series"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ps["x"], y=ps["smoothed"], mode='lines', name='Smoothed', line=dict(color='#3a7bd5', width=2)))
                    fig.add_trace(go.Scatter(x=ps["x"], y=ps["highs"], mode='lines', name='Highs', line=dict(color='rgba(0,255,136,0.3)', width=1)))
                    fig.add_trace(go.Scatter(x=ps["x"], y=ps["lows"], mode='lines', name='Lows', line=dict(color='rgba(255,68,68,0.3)', width=1), fill='tonexty', fillcolor='rgba(58,123,213,0.08)'))
                    for sh in structure.get("swing_highs", []):
                        fig.add_trace(go.Scatter(x=[sh["x"]], y=[sh["value"]], mode='markers', marker=dict(color='#ff4444', size=8, symbol='triangle-down'), showlegend=False))
                    for sl in structure.get("swing_lows", []):
                        fig.add_trace(go.Scatter(x=[sl["x"]], y=[sl["value"]], mode='markers', marker=dict(color='#00ff88', size=8, symbol='triangle-up'), showlegend=False))
                    fig.update_layout(title="Price Structure", height=350, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

        # ── TAB: S/R & Fibonacci ──
        with tabs[3]:
            st.markdown("### Support/Resistance & Fibonacci Levels")
            col_fib1, col_fib2 = st.columns([1, 1])

            with col_fib1:
                st.markdown("#### 📊 Support & Resistance")
                for i, s in enumerate(sr.get("support", [])):
                    st.markdown(f'<div class="fib-zone" style="border-color:#00ff88;"><span style="color:#00ff88;">🟢 Support #{i+1}</span> — Strength: {s.get("strength",0):.0%} — Touches: {s.get("touches",0)}</div>', unsafe_allow_html=True)
                for i, r in enumerate(sr.get("resistance", [])):
                    st.markdown(f'<div class="fib-zone" style="border-color:#ff4444;"><span style="color:#ff4444;">🔴 Resistance #{i+1}</span> — Strength: {r.get("strength",0):.0%} — Touches: {r.get("touches",0)}</div>', unsafe_allow_html=True)

                # Confluence zones
                if sr.get("key_zones"):
                    st.markdown("#### ⚡ Confluence Zones")
                    for z in sr["key_zones"]:
                        st.markdown(f'<div class="fib-zone" style="border-color:#ff8800;"><span style="color:#ff8800;">⚡ Confluence</span> at {z.get("level",0):.1f}</div>', unsafe_allow_html=True)

            with col_fib2:
                st.markdown("#### 📈 Fibonacci Retracement")
                retracements = fib.get("retracements", {})
                trend = fib.get("trend", "RANGING")
                st.markdown(f"*Swing: {fib.get('swing_low',{}).get('value',0):.1f} → {fib.get('swing_high',{}).get('value',0):.1f}*")

                for ratio in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
                    if ratio in retracements:
                        level = retracements[ratio]
                        color = "#ff8800" if level.get("importance") == "HIGH" else "#888888"
                        st.markdown(f'<div class="fib-zone" style="border-color:{color};"><span style="color:{color};">{level["label"]}</span> → <strong>{level["value"]:.2f}</strong> ({level.get("direction","")})</div>', unsafe_allow_html=True)

                # Golden Zone
                golden = fib.get("golden_zone", {})
                if golden:
                    st.markdown("#### 🏆 The Golden Zone")
                    st.success(f"**{golden['name']}**\n\nUpper: {golden['upper']:.2f} | Lower: {golden['lower']:.2f}\n\n{golden['description']}\n\n**Strategy:** {golden['strategy']}")

                # Fib Extensions
                st.markdown("#### 🎯 Fibonacci Extensions (TP Targets)")
                extensions = fib.get("extensions", {})
                for ratio in [1.272, 1.618, 2.0, 2.618]:
                    if ratio in extensions:
                        ext = extensions[ratio]
                        st.markdown(f"**{ext['label']}:** {ext['value']:.2f}")

                # Fib Trade Zones
                trade_zones = fib.get("trade_zones", [])
                if trade_zones:
                    st.markdown("#### 📍 Fib Trade Zones")
                    for tz in trade_zones:
                        st.markdown(f"- **{tz['zone_name']}** @ {tz['price']:.2f} — {tz['action']} ({tz['importance']})")

        # ── TAB: Divergences ──
        with tabs[4]:
            st.markdown("### Divergence Analysis")
            if divergences:
                for div in divergences:
                    icon = "🟢" if "BULLISH" in div.get("type", "") else "🔴"
                    with st.expander(f"{icon} {div['name']} — {div['confidence']:.0%}"):
                        st.markdown(f"**Type:** {div.get('type', 'N/A')}")
                        st.markdown(f"**Signal:** {div.get('signal', 'N/A')}")
                        st.markdown(f"**Description:** {div.get('description', 'N/A')}")
                        st.info(f"💡 **Action:** {div.get('implication', 'N/A')}")
            else:
                st.info("No divergences detected in the current chart.")

            # Show momentum oscillator
            if structure.get("price_series"):
                from analyzers.divergence_detector import DivergenceDetector as DD
                smoothed = np.array(structure["price_series"].get("smoothed", []))
                if len(smoothed) > 14:
                    dd = DD()
                    momentum = dd._calc_momentum_oscillator(smoothed, 14)
                    fig_mom = go.Figure()
                    fig_mom.add_trace(go.Scatter(y=smoothed, mode='lines', name='Price', line=dict(color='#3a7bd5')))
                    fig_mom.add_hline(y=50, line_dash='dash', line_color='#666666')
                    fig_mom.add_trace(go.Scatter(y=momentum, mode='lines', name='Momentum', line=dict(color='#ff88ff')))
                    fig_mom.update_layout(title="Price vs Momentum (Divergence Check)", height=250, template="plotly_dark")
                    st.plotly_chart(fig_mom, use_container_width=True)

        # ── TAB: Liquidity ──
        with tabs[5]:
            st.markdown("### 💧 Liquidity Zone Analysis")
            liq_summary = liquidity.get("summary", {})

            col_l1, col_l2, col_l3, col_l4 = st.columns(4)
            with col_l1:
                st.metric("Buy-Side Pools", liq_summary.get("buy_side_pools", 0))
            with col_l2:
                st.metric("Sell-Side Pools", liq_summary.get("sell_side_pools", 0))
            with col_l3:
                st.metric("Recent Sweeps", liq_summary.get("recent_sweeps", 0))
            with col_l4:
                st.metric("Pending Hunt Zones", liq_summary.get("pending_hunt_zones", 0))

            # Summary interpretation
            if liq_summary.get("interpretation"):
                st.info(f"🧠 **Liquidity Intelligence:** {liq_summary['interpretation']}")

            col_liq1, col_liq2 = st.columns([1, 1])

            with col_liq1:
                st.markdown("#### 🔴 Buy-Side Liquidity (Above Highs)")
                for bs in liquidity.get("buy_side_liquidity", []):
                    st.markdown(f'<div class="liquidity-zone" style="background:#3a1a1a;border:1px solid #ff444466;"><strong style="color:#ff4444;">⬆ {bs["source"]}</strong> at {bs["level"]:.1f}<br><span style="color:#8899aa;font-size:0.8rem;">{bs["description"]}</span></div>', unsafe_allow_html=True)

                st.markdown("#### 🟢 Sell-Side Liquidity (Below Lows)")
                for ss in liquidity.get("sell_side_liquidity", []):
                    st.markdown(f'<div class="liquidity-zone" style="background:#1a3a2a;border:1px solid #00ff8866;"><strong style="color:#00ff88;">⬇ {ss["source"]}</strong> at {ss["level"]:.1f}<br><span style="color:#8899aa;font-size:0.8rem;">{ss["description"]}</span></div>', unsafe_allow_html=True)

            with col_liq2:
                # Recent sweeps (actionable!)
                sweeps = liquidity.get("liquidity_sweeps", [])
                if sweeps:
                    st.markdown("#### ⚡ Recent Liquidity Sweeps (ACTIONABLE)")
                    for sweep in sweeps:
                        color = "#ff4444" if "BUY" in sweep.get("type", "") else "#00ff88"
                        st.markdown(f'<div class="liquidity-zone" style="background:{"#3a1a1a" if "BUY" in sweep.get("type","") else "#1a3a2a"};border:2px solid {color};"><strong style="color:{color};">⚡ {sweep["type"].replace("_"," ")}</strong> at {sweep["level"]:.1f}<br>{sweep["description"]}<br><strong>Action:</strong> {sweep["action"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("#### ⚡ No Recent Sweeps")
                    st.info("No liquidity sweeps detected recently. Pending zones are targets for smart money.")

                # Stop hunt zones
                stop_hunt_zones = liquidity.get("stop_hunt_zones", [])
                if stop_hunt_zones:
                    st.markdown("#### 🎯 Pending Stop Hunt Targets")
                    for zone in stop_hunt_zones:
                        st.warning(f"**Target:** {zone['level']:.1f} — {zone['description']}")

                # Equal levels
                equal_levels = liquidity.get("equal_levels", [])
                if equal_levels:
                    st.markdown("#### 🧲 Equal Highs/Lows (Strongest Magnets)")
                    for eq in equal_levels:
                        st.markdown(f"**{eq['type'].replace('_',' ').title()}:** {eq['level']:.1f} — {eq['description']}")

        # ── TAB: SL/TP & Risk ──
        with tabs[6]:
            st.markdown("### 🎯 SL/TP & Risk Management")

            # Best scenario
            best = sltp.get("best_scenario")
            if best:
                direction = best.get("direction", "BUY")
                st.markdown(f"""
                <div style="background:{'#1a3a2a' if direction=='BUY' else '#3a1a1a'};
                     border:2px solid {'#00ff88' if direction=='BUY' else '#ff4444'};
                     border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
                    <h3 style="color:{'#00ff88' if direction=='BUY' else '#ff4444'};margin:0;">⭐ {best.get('name','')}</h3>
                    <div style="display:flex;gap:1.5rem;margin-top:0.5rem;">
                        <span>📍 Entry: {best.get('entry',0):.2f}</span>
                        <span style="color:#ff44ff;">🛑 SL: {best.get('sl',0):.2f}</span>
                        <span style="color:#44ffff;">🎯 TP: {best.get('tp',0):.2f}</span>
                        <span style="color:#ffdd44;">📊 R:R 1:{best.get('risk_reward',0):.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Position sizing from risk manager
            best_risk = risk.get("best_scenario_risk", {})
            if best_risk:
                st.markdown("#### 💼 Position Sizing")
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("Recommended Lots", f"{best_risk.get('recommended_lots', 0):.2f}")
                with col_r2:
                    st.metric("Risk Amount", f"${best_risk.get('risk_amount_usd', 0):.2f}")
                with col_r3:
                    st.metric("Profit Potential", f"${best_risk.get('profit_potential_usd', 0):.2f}")
                with col_r4:
                    st.metric("SL Distance", f"{best_risk.get('sl_distance_pips', 0):.1f} pips")

            # Risk of ruin
            ror = risk.get("risk_of_ruin", {})
            if ror:
                st.markdown("#### 🏦 Risk of Ruin")
                ror_level = ror.get("level", "UNKNOWN")
                ror_color = {"VERY_LOW": "#00ff88", "LOW": "#88ff44", "MODERATE": "#ffdd44", "HIGH": "#ff8844", "VERY_HIGH": "#ff4444"}.get(ror_level, "#ffffff")
                st.markdown(f"**Level:** <span style='color:{ror_color};font-weight:700;'>{ror_level}</span> ({ror.get('ror', 0):.2%})", unsafe_allow_html=True)
                st.markdown(f"**Kelly %:** {ror.get('kelly_pct', 0):.1f}% — Recommended max risk: {ror.get('recommended_max_risk', 0):.2f}%")
                if ror.get("advice"):
                    st.info(ror["advice"])

            # Max drawdown
            max_dd = risk.get("max_drawdown_estimate", {})
            if max_dd:
                st.markdown("#### 📉 Max Drawdown Estimate")
                st.markdown(f"- **Expected:** {max_dd.get('expected_max_dd_pct', 0):.1f}%")
                st.markdown(f"- **Worst Case:** {max_dd.get('worst_case_dd_pct', 0):.1f}%")
                st.markdown(f"- **Recovery Needed:** {max_dd.get('recovery_needed', 0):.1f}%")

            # All scenarios
            st.markdown("#### All Trading Scenarios")
            for scenario in sltp.get("scenarios", []):
                direction = scenario.get("direction", "BUY")
                rr = scenario.get("risk_reward", 0)
                border = "#00ff88" if direction == "BUY" else "#ff4444"
                rr_color = "#00ff88" if rr >= 2 else "#ffdd44" if rr >= 1 else "#ff4444"
                st.markdown(f"""
                <div class="scenario-card" style="border-left:4px solid {border};">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{border};font-weight:700;">{scenario.get('name','')}</span>
                        <span style="color:{rr_color};font-weight:700;">R:R 1:{rr:.2f}</span>
                    </div>
                    <div style="font-size:0.85rem;color:#8899aa;margin-top:4px;">
                        📍 {scenario.get('entry',0):.2f} | 🛑 {scenario.get('sl',0):.2f} | 🎯 {scenario.get('tp',0):.2f} | 📊 {scenario.get('confidence',0):.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Risk rules
            st.markdown("#### 📋 Professional Risk Rules")
            for rule in risk.get("risk_rules", []):
                st.markdown(f"- {rule}")

        # ── TAB: Regime & Session ──
        with tabs[7]:
            st.markdown("### 🌐 Market Regime & Session Context")

            col_reg1, col_reg2 = st.columns([1, 1])

            with col_reg1:
                # Regime
                regime_name = regime.get("regime", "UNKNOWN")
                regime_icons = {"TRENDING": "📈", "RANGING": "↔️", "VOLATILE": "⚡", "TRANSITIONAL": "🔄"}
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem;">
                    <h3 style="margin:0;">{regime_icons.get(regime_name,'❓')} {regime_name}</h3>
                    <p style="color:#8899aa;">{regime.get('sub_regime','')}</p>
                    <p style="color:#6677aa;">Confidence: {regime.get('confidence',0):.0%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"**Trading Style:** {regime.get('trading_style','N/A')}")
                st.success(f"**Recommendation:** {regime.get('recommendation','N/A')}")

                # Regime indicators
                indicators_data = regime.get("indicators", {})
                if indicators_data:
                    st.markdown("#### Regime Indicators")
                    for key, value in indicators_data.items():
                        st.markdown(f"- **{key.replace('_',' ').title()}:** {value:.3f}")

            with col_reg2:
                # Session
                session_name = session.get("inferred_session", "UNKNOWN")
                session_info = session.get("session_info", {})
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem;">
                    <h3 style="margin:0;">⏰ {session_info.get('name', session_name)}</h3>
                    <p style="color:#8899aa;">{session_info.get('hours', 'N/A')}</p>
                    <p style="color:#6677aa;">{session_info.get('characteristics', '')}</p>
                </div>
                """, unsafe_allow_html=True)

                session_recs = session.get("recommendations", {})
                if session_recs:
                    st.markdown(f"**Approach:** {session_recs.get('approach', 'N/A')}")
                    st.markdown("✅ **Do:**")
                    for do in session_recs.get("do", []):
                        st.markdown(f"  - {do}")
                    st.markdown("🚫 **Avoid:**")
                    for avoid in session_recs.get("avoid", []):
                        st.markdown(f"  - {avoid}")

                # Transition alerts
                for alert in session.get("session_transitions", []):
                    st.warning(alert)

                # Best session for pair
                best_session = session.get("best_session_for_pair", {})
                if best_session:
                    st.info(f"🏆 **Best session for {pair}:** {best_session.get('best', 'N/A')}")

        # ── TAB: Confluence Score ──
        with tabs[8]:
            st.markdown("### 🎯 Confluence Score Engine")

            # Master score visualization
            master = confluence.get("master", {})
            grade = master.get("grade", "D")
            direction = master.get("direction", "NEUTRAL")
            confidence = master.get("confidence", 0)

            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])

            with col_c1:
                grade_css_class = grade_css.get(grade, "grade-d")
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.75rem;color:#8899aa;">CONFLUENCE GRADE</div>
                    <div class="{grade_css_class}">{grade}</div>
                    <div style="color:{'#00ff88' if direction=='BULLISH' else '#ff4444' if direction=='BEARISH' else '#ffdd44'};">{direction}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_c2:
                # Confluence breakdown by source
                breakdown = confluence.get("confluence_breakdown", {})
                st.markdown("#### Signal Breakdown by Source")
                for source, data in breakdown.items():
                    bull = data.get("bullish", 0)
                    bear = data.get("bearish", 0)
                    neut = data.get("neutral", 0)
                    st.markdown(f"**{source}:** 🟢{bull} / 🔴{bear} / 🟡{neut}")

            with col_c3:
                # Confluence gauge
                fig_conf = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#00ff88' if confidence > 0.65 else '#ffdd44' if confidence > 0.45 else '#ff4444'},
                        'steps': [
                            {'range': [0, 40], 'color': '#2a1a1a'},
                            {'range': [40, 65], 'color': '#2a2a1a'},
                            {'range': [65, 100], 'color': '#1a2a1a'}
                        ],
                    },
                    title={'text': f"Confluence Score ({direction})"}
                ))
                fig_conf.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_conf, use_container_width=True)

            # All signals
            st.markdown("#### 📋 All Detected Signals")
            for sig in confluence.get("signals", []):
                dir_color = "#00ff88" if sig["direction"] == "BULLISH" else "#ff4444" if sig["direction"] == "BEARISH" else "#ffdd44"
                dir_icon = "🟢" if sig["direction"] == "BULLISH" else "🔴" if sig["direction"] == "BEARISH" else "🟡"
                st.markdown(f"{dir_icon} **{sig['source']}:** {sig['name']} — Strength: {sig['strength']:.0%} (weight: {sig['weight']:.1f}x)")

            # Trade plan
            if trade_plan.get("action") != "DO NOT TRADE":
                st.markdown("#### 🚀 Execution Plan")
                for step in trade_plan.get("execution_steps", []):
                    st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ {trade_plan.get('reason', 'Conflicting signals — wait for clarity.')}")

            # MA crossovers
            if ma_crossovers:
                st.markdown("#### 📊 MA Crossovers Detected")
                for cross in ma_crossovers:
                    icon = "🟢" if cross.get("signal") == "BULLISH" else "🔴"
                    st.markdown(f"{icon} **{cross['name']}** — {cross['description']}")

        # ── TAB: Statistical Validation ──
        with tabs[9]:
            st.markdown("### 🔬 Statistical Validation — Is Your >85% Confidence Real?")
            st.warning(
                "⚠️ **Note:** The confluence engine uses *heuristic confidence scoring* — "
                "these are expert-system scores, NOT true statistical probabilities. "
                "This tab runs **4 premium statistical methods** to validate whether "
                "your confidence score is genuinely supported by the data."
            )

            if st.button("🧪 Run Full Statistical Validation", type="primary", use_container_width=True):
                validator = st.session_state.statistical_validator
                ps = structure.get("price_series", {})

                if not ps:
                    st.error("No price series data available.")
                else:
                    with st.spinner("🎰 Running Monte Carlo Simulation (2000 paths)..."):
                        mc = validator.monte_carlo_validation(ps, patterns, structure, n_simulations=2000)
                        st.session_state.mc_results = mc

                    with st.spinner("🔄 Running Bootstrap Resampling (5000 samples)..."):
                        bs = validator.bootstrap_confidence_interval(ps, n_bootstrap=5000)
                        st.session_state.bs_results = bs

                    with st.spinner("📐 Calculating Shannon Entropy..."):
                        ent = validator.shannon_entropy_analysis(ps)
                        st.session_state.ent_results = ent

                    with st.spinner("🔗 Running Markov Chain Analysis..."):
                        mk = validator.markov_chain_analysis(ps)
                        st.session_state.mk_results = mk

                    with st.spinner("🏁 Computing Final Verdict..."):
                        audit = validator.full_probability_audit(
                            ps, patterns, structure, confluence,
                            n_simulations=2000, n_bootstrap=5000
                        )
                        st.session_state.audit_results = audit

                    st.success("✅ Statistical validation complete!")

            # Display results if available
            if "audit_results" in st.session_state:
                audit = st.session_state.audit_results
                verdict = audit.get("final_verdict", {})

                # Final Verdict Card
                final_grade = verdict.get("final_grade", "UNKNOWN")
                grade_colors = {
                    "VALIDATED_85+": "#00ff88",
                    "PROBABLE_70-85": "#88ff44",
                    "POSSIBLE_55-70": "#ffdd44",
                    "UNRELIABLE_<55": "#ff4444"
                }
                grade_color = grade_colors.get(final_grade, "#ffffff")

                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                     border:3px solid {grade_color};border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;">
                    <h2 style="color:{grade_color};margin:0;text-align:center;">
                        🏁 FINAL STATISTICAL VERDICT: {final_grade}
                    </h2>
                    <p style="color:#ffffff;text-align:center;margin-top:0.5rem;">
                        {verdict.get('final_verdict', '')}
                    </p>
                    <p style="color:#ffdd44;text-align:center;font-size:0.9rem;margin-top:0.5rem;">
                        {verdict.get('honest_assessment', '')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Method votes
                st.markdown("#### 📊 Method Voting Results")
                votes = verdict.get("method_votes", [])
                for vote in votes:
                    v_color = "#00ff88" if vote["verdict"] == "PASS_85" else "#ffdd44" if vote["verdict"] in ["SIGNIFICANT", "MODERATE"] else "#ff4444"
                    v_icon = "✅" if vote["verdict"] == "PASS_85" else "🟡" if vote["verdict"] in ["SIGNIFICANT", "MODERATE"] else "❌"
                    st.markdown(f'{v_icon} **{vote["method"]}**: {vote["verdict"]} — Probability: {vote["prob"]:.1%}')

                # Detailed results per method
                col_sv1, col_sv2 = st.columns([1, 1])

                with col_sv1:
                    # Monte Carlo
                    mc = st.session_state.get("mc_results", {})
                    if mc and "error" not in mc:
                        with st.expander("🎰 Monte Carlo Simulation", expanded=True):
                            st.markdown(f"**Heuristic Confidence:** {mc.get('pattern_heuristic_confidence', 0):.1%}")
                            st.markdown(f"**Random Baseline:** {mc.get('random_baseline_win_rate', 0):.1%}")
                            st.markdown(f"**Statistical Probability:** {mc.get('statistical_probability', 0):.1%}")
                            st.markdown(f"**P-value:** {mc.get('p_value', 1):.4f}")
                            st.markdown(f"**Statistically Significant:** {'✅ Yes' if mc.get('is_statistically_significant') else '❌ No'}")
                            st.markdown(f"**Exceeds 85%:** {'✅ Yes' if mc.get('exceeds_85_confidence') else '❌ No'}")
                            st.info(mc.get('interpretation', ''))
                            if mc.get('warning'):
                                st.warning(mc['warning'])

                    # Entropy
                    ent = st.session_state.get("ent_results", {})
                    if ent and "error" not in ent:
                        with st.expander("📐 Shannon Entropy Analysis", expanded=True):
                            st.markdown(f"**Entropy:** {ent.get('entropy_bits', 0):.4f} bits (max 1.0)")
                            st.markdown(f"**Predictability:** {ent.get('predictability', 0):.1%}")
                            st.markdown(f"**Mutual Information:** {ent.get('mutual_information', 0):.4f} bits")
                            st.markdown(f"**P(up):** {ent.get('p_up', 0):.1%} | P(down):** {ent.get('p_down', 0):.1%}")
                            st.info(ent.get('interpretation', ''))

                with col_sv2:
                    # Bootstrap
                    bs = st.session_state.get("bs_results", {})
                    if bs and "error" not in bs:
                        with st.expander("🔄 Bootstrap Confidence Intervals", expanded=True):
                            pe = bs.get("point_estimates", {})
                            st.markdown(f"**Win Rate Estimate:** {pe.get('win_rate', 0):.1%}")
                            st.markdown(f"**Trend Strength:** {pe.get('trend_strength', 0):.3f}")

                            ci = bs.get("confidence_intervals", {})
                            for level, data in ci.items():
                                wr_ci = data.get("win_rate", (0, 1))
                                st.markdown(f"**{level.replace('ci_', '')}% CI for Win Rate:** [{wr_ci[0]:.1%}, {wr_ci[1]:.1%}]")

                            st.info(bs.get('interpretation', ''))

                    # Markov
                    mk = st.session_state.get("mk_results", {})
                    if mk and "error" not in mk:
                        with st.expander("🔗 Markov Chain Analysis", expanded=True):
                            st.markdown(f"**Current State:** {mk.get('current_state', '?')}")
                            st.markdown(f"**Persistence Probability:** {mk.get('persistence_probability', 0):.1%}")
                            st.markdown(f"**Next Bar Bullish:** {mk.get('next_bar_bullish_prob', 0):.1%}")
                            st.markdown(f"**Next Bar Bearish:** {mk.get('next_bar_bearish_prob', 0):.1%}")
                            st.markdown(f"**5-Bar Bullish:** {mk.get('five_bar_bullish_prob', 0):.1%}")
                            st.markdown(f"**Can Claim 85% (1-bar):** {'✅' if mk.get('can_claim_85_next_bar') else '❌'}")
                            st.markdown(f"**Can Claim 85% (5-bar):** {'✅' if mk.get('can_claim_85_five_bar') else '❌'}")
                            st.info(mk.get('interpretation', ''))

        # ── TAB: ML Engine ──
        with tabs[10]:
            st.markdown("### 🤖 Machine Learning Engine")
            st.caption("Ensemble ML | Isolation Forest | Walk-Forward Validation | Probability Calibration | Meta-Learner")

            if st.button("Run Full ML Pipeline", type="primary", use_container_width=True, key="run_ml"):
                ps = structure.get("price_series", {})
                if not ps:
                    st.error("No price series data available for ML analysis.")
                else:
                    with st.spinner("Engineering 50 ML features..."):
                        fe = st.session_state.feature_engineer
                        feat_result = fe.extract_features(ps)
                        st.session_state.feature_result = feat_result

                    with st.spinner("Training Random Forest + Gradient Boosting ensemble..."):
                        ml = st.session_state.ml_ensemble
                        ml_result = ml.train_and_predict(feat_result["feature_vector"], patterns, structure, regime, confluence)
                        st.session_state.ml_result = ml_result

                    with st.spinner("Detecting market anomalies (Isolation Forest + PCA)..."):
                        ad = st.session_state.ml_anomaly_detector
                        anomaly_result = ad.detect(feat_result["feature_vector"], ps)
                        st.session_state.ml_anomaly_result = anomaly_result

                    with st.spinner("Running walk-forward cross-validation..."):
                        wf = st.session_state.ml_walk_forward
                        wf_result = wf.validate(feat_result["feature_vector"], patterns, structure, confluence, n_windows=5)
                        st.session_state.wf_result = wf_result

                    with st.spinner("Calibrating probabilities (Platt + Isotonic)..."):
                        cal = st.session_state.ml_calibration
                        cal_result = cal.calibrate_and_predict(confluence, patterns, structure, regime, feat_result["feature_vector"])
                        st.session_state.cal_result = cal_result

                    if "audit_results" not in st.session_state:
                        with st.spinner("Running statistical validation..."):
                            sv = st.session_state.statistical_validator
                            stat_result = sv.full_probability_audit(ps, patterns, structure, confluence, n_simulations=1000, n_bootstrap=2000)
                            st.session_state.audit_results = stat_result
                    else:
                        stat_result = st.session_state.audit_results

                    with st.spinner("Running Meta-Learner (combining all models)..."):
                        meta = st.session_state.ml_meta_learner
                        meta_result = meta.predict_master(feat_result["feature_vector"], ml_result, anomaly_result, cal_result, stat_result, wf_result, confluence)
                        st.session_state.meta_result = meta_result

                    st.success("Full ML pipeline complete!")

            # Display ML results
            if "meta_result" in st.session_state:
                meta_result = st.session_state.meta_result
                master_prob = meta_result.get("master_probability", 0.5)
                master_dir = meta_result.get("master_direction", "NEUTRAL")
                master_grade = meta_result.get("master_grade", "D")
                agreement = meta_result.get("model_agreement", "UNKNOWN")
                ic = meta_result.get("information_coefficient", 0)
                risk_pos = meta_result.get("risk_adjusted_position", {})
                grade_colors_ml = {"A+": "#00ff88", "A": "#00ff88", "B": "#88ff44", "C": "#ffdd44", "D": "#ff4444"}
                g_color = grade_colors_ml.get(master_grade, "#ff4444")

                st.markdown(f"**ML MASTER PREDICTION:** Direction: **{master_dir}** | Grade: **{master_grade}** | Probability: **{master_prob:.1%}** | IC: **{ic:.3f}** | Agreement: **{agreement.replace('_',' ')}** | Position: **{risk_pos.get('position_strength','N/A')}** | Risk: **{risk_pos.get('recommended_risk_pct',0):.2f}%**")

                rec = meta_result.get("final_recommendation", "")
                st.info(f"{rec}")

                # Individual model predictions
                st.markdown("#### Individual Model Predictions")
                individual = meta_result.get("individual_models", {})
                for model_name, model_data in individual.items():
                    dir_color = "#00ff88" if model_data["direction"] == "BULLISH" else "#ff4444" if model_data["direction"] == "BEARISH" else "#ffdd44"
                    weight_pct = model_data["weight"] * 100
                    st.markdown(f"**{model_name.replace('_',' ').title()}**: {model_data['direction']} ({model_data['prob']:.1%}) — Weight: {weight_pct:.0f}% — Source: {model_data['source']}")

                # Model details in columns
                ml_c1, ml_c2, ml_c3 = st.columns(3)

                with ml_c1:
                    ml_r = st.session_state.get("ml_result", {})
                    if ml_r and "error" not in ml_r:
                        with st.expander("Ensemble Model Details"):
                            st.markdown(f"RF Probability: {ml_r.get('rf_probability', 0):.1%}")
                            st.markdown(f"GB Probability: {ml_r.get('gb_probability', 0):.1%}")
                            st.markdown(f"Agreement: {'Yes' if ml_r.get('agreement') == 'YES' else 'No'}")
                            cv = ml_r.get("cv_score", {})
                            st.markdown(f"CV Score (RF): {cv.get('rf_cv_mean', 0):.1%}")
                            st.markdown(f"CV Score (GB): {cv.get('gb_cv_mean', 0):.1%}")
                            st.markdown(f"Training Samples: {ml_r.get('training_samples', 0)}")
                            st.markdown("**Top Features:**")
                            for fi in ml_r.get("feature_importance", [])[:5]:
                                st.markdown(f"- {fi['feature']}: {fi['importance']:.3f}")

                    anom = st.session_state.get("ml_anomaly_result", {})
                    if anom and "anomaly_level" in anom:
                        with st.expander("Anomaly Detection"):
                            st.markdown(f"Level: **{anom['anomaly_level']}**")
                            st.markdown(f"Composite Score: {anom['anomaly_composite_score']:.3f}")
                            st.markdown(f"Risk Multiplier: {anom['risk_multiplier']}x")
                            if anom.get("recommendation"):
                                st.info(anom["recommendation"])

                with ml_c2:
                    wf_r = st.session_state.get("wf_result", {})
                    if wf_r and "error" not in wf_r:
                        with st.expander("Walk-Forward Validation"):
                            om = wf_r.get("overall_metrics", {})
                            st.markdown(f"OOS Accuracy: {om.get('accuracy', 0):.1%}")
                            st.markdown(f"Precision: {om.get('precision', 0):.1%}")
                            st.markdown(f"F1 Score: {om.get('f1_score', 0):.1%}")
                            for wr in wf_r.get("window_results", []):
                                st.markdown(f"Window {wr['window']}: Acc={wr['accuracy']:.1%}, PnL={wr['simulated_pnl']:.2f}, WR={wr['win_rate']:.1%}")
                            of = wf_r.get("overfitting_check", {})
                            st.markdown(f"Overfitting: {of.get('status', 'N/A')}")

                    cal_r = st.session_state.get("cal_result", {})
                    if cal_r and "main_confluence" in cal_r:
                        with st.expander("Probability Calibration"):
                            mc = cal_r["main_confluence"]
                            st.markdown(f"Raw Heuristic: {mc.get('raw_heuristic', 0):.1%}")
                            st.markdown(f"Platt Calibrated: {mc.get('platt_calibrated', 0):.1%}")
                            st.markdown(f"Isotonic Calibrated: {mc.get('isotonic_calibrated', 0):.1%}")
                            st.markdown(f"Best Calibrated: {mc.get('best_calibrated', 0):.1%}")
                            adj = mc.get('adjustment', 0)
                            st.markdown(f"Adjustment: {adj:+.1%}")
                            st.markdown(f"System is: {mc.get('direction', 'N/A')}")
                            cq = cal_r.get("calibration_quality", {})
                            st.markdown(f"Calibration Quality: {cq.get('quality', 'N/A')}")
                            if mc.get("warning"):
                                st.warning(mc["warning"])

                with ml_c3:
                    feat_r = st.session_state.get("feature_result", {})
                    if feat_r and feat_r.get("n_features", 0) > 0:
                        with st.expander("Feature Engineering"):
                            st.markdown(f"Features Extracted: {feat_r['n_features']}")
                            feats = feat_r.get("features", {})
                            sorted_feats = sorted(feats.items(), key=lambda x: abs(x[1]), reverse=True)
                            for name, value in sorted_feats[:15]:
                                st.markdown(f"- **{name}**: {value:.4f}")

        # ── TAB: Real Backtest DB ──
        with tabs[11]:
            st.markdown("### 🗄️ Real Backtest Database — MEASURED vs CLAIMED")
            db = st.session_state.trade_database
            db_stats = db.get_database_stats()

            st.markdown(f"**Database:** {db_stats['total_trades']} trades ({db_stats['backtest_trades']} backtest, {db_stats['live_trades']} live) | {db_stats['patterns_tracked']} patterns | {db_stats['pattern_combos']} combos | Overall WR: **{db_stats['overall_win_rate']:.1%}** | PF: **{db_stats['profit_factor']:.2f}**")

            if st.button("🔍 Backtest Current Analysis Against Real Data", type="primary", use_container_width=True, key="run_backtest"):
                bt = st.session_state.real_backtester
                confluence_grade = confluence.get("master", {}).get("grade", "D")
                regime_name = regime.get("regime", "TRENDING")
                session_name = session.get("inferred_session", "LONDON")
                bt_result = bt.backtest_current(patterns, confluence_grade, regime_name, session_name, pair)
                st.session_state.bt_result = bt_result
                st.success("Backtest complete!")

            if "bt_result" in st.session_state:
                bt_result = st.session_state.bt_result

                # Pattern backtests
                st.markdown("#### 📊 Pattern-by-Pattern: Heuristic vs MEASURED Win Rate")
                for pb in bt_result.get("pattern_backtests", []):
                    over = pb.get("overconfidence", 0)
                    over_color = "#ff4444" if over > 0.10 else "#ffdd44" if over > 0.05 else "#00ff88"
                    st.markdown(f"**{pb['pattern']}**: Heuristic **{pb['heuristic_confidence']:.0%}** → Measured **{pb['measured_win_rate']:.0%}** (n={pb['sample_size']}) | PF: {pb['measured_profit_factor']:.2f} | <span style='color:{over_color};'>Overconfidence: {over:+.0%}</span>", unsafe_allow_html=True)
                    st.markdown(f"&nbsp;&nbsp;{pb['verdict']}")
                    st.markdown("")

                # Grade backtest
                gb = bt_result.get("grade_backtest", {})
                if gb.get("sample_size", 0) > 0:
                    st.markdown(f"#### 🎯 Confluence Grade: **{gb['grade']}** → Measured WR: **{gb['measured_win_rate']:.1%}** (n={gb['sample_size']})")

                # Kelly from measured data
                kp = bt_result.get("kelly_params", {})
                if kp:
                    st.markdown(f"#### 📐 Real Kelly Criterion (Setup: {bt_result.get('setup_type','N/A')})")
                    st.markdown(f"- **Measured Win Rate:** {kp['win_rate']:.1%}")
                    st.markdown(f"- **Avg Win:** {kp['avg_win']:.1f} pips | **Avg Loss:** {kp['avg_loss']:.1f} pips")
                    st.markdown(f"- **Win/Loss Ratio:** {kp['win_loss_ratio']:.2f}")
                    st.markdown(f"- **Full Kelly:** {kp['kelly_fraction']:.2%} | **Half Kelly:** {kp['half_kelly']:.2%}")
                    if kp['kelly_fraction'] > 0:
                        st.success(f"✅ HAS EDGE: Half-Kelly recommends {kp['half_kelly']:.2%} risk per trade")
                    else:
                        st.error(f"🔴 NO EDGE: Kelly is negative — this setup loses money historically")

                # Overall verdict
                overall = bt_result.get("overall_verdict", {})
                if overall:
                    st.markdown("#### 🏁 Backtest Verdict")
                    st.info(overall.get("verdict", ""))
                    if overall.get("issues"):
                        st.markdown("**Issues:**")
                        for issue in overall["issues"]:
                            st.markdown(f"- 🔴 {issue}")
                    if overall.get("strengths"):
                        st.markdown("**Strengths:**")
                        for strength in overall["strengths"]:
                            st.markdown(f"- ✅ {strength}")

                # Real calibration
                rc = st.session_state.real_calibrator
                avg_heuristic = np.mean([p.get("confidence", 0.5) for p in patterns]) if patterns else 0.5
                cal_result = rc.calibrate(avg_heuristic, confluence.get("master", {}).get("grade", "D"), patterns)
                st.markdown("#### 🎯 Real Calibration (Database-Backed)")
                st.markdown(f"**Raw Heuristic:** {cal_result['raw_heuristic']:.1%} → **Calibrated:** {cal_result['calibrated_probability']:.1%} (Adjustment: {cal_result['adjustment']:+.1%})")
                if cal_result.get("is_overconfident"):
                    st.warning(f"⚠️ OVERCONFIDENT by {abs(cal_result['adjustment']):.0%}. Your system claims more than history supports.")
                elif cal_result.get("is_underconfident"):
                    st.success(f"🟢 UNDERCONFIDENT — signal is actually stronger than scored!")
                st.info(cal_result.get("honest_assessment", ""))

            # Pattern stats table
            st.markdown("#### 📋 All Pattern Statistics (from database)")
            all_stats = db.get_all_pattern_stats()
            for ps in all_stats:
                wr_color = "#00ff88" if ps["win_rate"] > 0.60 else "#ffdd44" if ps["win_rate"] > 0.50 else "#ff4444"
                st.markdown(f"<span style='color:{wr_color};'>**{ps['name']}**</span>: {ps['win_rate']:.1%} WR | {ps['total_occurrences']} trades | PF: {ps['profit_factor']:.2f} | AvgW: {ps['avg_win_pips']:.0f}p AvgL: {ps['avg_loss_pips']:.0f}p", unsafe_allow_html=True)

            # Trade journal entry
            st.markdown("---")
            st.markdown("#### 📝 Log a Trade (adds to database for future calibration)")
            with st.form("trade_form"):
                t_dir = st.selectbox("Direction", ["BUY", "SELL"])
                t_pair = st.selectbox("Pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"], key="t_pair")
                t_outcome = st.selectbox("Outcome", ["WIN", "LOSS"])
                t_pips = st.number_input("Pips Gained/Lost", value=50.0)
                t_pattern = st.text_input("Pattern Name", value=patterns[0]["name"] if patterns else "")
                t_grade = st.text_input("Confluence Grade", value=confluence.get("master", {}).get("grade", "C"))
                t_regime = st.text_input("Regime", value=regime.get("regime", "TRENDING"))
                t_session = st.text_input("Session", value=session.get("inferred_session", "LONDON"))
                t_notes = st.text_area("Notes")
                submitted = st.form_submit_button("💾 Save Trade")
                if submitted:
                    trade_id = db.insert_trade({
                        "trade_type": "live", "pair": t_pair, "direction": t_dir,
                        "entry_price": 0, "stop_loss": 0, "take_profit": 0,
                        "outcome": t_outcome, "pips_gained": t_pips,
                        "pattern_name": t_pattern, "confluence_grade": t_grade,
                        "regime": t_regime, "session": t_session, "notes": t_notes,
                    })
                    if t_pattern:
                        db.update_pattern_stats(t_pattern)
                    st.success(f"Trade #{trade_id} saved! Database now has {db.get_database_stats()['total_trades']} trades.")

        # ── TAB: Real Kelly ──
        with tabs[12]:
            st.markdown("### 📐 Real Kelly Criterion (from MEASURED data)")
            st.caption("The ONLY mathematically optimal position sizing method. Based on actual win rates, not estimates.")

            kelly_pair = st.selectbox("Pair for Kelly", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"], key="kelly_pair")
            sl_pips_input = st.number_input("Stop Loss (pips)", value=50.0, min_value=5.0)
            setup_input = st.selectbox("Setup Type", [
                "confluence_a_grade", "confluence_b_grade", "confluence_c_grade", "confluence_d_grade",
                "reversal_pattern", "continuation_pattern", "candlestick_signal",
                "fibonacci_entry", "divergence_signal", "liquidity_sweep", "ma_crossover"
            ])

            if st.button("📐 Calculate Real Kelly", type="primary", use_container_width=True):
                rk = st.session_state.real_kelly
                kelly_result = rk.calculate(setup_input, account_balance, sl_pips_input, kelly_pair)
                st.session_state.kelly_result = kelly_result

            if "kelly_result" in st.session_state:
                kr = st.session_state.kelly_result
                if "error" in kr:
                    st.error(kr["error"])
                    st.info(f"Fallback: Risk {kr.get('fallback_risk_pct', 0.5):.2%} → {kr.get('fallback_lots', 0):.2f} lots")
                else:
                    has_edge = kr.get("has_edge", False)
                    edge_color = "#00ff88" if has_edge else "#ff4444"

                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#0d1117,#161b22);
                         border:3px solid {edge_color};border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                        <h3 style="color:{edge_color};margin:0;">{'✅ EDGE DETECTED' if has_edge else '🔴 NO EDGE'}</h3>
                        <p style="color:#ffffff;margin:0.5rem 0;">
                            Measured WR: <strong>{kr['measured_win_rate']:.1%}</strong> |
                            Avg Win: <strong>{kr['measured_avg_win_pips']:.0f}p</strong> |
                            Avg Loss: <strong>{kr['measured_avg_loss_pips']:.0f}p</strong> |
                            W/L Ratio: <strong>{kr['win_loss_ratio']:.2f}</strong>
                        </p>
                        <p style="color:#8899aa;">
                            Sample: {kr['sample_size']} trades |
                            Full Kelly: {kr['full_kelly']:.2%} |
                            <strong style="color:#ffdd44;">Half Kelly: {kr['half_kelly']:.2%}</strong> |
                            Adjusted: {kr['adjusted_half_kelly']:.2%}
                        </p>
                        <p style="color:#ffdd44;font-size:1.2rem;font-weight:700;">
                            Final Risk: {kr['final_risk_pct']:.2%} → {kr['recommended_lots']:.2f} lots (${kr['risk_amount_usd']:.2f} risk)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info(kr.get("recommendation", ""))

                    # Additional Kelly metrics
                    st.markdown(f"**Expected Value:** {kr['expected_value_pips']:+.1f} pips per trade")
                    st.markdown(f"**Risk of Ruin:** {kr['risk_of_ruin']:.2%}")
                    st.markdown(f"**Trades to Double Account:** {kr['trades_to_double']}")

                    # Comparison table
                    st.markdown("#### Kelly Comparison")
                    st.markdown(f"| Strategy | Risk % | Lots | Notes |")
                    st.markdown(f"|----------|--------|------|-------|")
                    st.markdown(f"| Full Kelly | {kr['full_kelly']:.2%} | {st.session_state.real_kelly._calc_lots(account_balance, kr['full_kelly']/100 if kr['full_kelly']>0 else 0, sl_pips_input, kelly_pair):.2f} | ⚠️ Too aggressive |")
                    st.markdown(f"| **Half Kelly** | **{kr['half_kelly']:.2%}** | **{kr['recommended_lots']:.2f}** | **Recommended** |")
                    st.markdown(f"| Quarter Kelly | {kr['quarter_kelly']:.2%} | {st.session_state.real_kelly._calc_lots(account_balance, kr['quarter_kelly']/100 if kr['quarter_kelly']>0 else 0, sl_pips_input, kelly_pair):.2f} | Conservative |")

        # ── Export ──
        st.markdown("---")
        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            if st.button("📥 Download Annotated Image", use_container_width=True, key="dl_img"):
                if st.session_state.annotated_image is not None:
                    pil_img = Image.fromarray(st.session_state.annotated_image)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.download_button("💾 Save", data=buf.getvalue(), file_name="forex_analysis_v2.png", mime="image/png", use_container_width=True)

        with col_exp2:
            export = {
                "confluence": {"grade": grade, "direction": direction, "bull_score": confluence.get("bull_score"), "bear_score": confluence.get("bear_score")},
                "regime": regime, "session": session, "structure": {k: v for k, v in structure.items() if k != "price_series"},
                "geometric_patterns": [{"name": p["name"], "type": p["type"], "confidence": p["confidence"]} for p in patterns],
                "candlestick_patterns": [{"name": c["name"], "signal": c["signal"], "confidence": c["confidence"]} for c in candlesticks],
                "divergences": [{"name": d["name"], "type": d["type"], "confidence": d["confidence"]} for d in divergences],
                "fibonacci": fib, "liquidity": liquidity,
                "sr": {"support": [{"level": s["price_level"], "strength": s["strength"]} for s in sr.get("support", [])],
                        "resistance": [{"level": r["price_level"], "strength": r["strength"]} for r in sr.get("resistance", [])]},
                "sltp": sltp, "risk_management": risk,
                "trade_plan": trade_plan,
            }
            st.download_button("📄 Download Full Report (JSON)", data=json.dumps(export, indent=2, default=str), file_name="forex_analysis_v2.json", mime="application/json", use_container_width=True)

        with col_exp3:
            # Quick summary copy
            summary_text = f"""FOREX CHART ANALYSIS v2
========================
Pair: {pair} | Balance: ${account_balance:,.0f} | Risk: {risk_pct}%
Confluence: Grade {grade} | Direction: {direction} | Bull: {confluence.get('bull_score',0):.0%} Bear: {confluence.get('bear_score',0):.0%}
Regime: {regime.get('regime','N/A')} | Trend: {structure.get('trend_direction','N/A')}
Patterns: {len(patterns)} geometric, {len(candlesticks)} candlestick
Divergences: {len(divergences)} | Liquidity Sweeps: {len(liquidity.get('liquidity_sweeps',[]))}
Best Setup: {sltp.get('best_scenario',{}).get('name','N/A')} | R:R 1:{sltp.get('best_scenario',{}).get('risk_reward',0):.2f}
Lots: {risk.get('best_scenario_risk',{}).get('recommended_lots',0):.2f} | Risk: ${risk.get('best_scenario_risk',{}).get('risk_amount_usd',0):.2f}
Session: {session.get('inferred_session','N/A')}
"""
            st.download_button("📋 Download Summary (TXT)", data=summary_text, file_name="forex_summary.txt", mime="text/plain", use_container_width=True)

else:
    # ── Landing Page ──
    st.markdown("""
    <div style="text-align:center;padding:2rem 1rem;">
        <h2 style="color:#3a7bd5;">Upload a Forex Chart to Begin Analysis</h2>
        <p style="color:#8899aa;font-size:1rem;max-width:700px;margin:1rem auto;">
            Automated analysis: 10+ geometric patterns, 17 candlestick patterns, Fibonacci levels,
            divergence detection, liquidity zones, confluence scoring, risk management, and session context.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    features = [
        ("🧩", "10+ Geometric Patterns", "H&S, Triangles, Wedges, Flags, Channels, Breakouts"),
        ("🕯️", "17 Candlestick Patterns", "Doji, Hammer, Engulfing, Pin Bar, Morning/Evening Star"),
        ("📈", "Fibonacci Engine", "Retracements, Extensions, Golden Zone, Trade Zones"),
        ("📉", "Divergence Detection", "Regular & Hidden, Bullish & Bearish"),
        ("💧", "Liquidity Zones", "Buy/Sell-side pools, Stop Hunts, Equal Levels"),
        ("🎯", "Confluence Score", "Compounded signals, Grade A+-D, Trade Plan"),
        ("⚖️", "Risk Management", "Position Sizing, Risk of Ruin, Max Drawdown"),
        ("⏰", "Session Context", "Asian/London/NY, Volatility Expectations"),
        ("📊", "Indicator Detection", "MAs, Bollinger Bands, Trend Lines, Crossovers"),
        ("🚀", "Execution Plan", "Step-by-step: Entry, SL, TP, Confirmation"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 5]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.8rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <h4 style="color:#ffffff;margin:0.3rem 0;font-size:0.9rem;">{title}</h4>
                <p style="color:#8899aa;font-size:0.7rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#6677aa;">⬆️ Upload a chart image using the sidebar to get started</p>', unsafe_allow_html=True)

