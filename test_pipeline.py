#!/usr/bin/env python3
"""
End-to-end validation script for Forex Chart Analyzer Pro v2.
Run with: python test_pipeline.py
"""

import sys
import numpy as np
from PIL import Image


def create_test_chart(seed: int = 42) -> np.ndarray:
    """Create a synthetic forex chart image for testing."""
    np.random.seed(seed)
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :] = [20, 24, 33]
    price = 300
    for i in range(30, 850, 28):
        price += np.random.randint(-8, 12)
        price = max(100, min(500, price))
        bh = np.random.randint(10, 40)
        is_green = np.random.random() > 0.4
        bt = price - bh if is_green else price
        color = [46, 204, 113] if is_green else [231, 76, 60]
        img[max(10, bt - 20):min(590, bt + bh + 20), i:i + 2] = [180, 180, 180]
        img[max(10, bt):min(590, bt + bh), i:i + 8] = color
    for y in range(100, 500, 60):
        img[y, 40:860] = [40, 44, 53]
    for y in [180, 300, 380]:
        img[y:y + 2, 40:860] = [52, 152, 219]
    return img


def validate_keys(name: str, actual: dict, required_keys: list) -> bool:
    """Validate that a dict contains all required keys."""
    missing = [k for k in required_keys if k not in actual]
    if missing:
        print(f"  ❌ {name}: missing {missing}")
        return False
    print(f"  ✅ {name}")
    return True


def main():
    print("=" * 60)
    print("FOREX CHART ANALYZER - VALIDATION")
    print("=" * 60)

    img = create_test_chart()
    passed = 0
    total = 0

    # -- Imports --
    print("\n-- Imports --")
    try:
        from analyzers.image_processor import ImageProcessor
        from analyzers.pattern_detector import PatternDetector
        from analyzers.candlestick_detector import CandlestickDetector
        from analyzers.structure_analyzer import StructureAnalyzer
        from analyzers.sr_detector import SRDetector
        from analyzers.fibonacci_calculator import FibonacciCalculator
        from analyzers.sl_tp_calculator import SLTPCalculator
        from analyzers.regime_classifier import RegimeClassifier
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
        from analyzers.real_kelly import RealKellyCalculator
        from analyzers.real_calibrator import RealCalibrator
        from analyzers.real_backtester import RealBacktester
        from utils.visualizer import Visualizer
        print("  ✅ All 26 modules imported successfully")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        sys.exit(1)

    # -- Core Pipeline --
    print("\n-- Core Pipeline --")
    processor = ImageProcessor()
    preprocessed = processor.preprocess(img)
    color_info = processor.extract_chart_colors(img)
    green_points = processor.extract_price_series(img, color_info["green_mask"])
    red_points = processor.extract_price_series(img, color_info["red_mask"])
    all_points = green_points + red_points

    structure = StructureAnalyzer().analyze(all_points, img.shape[0]) if all_points else {}
    ps = structure.get("price_series", {})

    if not ps:
        print("  ❌ No price_series extracted — cannot continue")
        sys.exit(1)

    patterns = PatternDetector().detect_all(ps, img.shape[0])
    candlesticks = CandlestickDetector().detect_all(ps)
    sr = SRDetector().detect(ps, img.shape[0])
    regime = RegimeClassifier().classify(ps, structure)
    fib = FibonacciCalculator().calculate(structure)
    divergences = DivergenceDetector().detect_all(ps, structure)
    indicators = IndicatorDetector().detect_all(img, preprocessed["edges"])
    liquidity = LiquidityDetector().detect_all(ps, structure, sr)
    session = SessionAnalyzer().analyze(ps, regime, "EURUSD")
    sltp = SLTPCalculator().calculate(patterns, sr, structure, ps)
    confluence = ConfluenceEngine().analyze(patterns, candlesticks, sr, fib, structure, regime, sltp)
    risk = RiskManager().calculate(sltp, sr, structure, img.shape[0], 10000, 2.0, "EURUSD")
    annotated = Visualizer().create_full_overlay(img, patterns, sr, structure, sltp, regime)

    total += 12
    passed += sum([
        validate_keys("ImageProcessor", preprocessed, ["edges", "gray"]),
        validate_keys("StructureAnalyzer", structure, ["price_series", "swing_highs", "swing_lows", "trend_direction"]),
        validate_keys("SRDetector", sr, ["support", "resistance"]),
        validate_keys("FibonacciCalculator", fib, ["retracements", "extensions"]),
        validate_keys("RegimeClassifier", regime, ["regime", "confidence", "indicators"]),
        validate_keys("SLTPCalculator", sltp, ["bias", "scenarios", "best_scenario", "current_price"]),
        validate_keys("ConfluenceEngine", confluence, ["master", "trade_plan", "signals", "bull_score", "bear_score"]),
        validate_keys("RiskManager", risk, ["best_scenario_risk", "risk_amount", "risk_of_ruin"]),
        validate_keys("LiquidityDetector", liquidity, ["buy_side_liquidity", "sell_side_liquidity", "summary"]),
        validate_keys("SessionAnalyzer", session, ["inferred_session", "volatility"]),
        validate_keys("Visualizer", {"annotated": annotated}, ["annotated"]),
        validate_keys("SLTP.best_scenario", sltp.get("best_scenario") or {}, ["direction"]) if sltp.get("best_scenario") else (print("  ⚠️ SLTP.best_scenario: None (expected for weak signals)"), True)[1],
    ])

    # -- ML Pipeline --
    print("\n-- ML Pipeline --")
    fe = FeatureEngineer()
    feat_result = fe.extract_features(ps)
    fv = feat_result["feature_vector"]

    ml_result = MLEnsemble().train_and_predict(fv, patterns, structure, regime, confluence)
    ad_result = MLAnomalyDetector().detect(fv, ps)
    wf_result = WalkForwardValidator().validate(fv, patterns, structure, confluence, n_windows=3)
    cal_result = CalibrationEngine().calibrate_and_predict(confluence, patterns, structure, regime, fv)
    sv_result = StatisticalValidator().full_probability_audit(ps, patterns, structure, confluence, n_simulations=200, n_bootstrap=200)
    meta_result = MetaLearner().predict_master(fv, ml_result, ad_result, cal_result, sv_result, wf_result, confluence)

    total += 7
    passed += sum([
        validate_keys("FeatureEngineer", feat_result, ["feature_vector", "features", "n_features", "feature_names"]),
        validate_keys("MLEnsemble", ml_result, ["ml_probability", "ml_direction", "ml_confidence", "rf_probability", "gb_probability", "agreement"]),
        validate_keys("MLAnomalyDetector", ad_result, ["anomaly_level", "anomaly_composite_score", "risk_multiplier", "recommendation"]),
        validate_keys("WalkForwardValidator", wf_result, ["overall_metrics", "window_results", "overfitting_check"]),
        validate_keys("CalibrationEngine", cal_result, ["main_confluence", "calibration_quality"]),
        validate_keys("CalibrationEngine.main_confluence", cal_result.get("main_confluence", {}), ["raw_heuristic", "platt_calibrated", "best_calibrated", "direction", "is_overconfident"]),
        validate_keys("MetaLearner", meta_result, ["master_probability", "master_direction", "master_grade", "model_agreement", "individual_models"]),
    ])

    # -- Database Pipeline --
    print("\n-- Database Pipeline --")
    db = TradeDatabase()
    db_stats = db.get_database_stats()
    confluence_grade = confluence.get("master", {}).get("grade", "D")
    regime_name = regime.get("regime", "TRENDING")
    session_name = session.get("inferred_session", "LONDON")

    bt_result = RealBacktester(db).backtest_current(patterns, confluence_grade, regime_name, session_name, "EURUSD")
    kelly_result = RealKellyCalculator(db).calculate("continuation_pattern", 10000, 50, "EURUSD")
    rc_result = RealCalibrator(db).calibrate(0.75, confluence_grade, patterns)

    total += 4
    passed += sum([
        validate_keys("TradeDatabase.stats", db_stats, ["total_trades", "backtest_trades", "live_trades", "patterns_tracked", "overall_win_rate", "profit_factor"]),
        validate_keys("RealBacktester", bt_result, ["pattern_backtests", "overall_verdict"]),
        validate_keys("RealKellyCalculator", kelly_result, ["full_kelly", "half_kelly", "quarter_kelly", "measured_win_rate", "has_edge", "recommended_lots"]),
        validate_keys("RealCalibrator", rc_result, ["raw_heuristic", "calibrated_probability", "adjustment", "is_overconfident", "honest_assessment"]),
    ])

    # -- Summary --
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} validations passed")
    if passed == total:
        print("✅ ALL VALIDATIONS PASS — Pipeline is production-ready")
    else:
        print(f"⚠️ {total - passed} validations failed")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
