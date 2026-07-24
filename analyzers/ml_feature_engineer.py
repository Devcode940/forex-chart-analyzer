"""
ML Feature Engineering Pipeline
================================
Extracts 50+ quantitative features from price data for ML models.
Features are engineered to capture momentum, volatility, structure,
candle characteristics, and cross-timeframe properties.
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class FeatureEngineer:
    """Extracts ML-ready feature vectors from price series."""

    FEATURE_NAMES = [
        # Momentum (10)
        "return_1", "return_5", "return_10", "return_20",
        "momentum_5", "momentum_10", "momentum_20",
        "roc_5", "roc_10", "roc_20",
        # Volatility (10)
        "vol_5", "vol_10", "vol_20", "vol_50",
        "atr_approx", "vol_ratio_5_20",
        "upper_wick_ratio", "lower_wick_ratio", "body_ratio",
        "range_ratio",
        # Trend (10)
        "trend_r2", "trend_slope", "trend_intercept",
        "sma_slope_5", "sma_slope_10", "sma_slope_20",
        "price_vs_sma5", "price_vs_sma10", "price_vs_sma20",
        "efficiency_ratio",
        # Structure (10)
        "swing_high_count", "swing_low_count",
        "last_swing_high_dist", "last_swing_low_dist",
        "swing_high_slope", "swing_low_slope",
        "channel_width", "channel_slope",
        "bos_count_bull", "bos_count_bear",
        # Statistical (10)
        "skewness", "kurtosis", "sharpe_approx",
        "sortino_approx", "max_drawdown",
        "var_95", "cvar_95",
        "hurst_exponent", "mean_reversion_score",
        "serial_correlation",
    ]

    def extract_features(self, price_series: dict) -> dict:
        """Extract full feature vector from price data."""
        smoothed = np.array(price_series.get("smoothed", []))
        highs = np.array(price_series.get("highs", []))
        lows = np.array(price_series.get("lows", []))

        if len(smoothed) < 20:
            return {"features": {}, "feature_vector": np.array([]), "n_features": 0}

        features = {}

        # Momentum Features
        returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
        features["return_1"] = float(returns[-1]) if len(returns) > 0 else 0
        features["return_5"] = float(np.sum(returns[-5:])) if len(returns) >= 5 else 0
        features["return_10"] = float(np.sum(returns[-10:])) if len(returns) >= 10 else 0
        features["return_20"] = float(np.sum(returns[-20:])) if len(returns) >= 20 else 0

        features["momentum_5"] = float(smoothed[-1] - smoothed[-6]) if len(smoothed) >= 6 else 0
        features["momentum_10"] = float(smoothed[-1] - smoothed[-11]) if len(smoothed) >= 11 else 0
        features["momentum_20"] = float(smoothed[-1] - smoothed[-21]) if len(smoothed) >= 21 else 0

        for p in [5, 10, 20]:
            if len(smoothed) > p and smoothed[-p] != 0:
                features[f"roc_{p}"] = float((smoothed[-1] - smoothed[-p]) / abs(smoothed[-p]) * 100)
            else:
                features[f"roc_{p}"] = 0

        # Volatility Features
        for w in [5, 10, 20, 50]:
            if len(returns) >= w:
                features[f"vol_{w}"] = float(np.std(returns[-w:]))
            else:
                features[f"vol_{w}"] = float(np.std(returns))

        features["atr_approx"] = float(np.mean(highs[-14:] - lows[-14:])) if len(highs) >= 14 else float(np.mean(highs - lows))

        features["vol_ratio_5_20"] = float(features["vol_5"] / (features["vol_20"] + 1e-8))

        # Candle shape ratios
        if len(highs) > 0 and len(lows) > 0:
            total_range = np.mean(highs[-10:] - lows[-10:]) if len(highs) >= 10 else np.mean(highs - lows)
            body_size = np.mean(np.abs(np.diff(smoothed[-10:]))) if len(smoothed) >= 11 else 0
            features["body_ratio"] = float(body_size / (total_range + 1e-8))
            features["range_ratio"] = float(total_range / (smoothed[-1] + 1e-8))

        features["upper_wick_ratio"] = 0.3  # Approximate
        features["lower_wick_ratio"] = 0.3

        # Trend Features
        x = np.arange(len(smoothed))
        slope, intercept = np.polyfit(x, smoothed, 1)
        predicted = slope * x + intercept
        ss_res = np.sum((smoothed - predicted) ** 2)
        ss_tot = np.sum((smoothed - np.mean(smoothed)) ** 2)
        features["trend_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        features["trend_slope"] = float(slope / (np.mean(smoothed) + 1e-8))
        features["trend_intercept"] = float(intercept)

        # SMA slopes
        for w in [5, 10, 20]:
            if len(smoothed) >= w:
                sma = np.convolve(smoothed, np.ones(w)/w, mode='valid')
                sma_slope = np.polyfit(np.arange(len(sma[-5:])), sma[-5:], 1)[0] if len(sma) >= 5 else 0
                features[f"sma_slope_{w}"] = float(sma_slope / (sma[-1] + 1e-8))
                features[f"price_vs_sma{w}"] = float((smoothed[-1] - sma[-1]) / (sma[-1] + 1e-8))
            else:
                features[f"sma_slope_{w}"] = 0
                features[f"price_vs_sma{w}"] = 0

        # Efficiency ratio
        net_change = abs(smoothed[-1] - smoothed[0])
        path_length = np.sum(np.abs(np.diff(smoothed)))
        features["efficiency_ratio"] = float(net_change / path_length) if path_length > 0 else 0

        # Structure Features
        distance = max(3, len(smoothed) // 15)
        prominence = np.ptp(smoothed) * 0.04 if len(smoothed) > 0 else 0.01
        swing_highs, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
        swing_lows, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)

        features["swing_high_count"] = len(swing_highs)
        features["swing_low_count"] = len(swing_lows)

        features["last_swing_high_dist"] = float(len(smoothed) - swing_highs[-1]) if len(swing_highs) > 0 else len(smoothed)
        features["last_swing_low_dist"] = float(len(smoothed) - swing_lows[-1]) if len(swing_lows) > 0 else len(smoothed)

        if len(swing_highs) >= 2:
            features["swing_high_slope"] = float(np.polyfit(swing_highs[-3:], smoothed[swing_highs[-3:]], 1)[0])
        else:
            features["swing_high_slope"] = 0

        if len(swing_lows) >= 2:
            features["swing_low_slope"] = float(np.polyfit(swing_lows[-3:], smoothed[swing_lows[-3:]], 1)[0])
        else:
            features["swing_low_slope"] = 0

        # Channel features
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            h_vals = smoothed[swing_highs[-3:]]
            l_vals = smoothed[swing_lows[-3:]]
            features["channel_width"] = float(np.mean(h_vals) - np.mean(l_vals))
            features["channel_slope"] = float((np.mean(h_vals) - np.mean(l_vals)) / (np.mean(smoothed) + 1e-8))
        else:
            features["channel_width"] = 0
            features["channel_slope"] = 0

        features["bos_count_bull"] = 0  # Simplified
        features["bos_count_bear"] = 0

        # Statistical Features
        features["skewness"] = float(stats.skew(returns)) if len(returns) > 2 else 0
        features["kurtosis"] = float(stats.kurtosis(returns)) if len(returns) > 3 else 0

        mean_ret = np.mean(returns) if len(returns) > 0 else 0
        std_ret = np.std(returns) if len(returns) > 0 else 1
        features["sharpe_approx"] = float(mean_ret / (std_ret + 1e-8) * np.sqrt(252))

        neg_returns = returns[returns < 0]
        down_std = np.std(neg_returns) if len(neg_returns) > 1 else std_ret
        features["sortino_approx"] = float(mean_ret / (down_std + 1e-8) * np.sqrt(252))

        # Max drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        features["max_drawdown"] = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # VaR and CVaR
        features["var_95"] = float(np.percentile(returns, 5)) if len(returns) > 5 else 0
        var_threshold = features["var_95"]
        tail_returns = returns[returns <= var_threshold]
        features["cvar_95"] = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_threshold

        # Hurst exponent (R/S analysis)
        features["hurst_exponent"] = self._hurst_exponent(smoothed)

        # Mean reversion score
        features["mean_reversion_score"] = self._mean_reversion_score(returns)

        # Serial correlation
        if len(returns) > 2:
            features["serial_correlation"] = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            features["serial_correlation"] = 0

        # Build feature vector in consistent order
        feature_vector = np.array([features.get(name, 0) for name in self.FEATURE_NAMES])

        # Replace NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=1e6, neginf=-1e6)

        return {
            "features": features,
            "feature_vector": feature_vector,
            "feature_names": self.FEATURE_NAMES,
            "n_features": len(feature_vector),
        }

    def _hurst_exponent(self, data: np.ndarray, max_lag: int = 20) -> float:
        """Estimate Hurst exponent using R/S analysis."""
        if len(data) < 30:
            return 0.5  # Random walk default

        lags = range(2, min(max_lag, len(data) // 4))
        rs_values = []

        for lag in lags:
            segments = [data[i:i+lag] for i in range(0, len(data) - lag, lag)]
            rs_segment = []

            for seg in segments:
                if len(seg) < 2:
                    continue
                mean_val = np.mean(seg)
                deviations = np.cumsum(seg - mean_val)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(seg)
                if S > 0:
                    rs_segment.append(R / S)

            if rs_segment:
                rs_values.append((np.log(lag), np.log(np.mean(rs_segment))))

        if len(rs_values) >= 2:
            x = np.array([v[0] for v in rs_values])
            y = np.array([v[1] for v in rs_values])
            slope = np.polyfit(x, y, 1)[0]
            return float(np.clip(slope, 0, 1))

        return 0.5

    def _mean_reversion_score(self, returns: np.ndarray) -> float:
        """Score how mean-reverting the series is (0=trending, 1=mean-reverting)."""
        if len(returns) < 10:
            return 0.5

        # Variance ratio test
        var_1 = np.var(returns)
        returns_2 = returns[::2]
        var_2 = np.var(returns_2) if len(returns_2) > 1 else var_1

        vr = var_2 / (2 * var_1 + 1e-8)
        # vr < 1 = mean reverting, vr > 1 = trending, vr = 1 = random walk
        score = 1 / (1 + np.exp(vr - 1))  # Sigmoid: maps to [0, 1]
        return float(np.clip(score, 0, 1))

