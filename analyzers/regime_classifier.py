"""
Market Regime Classifier Module
Classifies the current market regime: Trending, Ranging, Volatile, Quiet.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

class RegimeClassifier:
    """Classifies the current market regime."""

    def __init__(self):
        self.regime = "UNKNOWN"
        self.sub_regime = "UNKNOWN"

    def classify(self, price_series: dict, structure_results: dict) -> dict:
        """Classify the market regime from price data and structure."""
        smoothed = np.array(price_series.get("smoothed", []))

        if len(smoothed) < 10:
            return {
                "regime": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "details": "Not enough price data"
            }

        # Calculate regime indicators
        trend_strength = self._calc_trend_strength(smoothed)
        volatility = self._calc_volatility(smoothed)
        efficiency = self._calc_efficiency_ratio(smoothed)
        adx_approx = self._approximate_adx(smoothed)

        # Classify primary regime
        if efficiency > 0.4 and adx_approx > 0.5:
            self.regime = "TRENDING"
            if trend_strength > 0:
                self.sub_regime = "UPTREND"
            else:
                self.sub_regime = "DOWNTREND"
            confidence = min(0.95, 0.6 + efficiency * 0.35)
        elif efficiency < 0.25 and adx_approx < 0.3:
            self.regime = "RANGING"
            self.sub_regime = "CONSOLIDATION"
            confidence = min(0.9, 0.5 + (1 - efficiency) * 0.4)
        elif volatility > np.percentile([self._calc_volatility(smoothed[i:i+20])
                                          for i in range(0, len(smoothed)-20, 10)], 75):
            self.regime = "VOLATILE"
            self.sub_regime = "HIGH_VOLATILITY"
            confidence = 0.6
        else:
            self.regime = "TRANSITIONAL"
            self.sub_regime = "POTENTIAL_BREAKOUT"
            confidence = 0.5

        # Incorporate structure analysis
        structure_trend = structure_results.get("trend_direction", "RANGING")
        structure_strength = structure_results.get("trend_strength", 0.5)

        if structure_trend == "UPTREND" and self.regime == "RANGING":
            self.sub_regime = "ACCUMULATION"
            confidence *= 0.8
        elif structure_trend == "DOWNTREND" and self.regime == "RANGING":
            self.sub_regime = "DISTRIBUTION"
            confidence *= 0.8
        elif structure_trend in ["UPTREND", "DOWNTREND"] and self.regime == "TRENDING":
            confidence = min(0.95, confidence * 1.1)

        # Generate trading recommendations based on regime
        trading_style = self._get_trading_style()
        risk_level = self._get_risk_level()

        return {
            "regime": self.regime,
            "sub_regime": self.sub_regime,
            "confidence": round(confidence, 2),
            "indicators": {
                "trend_strength": round(float(trend_strength), 3),
                "volatility": round(float(volatility), 3),
                "efficiency_ratio": round(float(efficiency), 3),
                "adx_approximation": round(float(adx_approx), 3)
            },
            "trading_style": trading_style,
            "risk_level": risk_level,
            "recommendation": self._get_regime_recommendation()
        }

    def _calc_trend_strength(self, smoothed: np.ndarray) -> float:
        """Calculate trend strength using linear regression R²."""
        x = np.arange(len(smoothed))
        slope, intercept = np.polyfit(x, smoothed, 1)
        predicted = slope * x + intercept
        ss_res = np.sum((smoothed - predicted) ** 2)
        ss_tot = np.sum((smoothed - np.mean(smoothed)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return float(np.sign(slope) * r_squared)

    def _calc_volatility(self, data: np.ndarray) -> float:
        """Calculate normalized volatility."""
        if len(data) < 2:
            return 0
        returns = np.diff(data) / (data[:-1] + 1e-6)
        return float(np.std(returns))

    def _calc_efficiency_ratio(self, smoothed: np.ndarray) -> float:
        """Kaufman Efficiency Ratio - measures trend efficiency."""
        if len(smoothed) < 2:
            return 0
        net_change = abs(smoothed[-1] - smoothed[0])
        path_length = np.sum(np.abs(np.diff(smoothed)))
        return float(net_change / path_length) if path_length > 0 else 0

    def _approximate_adx(self, smoothed: np.ndarray) -> float:
        """Approximate ADX (Average Directional Index)."""
        if len(smoothed) < 10:
            return 0

        # Use directional movement approximation
        up_moves = np.maximum(np.diff(smoothed), 0)
        down_moves = np.maximum(-np.diff(smoothed), 0)

        window = min(14, len(up_moves))
        avg_up = np.mean(up_moves[-window:])
        avg_down = np.mean(down_moves[-window:])

        if avg_up + avg_down < 1e-6:
            return 0

        dx = abs(avg_up - avg_down) / (avg_up + avg_down)
        return float(dx)

    def _get_trading_style(self) -> str:
        """Get recommended trading style based on regime."""
        styles = {
            "TRENDING": "Trend Following (ride the momentum)",
            "RANGING": "Mean Reversion (buy support, sell resistance)",
            "VOLATILE": "Reduced Position Sizing (wait for clarity)",
            "TRANSITIONAL": "Breakout Strategy (wait for confirmation)"
        }
        return styles.get(self.regime, "Cautious")

    def _get_risk_level(self) -> dict:
        """Get risk level recommendation."""
        levels = {
            "TRENDING": {"level": "MODERATE", "position_sizing": "Standard (1-2% risk)"},
            "RANGING": {"level": "LOW", "position_sizing": "Reduced (0.5-1% risk)"},
            "VOLATILE": {"level": "HIGH", "position_sizing": "Minimum (0.25-0.5% risk)"},
            "TRANSITIONAL": {"level": "MODERATE-HIGH", "position_sizing": "Cautious (0.5-1% risk)"}
        }
        return levels.get(self.regime, {"level": "UNKNOWN", "position_sizing": "Minimal"})

    def _get_regime_recommendation(self) -> str:
        """Get specific trading recommendation based on regime."""
        recommendations = {
            "TRENDING": "Look for pullback entries in the trend direction. Use moving averages or trendlines for entries. Avoid counter-trend trades.",
            "RANGING": "Trade between support and resistance. Use oscillators (RSI, Stochastic) for entry timing. Avoid breakout trades without confirmation.",
            "VOLATILE": "Reduce position sizes and widen stops. Wait for volatility to compress before entering. Consider options strategies if available.",
            "TRANSITIONAL": "Set alerts at key levels. Wait for confirmed breakout with volume. Have orders ready but don't preempt the move."
        }
        return recommendations.get(self.regime, "Wait for clearer market conditions.")

