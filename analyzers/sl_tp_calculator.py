"""
Stop Loss / Take Profit Calculator Module
Calculates intelligent SL and TP levels based on pattern, structure, and S/R analysis.
"""

import numpy as np


class SLTPCalculator:
    """Calculates Stop Loss and Take Profit levels."""

    def __init__(self):
        self.recommendations = []

    def calculate(self, pattern_results: list, sr_results: dict,
                  structure_results: dict, price_series: dict) -> dict:
        """
        Calculate SL/TP based on all analysis results.
        Returns multiple scenario recommendations.
        """
        self.recommendations = []
        smoothed = np.array(price_series.get("smoothed", []))

        if len(smoothed) < 5:
            return {"scenarios": [], "risk_reward_ratios": []}

        current_price = float(smoothed[-1])
        price_range = float(np.max(smoothed) - np.min(smoothed))

        if price_range < 1e-6:
            return {"scenarios": [], "risk_reward_ratios": []}

        # Determine dominant bias
        bias = self._determine_bias(pattern_results, structure_results)

        # Generate scenarios based on different approaches
        self._pattern_based_scenarios(pattern_results, current_price, price_range)
        self._sr_based_scenarios(sr_results, current_price, price_range, bias)
        self._structure_based_scenarios(structure_results, current_price, price_range, smoothed)
        self._atr_based_scenarios(smoothed, current_price, bias)

        # Add a conservative and aggressive scenario
        self._conservative_scenario(bias, current_price, price_range)
        self._aggressive_scenario(bias, current_price, price_range)

        # Calculate risk/reward ratios
        rr_ratios = []
        for rec in self.recommendations:
            risk = abs(current_price - rec["sl"])
            reward = abs(rec["tp"] - current_price)
            rr = reward / risk if risk > 1e-6 else 0
            rr_ratios.append({
                "scenario": rec["name"],
                "risk_reward": round(rr, 2),
                "risk_pips": round(risk, 1),
                "reward_pips": round(reward, 1)
            })
            rec["risk_reward"] = round(rr, 2)

        # Sort by risk/reward
        self.recommendations.sort(key=lambda x: x.get("risk_reward", 0), reverse=True)

        return {
            "bias": bias,
            "current_price": current_price,
            "scenarios": self.recommendations,
            "risk_reward_ratios": rr_ratios,
            "best_scenario": self.recommendations[0] if self.recommendations else None
        }

    def _determine_bias(self, patterns: list, structure: dict) -> str:
        """Determine overall market bias from patterns and structure."""
        bullish_score = 0
        bearish_score = 0

        # Pattern signals
        for p in patterns:
            if "BULLISH" in p.get("type", ""):
                bullish_score += p.get("confidence", 0.5)
            elif "BEARISH" in p.get("type", ""):
                bearish_score += p.get("confidence", 0.5)

        # Structure signals
        trend = structure.get("trend_direction", "RANGING")
        if trend == "UPTREND":
            bullish_score += structure.get("trend_strength", 0.5)
        elif trend == "DOWNTREND":
            bearish_score += structure.get("trend_strength", 0.5)

        if bullish_score > bearish_score + 0.2:
            return "BULLISH"
        elif bearish_score > bullish_score + 0.2:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _pattern_based_scenarios(self, patterns: list, current_price: float, price_range: float):
        """Generate SL/TP based on detected patterns."""
        for p in patterns:
            name = p.get("name", "")
            direction = p.get("target_direction", "PENDING")
            confidence = p.get("confidence", 0.5)

            if direction == "UP":
                sl_distance = price_range * 0.15
                tp_distance = price_range * (0.3 + confidence * 0.4)
                self.recommendations.append({
                    "name": f"{name} - Long Entry",
                    "direction": "BUY",
                    "entry": round(current_price, 2),
                    "sl": round(current_price - sl_distance, 2),
                    "tp": round(current_price + tp_distance, 2),
                    "confidence": confidence,
                    "source": f"Pattern: {name}"
                })

            elif direction == "DOWN":
                sl_distance = price_range * 0.15
                tp_distance = price_range * (0.3 + confidence * 0.4)
                self.recommendations.append({
                    "name": f"{name} - Short Entry",
                    "direction": "SELL",
                    "entry": round(current_price, 2),
                    "sl": round(current_price + sl_distance, 2),
                    "tp": round(current_price - tp_distance, 2),
                    "confidence": confidence,
                    "source": f"Pattern: {name}"
                })

    def _sr_based_scenarios(self, sr: dict, current_price: float, price_range: float, bias: str):
        """Generate SL/TP based on support/resistance levels."""
        supports = sr.get("support", [])
        resistances = sr.get("resistance", [])

        # Find nearest support and resistance
        nearest_support = None
        nearest_resistance = None

        for s in supports:
            if s["price_level"] < current_price:
                if nearest_support is None or s["price_level"] > nearest_support:
                    nearest_support = s["price_level"]

        for r in resistances:
            if r["price_level"] > current_price:
                if nearest_resistance is None or r["price_level"] < nearest_resistance:
                    nearest_resistance = r["price_level"]

        # Buy scenario
        if bias in ["BULLISH", "NEUTRAL"] and nearest_support:
            sl = nearest_support - price_range * 0.02  # Below support
            tp = nearest_resistance if nearest_resistance else current_price + (current_price - sl) * 2
            self.recommendations.append({
                "name": "Support Bounce - Long",
                "direction": "BUY",
                "entry": round(current_price, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "confidence": 0.65,
                "source": f"Support at {round(nearest_support, 2)}"
            })

        # Sell scenario
        if bias in ["BEARISH", "NEUTRAL"] and nearest_resistance:
            sl = nearest_resistance + price_range * 0.02  # Above resistance
            tp = nearest_support if nearest_support else current_price - (sl - current_price) * 2
            self.recommendations.append({
                "name": "Resistance Rejection - Short",
                "direction": "SELL",
                "entry": round(current_price, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "confidence": 0.65,
                "source": f"Resistance at {round(nearest_resistance, 2)}"
            })

    def _structure_based_scenarios(self, structure: dict, current_price: float,
                                   price_range: float, smoothed: np.ndarray):
        """Generate SL/TP based on market structure."""
        swing_highs = structure.get("swing_highs", [])
        swing_lows = structure.get("swing_lows", [])
        breaks = structure.get("structure_breaks", [])

        # If bullish BOS detected
        bullish_breaks = [b for b in breaks if b["type"] == "BULLISH_BOS"]
        if bullish_breaks:
            recent_low = min([sl["value"] for sl in swing_lows[-2:]]) if len(swing_lows) >= 2 else current_price - price_range * 0.1
            recent_high = max([sh["value"] for sh in swing_highs[-2:]]) if len(swing_highs) >= 2 else current_price + price_range * 0.3
            self.recommendations.append({
                "name": "Bullish BOS - Trend Following",
                "direction": "BUY",
                "entry": round(current_price, 2),
                "sl": round(recent_low - price_range * 0.02, 2),
                "tp": round(recent_high + price_range * 0.2, 2),
                "confidence": 0.7,
                "source": "Bullish Break of Structure"
            })

        # If bearish BOS detected
        bearish_breaks = [b for b in breaks if b["type"] == "BEARISH_BOS"]
        if bearish_breaks:
            recent_high = max([sh["value"] for sh in swing_highs[-2:]]) if len(swing_highs) >= 2 else current_price + price_range * 0.1
            recent_low = min([sl["value"] for sl in swing_lows[-2:]]) if len(swing_lows) >= 2 else current_price - price_range * 0.3
            self.recommendations.append({
                "name": "Bearish BOS - Trend Following",
                "direction": "SELL",
                "entry": round(current_price, 2),
                "sl": round(recent_high + price_range * 0.02, 2),
                "tp": round(recent_low - price_range * 0.2, 2),
                "confidence": 0.7,
                "source": "Bearish Break of Structure"
            })

    def _atr_based_scenarios(self, smoothed: np.ndarray, current_price: float, bias: str):
        """Generate SL/TP based on average true range approximation."""
        if len(smoothed) < 5:
            return

        # Approximate ATR
        diffs = np.abs(np.diff(smoothed))
        atr = float(np.mean(diffs[-20:])) if len(diffs) >= 20 else float(np.mean(diffs))

        if bias in ["BULLISH", "NEUTRAL"]:
            self.recommendations.append({
                "name": "ATR-Based Long",
                "direction": "BUY",
                "entry": round(current_price, 2),
                "sl": round(current_price - 1.5 * atr, 2),
                "tp": round(current_price + 3.0 * atr, 2),
                "confidence": 0.55,
                "source": f"ATR approximation: {round(atr, 2)}"
            })

        if bias in ["BEARISH", "NEUTRAL"]:
            self.recommendations.append({
                "name": "ATR-Based Short",
                "direction": "SELL",
                "entry": round(current_price, 2),
                "sl": round(current_price + 1.5 * atr, 2),
                "tp": round(current_price - 3.0 * atr, 2),
                "confidence": 0.55,
                "source": f"ATR approximation: {round(atr, 2)}"
            })

    def _conservative_scenario(self, bias: str, current_price: float, price_range: float):
        """Conservative SL/TP with tight stops."""
        if bias in ["BULLISH", "NEUTRAL"]:
            self.recommendations.append({
                "name": "Conservative Long",
                "direction": "BUY",
                "entry": round(current_price, 2),
                "sl": round(current_price - price_range * 0.08, 2),
                "tp": round(current_price + price_range * 0.16, 2),
                "confidence": 0.7,
                "source": "Conservative (tight SL)"
            })

        if bias in ["BEARISH", "NEUTRAL"]:
            self.recommendations.append({
                "name": "Conservative Short",
                "direction": "SELL",
                "entry": round(current_price, 2),
                "sl": round(current_price + price_range * 0.08, 2),
                "tp": round(current_price - price_range * 0.16, 2),
                "confidence": 0.7,
                "source": "Conservative (tight SL)"
            })

    def _aggressive_scenario(self, bias: str, current_price: float, price_range: float):
        """Aggressive SL/TP with wide stops and big targets."""
        if bias in ["BULLISH", "NEUTRAL"]:
            self.recommendations.append({
                "name": "Aggressive Long",
                "direction": "BUY",
                "entry": round(current_price, 2),
                "sl": round(current_price - price_range * 0.25, 2),
                "tp": round(current_price + price_range * 0.75, 2),
                "confidence": 0.4,
                "source": "Aggressive (wide SL, big target)"
            })

        if bias in ["BEARISH", "NEUTRAL"]:
            self.recommendations.append({
                "name": "Aggressive Short",
                "direction": "SELL",
                "entry": round(current_price, 2),
                "sl": round(current_price + price_range * 0.25, 2),
                "tp": round(current_price - price_range * 0.75, 2),
                "confidence": 0.4,
                "source": "Aggressive (wide SL, big target)"
            })
