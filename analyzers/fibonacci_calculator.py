"""
Fibonacci Retracement & Extension Calculator
Auto-calculates Fib levels from swing points for entry, SL, and TP zones.
"""

from typing import Optional

import numpy as np


class FibonacciCalculator:
    """Calculates Fibonacci retracement and extension levels."""

    # Standard Fibonacci ratios
    RETRACEMENT_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_RATIOS = [0.0, 0.272, 0.414, 0.618, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    FIB_LABELS = {
        0.0: "0%",
        0.236: "23.6%",
        0.382: "38.2%",
        0.5: "50%",
        0.618: "61.8%",
        0.786: "78.6%",
        1.0: "100%",
        1.272: "127.2%",
        1.414: "141.4%",
        1.618: "161.8%",
        2.0: "200%",
        2.618: "261.8%",
    }

    # Zone importance
    ZONE_IMPORTANCE = {
        0.382: "MEDIUM",  # Shallow pullback — strong trend
        0.5: "MEDIUM",  # Deep pullback — moderate trend
        0.618: "HIGH",  # Golden ratio — most watched level
        0.786: "MEDIUM",  # Deep pullback — potential reversal zone
        1.618: "HIGH",  # Golden extension — common TP target
        2.618: "MEDIUM",  # Extended TP — for strong trends
    }

    def __init__(self):
        self.retracements = {}
        self.extensions = {}

    def calculate(self, structure_results: dict, current_price: float = None) -> dict:
        """
        Calculate Fibonacci levels from the most significant swing points.
        Returns retracement zones, extension targets, and trade zones.
        """
        swing_highs = structure_results.get("swing_highs", [])
        swing_lows = structure_results.get("swing_lows", [])
        trend = structure_results.get("trend_direction", "RANGING")

        if not swing_highs or not swing_lows:
            # Fallback: use price series min/max as swing points
            price_series_data = structure_results.get("price_series", {})
            smoothed = price_series_data.get("smoothed", [])
            if len(smoothed) < 5:
                return {
                    "retracements": {},
                    "extensions": {},
                    "trade_zones": [],
                    "fib_confluence": [],
                }
            high_val = max(smoothed)
            low_val = min(smoothed)
            sig_high = {
                "value": high_val,
                "index": (
                    smoothed.index(high_val)
                    if isinstance(smoothed, list)
                    else int(np.argmax(smoothed))
                ),
            }
            sig_low = {
                "value": low_val,
                "index": (
                    smoothed.index(low_val)
                    if isinstance(smoothed, list)
                    else int(np.argmin(smoothed))
                ),
            }
        else:

            sig_high = max(swing_highs, key=lambda s: s["value"])
            sig_low = min(swing_lows, key=lambda s: s["value"])

        high_val = sig_high["value"]
        low_val = sig_low["value"]
        price_range = high_val - low_val

        if price_range < 1e-6:
            return {
                "retracements": {},
                "extensions": {},
                "trade_zones": [],
                "fib_confluence": [],
            }

        # Calculate retracements based on trend
        if trend == "UPTREND":
            # Pullback levels from the high (where to buy)
            retracements = self._calc_uptrend_retracement(high_val, low_val)
            extensions = self._calc_uptrend_extensions(high_val, low_val)
        elif trend == "DOWNTREND":
            # Pullback levels from the low (where to sell)
            retracements = self._calc_downtrend_retracement(high_val, low_val)
            extensions = self._calc_downtrend_extensions(high_val, low_val)
        else:
            # Ranging — calculate both
            retracements = self._calc_uptrend_retracement(high_val, low_val)
            extensions = self._calc_uptrend_extensions(high_val, low_val)

        # Identify trade zones (where Fib levels confluence with S/R)
        trade_zones = self._identify_trade_zones(retracements, extensions, trend)

        # Find Fib confluence zones (where multiple Fib levels cluster)
        fib_confluence = self._find_fib_confluence(retracements)

        return {
            "trend": trend,
            "swing_high": {"value": high_val, "index": sig_high.get("index", 0)},
            "swing_low": {"value": low_val, "index": sig_low.get("index", 0)},
            "price_range": float(price_range),
            "retracements": retracements,
            "extensions": extensions,
            "trade_zones": trade_zones,
            "fib_confluence": fib_confluence,
            "golden_zone": self._get_golden_zone(retracements, trend),
        }

    def _calc_uptrend_retracement(self, high: float, low: float) -> dict:
        """Fib retracement for uptrend (levels where price may pull back to)."""
        price_range = high - low
        levels = {}
        for ratio in self.RETRACEMENT_RATIOS:
            level = high - price_range * ratio
            levels[ratio] = {
                "value": round(level, 2),
                "label": self.FIB_LABELS.get(ratio, f"{ratio:.1%}"),
                "importance": self.ZONE_IMPORTANCE.get(ratio, "LOW"),
                "type": "RETRACEMENT",
                "direction": "BUY_ZONE",
                "image_y_note": f"Price at {round(level, 2)}",
            }
        return levels

    def _calc_downtrend_retracement(self, high: float, low: float) -> dict:
        """Fib retracement for downtrend."""
        price_range = high - low
        levels = {}
        for ratio in self.RETRACEMENT_RATIOS:
            level = low + price_range * ratio
            levels[ratio] = {
                "value": round(level, 2),
                "label": self.FIB_LABELS.get(ratio, f"{ratio:.1%}"),
                "importance": self.ZONE_IMPORTANCE.get(ratio, "LOW"),
                "type": "RETRACEMENT",
                "direction": "SELL_ZONE",
                "image_y_note": f"Price at {round(level, 2)}",
            }
        return levels

    def _calc_uptrend_extensions(self, high: float, low: float) -> dict:
        """Fib extension targets for uptrend."""
        price_range = high - low
        levels = {}
        for ratio in self.EXTENSION_RATIOS:
            level = high + price_range * ratio
            levels[ratio] = {
                "value": round(level, 2),
                "label": self.FIB_LABELS.get(ratio, f"{ratio:.1%}"),
                "importance": self.ZONE_IMPORTANCE.get(ratio, "LOW"),
                "type": "EXTENSION",
                "direction": "TAKE_PROFIT",
            }
        return levels

    def _calc_downtrend_extensions(self, high: float, low: float) -> dict:
        """Fib extension targets for downtrend."""
        price_range = high - low
        levels = {}
        for ratio in self.EXTENSION_RATIOS:
            level = low - price_range * ratio
            levels[ratio] = {
                "value": round(level, 2),
                "label": self.FIB_LABELS.get(ratio, f"{ratio:.1%}"),
                "importance": self.ZONE_IMPORTANCE.get(ratio, "LOW"),
                "type": "EXTENSION",
                "direction": "TAKE_PROFIT",
            }
        return levels

    def _get_golden_zone(self, retracements: dict, trend: str) -> dict:
        """
        The 'Golden Zone' is the 61.8%-78.6% retracement area.
        This is where most professional traders look for entries.
        """
        if 0.618 in retracements and 0.786 in retracements:
            return {
                "name": "Golden Zone (61.8% - 78.6%)",
                "upper": retracements[0.618]["value"],
                "lower": retracements[0.786]["value"],
                "description": "The highest-probability reversal zone. Professional traders scale in here.",
                "strategy": (
                    "Place limit orders at 61.8% and 78.6%. "
                    "SL below 78.6% (for uptrend) or above 61.8% (for downtrend). "
                    "Target 161.8% extension."
                ),
                "trend": trend,
            }
        return {}

    def _identify_trade_zones(
        self, retracements: dict, extensions: dict, trend: str
    ) -> list:
        """Identify actionable trade zones based on Fib + trend."""
        zones = []

        # Key retracement entry zones
        for ratio in [0.382, 0.5, 0.618, 0.786]:
            if ratio in retracements:
                level = retracements[ratio]
                entry_type = "BUY" if trend in ["UPTREND", "RANGING"] else "SELL"

                zones.append(
                    {
                        "zone_name": f"Fib {level['label']} Retracement",
                        "price": level["value"],
                        "action": f"LOOK_FOR_{entry_type}_ENTRY",
                        "importance": level["importance"],
                        "reason": (
                            f"Price pulling back to {level['label']} in {trend}. "
                            f"{'Shallow pullback = strong trend' if ratio < 0.5 else 'Deep pullback = potential value entry'}."
                        ),
                    }
                )

        # Key extension target zones
        for ratio in [1.0, 1.272, 1.618, 2.618]:
            if ratio in extensions:
                level = extensions[ratio]
                zones.append(
                    {
                        "zone_name": f"Fib {level['label']} Extension",
                        "price": level["value"],
                        "action": "TAKE_PROFIT_TARGET",
                        "importance": level["importance"],
                        "reason": (
                            f"Fibonacci extension at {level['label']}. "
                            f"{'Most common TP target' if ratio == 1.618 else 'Extended target for strong moves'}."
                        ),
                    }
                )

        return zones

    def _find_fib_confluence(self, retracements: dict) -> list:
        """Find where Fib levels cluster (confluence = stronger zone)."""
        confluences = []

        # Check for clustering of important levels
        important_ratios = [r for r in [0.5, 0.618, 0.786] if r in retracements]
        if len(important_ratios) >= 2:
            values = [retracements[r]["value"] for r in important_ratios]
            price_range = max(values) - min(values)
            avg_value = sum(values) / len(values)

            if price_range < abs(avg_value) * 0.05:  # Within 5% of each other
                confluences.append(
                    {
                        "name": "Fib Confluence Zone",
                        "center": round(avg_value, 2),
                        "range": round(price_range, 2),
                        "ratios": [
                            self.FIB_LABELS.get(r, f"{r:.1%}") for r in important_ratios
                        ],
                        "strength": "HIGH",
                        "note": "Multiple Fib levels cluster here — high-probability reaction zone",
                    }
                )

        return confluences
