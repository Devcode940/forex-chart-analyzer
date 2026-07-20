"""
Pattern Detector Module
Detects chart patterns: Head & Shoulders, Double Top/Bottom, Wedges, Triangles, Channels, Flags, Breakouts.
"""

import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d


class PatternDetector:
    """Detects geometric chart patterns from price data."""

    def __init__(self):
        self.detected_patterns = []

    def detect_all(self, price_series: dict, image_height: int) -> list:
        """Run all pattern detection algorithms."""
        self.detected_patterns = []

        smoothed = np.array(price_series.get("smoothed", []))
        x_positions = price_series.get("x", [])
        highs = price_series.get("highs", [])
        lows = price_series.get("lows", [])

        if len(smoothed) < 10:
            return self.detected_patterns

        # Detect swing points for pattern recognition
        swing_points = self._get_swing_points(smoothed)

        # Run individual pattern detectors
        self._detect_head_and_shoulders(swing_points, x_positions, image_height, smoothed)
        self._detect_inverse_head_and_shoulders(swing_points, x_positions, image_height, smoothed)
        self._detect_double_top(swing_points, x_positions, image_height, smoothed)
        self._detect_double_bottom(swing_points, x_positions, image_height, smoothed)
        self._detect_ascending_triangle(swing_points, x_positions, image_height, smoothed)
        self._detect_descending_triangle(swing_points, x_positions, image_height, smoothed)
        self._detect_symmetric_triangle(swing_points, x_positions, image_height, smoothed)
        self._detect_rising_wedge(swing_points, x_positions, image_height, smoothed)
        self._detect_falling_wedge(swing_points, x_positions, image_height, smoothed)
        self._detect_bull_flag(swing_points, x_positions, image_height, smoothed)
        self._detect_bear_flag(swing_points, x_positions, image_height, smoothed)
        self._detect_channel(swing_points, x_positions, image_height, smoothed)
        self._detect_breakout(smoothed, x_positions, image_height)

        # Sort by confidence
        self.detected_patterns.sort(key=lambda p: p["confidence"], reverse=True)

        return self.detected_patterns

    def _get_swing_points(self, smoothed: np.ndarray) -> dict:
        """Get refined swing highs and lows."""
        order = max(3, len(smoothed) // 15)

        maxima_idx = argrelextrema(smoothed, np.greater, order=order)[0]
        minima_idx = argrelextrema(smoothed, np.less, order=order)[0]

        return {
            "highs": [{"idx": int(i), "val": float(smoothed[i])} for i in maxima_idx],
            "lows": [{"idx": int(i), "val": float(smoothed[i])} for i in minima_idx]
        }

    def _detect_head_and_shoulders(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Head and Shoulders pattern (bearish reversal)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 3 or len(lows) < 2:
            return

        for i in range(len(highs) - 2):
            left_shoulder = highs[i]
            head = highs[i + 1]
            right_shoulder = highs[i + 2] if i + 2 < len(highs) else None

            if right_shoulder is None:
                continue

            # Head should be highest, shoulders roughly equal
            if head["val"] > left_shoulder["val"] and head["val"] > right_shoulder["val"]:
                shoulder_diff = abs(left_shoulder["val"] - right_shoulder["val"])
                head_prominence = head["val"] - max(left_shoulder["val"], right_shoulder["val"])

                if head_prominence > 0 and shoulder_diff / (head_prominence + 1e-6) < 2.0:
                    # Find neckline
                    relevant_lows = [l for l in lows if left_shoulder["idx"] <= l["idx"] <= right_shoulder["idx"]]
                    if len(relevant_lows) >= 2:
                        confidence = min(0.95, 0.6 + head_prominence / (np.max(smoothed) - np.min(smoothed) + 1e-6))

                        self.detected_patterns.append({
                            "name": "Head & Shoulders",
                            "type": "BEARISH_REVERSAL",
                            "confidence": confidence,
                            "description": "Classic bearish reversal pattern with left shoulder, head, and right shoulder. Watch for neckline break.",
                            "key_points": {
                                "left_shoulder": {"idx": left_shoulder["idx"], "val": left_shoulder["val"]},
                                "head": {"idx": head["idx"], "val": head["val"]},
                                "right_shoulder": {"idx": right_shoulder["idx"], "val": right_shoulder["val"]},
                                "neckline_lows": [{"idx": l["idx"], "val": l["val"]} for l in relevant_lows[:2]]
                            },
                            "x_range": (
                                x_pos[left_shoulder["idx"]] if left_shoulder["idx"] < len(x_pos) else 0,
                                x_pos[right_shoulder["idx"]] if right_shoulder["idx"] < len(x_pos) else 0
                            ),
                            "target_direction": "DOWN"
                        })

    def _detect_inverse_head_and_shoulders(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Inverse Head and Shoulders (bullish reversal)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(lows) < 3 or len(highs) < 2:
            return

        for i in range(len(lows) - 2):
            left_shoulder = lows[i]
            head = lows[i + 1]
            right_shoulder = lows[i + 2] if i + 2 < len(lows) else None

            if right_shoulder is None:
                continue

            # Head should be lowest, shoulders roughly equal
            if head["val"] < left_shoulder["val"] and head["val"] < right_shoulder["val"]:
                shoulder_diff = abs(left_shoulder["val"] - right_shoulder["val"])
                head_prominence = min(left_shoulder["val"], right_shoulder["val"]) - head["val"]

                if head_prominence > 0 and shoulder_diff / (head_prominence + 1e-6) < 2.0:
                    confidence = min(0.95, 0.6 + head_prominence / (np.max(smoothed) - np.min(smoothed) + 1e-6))

                    self.detected_patterns.append({
                        "name": "Inverse Head & Shoulders",
                        "type": "BULLISH_REVERSAL",
                        "confidence": confidence,
                        "description": "Bullish reversal pattern. Watch for neckline breakout to confirm trend change upward.",
                        "key_points": {
                            "left_shoulder": {"idx": left_shoulder["idx"], "val": left_shoulder["val"]},
                            "head": {"idx": head["idx"], "val": head["val"]},
                            "right_shoulder": {"idx": right_shoulder["idx"], "val": right_shoulder["val"]}
                        },
                        "x_range": (
                            x_pos[left_shoulder["idx"]] if left_shoulder["idx"] < len(x_pos) else 0,
                            x_pos[right_shoulder["idx"]] if right_shoulder["idx"] < len(x_pos) else 0
                        ),
                        "target_direction": "UP"
                    })

    def _detect_double_top(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Double Top pattern (bearish reversal)."""
        highs = swings["highs"]
        if len(highs) < 2:
            return

        price_range = np.max(smoothed) - np.min(smoothed) + 1e-6

        for i in range(len(highs) - 1):
            h1 = highs[i]
            h2 = highs[i + 1]

            diff = abs(h1["val"] - h2["val"])
            if diff / price_range < 0.08:  # Within 8% of each other
                confidence = min(0.90, 0.65 + (1 - diff / price_range) * 0.25)

                self.detected_patterns.append({
                    "name": "Double Top",
                    "type": "BEARISH_REVERSAL",
                    "confidence": confidence,
                    "description": "Two peaks at similar price levels. A break below the valley between them confirms bearish reversal.",
                    "key_points": {
                        "top1": {"idx": h1["idx"], "val": h1["val"]},
                        "top2": {"idx": h2["idx"], "val": h2["val"]}
                    },
                    "x_range": (
                        x_pos[h1["idx"]] if h1["idx"] < len(x_pos) else 0,
                        x_pos[h2["idx"]] if h2["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "DOWN"
                })

    def _detect_double_bottom(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Double Bottom pattern (bullish reversal)."""
        lows = swings["lows"]
        if len(lows) < 2:
            return

        price_range = np.max(smoothed) - np.min(smoothed) + 1e-6

        for i in range(len(lows) - 1):
            l1 = lows[i]
            l2 = lows[i + 1]

            diff = abs(l1["val"] - l2["val"])
            if diff / price_range < 0.08:
                confidence = min(0.90, 0.65 + (1 - diff / price_range) * 0.25)

                self.detected_patterns.append({
                    "name": "Double Bottom (W Pattern)",
                    "type": "BULLISH_REVERSAL",
                    "confidence": confidence,
                    "description": "Two valleys at similar price levels. A break above the peak between them confirms bullish reversal.",
                    "key_points": {
                        "bottom1": {"idx": l1["idx"], "val": l1["val"]},
                        "bottom2": {"idx": l2["idx"], "val": l2["val"]}
                    },
                    "x_range": (
                        x_pos[l1["idx"]] if l1["idx"] < len(x_pos) else 0,
                        x_pos[l2["idx"]] if l2["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "UP"
                })

    def _detect_ascending_triangle(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Ascending Triangle (bullish continuation)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            return

        # Check if highs are flat and lows are rising
        high_vals = [h["val"] for h in highs[-3:]]
        low_vals = [l["val"] for l in lows[-3:]]

        if len(high_vals) >= 2:
            high_range = max(high_vals) - min(high_vals)
            price_range = np.max(smoothed) - np.min(smoothed) + 1e-6

            if high_range / price_range < 0.06:
                # Flat resistance
                low_slope = np.polyfit(range(len(low_vals)), low_vals, 1)[0] if len(low_vals) >= 2 else 0
                if low_slope > 0:
                    confidence = min(0.85, 0.6 + low_slope * 10)

                    self.detected_patterns.append({
                        "name": "Ascending Triangle",
                        "type": "BULLISH_CONTINUATION",
                        "confidence": confidence,
                        "description": "Flat resistance with rising support. Expect breakout upward with increasing probability.",
                        "key_points": {
                            "resistance": float(np.mean(high_vals)),
                            "rising_lows": low_vals
                        },
                        "x_range": (
                            x_pos[highs[0]["idx"]] if highs[0]["idx"] < len(x_pos) else 0,
                            x_pos[highs[-1]["idx"]] if highs[-1]["idx"] < len(x_pos) else 0
                        ),
                        "target_direction": "UP"
                    })

    def _detect_descending_triangle(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Descending Triangle (bearish continuation)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            return

        low_vals = [l["val"] for l in lows[-3:]]
        high_vals = [h["val"] for h in highs[-3:]]

        if len(low_vals) >= 2:
            low_range = max(low_vals) - min(low_vals)
            price_range = np.max(smoothed) - np.min(smoothed) + 1e-6

            if low_range / price_range < 0.06:
                high_slope = np.polyfit(range(len(high_vals)), high_vals, 1)[0] if len(high_vals) >= 2 else 0
                if high_slope < 0:
                    confidence = min(0.85, 0.6 + abs(high_slope) * 10)

                    self.detected_patterns.append({
                        "name": "Descending Triangle",
                        "type": "BEARISH_CONTINUATION",
                        "confidence": confidence,
                        "description": "Flat support with descending highs. Expect breakdown with selling pressure building.",
                        "key_points": {
                            "support": float(np.mean(low_vals)),
                            "descending_highs": high_vals
                        },
                        "x_range": (
                            x_pos[lows[0]["idx"]] if lows[0]["idx"] < len(x_pos) else 0,
                            x_pos[lows[-1]["idx"]] if lows[-1]["idx"] < len(x_pos) else 0
                        ),
                        "target_direction": "DOWN"
                    })

    def _detect_symmetric_triangle(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Symmetric Triangle (continuation, direction depends on breakout)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            return

        high_vals = [h["val"] for h in highs[-3:]]
        low_vals = [l["val"] for l in lows[-3:]]

        if len(high_vals) >= 2 and len(low_vals) >= 2:
            high_slope = np.polyfit(range(len(high_vals)), high_vals, 1)[0]
            low_slope = np.polyfit(range(len(low_vals)), low_vals, 1)[0]

            if high_slope < -0.01 and low_slope > 0.01:
                confidence = 0.65

                self.detected_patterns.append({
                    "name": "Symmetric Triangle",
                    "type": "CONTINUATION",
                    "confidence": confidence,
                    "description": "Converging trend lines suggest a breakout is imminent. Watch for volume confirmation.",
                    "key_points": {
                        "descending_highs": high_vals,
                        "rising_lows": low_vals
                    },
                    "x_range": (
                        x_pos[highs[0]["idx"]] if highs[0]["idx"] < len(x_pos) else 0,
                        x_pos[highs[-1]["idx"]] if highs[-1]["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "PENDING"
                })

    def _detect_rising_wedge(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Rising Wedge (bearish reversal)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            return

        high_vals = [h["val"] for h in highs[-3:]]
        low_vals = [l["val"] for l in lows[-3:]]

        if len(high_vals) >= 2 and len(low_vals) >= 2:
            high_slope = np.polyfit(range(len(high_vals)), high_vals, 1)[0]
            low_slope = np.polyfit(range(len(low_vals)), low_vals, 1)[0]

            # Both rising, but lows rising faster (converging upward)
            if high_slope > 0.01 and low_slope > 0.01 and low_slope > high_slope:
                confidence = 0.7

                self.detected_patterns.append({
                    "name": "Rising Wedge",
                    "type": "BEARISH_REVERSAL",
                    "confidence": confidence,
                    "description": "Both trend lines rising but converging. Typically breaks downward despite upward appearance.",
                    "key_points": {
                        "rising_highs": high_vals,
                        "rising_lows": low_vals
                    },
                    "x_range": (
                        x_pos[highs[0]["idx"]] if highs[0]["idx"] < len(x_pos) else 0,
                        x_pos[highs[-1]["idx"]] if highs[-1]["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "DOWN"
                })

    def _detect_falling_wedge(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Falling Wedge (bullish reversal)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            return

        high_vals = [h["val"] for h in highs[-3:]]
        low_vals = [l["val"] for l in lows[-3:]]

        if len(high_vals) >= 2 and len(low_vals) >= 2:
            high_slope = np.polyfit(range(len(high_vals)), high_vals, 1)[0]
            low_slope = np.polyfit(range(len(low_vals)), low_vals, 1)[0]

            # Both falling, but highs falling faster (converging downward)
            if high_slope < -0.01 and low_slope < -0.01 and high_slope < low_slope:
                confidence = 0.7

                self.detected_patterns.append({
                    "name": "Falling Wedge",
                    "type": "BULLISH_REVERSAL",
                    "confidence": confidence,
                    "description": "Both trend lines falling but converging. Typically breaks upward despite downward appearance.",
                    "key_points": {
                        "falling_highs": high_vals,
                        "falling_lows": low_vals
                    },
                    "x_range": (
                        x_pos[highs[0]["idx"]] if highs[0]["idx"] < len(x_pos) else 0,
                        x_pos[highs[-1]["idx"]] if highs[-1]["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "UP"
                })

    def _detect_bull_flag(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Bull Flag (bullish continuation)."""
        n = len(smoothed)
        if n < 10:
            return

        # Look for a sharp rise followed by a slight downward channel
        first_half = smoothed[:n // 3]
        second_half = smoothed[n // 3:2 * n // 3]
        last_part = smoothed[2 * n // 3:]

        if len(first_half) > 0 and len(second_half) > 0:
            rise = first_half[-1] - first_half[0]
            consolidation_slope = np.polyfit(range(len(second_half)), second_half, 1)[0] if len(second_half) >= 2 else 0

            if rise > 0 and -0.02 < consolidation_slope < 0:
                confidence = 0.65

                self.detected_patterns.append({
                    "name": "Bull Flag",
                    "type": "BULLISH_CONTINUATION",
                    "confidence": confidence,
                    "description": "Strong upward move followed by slight downward consolidation. Expect continuation upward.",
                    "key_points": {
                        "pole_rise": float(rise),
                        "flag_slope": float(consolidation_slope)
                    },
                    "x_range": (0, x_pos[-1] if x_pos else 0),
                    "target_direction": "UP"
                })

    def _detect_bear_flag(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect Bear Flag (bearish continuation)."""
        n = len(smoothed)
        if n < 10:
            return

        first_half = smoothed[:n // 3]
        second_half = smoothed[n // 3:2 * n // 3]

        if len(first_half) > 0 and len(second_half) > 0:
            drop = first_half[-1] - first_half[0]
            consolidation_slope = np.polyfit(range(len(second_half)), second_half, 1)[0] if len(second_half) >= 2 else 0

            if drop < 0 and 0 < consolidation_slope < 0.02:
                confidence = 0.65

                self.detected_patterns.append({
                    "name": "Bear Flag",
                    "type": "BEARISH_CONTINUATION",
                    "confidence": confidence,
                    "description": "Strong downward move followed by slight upward consolidation. Expect continuation downward.",
                    "key_points": {
                        "pole_drop": float(drop),
                        "flag_slope": float(consolidation_slope)
                    },
                    "x_range": (0, x_pos[-1] if x_pos else 0),
                    "target_direction": "DOWN"
                })

    def _detect_channel(self, swings: dict, x_pos: list, img_h: int, smoothed: np.ndarray):
        """Detect price channel (rising or falling)."""
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            return

        high_vals = [h["val"] for h in highs]
        high_xs = [h["idx"] for h in highs]
        low_vals = [l["val"] for l in lows]
        low_xs = [l["idx"] for l in lows]

        if len(high_vals) >= 2 and len(low_vals) >= 2:
            high_slope = np.polyfit(high_xs, high_vals, 1)[0]
            low_slope = np.polyfit(low_xs, low_vals, 1)[0]

            # Both slopes should be similar for a channel
            slope_diff = abs(high_slope - low_slope)
            avg_slope = (high_slope + low_slope) / 2

            if slope_diff < abs(avg_slope) * 0.5 + 0.02 and abs(avg_slope) > 0.01:
                direction = "RISING" if avg_slope > 0 else "FALLING"
                bias = "BULLISH" if avg_slope > 0 else "BEARISH"
                confidence = 0.60

                self.detected_patterns.append({
                    "name": f"{direction} Channel",
                    "type": f"{bias}_CONTINUATION",
                    "confidence": confidence,
                    "description": f"Parallel {direction.lower()} trend lines form a channel. Trade within channel or wait for breakout.",
                    "key_points": {
                        "upper_slope": float(high_slope),
                        "lower_slope": float(low_slope),
                        "channel_width": float(np.mean(high_vals) - np.mean(low_vals))
                    },
                    "x_range": (
                        x_pos[highs[0]["idx"]] if highs[0]["idx"] < len(x_pos) else 0,
                        x_pos[highs[-1]["idx"]] if highs[-1]["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "UP" if avg_slope > 0 else "DOWN"
                })

    def _detect_breakout(self, smoothed: np.ndarray, x_pos: list, img_h: int):
        """Detect potential breakout zones where price approaches key levels."""
        if len(smoothed) < 5:
            return

        n = len(smoothed)
        recent = smoothed[-n // 4:]

        if len(recent) < 3:
            return

        # Calculate recent momentum
        recent_slope = np.polyfit(range(len(recent)), recent, 1)[0]
        recent_volatility = np.std(recent)

        # Check if price is near recent extremes
        near_high = smoothed[-1] > np.percentile(smoothed, 85)
        near_low = smoothed[-1] < np.percentile(smoothed, 15)

        if near_high and recent_slope > 0:
            self.detected_patterns.append({
                "name": "Bullish Breakout Attempt",
                "type": "BREAKOUT",
                "confidence": 0.55,
                "description": "Price approaching resistance with upward momentum. Watch for volume and candle confirmation.",
                "key_points": {
                    "resistance_level": float(np.percentile(smoothed, 95)),
                    "current_momentum": float(recent_slope)
                },
                "x_range": (x_pos[-n // 4] if len(x_pos) > n // 4 else 0, x_pos[-1] if x_pos else 0),
                "target_direction": "UP"
            })

        if near_low and recent_slope < 0:
            self.detected_patterns.append({
                "name": "Bearish Breakdown Attempt",
                "type": "BREAKDOWN",
                "confidence": 0.55,
                "description": "Price approaching support with downward momentum. Watch for breakdown confirmation.",
                "key_points": {
                    "support_level": float(np.percentile(smoothed, 5)),
                    "current_momentum": float(recent_slope)
                },
                "x_range": (x_pos[-n // 4] if len(x_pos) > n // 4 else 0, x_pos[-1] if x_pos else 0),
                "target_direction": "DOWN"
            })
