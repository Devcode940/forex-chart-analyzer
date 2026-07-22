"""
Pattern Detector Module — Enhanced v2
Detects chart patterns with improved swing detection (scipy find_peaks),
symmetry scoring, quality metrics, and Cup & Handle detection.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class PatternDetector:
    """Detects geometric chart patterns from price data with enhanced accuracy."""

    def __init__(self):
        self.detected_patterns = []

    def detect_all(self, price_series: dict, image_height: int,
                   structure_context: dict = None) -> list:
        """Run all pattern detection algorithms with optional structure context.

                'trend_direction' for trend-alignment confidence scoring.

        """
        self.detected_patterns = []

        smoothed = np.array(price_series.get("smoothed", []))
        x_positions = price_series.get("x", [])

        if len(smoothed) < 15:
            return self.detected_patterns

        swing_points = self._get_swing_points(smoothed)
        trend_dir = (structure_context or {}).get("trend_direction", "NEUTRAL")

        # Core patterns
        self._detect_head_and_shoulders(swing_points, x_positions, smoothed, trend_dir)
        self._detect_inverse_head_and_shoulders(swing_points, x_positions, smoothed, trend_dir)
        self._detect_double_top(swing_points, x_positions, smoothed, trend_dir)
        self._detect_double_bottom(swing_points, x_positions, smoothed, trend_dir)
        self._detect_ascending_triangle(swing_points, x_positions, smoothed)
        self._detect_descending_triangle(swing_points, x_positions, smoothed)
        self._detect_symmetric_triangle(swing_points, x_positions, smoothed)
        self._detect_rising_wedge(swing_points, x_positions, smoothed)
        self._detect_falling_wedge(swing_points, x_positions, smoothed)
        self._detect_bull_flag(swing_points, x_positions, smoothed)
        self._detect_bear_flag(swing_points, x_positions, smoothed)
        self._detect_channel(swing_points, x_positions, smoothed)
        self._detect_breakout(smoothed, x_positions)
        self._detect_cup_and_handle(swing_points, x_positions, smoothed)

        self.detected_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return self.detected_patterns

    # ── Swing detection ──────────────────────────────────────────────

    def _get_swing_points(self, smoothed: np.ndarray) -> dict:
        """Improved swing detection with Gaussian smoothing and prominence."""
        smoothed = gaussian_filter1d(smoothed, sigma=1.5)
        distance = max(5, len(smoothed) // 25)
        prominence = np.ptp(smoothed) * 0.045

        highs_idx, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
        lows_idx, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)

        return {
            "highs": [{"idx": int(i), "val": float(smoothed[i])} for i in highs_idx],
            "lows": [{"idx": int(i), "val": float(smoothed[i])} for i in lows_idx]
        }

    # ── Confidence calculator ────────────────────────────────────────

    def _calculate_confidence(self, base: float, symmetry: float = 1.0,
                              trend_align: float = 1.0, quality: float = 1.0) -> float:
        """Compounded confidence from multiple independent factors.

                pattern matches expected trend context.
            quality: Data quality / noise score (0.7–1.0).

        """
        conf = base * symmetry * trend_align * quality
        return max(0.1, min(0.95, conf))

    def _symmetry_score(self, val1: float, val2: float, reference: float) -> float:
        """Compute symmetry between two values relative to a reference range.

        Returns 1.0 for perfect symmetry, decreasing toward 0.0 as asymmetry
        increases relative to the reference magnitude.
        """
        if reference < 1e-9:
            return 0.5
        diff = abs(val1 - val2)
        return max(0.0, 1.0 - diff / reference)

    def _trend_alignment(self, pattern_direction: str, trend_dir: str) -> float:
        """Score how well a pattern's expected direction aligns with the trend.

        Returns 1.2 for strong alignment, 0.8 for counter-trend, 1.0 for neutral.
        """
        if trend_dir == "NEUTRAL":
            return 1.0
        alignment_map = {
            ("UP", "UPTREND"): 1.2,
            ("DOWN", "DOWNTREND"): 1.2,
            ("UP", "DOWNTREND"): 0.8,
            ("DOWN", "UPTREND"): 0.8,
        }
        return alignment_map.get((pattern_direction, trend_dir), 1.0)

    def _quality_score(self, smoothed: np.ndarray, idx_start: int, idx_end: int) -> float:
        """Assess data quality in a pattern region based on noise level.

        Low noise = higher quality. Returns 0.7–1.0.
        """
        if idx_end <= idx_start + 2:
            return 0.7
        segment = smoothed[idx_start:idx_end + 1]
        if len(segment) < 3:
            return 0.7
        noise = np.std(np.diff(segment))
        signal = np.ptp(segment) + 1e-9
        snr = signal / (noise + 1e-9)
        return max(0.7, min(1.0, 0.5 + snr * 0.1))

    # ── Pattern: Head & Shoulders ────────────────────────────────────

    def _detect_head_and_shoulders(self, swings: dict, x_pos: list,
                                   smoothed: np.ndarray, trend_dir: str):
        """Detect Head and Shoulders pattern (bearish reversal)."""
        highs = swings["highs"]
        lows = swings["lows"]
        if len(highs) < 3 or len(lows) < 2:
            return

        price_range = np.ptp(smoothed) + 1e-9

        for i in range(len(highs) - 2):
            left_shoulder = highs[i]
            head = highs[i + 1]
            right_shoulder = highs[i + 2]

            # Head must be highest
            if not (head["val"] > left_shoulder["val"] and head["val"] > right_shoulder["val"]):
                continue

            head_prominence = head["val"] - max(left_shoulder["val"], right_shoulder["val"])
            if head_prominence <= 0:
                continue

            shoulder_diff = abs(left_shoulder["val"] - right_shoulder["val"])
            symmetry = self._symmetry_score(left_shoulder["val"], right_shoulder["val"], head_prominence)

            # Shoulders should be roughly equal relative to head prominence
            if shoulder_diff / (head_prominence + 1e-6) > 2.0:
                continue

            # Find neckline lows between shoulders
            relevant_lows = [l for l in lows
                             if left_shoulder["idx"] <= l["idx"] <= right_shoulder["idx"]]
            if len(relevant_lows) < 2:
                continue

            quality = self._quality_score(smoothed, left_shoulder["idx"], right_shoulder["idx"])
            trend_al = self._trend_alignment("DOWN", trend_dir)

            confidence = self._calculate_confidence(
                base=0.70,
                symmetry=symmetry,
                trend_align=trend_al,
                quality=quality
            )

            self.detected_patterns.append({
                "name": "Head & Shoulders",
                "type": "BEARISH_REVERSAL",
                "confidence": confidence,
                "symmetry": round(symmetry, 3),
                "quality_score": round(quality, 3),
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

    # ── Pattern: Inverse Head & Shoulders ────────────────────────────

    def _detect_inverse_head_and_shoulders(self, swings: dict, x_pos: list,
                                           smoothed: np.ndarray, trend_dir: str):
        """Detect Inverse Head and Shoulders (bullish reversal)."""
        highs = swings["highs"]
        lows = swings["lows"]
        if len(lows) < 3 or len(highs) < 2:
            return

        price_range = np.ptp(smoothed) + 1e-9

        for i in range(len(lows) - 2):
            left_shoulder = lows[i]
            head = lows[i + 1]
            right_shoulder = lows[i + 2]

            # Head must be lowest
            if not (head["val"] < left_shoulder["val"] and head["val"] < right_shoulder["val"]):
                continue

            head_prominence = min(left_shoulder["val"], right_shoulder["val"]) - head["val"]
            if head_prominence <= 0:
                continue

            shoulder_diff = abs(left_shoulder["val"] - right_shoulder["val"])
            symmetry = self._symmetry_score(left_shoulder["val"], right_shoulder["val"], head_prominence)

            if shoulder_diff / (head_prominence + 1e-6) > 2.0:
                continue

            relevant_highs = [h for h in highs
                              if left_shoulder["idx"] <= h["idx"] <= right_shoulder["idx"]]
            if len(relevant_highs) < 2:
                continue

            quality = self._quality_score(smoothed, left_shoulder["idx"], right_shoulder["idx"])
            trend_al = self._trend_alignment("UP", trend_dir)

            confidence = self._calculate_confidence(
                base=0.70,
                symmetry=symmetry,
                trend_align=trend_al,
                quality=quality
            )

            self.detected_patterns.append({
                "name": "Inverse Head & Shoulders",
                "type": "BULLISH_REVERSAL",
                "confidence": confidence,
                "symmetry": round(symmetry, 3),
                "quality_score": round(quality, 3),
                "description": "Bullish reversal pattern. Watch for neckline breakout to confirm trend change upward.",
                "key_points": {
                    "left_shoulder": {"idx": left_shoulder["idx"], "val": left_shoulder["val"]},
                    "head": {"idx": head["idx"], "val": head["val"]},
                    "right_shoulder": {"idx": right_shoulder["idx"], "val": right_shoulder["val"]},
                    "neckline_highs": [{"idx": h["idx"], "val": h["val"]} for h in relevant_highs[:2]]
                },
                "x_range": (
                    x_pos[left_shoulder["idx"]] if left_shoulder["idx"] < len(x_pos) else 0,
                    x_pos[right_shoulder["idx"]] if right_shoulder["idx"] < len(x_pos) else 0
                ),
                "target_direction": "UP"
            })

    # ── Pattern: Double Top ──────────────────────────────────────────

    def _detect_double_top(self, swings: dict, x_pos: list,
                           smoothed: np.ndarray, trend_dir: str):
        """Detect Double Top pattern (bearish reversal)."""
        highs = swings["highs"]
        if len(highs) < 2:
            return

        price_range = np.ptp(smoothed) + 1e-9

        for i in range(len(highs) - 1):
            h1 = highs[i]
            h2 = highs[i + 1]
            diff = abs(h1["val"] - h2["val"])

            if diff / price_range < 0.08:
                symmetry = self._symmetry_score(h1["val"], h2["val"], price_range)
                quality = self._quality_score(smoothed, h1["idx"], h2["idx"])
                trend_al = self._trend_alignment("DOWN", trend_dir)

                confidence = self._calculate_confidence(
                    base=0.65,
                    symmetry=symmetry,
                    trend_align=trend_al,
                    quality=quality
                )

                self.detected_patterns.append({
                    "name": "Double Top",
                    "type": "BEARISH_REVERSAL",
                    "confidence": confidence,
                    "symmetry": round(symmetry, 3),
                    "quality_score": round(quality, 3),
                    "description": "Two peaks at similar price levels. A break below the valley confirms bearish reversal.",
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

    # ── Pattern: Double Bottom ───────────────────────────────────────

    def _detect_double_bottom(self, swings: dict, x_pos: list,
                              smoothed: np.ndarray, trend_dir: str):
        """Detect Double Bottom pattern (bullish reversal)."""
        lows = swings["lows"]
        if len(lows) < 2:
            return

        price_range = np.ptp(smoothed) + 1e-9

        for i in range(len(lows) - 1):
            l1 = lows[i]
            l2 = lows[i + 1]
            diff = abs(l1["val"] - l2["val"])

            if diff / price_range < 0.08:
                symmetry = self._symmetry_score(l1["val"], l2["val"], price_range)
                quality = self._quality_score(smoothed, l1["idx"], l2["idx"])
                trend_al = self._trend_alignment("UP", trend_dir)

                confidence = self._calculate_confidence(
                    base=0.65,
                    symmetry=symmetry,
                    trend_align=trend_al,
                    quality=quality
                )

                self.detected_patterns.append({
                    "name": "Double Bottom (W Pattern)",
                    "type": "BULLISH_REVERSAL",
                    "confidence": confidence,
                    "symmetry": round(symmetry, 3),
                    "quality_score": round(quality, 3),
                    "description": "Two valleys at similar price levels. A break above the peak confirms bullish reversal.",
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

    # ── Pattern: Ascending Triangle ──────────────────────────────────

    def _detect_ascending_triangle(self, swings: dict, x_pos: list,
                                   smoothed: np.ndarray):
        """Detect Ascending Triangle (bullish continuation)."""
        highs = swings["highs"]
        lows = swings["lows"]
        if len(highs) < 2 or len(lows) < 2:
            return

        high_vals = [h["val"] for h in highs[-3:]]
        low_vals = [l["val"] for l in lows[-3:]]
        price_range = np.ptp(smoothed) + 1e-9

        if len(high_vals) >= 2:
            high_range = max(high_vals) - min(high_vals)
            if high_range / price_range < 0.06:
                if len(low_vals) >= 2:
                    low_slope = np.polyfit(range(len(low_vals)), low_vals, 1)[0]
                    if low_slope > 0:
                        quality = self._quality_score(smoothed, highs[0]["idx"], highs[-1]["idx"])
                        confidence = self._calculate_confidence(
                            base=0.65, symmetry=1.0, trend_align=1.1, quality=quality
                        )

                        self.detected_patterns.append({
                            "name": "Ascending Triangle",
                            "type": "BULLISH_CONTINUATION",
                            "confidence": confidence,
                            "quality_score": round(quality, 3),
                            "description": "Flat resistance with rising support. Expect breakout upward.",
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

    # ── Pattern: Descending Triangle ─────────────────────────────────

    def _detect_descending_triangle(self, swings: dict, x_pos: list,
                                    smoothed: np.ndarray):
        """Detect Descending Triangle (bearish continuation)."""
        highs = swings["highs"]
        lows = swings["lows"]
        if len(highs) < 2 or len(lows) < 2:
            return

        low_vals = [l["val"] for l in lows[-3:]]
        high_vals = [h["val"] for h in highs[-3:]]
        price_range = np.ptp(smoothed) + 1e-9

        if len(low_vals) >= 2:
            low_range = max(low_vals) - min(low_vals)
            if low_range / price_range < 0.06:
                if len(high_vals) >= 2:
                    high_slope = np.polyfit(range(len(high_vals)), high_vals, 1)[0]
                    if high_slope < 0:
                        quality = self._quality_score(smoothed, lows[0]["idx"], lows[-1]["idx"])
                        confidence = self._calculate_confidence(
                            base=0.65, symmetry=1.0, trend_align=1.1, quality=quality
                        )

                        self.detected_patterns.append({
                            "name": "Descending Triangle",
                            "type": "BEARISH_CONTINUATION",
                            "confidence": confidence,
                            "quality_score": round(quality, 3),
                            "description": "Flat support with descending highs. Expect breakdown.",
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

    # ── Pattern: Symmetric Triangle ──────────────────────────────────

    def _detect_symmetric_triangle(self, swings: dict, x_pos: list,
                                   smoothed: np.ndarray):
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
                quality = self._quality_score(smoothed, highs[0]["idx"], highs[-1]["idx"])
                confidence = self._calculate_confidence(base=0.60, quality=quality)

                self.detected_patterns.append({
                    "name": "Symmetric Triangle",
                    "type": "CONTINUATION",
                    "confidence": confidence,
                    "quality_score": round(quality, 3),
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

    # ── Pattern: Rising Wedge ────────────────────────────────────────

    def _detect_rising_wedge(self, swings: dict, x_pos: list,
                             smoothed: np.ndarray):
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

            if high_slope > 0.01 and low_slope > 0.01 and low_slope > high_slope:
                quality = self._quality_score(smoothed, highs[0]["idx"], highs[-1]["idx"])
                confidence = self._calculate_confidence(base=0.65, trend_align=1.05, quality=quality)

                self.detected_patterns.append({
                    "name": "Rising Wedge",
                    "type": "BEARISH_REVERSAL",
                    "confidence": confidence,
                    "quality_score": round(quality, 3),
                    "description": "Both trend lines rising but converging. Typically breaks downward.",
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

    # ── Pattern: Falling Wedge ───────────────────────────────────────

    def _detect_falling_wedge(self, swings: dict, x_pos: list,
                              smoothed: np.ndarray):
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

            if high_slope < -0.01 and low_slope < -0.01 and high_slope < low_slope:
                quality = self._quality_score(smoothed, highs[0]["idx"], highs[-1]["idx"])
                confidence = self._calculate_confidence(base=0.65, trend_align=1.05, quality=quality)

                self.detected_patterns.append({
                    "name": "Falling Wedge",
                    "type": "BULLISH_REVERSAL",
                    "confidence": confidence,
                    "quality_score": round(quality, 3),
                    "description": "Both trend lines falling but converging. Typically breaks upward.",
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

    # ── Pattern: Bull Flag ───────────────────────────────────────────

    def _detect_bull_flag(self, swings: dict, x_pos: list,
                          smoothed: np.ndarray):
        """Detect Bull Flag (bullish continuation)."""
        n = len(smoothed)
        if n < 12:
            return

        pole_end = n // 3
        flag_start = pole_end
        flag_end = 2 * n // 3

        pole = smoothed[:pole_end]
        flag = smoothed[flag_start:flag_end]

        if len(pole) < 3 or len(flag) < 3:
            return

        rise = pole[-1] - pole[0]
        flag_slope = np.polyfit(range(len(flag)), flag, 1)[0]

        if rise > 0 and -0.02 < flag_slope < 0:
            pole_strength = rise / (np.ptp(smoothed) + 1e-9)
            quality = self._quality_score(smoothed, 0, flag_end)
            confidence = self._calculate_confidence(
                base=0.60 + pole_strength * 0.15,
                symmetry=1.0,
                trend_align=1.1,
                quality=quality
            )

            self.detected_patterns.append({
                "name": "Bull Flag",
                "type": "BULLISH_CONTINUATION",
                "confidence": confidence,
                "quality_score": round(quality, 3),
                "description": "Strong upward move followed by slight downward consolidation. Expect continuation upward.",
                "key_points": {
                    "pole_rise": float(rise),
                    "flag_slope": float(flag_slope),
                    "pole_strength": round(float(pole_strength), 3)
                },
                "x_range": (x_pos[0] if x_pos else 0, x_pos[flag_end] if flag_end < len(x_pos) else 0),
                "target_direction": "UP"
            })

    # ── Pattern: Bear Flag ───────────────────────────────────────────

    def _detect_bear_flag(self, swings: dict, x_pos: list,
                          smoothed: np.ndarray):
        """Detect Bear Flag (bearish continuation)."""
        n = len(smoothed)
        if n < 12:
            return

        pole_end = n // 3
        flag_start = pole_end
        flag_end = 2 * n // 3

        pole = smoothed[:pole_end]
        flag = smoothed[flag_start:flag_end]

        if len(pole) < 3 or len(flag) < 3:
            return

        drop = pole[-1] - pole[0]
        flag_slope = np.polyfit(range(len(flag)), flag, 1)[0]

        if drop < 0 and 0 < flag_slope < 0.02:
            pole_strength = abs(drop) / (np.ptp(smoothed) + 1e-9)
            quality = self._quality_score(smoothed, 0, flag_end)
            confidence = self._calculate_confidence(
                base=0.60 + pole_strength * 0.15,
                symmetry=1.0,
                trend_align=1.1,
                quality=quality
            )

            self.detected_patterns.append({
                "name": "Bear Flag",
                "type": "BEARISH_CONTINUATION",
                "confidence": confidence,
                "quality_score": round(quality, 3),
                "description": "Strong downward move followed by slight upward consolidation. Expect continuation downward.",
                "key_points": {
                    "pole_drop": float(drop),
                    "flag_slope": float(flag_slope),
                    "pole_strength": round(float(pole_strength), 3)
                },
                "x_range": (x_pos[0] if x_pos else 0, x_pos[flag_end] if flag_end < len(x_pos) else 0),
                "target_direction": "DOWN"
            })

    # ── Pattern: Channel ─────────────────────────────────────────────

    def _detect_channel(self, swings: dict, x_pos: list,
                        smoothed: np.ndarray):
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

            slope_diff = abs(high_slope - low_slope)
            avg_slope = (high_slope + low_slope) / 2

            if slope_diff < abs(avg_slope) * 0.5 + 0.02 and abs(avg_slope) > 0.01:
                direction = "RISING" if avg_slope > 0 else "FALLING"
                bias = "BULLISH" if avg_slope > 0 else "BEARISH"

                channel_width = float(np.mean(high_vals) - np.mean(low_vals))
                price_range = np.ptp(smoothed) + 1e-9
                parallelism = max(0.0, 1.0 - slope_diff / (abs(avg_slope) + 0.01))
                quality = self._quality_score(smoothed, high_xs[0], high_xs[-1])

                confidence = self._calculate_confidence(
                    base=0.55,
                    symmetry=parallelism,
                    quality=quality
                )

                self.detected_patterns.append({
                    "name": f"{direction} Channel",
                    "type": f"{bias}_CONTINUATION",
                    "confidence": confidence,
                    "symmetry": round(parallelism, 3),
                    "quality_score": round(quality, 3),
                    "description": f"Parallel {direction.lower()} trend lines form a channel. Trade within channel or wait for breakout.",
                    "key_points": {
                        "upper_slope": float(high_slope),
                        "lower_slope": float(low_slope),
                        "channel_width": channel_width
                    },
                    "x_range": (
                        x_pos[highs[0]["idx"]] if highs[0]["idx"] < len(x_pos) else 0,
                        x_pos[highs[-1]["idx"]] if highs[-1]["idx"] < len(x_pos) else 0
                    ),
                    "target_direction": "UP" if avg_slope > 0 else "DOWN"
                })

    # ── Pattern: Breakout ────────────────────────────────────────────

    def _detect_breakout(self, smoothed: np.ndarray, x_pos: list):
        """Detect potential breakout zones where price approaches key levels."""
        if len(smoothed) < 5:
            return

        n = len(smoothed)
        recent = smoothed[-n // 4:]

        if len(recent) < 3:
            return

        recent_slope = np.polyfit(range(len(recent)), recent, 1)[0]
        near_high = smoothed[-1] > np.percentile(smoothed, 85)
        near_low = smoothed[-1] < np.percentile(smoothed, 15)

        if near_high and recent_slope > 0:
            self.detected_patterns.append({
                "name": "Bullish Breakout Attempt",
                "type": "BREAKOUT",
                "confidence": 0.55,
                "quality_score": 0.7,
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
                "quality_score": 0.7,
                "description": "Price approaching support with downward momentum. Watch for breakdown confirmation.",
                "key_points": {
                    "support_level": float(np.percentile(smoothed, 5)),
                    "current_momentum": float(recent_slope)
                },
                "x_range": (x_pos[-n // 4] if len(x_pos) > n // 4 else 0, x_pos[-1] if x_pos else 0),
                "target_direction": "DOWN"
            })

    # ── Pattern: Cup and Handle ──────────────────────────────────────

    def _detect_cup_and_handle(self, swings: dict, x_pos: list,
                               smoothed: np.ndarray):
        """Detect Cup and Handle pattern (bullish continuation).

        The cup is a U-shaped price decline and recovery. The handle is a
        small downward consolidation near the right rim of the cup. Breakout
        above the handle signals bullish continuation.

        Requirements:
        - At least 3 swing lows forming the U-shape
        - At least 2 swing highs defining the rim
        - Handle forms a slight downward drift after the right rim
        """
        highs = swings["highs"]
        lows = swings["lows"]
        if len(lows) < 3 or len(highs) < 2:
            return

        price_range = np.ptp(smoothed) + 1e-9

        for i in range(len(highs) - 1):
            left_rim = highs[i]
            right_rim_candidate = None

            for j in range(i + 1, len(highs)):
                candidate = highs[j]
                rim_diff = abs(candidate["val"] - left_rim["val"])
                if rim_diff / price_range < 0.10:
                    right_rim_candidate = candidate
                    break

            if right_rim_candidate is None:
                continue

            # Check for U-shape: lows between the rims should form a bowl
            cup_lows = [l for l in lows
                        if left_rim["idx"] <= l["idx"] <= right_rim_candidate["idx"]]
            if len(cup_lows) < 1:
                continue

            # The lowest point in the cup should be roughly in the middle
            cup_bottom = min(cup_lows, key=lambda l: l["val"])
            rim_avg = (left_rim["val"] + right_rim_candidate["val"]) / 2
            cup_depth = rim_avg - cup_bottom["val"]

            if cup_depth <= 0:
                continue

            # Cup depth should be meaningful (at least 8% of price range)
            if cup_depth / price_range < 0.08:
                continue

            # Cup bottom should be in the middle third of the cup span
            cup_span = right_rim_candidate["idx"] - left_rim["idx"]
            if cup_span < 5:
                continue

            bottom_relative_pos = (cup_bottom["idx"] - left_rim["idx"]) / cup_span
            if not (0.2 <= bottom_relative_pos <= 0.8):
                continue

            # Symmetry: left half depth vs right half depth
            left_half_lows = [l for l in cup_lows
                              if l["idx"] <= cup_bottom["idx"]]
            right_half_lows = [l for l in cup_lows
                               if l["idx"] > cup_bottom["idx"]]

            left_depth = rim_avg - min(l["val"] for l in left_half_lows) if left_half_lows else cup_depth
            right_depth = rim_avg - min(l["val"] for l in right_half_lows) if right_half_lows else cup_depth
            symmetry = self._symmetry_score(left_depth, right_depth, cup_depth)

            # Check for handle: slight pullback after right rim
            handle_found = False
            handle_slope = 0.0
            handle_lows = [l for l in lows
                           if l["idx"] > right_rim_candidate["idx"]]

            if handle_lows:
                # Handle should be a shallow downward drift within 20% of cup span
                handle_end_idx = min(
                    right_rim_candidate["idx"] + int(cup_span * 0.3),
                    len(smoothed) - 1
                )
                handle_segment = smoothed[right_rim_candidate["idx"]:handle_end_idx + 1]
                if len(handle_segment) >= 3:
                    handle_slope = np.polyfit(range(len(handle_segment)), handle_segment, 1)[0]
                    # Handle should drift slightly downward (or sideways)
                    if -0.03 <= handle_slope <= 0.005:
                        handle_found = True

            # Compute confidence
            quality = self._quality_score(smoothed, left_rim["idx"],
                                          right_rim_candidate["idx"])
            handle_bonus = 1.1 if handle_found else 0.9

            confidence = self._calculate_confidence(
                base=0.60,
                symmetry=symmetry,
                trend_align=handle_bonus,
                quality=quality
            )

            handle_desc = " Handle consolidation confirms bullish setup." if handle_found else " No clear handle detected — watch for pullback entry."

            self.detected_patterns.append({
                "name": "Cup and Handle",
                "type": "BULLISH_CONTINUATION",
                "confidence": confidence,
                "symmetry": round(symmetry, 3),
                "quality_score": round(quality, 3),
                "has_handle": handle_found,
                "description": f"U-shaped cup with depth {cup_depth / price_range:.0%} of price range.{handle_desc}",
                "key_points": {
                    "left_rim": {"idx": left_rim["idx"], "val": left_rim["val"]},
                    "cup_bottom": {"idx": cup_bottom["idx"], "val": cup_bottom["val"]},
                    "right_rim": {"idx": right_rim_candidate["idx"], "val": right_rim_candidate["val"]},
                    "cup_depth_pct": round(cup_depth / price_range, 3),
                    "handle_slope": round(float(handle_slope), 4)
                },
                "x_range": (
                    x_pos[left_rim["idx"]] if left_rim["idx"] < len(x_pos) else 0,
                    x_pos[right_rim_candidate["idx"]] if right_rim_candidate["idx"] < len(x_pos) else 0
                ),
                "target_direction": "UP"
            })
