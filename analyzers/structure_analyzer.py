"""
Market Structure Analyzer Module
Analyzes market structure: trend direction, swing highs/lows, market phases.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


class StructureAnalyzer:
    """Analyzes market structure from extracted price data."""

    def __init__(self):
        self.swing_highs = []
        self.swing_lows = []
        self.trend_direction = "RANGING"
        self.structure_points = []

    def analyze(self, price_points: list, image_height: int) -> dict:
        """
        Full structure analysis pipeline.
        In images, y=0 is top, so lower y = higher price.
        """
        if len(price_points) < 5:
            return {"error": "Insufficient price data for structure analysis"}

        # Extract center y-values as a "price" series (inverted because image y-axis)
        centers = [p["center_y"] for p in price_points]
        tops = [p["top_y"] for p in price_points]
        bottoms = [p["bottom_y"] for p in price_points]
        x_positions = [p["x"] for p in price_points]

        # Invert: in image, low y = high price
        inverted_centers = [image_height - c for c in centers]
        inverted_tops = [image_height - t for t in tops]
        inverted_bottoms = [image_height - b for b in bottoms]

        # Smooth the series for swing detection
        smoothed = gaussian_filter1d(np.array(inverted_centers), sigma=3)

        # Detect swing highs and lows
        self._detect_swings(smoothed, x_positions, image_height)

        # Determine trend
        trend = self._determine_trend()

        # Identify market structure breaks
        breaks = self._detect_structure_breaks()

        # Identify market phases
        phases = self._identify_phases(smoothed)

        return {
            "trend_direction": trend["direction"],
            "trend_strength": trend["strength"],
            "swing_highs": self.swing_highs,
            "swing_lows": self.swing_lows,
            "structure_breaks": breaks,
            "phases": phases,
            "price_series": {
                "x": x_positions,
                "centers": inverted_centers,
                "highs": inverted_tops,
                "lows": inverted_bottoms,
                "smoothed": smoothed.tolist(),
            },
        }

    def _detect_swings(
        self, smoothed: np.ndarray, x_positions: list, image_height: int
    ):
        """Detect swing highs and swing lows in the price series."""
        order = max(3, len(smoothed) // 20)

        # Local maxima = swing highs (in inverted coords)
        maxima_indices = argrelextrema(smoothed, np.greater, order=order)[0]
        # Local minima = swing lows
        minima_indices = argrelextrema(smoothed, np.less, order=order)[0]

        self.swing_highs = []
        for idx in maxima_indices:
            if idx < len(x_positions):
                self.swing_highs.append(
                    {
                        "index": int(idx),
                        "x": x_positions[idx],
                        "price_y": int(
                            image_height - smoothed[idx]
                        ),  # Convert back to image coords
                        "value": float(smoothed[idx]),
                    }
                )

        self.swing_lows = []
        for idx in minima_indices:
            if idx < len(x_positions):
                self.swing_lows.append(
                    {
                        "index": int(idx),
                        "x": x_positions[idx],
                        "price_y": int(image_height - smoothed[idx]),
                        "value": float(smoothed[idx]),
                    }
                )

    def _determine_trend(self) -> dict:
        """Determine the overall trend direction and strength."""
        if len(self.swing_highs) < 2 and len(self.swing_lows) < 2:
            return {"direction": "RANGING", "strength": 0.3}

        sh_values = [s["value"] for s in self.swing_highs]
        sl_values = [s["value"] for s in self.swing_lows]

        high_trend = self._calc_trend_slope(sh_values)
        low_trend = self._calc_trend_slope(sl_values)

        # Both trending up = uptrend
        if high_trend > 0.1 and low_trend > 0.1:
            strength = min(abs(high_trend), abs(low_trend))
            return {"direction": "UPTREND", "strength": min(strength * 5, 1.0)}

        # Both trending down = downtrend
        if high_trend < -0.1 and low_trend < -0.1:
            strength = min(abs(high_trend), abs(low_trend))
            return {"direction": "DOWNTREND", "strength": min(strength * 5, 1.0)}

        # Mixed = ranging
        return {"direction": "RANGING", "strength": 0.5}

    def _calc_trend_slope(self, values: list) -> float:
        """Calculate the slope of a trend line."""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        # Normalize
        return slope / (np.mean(y) + 1e-6)

    def _detect_structure_breaks(self) -> list:
        """Detect market structure breaks (BOS - Break of Structure)."""
        breaks = []
        all_swings = [(s, "high") for s in self.swing_highs] + [
            (s, "low") for s in self.swing_lows
        ]
        all_swings.sort(key=lambda x: x[0]["index"])

        for i in range(1, len(all_swings)):
            prev = all_swings[i - 1]
            curr = all_swings[i]

            # Bullish BOS: current swing low breaks above previous swing high
            if prev[1] == "high" and curr[1] == "low":
                if curr[0]["value"] > prev[0]["value"]:
                    breaks.append(
                        {
                            "type": "BULLISH_BOS",
                            "index": curr[0]["index"],
                            "x": curr[0]["x"],
                            "price_y": curr[0]["price_y"],
                        }
                    )

            # Bearish BOS: current swing high breaks below previous swing low
            if prev[1] == "low" and curr[1] == "high":
                if curr[0]["value"] < prev[0]["value"]:
                    breaks.append(
                        {
                            "type": "BEARISH_BOS",
                            "index": curr[0]["index"],
                            "x": curr[0]["x"],
                            "price_y": curr[0]["price_y"],
                        }
                    )

        return breaks

    def _identify_phases(self, smoothed: np.ndarray) -> list:
        """Identify market phases (accumulation, distribution, markup, markdown)."""
        phases = []
        n = len(smoothed)

        if n < 10:
            return [{"phase": "UNKNOWN", "start": 0, "end": n}]

        # Simple phase detection based on slope and volatility
        window = max(5, n // 6)
        for i in range(0, n - window, window // 2):
            segment = smoothed[i : i + window]
            slope = self._calc_trend_slope(segment.tolist())
            volatility = np.std(segment)

            if slope > 0.05:
                phase = "MARKUP" if volatility < np.std(smoothed) else "STRONG_UPTREND"
            elif slope < -0.05:
                phase = (
                    "MARKDOWN" if volatility < np.std(smoothed) else "STRONG_DOWNTREND"
                )
            else:
                phase = (
                    "ACCUMULATION"
                    if i < n // 3
                    else "DISTRIBUTION" if i > 2 * n // 3 else "CONSOLIDATION"
                )

            phases.append(
                {
                    "phase": phase,
                    "start": i,
                    "end": min(i + window, n),
                    "slope": float(slope),
                    "volatility": float(volatility),
                }
            )

        return phases
