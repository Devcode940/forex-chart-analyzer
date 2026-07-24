"""
Indicator Line Detector Module
Detects technical indicators already drawn on the chart image:
- Moving Averages (colored lines — typically yellow, blue, red)
- Bollinger Bands (typically light blue/gray bands)
- Horizontal/diagonal trend lines
"""

import cv2
import numpy as np

class IndicatorDetector:
    """Detects technical indicator lines from the chart image."""

    def __init__(self):
        self.detected_indicators = []

    def detect_all(self, image: np.ndarray, edges: np.ndarray) -> list:
        """
        Detect indicator lines on the chart.
        Uses color segmentation and line tracking.
        """
        self.detected_indicators = []

        # 1. Detect moving average lines
        ma_lines = self._detect_moving_averages(image)
        self.detected_indicators.extend(ma_lines)

        # 2. Detect Bollinger Bands
        bollinger = self._detect_bollinger_bands(image)
        if bollinger:
            self.detected_indicators.append(bollinger)

        # 3. Detect trend lines (diagonal)
        trend_lines = self._detect_trend_lines(edges)
        self.detected_indicators.extend(trend_lines)

        # 4. Detect horizontal lines (manual S/R or pivot levels)
        horiz_lines = self._detect_horizontal_lines(edges, image)
        self.detected_indicators.extend(horiz_lines)

        return self.detected_indicators

    def _detect_moving_averages(self, image: np.ndarray) -> list:
        """
        Detect moving average lines by color.
        Common MA colors: Yellow (20 SMA), Blue (50 SMA), Red (200 SMA).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ma_lines = []

        # Yellow MA (common: 20 SMA)
        yellow_mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
        yellow_points = self._extract_line_points(yellow_mask)
        if len(yellow_points) > 10:
            ma_lines.append({
                "name": "Fast Moving Average (yellow)",
                "type": "MA_FAST",
                "color": "YELLOW",
                "likely_period": "20 SMA / 9 EMA",
                "points": yellow_points[:50],
                "point_count": len(yellow_points),
                "significance": "Short-term trend direction and dynamic support/resistance",
            })

        # Blue/Cyan MA (common: 50 SMA)
        blue_mask = cv2.inRange(hsv, np.array([100, 60, 60]), np.array([130, 255, 255]))
        blue_points = self._extract_line_points(blue_mask)
        if len(blue_points) > 10:
            ma_lines.append({
                "name": "Medium Moving Average (blue)",
                "type": "MA_MEDIUM",
                "color": "BLUE",
                "likely_period": "50 SMA / 21 EMA",
                "points": blue_points[:50],
                "point_count": len(blue_points),
                "significance": "Medium-term trend and institutional level",
            })

        # Red/Orange MA (common: 200 SMA)
        red_mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 60, 60]), np.array([180, 255, 255]))
        red_mask = red_mask | red_mask2
        red_points = self._extract_line_points(red_mask)
        if len(red_points) > 10:
            ma_lines.append({
                "name": "Slow Moving Average (red)",
                "type": "MA_SLOW",
                "color": "RED",
                "likely_period": "200 SMA / 100 EMA",
                "points": red_points[:50],
                "point_count": len(red_points),
                "significance": "Long-term trend — institutional traders watch this closely",
            })

        # White/Gray MA
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
        gray_points = self._extract_line_points(gray_mask)
        if len(gray_points) > 20:  # Need more points to distinguish from background
            ma_lines.append({
                "name": "Moving Average (white/gray)",
                "type": "MA_OTHER",
                "color": "WHITE",
                "likely_period": "Unknown",
                "points": gray_points[:50],
                "point_count": len(gray_points),
                "significance": "Dynamic trend indicator",
            })

        return ma_lines

    def _detect_bollinger_bands(self, image: np.ndarray) -> dict:
        """
        Detect Bollinger Bands by looking for paired parallel lines
        with a middle line, typically in light blue/gray.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Light blue/gray for Bollinger Bands
        band_mask = cv2.inRange(hsv, np.array([95, 20, 150]), np.array([115, 80, 220]))
        band_points = self._extract_line_points(band_mask)

        if len(band_points) > 20:
            return {
                "name": "Bollinger Bands",
                "type": "BOLLINGER",
                "points": band_points[:50],
                "point_count": len(band_points),
                "significance": (
                    "Price touching upper band = overbought. "
                    "Price touching lower band = oversold. "
                    "Band squeeze = potential breakout coming."
                ),
            }
        return None

    def _detect_trend_lines(self, edges: np.ndarray) -> list:
        """Detect diagonal trend lines using Hough transform."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=60, minLineLength=80, maxLineGap=15
        )

        trend_lines = []
        if lines is not None:
            for line in lines:
                coords = line.reshape(-1)
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                if 10 < abs(angle) < 80 and length > 80:
                    if angle < 0:  # Rising line (image coords: y decreases upward)
                        line_type = "RISING_TRENDLINE"
                        significance = "Support trend line — price bouncing off this suggests uptrend"
                    else:
                        line_type = "FALLING_TRENDLINE"
                        significance = "Resistance trend line — price rejecting here suggests downtrend"

                    trend_lines.append({
                        "name": f"Trend Line ({'rising' if angle < 0 else 'falling'})",
                        "type": line_type,
                        "start": (int(x1), int(y1)),
                        "end": (int(x2), int(y2)),
                        "angle": round(float(angle), 1),
                        "length": round(float(length), 1),
                        "significance": significance,
                    })

        return trend_lines[:5]  # Limit to top 5

    def _detect_horizontal_lines(self, edges: np.ndarray, image: np.ndarray) -> list:
        """Detect manually drawn horizontal lines (not grid lines)."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 2,
            threshold=100, minLineLength=150, maxLineGap=5
        )

        h_lines = []
        if lines is not None:
            for line in lines:
                coords = line.reshape(-1)
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                if abs(y2 - y1) < 3:  # Nearly horizontal

                    h, w = image.shape[:2]
                    if 0 < y1 < h and 0 < x1 < w and 0 < x2 < w:
                        # Sample colors along the line
                        mid_x = (x1 + x2) // 2
                        pixel_color = image[y1, mid_x]

                        # Not a grid line if it's colored (not just gray)
                        is_colored = (max(pixel_color) - min(pixel_color) > 50 or
                                     pixel_color[1] > 150 or pixel_color[0] > 150)

                        if is_colored:
                            h_lines.append({
                                "name": f"Horizontal Level at y={y1}",
                                "type": "HORIZONTAL_LEVEL",
                                "y_position": int(y1),
                                "x_range": (int(x1), int(x2)),
                                "significance": "Manually drawn level — likely a key S/R or alert level",
                            })

        return h_lines[:5]

    def _extract_line_points(self, mask: np.ndarray) -> list:
        """Extract ordered points from a binary mask (for MA line tracking)."""
        h, w = mask.shape
        points = []

        # Divide into vertical slices
        num_slices = min(w, 100)
        slice_width = max(w // num_slices, 1)

        for i in range(num_slices):
            x_start = i * slice_width
            x_end = min((i + 1) * slice_width, w)
            column = mask[:, x_start:x_end]

            rows = np.where(column > 0)
            if len(rows[0]) > 0:
                center_y = int(np.mean(rows[0]))
                center_x = (x_start + x_end) // 2
                points.append({"x": center_x, "y": center_y})

        return points

    def get_ma_crossovers(self, ma_lines: list) -> list:
        """
        Detect moving average crossovers (Golden Cross / Death Cross).
        Returns crossover events if multiple MAs are detected.
        """
        crossovers = []

        if len(ma_lines) < 2:
            return crossovers

        # Get points from different MAs
        fast_points = None
        slow_points = None

        for ma in ma_lines:
            if ma.get("type") == "MA_FAST":
                fast_points = ma.get("points", [])
            elif ma.get("type") == "MA_SLOW":
                slow_points = ma.get("points", [])

        if not fast_points or not slow_points:
            return crossovers

        # Find intersection points
        min_len = min(len(fast_points), len(slow_points))

        for i in range(1, min_len):
            fast_above_prev = fast_points[i - 1]["y"] < slow_points[i - 1]["y"]  # y inverted
            fast_above_curr = fast_points[i]["y"] < slow_points[i]["y"]

            if fast_above_prev != fast_above_curr:
                # Crossover detected
                if fast_above_curr:  # Fast crossed above slow
                    cross_type = "GOLDEN_CROSS"
                    signal = "BULLISH"
                    desc = "Fast MA crossed above slow MA (Golden Cross) — bullish signal"
                else:  # Fast crossed below slow
                    cross_type = "DEATH_CROSS"
                    signal = "BEARISH"
                    desc = "Fast MA crossed below slow MA (Death Cross) — bearish signal"

                crossovers.append({
                    "name": cross_type.replace("_", " ").title(),
                    "type": cross_type,
                    "signal": signal,
                    "x": fast_points[i]["x"],
                    "description": desc,
                    "significance": "MA crossovers are lagging signals — confirm with price action"
                })

        return crossovers

