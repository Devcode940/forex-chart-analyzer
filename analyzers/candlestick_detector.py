"""
Candlestick Pattern Detector Module
Detects individual candlestick patterns from extracted OHLC-like data:
Doji, Hammer, Shooting Star, Engulfing, Pin Bar, Morning/Evening Star, etc.
"""

from typing import Optional

import numpy as np


class CandlestickDetector:
    """Detects individual candlestick patterns from price series data."""

    def __init__(self):
        self.detected = []

    def detect_all(self, price_series: dict) -> list:
        """
        Detect candlestick patterns from extracted price data.
        Uses high/low/center series to approximate OHLC.
        """
        self.detected = []

        highs = np.array(price_series.get("highs", []))
        lows = np.array(price_series.get("lows", []))
        centers = np.array(price_series.get("smoothed", []))
        x_positions = price_series.get("x", [])

        if len(highs) < 5 or len(lows) < 5:
            return self.detected

        # Build approximate OHLC candles
        candles = self._build_candles(highs, lows, centers, x_positions)

        if len(candles) < 3:
            return self.detected

        # Run all detectors
        self._detect_doji(candles)
        self._detect_hammer(candles)
        self._detect_inverted_hammer(candles)
        self._detect_shooting_star(candles)
        self._detect_bullish_engulfing(candles)
        self._detect_bearish_engulfing(candles)
        self._detect_morning_star(candles)
        self._detect_evening_star(candles)
        self._detect_pin_bar_bull(candles)
        self._detect_pin_bar_bear(candles)
        self._detect_three_white_soldiers(candles)
        self._detect_three_black_crows(candles)
        self._detect_spinning_top(candles)
        self._detect_marubozu_bull(candles)
        self._detect_marubozu_bear(candles)
        self._detect_tweezer_top(candles)
        self._detect_tweezer_bottom(candles)

        self.detected.sort(key=lambda p: p["confidence"], reverse=True)
        return self.detected

    def _build_candles(self, highs, lows, centers, x_positions) -> list:
        """Build approximate OHLC candle structures."""
        candles = []
        for i in range(len(highs)):
            # Approximate open/close from center movement
            if i < len(centers) - 1:
                is_bullish = (
                    centers[i + 1] > centers[i] if i + 1 < len(centers) else True
                )
            else:
                is_bullish = True

            if is_bullish:
                open_price = lows[i] + (highs[i] - lows[i]) * 0.3
                close_price = lows[i] + (highs[i] - lows[i]) * 0.7
            else:
                open_price = lows[i] + (highs[i] - lows[i]) * 0.7
                close_price = lows[i] + (highs[i] - lows[i]) * 0.3

            body = abs(close_price - open_price)
            total_range = highs[i] - lows[i]

            upper_wick = highs[i] - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - lows[i]

            candle = {
                "index": i,
                "x": x_positions[i] if i < len(x_positions) else 0,
                "high": float(highs[i]),
                "low": float(lows[i]),
                "open": float(open_price),
                "close": float(close_price),
                "body": float(body),
                "range": float(total_range),
                "upper_wick": float(upper_wick),
                "lower_wick": float(lower_wick),
                "is_bullish": is_bullish,
                "body_ratio": float(body / total_range) if total_range > 0 else 0,
            }
            candles.append(candle)
        return candles

    def _add_pattern(
        self,
        name: str,
        category: str,
        signal: str,
        index: int,
        x: int,
        confidence: float,
        description: str,
        implication: str,
    ):
        self.detected.append(
            {
                "name": name,
                "category": category,
                "signal": signal,
                "index": index,
                "x": x,
                "confidence": min(confidence, 0.95),
                "description": description,
                "implication": implication,
            }
        )

    def _detect_doji(self, candles: list):
        """Doji: very small body relative to range — indecision."""
        for c in candles:
            if c["range"] < 1e-6:
                continue
            body_ratio = c["body"] / c["range"]
            if body_ratio < 0.1:
                conf = 0.8 * (1 - body_ratio / 0.1)
                self._add_pattern(
                    "Doji",
                    "indecision",
                    "REVERSAL_POSSIBLE",
                    c["index"],
                    c["x"],
                    conf,
                    "Very small body with long wicks. Market indecision — potential reversal.",
                    "Wait for confirmation candle. If at S/R, high reversal probability.",
                )

    def _detect_hammer(self, candles: list):
        """Hammer: small body at top, long lower wick — bullish reversal."""
        for i, c in enumerate(candles):
            if c["range"] < 1e-6:
                continue
            # Lower wick at least 2x body, upper wick small
            if (
                c["lower_wick"] > c["body"] * 2
                and c["upper_wick"] < c["body"] * 0.5
                and c["body_ratio"] < 0.4
            ):
                # Context: better after downtrend
                context_bonus = (
                    0.15 if i > 2 and candles[i - 2]["is_bullish"] is False else 0
                )
                self._add_pattern(
                    "Hammer",
                    "bullish_reversal",
                    "BUY",
                    c["index"],
                    c["x"],
                    0.7 + context_bonus,
                    "Small body at top with long lower wick. Sellers pushed down but buyers reclaimed.",
                    "Enter long on break above hammer high. SL below hammer low.",
                )

    def _detect_inverted_hammer(self, candles: list):
        """Inverted Hammer: small body at bottom, long upper wick — potential bullish reversal."""
        for i, c in enumerate(candles):
            if c["range"] < 1e-6:
                continue
            if (
                c["upper_wick"] > c["body"] * 2
                and c["lower_wick"] < c["body"] * 0.5
                and c["body_ratio"] < 0.4
                and not c["is_bullish"]
            ):
                self._add_pattern(
                    "Inverted Hammer",
                    "bullish_reversal",
                    "BUY",
                    c["index"],
                    c["x"],
                    0.6,
                    "Small body at bottom with long upper wick. Buyers tested upward.",
                    "Needs confirmation. Wait for bullish candle close above this high.",
                )

    def _detect_shooting_star(self, candles: list):
        """Shooting Star: small body at bottom, long upper wick at top — bearish reversal."""
        for i, c in enumerate(candles):
            if c["range"] < 1e-6:
                continue
            if (
                c["upper_wick"] > c["body"] * 2
                and c["lower_wick"] < c["body"] * 0.5
                and c["is_bullish"] is False
            ):
                context_bonus = (
                    0.15 if i > 2 and candles[i - 2].get("is_bullish", False) else 0
                )
                self._add_pattern(
                    "Shooting Star",
                    "bearish_reversal",
                    "SELL",
                    c["index"],
                    c["x"],
                    0.7 + context_bonus,
                    "Small body at bottom with long upper wick after rally. Buyers got rejected.",
                    "Enter short on break below shooting star low. SL above the high.",
                )

    def _detect_bullish_engulfing(self, candles: list):
        """Bullish Engulfing: current bullish candle fully engulfs previous bearish."""
        for i in range(1, len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            if (
                not prev["is_bullish"]
                and curr["is_bullish"]
                and curr["open"] <= prev["close"]
                and curr["close"] >= prev["open"]
                and curr["body"] > prev["body"]
            ):
                self._add_pattern(
                    "Bullish Engulfing",
                    "bullish_reversal",
                    "BUY",
                    curr["index"],
                    curr["x"],
                    0.75,
                    "Bullish candle completely engulfs prior bearish candle. Strong shift in momentum.",
                    "Enter long at close. SL below engulfing candle low. TP at next resistance.",
                )

    def _detect_bearish_engulfing(self, candles: list):
        """Bearish Engulfing: current bearish candle fully engulfs previous bullish."""
        for i in range(1, len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            if (
                prev["is_bullish"]
                and not curr["is_bullish"]
                and curr["open"] >= prev["close"]
                and curr["close"] <= prev["open"]
                and curr["body"] > prev["body"]
            ):
                self._add_pattern(
                    "Bearish Engulfing",
                    "bearish_reversal",
                    "SELL",
                    curr["index"],
                    curr["x"],
                    0.75,
                    "Bearish candle completely engulfs prior bullish candle. Sellers taking control.",
                    "Enter short at close. SL above engulfing candle high. TP at next support.",
                )

    def _detect_morning_star(self, candles: list):
        """Morning Star: bearish → small body → bullish — strong bullish reversal."""
        for i in range(2, len(candles)):
            first = candles[i - 2]
            second = candles[i - 1]
            third = candles[i]

            if (
                not first["is_bullish"]
                and second["body_ratio"] < 0.3
                and third["is_bullish"]
                and third["close"] > (first["open"] + first["close"]) / 2
            ):
                self._add_pattern(
                    "Morning Star",
                    "bullish_reversal",
                    "BUY",
                    third["index"],
                    third["x"],
                    0.8,
                    "3-candle reversal: bearish → indecision → bullish. High reliability pattern.",
                    "Enter long. SL below the star (middle candle) low. Strong reversal signal.",
                )

    def _detect_evening_star(self, candles: list):
        """Evening Star: bullish → small body → bearish — strong bearish reversal."""
        for i in range(2, len(candles)):
            first = candles[i - 2]
            second = candles[i - 1]
            third = candles[i]

            if (
                first["is_bullish"]
                and second["body_ratio"] < 0.3
                and not third["is_bullish"]
                and third["close"] < (first["open"] + first["close"]) / 2
            ):
                self._add_pattern(
                    "Evening Star",
                    "bearish_reversal",
                    "SELL",
                    third["index"],
                    third["x"],
                    0.8,
                    "3-candle reversal: bullish → indecision → bearish. High reliability pattern.",
                    "Enter short. SL above the star (middle candle) high. Strong reversal signal.",
                )

    def _detect_pin_bar_bull(self, candles: list):
        """Bullish Pin Bar: very long lower wick, tiny body at top — rejection of lower prices."""
        for c in candles:
            if c["range"] < 1e-6:
                continue
            wick_ratio = c["lower_wick"] / c["range"]
            if wick_ratio > 0.65 and c["body_ratio"] < 0.25 and c["is_bullish"]:
                self._add_pattern(
                    "Bullish Pin Bar",
                    "bullish_reversal",
                    "BUY",
                    c["index"],
                    c["x"],
                    0.75,
                    "Long lower wick rejecting lower prices. Classic Forex reversal signal.",
                    "Enter long on 50% retracement of pin bar. SL below the wick low.",
                )

    def _detect_pin_bar_bear(self, candles: list):
        """Bearish Pin Bar: very long upper wick, tiny body at bottom — rejection of higher prices."""
        for c in candles:
            if c["range"] < 1e-6:
                continue
            wick_ratio = c["upper_wick"] / c["range"]
            if wick_ratio > 0.65 and c["body_ratio"] < 0.25 and not c["is_bullish"]:
                self._add_pattern(
                    "Bearish Pin Bar",
                    "bearish_reversal",
                    "SELL",
                    c["index"],
                    c["x"],
                    0.75,
                    "Long upper wick rejecting higher prices. Classic Forex reversal signal.",
                    "Enter short on 50% retracement of pin bar. SL above the wick high.",
                )

    def _detect_three_white_soldiers(self, candles: list):
        """Three White Soldiers: 3 consecutive bullish candles, each opening within previous body."""
        for i in range(2, len(candles)):
            first = candles[i - 2]
            second = candles[i - 1]
            third = candles[i]

            if (
                first["is_bullish"]
                and second["is_bullish"]
                and third["is_bullish"]
                and second["open"] >= first["open"]
                and second["close"] > first["close"]
                and third["open"] >= second["open"]
                and third["close"] > second["close"]
            ):
                self._add_pattern(
                    "Three White Soldiers",
                    "bullish_continuation",
                    "BUY",
                    third["index"],
                    third["x"],
                    0.8,
                    "3 consecutive bullish candles with progressive closes. Very strong bullish signal.",
                    "Enter long. SL below first soldier. Strong trend confirmation.",
                )

    def _detect_three_black_crows(self, candles: list):
        """Three Black Crows: 3 consecutive bearish candles."""
        for i in range(2, len(candles)):
            first = candles[i - 2]
            second = candles[i - 1]
            third = candles[i]

            if (
                not first["is_bullish"]
                and not second["is_bullish"]
                and not third["is_bullish"]
                and second["open"] <= first["open"]
                and second["close"] < first["close"]
                and third["open"] <= second["open"]
                and third["close"] < second["close"]
            ):
                self._add_pattern(
                    "Three Black Crows",
                    "bearish_continuation",
                    "SELL",
                    third["index"],
                    third["x"],
                    0.8,
                    "3 consecutive bearish candles with progressive closes. Very strong bearish signal.",
                    "Enter short. SL above first crow. Strong trend confirmation.",
                )

    def _detect_spinning_top(self, candles: list):
        """Spinning Top: small body, roughly equal wicks — indecision."""
        for c in candles:
            if c["range"] < 1e-6:
                continue
            wick_diff = abs(c["upper_wick"] - c["lower_wick"]) / c["range"]
            if c["body_ratio"] < 0.2 and wick_diff < 0.2:
                self._add_pattern(
                    "Spinning Top",
                    "indecision",
                    "WAIT",
                    c["index"],
                    c["x"],
                    0.5,
                    "Small body with balanced wicks. Market is undecided — consolidation likely.",
                    "Do not enter. Wait for directional breakout from this zone.",
                )

    def _detect_marubozu_bull(self, candles: list):
        """Bullish Marubozu: large body, almost no wicks — extreme buying pressure."""
        for c in candles:
            if c["range"] < 1e-6:
                continue
            wick_total = (c["upper_wick"] + c["lower_wick"]) / c["range"]
            if c["is_bullish"] and c["body_ratio"] > 0.85 and wick_total < 0.15:
                self._add_pattern(
                    "Bullish Marubozu",
                    "bullish_continuation",
                    "BUY",
                    c["index"],
                    c["x"],
                    0.7,
                    "Large body with virtually no wicks. Buyers dominated the entire session.",
                    "Enter long on next candle. SL below marubozu open. Momentum is strong.",
                )

    def _detect_marubozu_bear(self, candles: list):
        """Bearish Marubozu: large body, almost no wicks — extreme selling pressure."""
        for c in candles:
            if c["range"] < 1e-6:
                continue
            wick_total = (c["upper_wick"] + c["lower_wick"]) / c["range"]
            if not c["is_bullish"] and c["body_ratio"] > 0.85 and wick_total < 0.15:
                self._add_pattern(
                    "Bearish Marubozu",
                    "bearish_continuation",
                    "SELL",
                    c["index"],
                    c["x"],
                    0.7,
                    "Large body with virtually no wicks. Sellers dominated the entire session.",
                    "Enter short on next candle. SL above marubozu open. Momentum is strong.",
                )

    def _detect_tweezer_top(self, candles: list):
        """Tweezer Top: two candles with same high — bearish reversal."""
        for i in range(1, len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            if (
                prev["is_bullish"]
                and not curr["is_bullish"]
                and abs(prev["high"] - curr["high"]) / max(prev["range"], 1) < 0.03
            ):
                self._add_pattern(
                    "Tweezer Top",
                    "bearish_reversal",
                    "SELL",
                    curr["index"],
                    curr["x"],
                    0.65,
                    "Two candles with matching highs. Double rejection at resistance level.",
                    "Enter short below the low of the second candle. SL above the shared high.",
                )

    def _detect_tweezer_bottom(self, candles: list):
        """Tweezer Bottom: two candles with same low — bullish reversal."""
        for i in range(1, len(candles)):
            prev = candles[i - 1]
            curr = candles[i]

            if (
                not prev["is_bullish"]
                and curr["is_bullish"]
                and abs(prev["low"] - curr["low"]) / max(prev["range"], 1) < 0.03
            ):
                self._add_pattern(
                    "Tweezer Bottom",
                    "bullish_reversal",
                    "BUY",
                    curr["index"],
                    curr["x"],
                    0.65,
                    "Two candles with matching lows. Double support at this level.",
                    "Enter long above the high of the second candle. SL below the shared low.",
                )
