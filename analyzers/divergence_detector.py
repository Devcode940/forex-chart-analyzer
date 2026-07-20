"""
Divergence Detector Module
Detects regular and hidden divergences between price and momentum.
Divergence is one of the most powerful reversal/continuation signals in forex.

Regular Divergence = Trend Reversal Signal
Hidden Divergence = Trend Continuation Signal
"""

import numpy as np
from scipy.signal import argrelextrema


class DivergenceDetector:
    """
    Detects divergences by comparing price swing structure
    with an approximate momentum oscillator derived from price.
    """

    def __init__(self):
        self.divergences = []

    def detect_all(self, price_series: dict, structure_results: dict) -> list:
        """
        Detect all types of divergence.
        Since we don't have RSI/MACD from the image, we approximate momentum
        using rate-of-change and price velocity.
        """
        self.divergences = []

        smoothed = np.array(price_series.get("smoothed", []))
        x_positions = price_series.get("x", [])

        if len(smoothed) < 15:
            return self.divergences

        # Approximate momentum indicator (rate of change)
        roc = self._calc_rate_of_change(smoothed, period=5)

        # Approximate momentum oscillator (similar to RSI concept)
        momentum = self._calc_momentum_oscillator(smoothed, period=14)

        # Find swings in both price and momentum
        order = max(3, len(smoothed) // 12)
        price_highs_idx = argrelextrema(smoothed, np.greater, order=order)[0]
        price_lows_idx = argrelextrema(smoothed, np.less, order=order)[0]
        mom_highs_idx = argrelextrema(momentum, np.greater, order=order)[0]
        mom_lows_idx = argrelextrema(momentum, np.less, order=order)[0]

        # Detect Regular Bearish Divergence (price HH + momentum LH)
        self._detect_regular_bearish(smoothed, momentum, price_highs_idx, mom_highs_idx, x_positions)

        # Detect Regular Bullish Divergence (price LL + momentum HL)
        self._detect_regular_bullish(smoothed, momentum, price_lows_idx, mom_lows_idx, x_positions)

        # Detect Hidden Bullish Divergence (price HL + momentum LL)
        self._detect_hidden_bullish(smoothed, momentum, price_lows_idx, mom_lows_idx, x_positions)

        # Detect Hidden Bearish Divergence (price LH + momentum HH)
        self._detect_hidden_bearish(smoothed, momentum, price_highs_idx, mom_highs_idx, x_positions)

        self.divergences.sort(key=lambda d: d["confidence"], reverse=True)
        return self.divergences

    def _calc_rate_of_change(self, data: np.ndarray, period: int = 5) -> np.ndarray:
        """Calculate Rate of Change (momentum proxy)."""
        roc = np.zeros_like(data)
        for i in range(period, len(data)):
            if data[i - period] != 0:
                roc[i] = (data[i] - data[i - period]) / abs(data[i - period]) * 100
        return roc

    def _calc_momentum_oscillator(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate a momentum oscillator similar to RSI.
        Uses up/down moves to create a bounded 0-100 oscillator.
        """
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Simple moving average of gains/losses
        osc = np.zeros(len(data))

        if len(gains) < period:
            return osc

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(data)):
            if i - 1 < len(gains):
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                osc[i] = 100 - (100 / (1 + rs))
            else:
                osc[i] = 100

        return osc

    def _detect_regular_bearish(self, price, momentum, price_highs, mom_highs, x_pos):
        """Regular Bearish: Price makes Higher High, but momentum makes Lower High → Reversal DOWN"""
        if len(price_highs) < 2 or len(mom_highs) < 2:
            return

        for i in range(len(price_highs) - 1):
            p_idx1 = price_highs[i]
            p_idx2 = price_highs[i + 1]

            if p_idx2 >= len(price) or p_idx1 >= len(price):
                continue

            # Price: higher high
            if price[p_idx2] > price[p_idx1]:
                # Find corresponding momentum highs
                mom_before = [m for m in mom_highs if m <= p_idx1]
                mom_after = [m for m in mom_highs if p_idx1 < m <= p_idx2]

                if mom_before and mom_after:
                    m_idx1 = mom_before[-1]
                    m_idx2 = mom_after[-1]

                    # Momentum: lower high
                    if m_idx2 < len(momentum) and m_idx1 < len(momentum):
                        if momentum[m_idx2] < momentum[m_idx1]:
                            price_diff = (price[p_idx2] - price[p_idx1]) / abs(price[p_idx1]) * 100
                            mom_diff = momentum[m_idx1] - momentum[m_idx2]

                            confidence = min(0.9, 0.5 + abs(price_diff) * 0.05 + mom_diff * 0.01)

                            self.divergences.append({
                                "name": "Regular Bearish Divergence",
                                "type": "BEARISH_REVERSAL",
                                "signal": "SELL",
                                "confidence": confidence,
                                "description": (
                                    f"Price made higher high ({price[p_idx1]:.1f} → {price[p_idx2]:.1f}) "
                                    f"but momentum made lower high ({momentum[m_idx1]:.1f} → {momentum[m_idx2]:.1f}). "
                                    f"Upward momentum is fading — bearish reversal likely."
                                ),
                                "implication": (
                                    "Enter short on confirmation (bearish candle close below recent swing low). "
                                    "SL above the divergence high. TP at next support or Fib level."
                                ),
                                "price_high1_idx": int(p_idx1),
                                "price_high2_idx": int(p_idx2),
                                "x_range": (
                                    x_pos[p_idx1] if p_idx1 < len(x_pos) else 0,
                                    x_pos[p_idx2] if p_idx2 < len(x_pos) else 0
                                )
                            })

    def _detect_regular_bullish(self, price, momentum, price_lows, mom_lows, x_pos):
        """Regular Bullish: Price makes Lower Low, but momentum makes Higher Low → Reversal UP"""
        if len(price_lows) < 2 or len(mom_lows) < 2:
            return

        for i in range(len(price_lows) - 1):
            p_idx1 = price_lows[i]
            p_idx2 = price_lows[i + 1]

            if p_idx2 >= len(price) or p_idx1 >= len(price):
                continue

            # Price: lower low
            if price[p_idx2] < price[p_idx1]:
                # Find corresponding momentum lows
                mom_before = [m for m in mom_lows if m <= p_idx1]
                mom_after = [m for m in mom_lows if p_idx1 < m <= p_idx2]

                if mom_before and mom_after:
                    m_idx1 = mom_before[-1]
                    m_idx2 = mom_after[-1]

                    if m_idx2 < len(momentum) and m_idx1 < len(momentum):
                        # Momentum: higher low
                        if momentum[m_idx2] > momentum[m_idx1]:
                            price_diff = (price[p_idx1] - price[p_idx2]) / abs(price[p_idx1]) * 100
                            mom_diff = momentum[m_idx2] - momentum[m_idx1]

                            confidence = min(0.9, 0.5 + abs(price_diff) * 0.05 + mom_diff * 0.01)

                            self.divergences.append({
                                "name": "Regular Bullish Divergence",
                                "type": "BULLISH_REVERSAL",
                                "signal": "BUY",
                                "confidence": confidence,
                                "description": (
                                    f"Price made lower low ({price[p_idx1]:.1f} → {price[p_idx2]:.1f}) "
                                    f"but momentum made higher low ({momentum[m_idx1]:.1f} → {momentum[m_idx2]:.1f}). "
                                    f"Selling pressure is fading — bullish reversal likely."
                                ),
                                "implication": (
                                    "Enter long on confirmation (bullish candle close above recent swing high). "
                                    "SL below the divergence low. TP at next resistance or Fib level."
                                ),
                                "price_low1_idx": int(p_idx1),
                                "price_low2_idx": int(p_idx2),
                                "x_range": (
                                    x_pos[p_idx1] if p_idx1 < len(x_pos) else 0,
                                    x_pos[p_idx2] if p_idx2 < len(x_pos) else 0
                                )
                            })

    def _detect_hidden_bullish(self, price, momentum, price_lows, mom_lows, x_pos):
        """Hidden Bullish: Price makes Higher Low, momentum makes Lower Low → Continuation UP"""
        if len(price_lows) < 2 or len(mom_lows) < 2:
            return

        for i in range(len(price_lows) - 1):
            p_idx1 = price_lows[i]
            p_idx2 = price_lows[i + 1]

            if p_idx2 >= len(price) or p_idx1 >= len(price):
                continue

            # Price: higher low
            if price[p_idx2] > price[p_idx1]:
                mom_before = [m for m in mom_lows if m <= p_idx1]
                mom_after = [m for m in mom_lows if p_idx1 < m <= p_idx2]

                if mom_before and mom_after:
                    m_idx1 = mom_before[-1]
                    m_idx2 = mom_after[-1]

                    if m_idx2 < len(momentum) and m_idx1 < len(momentum):
                        # Momentum: lower low
                        if momentum[m_idx2] < momentum[m_idx1]:
                            confidence = 0.65

                            self.divergences.append({
                                "name": "Hidden Bullish Divergence",
                                "type": "BULLISH_CONTINUATION",
                                "signal": "BUY",
                                "confidence": confidence,
                                "description": (
                                    f"Price made higher low but momentum made lower low. "
                                    f"This is a hidden divergence — trend is still strong, expect continuation UP."
                                ),
                                "implication": (
                                    "Buy the dip. SL below the higher low. "
                                    "This confirms the uptrend is intact."
                                ),
                                "price_low1_idx": int(p_idx1),
                                "price_low2_idx": int(p_idx2),
                                "x_range": (
                                    x_pos[p_idx1] if p_idx1 < len(x_pos) else 0,
                                    x_pos[p_idx2] if p_idx2 < len(x_pos) else 0
                                )
                            })

    def _detect_hidden_bearish(self, price, momentum, price_highs, mom_highs, x_pos):
        """Hidden Bearish: Price makes Lower High, momentum makes Higher High → Continuation DOWN"""
        if len(price_highs) < 2 or len(mom_highs) < 2:
            return

        for i in range(len(price_highs) - 1):
            p_idx1 = price_highs[i]
            p_idx2 = price_highs[i + 1]

            if p_idx2 >= len(price) or p_idx1 >= len(price):
                continue

            # Price: lower high
            if price[p_idx2] < price[p_idx1]:
                mom_before = [m for m in mom_highs if m <= p_idx1]
                mom_after = [m for m in mom_highs if p_idx1 < m <= p_idx2]

                if mom_before and mom_after:
                    m_idx1 = mom_before[-1]
                    m_idx2 = mom_after[-1]

                    if m_idx2 < len(momentum) and m_idx1 < len(momentum):
                        # Momentum: higher high
                        if momentum[m_idx2] > momentum[m_idx1]:
                            confidence = 0.65

                            self.divergences.append({
                                "name": "Hidden Bearish Divergence",
                                "type": "BEARISH_CONTINUATION",
                                "signal": "SELL",
                                "confidence": confidence,
                                "description": (
                                    f"Price made lower high but momentum made higher high. "
                                    f"This is a hidden divergence — downtrend is still strong, expect continuation DOWN."
                                ),
                                "implication": (
                                    "Sell the bounce. SL above the lower high. "
                                    "This confirms the downtrend is intact."
                                ),
                                "price_high1_idx": int(p_idx1),
                                "price_high2_idx": int(p_idx2),
                                "x_range": (
                                    x_pos[p_idx1] if p_idx1 < len(x_pos) else 0,
                                    x_pos[p_idx2] if p_idx2 < len(x_pos) else 0
                                )
                            })
