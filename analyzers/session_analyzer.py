"""
Session Context Analyzer Module
Identifies the trading session context (Asian, London, New York) and
provides session-based volatility expectations.

This is critical because:
- Asian session: Low volatility, range-bound
- London session: High volatility, breakout-prone
- NY session: Highest volume, trend continuation
- London/NY overlap: Most volatile period
"""

import numpy as np


class SessionAnalyzer:
    """
    Analyzes the trading session context based on the chart's
    time axis (if detectable) and price behavior characteristics.
    """

    SESSIONS = {
        "ASIAN": {
            "name": "Asian Session",
            "hours": "00:00 - 08:00 GMT",
            "characteristics": "Low volatility, tight ranges, false breakouts",
            "strategy": "Range trading, fade breakouts, wait for London",
            "pairs": "AUD, NZD, JPY crosses",
            "avg_pip_range": {"EURUSD": 30, "GBPUSD": 40, "USDJPY": 35},
        },
        "LONDON": {
            "name": "London Session",
            "hours": "08:00 - 16:00 GMT",
            "characteristics": "High volatility, trend-setting, breakouts",
            "strategy": "Breakout trading, trend following, momentum entries",
            "pairs": "EUR, GBP, CHF crosses",
            "avg_pip_range": {"EURUSD": 80, "GBPUSD": 100, "USDJPY": 70},
        },
        "NEW_YORK": {
            "name": "New York Session",
            "hours": "13:00 - 22:00 GMT",
            "characteristics": "High volume, news-driven, reversals at London highs/lows",
            "strategy": "News trading, reversal at London extremes, continuation",
            "pairs": "USD pairs, all majors",
            "avg_pip_range": {"EURUSD": 70, "GBPUSD": 90, "USDJPY": 60},
        },
        "LONDON_NY_OVERLAP": {
            "name": "London/NY Overlap",
            "hours": "13:00 - 16:00 GMT",
            "characteristics": "HIGHEST volume and volatility of the day",
            "strategy": "Best time for breakout and trend trades. Expect big moves.",
            "pairs": "All major pairs",
            "avg_pip_range": {"EURUSD": 50, "GBPUSD": 60, "USDJPY": 45},
        },
    }

    def __init__(self):
        self.session = None
        self.volatility_expectation = "MODERATE"

    def analyze(
        self, price_series: dict, regime_results: dict, pair: str = "EURUSD"
    ) -> dict:
        """
        Infer session context from price behavior and provide
        session-specific trading recommendations.
        """
        smoothed = np.array(price_series.get("smoothed", []))

        if len(smoothed) < 5:
            return {"session": "UNKNOWN", "recommendation": "Insufficient data"}

        # Calculate session characteristics from price behavior
        volatility = self._calc_intraday_volatility(smoothed)
        range_expansion = self._calc_range_expansion(smoothed)

        # Infer session from behavior
        inferred_session = self._infer_session(
            volatility, range_expansion, regime_results
        )

        # Generate session-specific recommendations
        recommendations = self._get_session_recommendations(
            inferred_session, pair, regime_results
        )

        # Volatility expectations
        vol_expectation = self._get_volatility_expectation(inferred_session, pair)

        # Session transition alerts
        transitions = self._get_session_transition_alerts(inferred_session)

        return {
            "inferred_session": inferred_session,
            "session_info": self.SESSIONS.get(inferred_session, {}),
            "volatility": volatility,
            "range_expansion": range_expansion,
            "volatility_expectation": vol_expectation,
            "recommendations": recommendations,
            "session_transitions": transitions,
            "pair": pair,
            "best_session_for_pair": self._best_session_for_pair(pair),
        }

    def _calc_intraday_volatility(self, smoothed: np.ndarray) -> dict:
        """Calculate volatility metrics."""
        if len(smoothed) < 5:
            return {"level": "LOW", "value": 0}

        returns = np.diff(smoothed) / (smoothed[:-1] + 1e-6)
        vol = float(np.std(returns))

        if vol < 0.005:
            level = "LOW"
        elif vol < 0.015:
            level = "MODERATE"
        elif vol < 0.03:
            level = "HIGH"
        else:
            level = "VERY_HIGH"

        return {"level": level, "value": round(vol, 4)}

    def _calc_range_expansion(self, smoothed: np.ndarray) -> dict:
        """Calculate if the range is expanding or contracting."""
        if len(smoothed) < 10:
            return {"direction": "STABLE", "ratio": 1.0}

        # Compare recent range to earlier range
        n = len(smoothed)
        first_half_range = np.max(smoothed[: n // 2]) - np.min(smoothed[: n // 2])
        second_half_range = np.max(smoothed[n // 2 :]) - np.min(smoothed[n // 2 :])

        if first_half_range > 0:
            ratio = second_half_range / first_half_range
        else:
            ratio = 1.0

        if ratio > 1.5:
            direction = "EXPANDING"
        elif ratio < 0.7:
            direction = "CONTRACTING"
        else:
            direction = "STABLE"

        return {"direction": direction, "ratio": round(ratio, 2)}

    def _infer_session(
        self, volatility: dict, range_expansion: dict, regime: dict
    ) -> str:
        """Infer which session the chart represents based on behavior."""
        vol_level = volatility.get("level", "MODERATE")
        range_dir = range_expansion.get("direction", "STABLE")

        if vol_level in ["LOW", "MODERATE"] and range_dir in ["STABLE", "CONTRACTING"]:
            return "ASIAN"
        elif vol_level in ["HIGH", "VERY_HIGH"] and range_dir == "EXPANDING":
            return "LONDON_NY_OVERLAP"
        elif vol_level in ["HIGH"] and range_dir in ["EXPANDING", "STABLE"]:
            return "LONDON"
        elif vol_level in ["MODERATE", "HIGH"] and range_dir in ["STABLE", "EXPANDING"]:
            return "NEW_YORK"
        else:
            return "LONDON"  # Default most common active session

    def _get_session_recommendations(
        self, session: str, pair: str, regime: dict
    ) -> dict:
        """Get trading recommendations specific to the current session."""
        base_recs = {
            "ASIAN": {
                "approach": "RANGE TRADING",
                "do": [
                    "Trade within the Asian range (buy low, sell high)",
                    "Use tight stops — volatility is low",
                    "Look for fake breakouts to fade",
                    "Set alerts for London open breakout",
                ],
                "avoid": [
                    "Avoid breakout trades (most fail in Asian)",
                    "Don't expect big moves — position for small targets",
                    "Avoid over-leveraging to compensate for small moves",
                ],
            },
            "LONDON": {
                "approach": "BREAKOUT & TREND TRADING",
                "do": [
                    "Trade breakouts of Asian session range",
                    "Follow the London trend direction",
                    "Enter on pullbacks to EMAs or Fib levels",
                    "Use wider stops — volatility is higher",
                ],
                "avoid": [
                    "Avoid fading breakouts in first hour",
                    "Don't ignore the London open direction",
                    "Avoid trading against the established trend",
                ],
            },
            "NEW_YORK": {
                "approach": "NEWS & REVERSAL TRADING",
                "do": [
                    "Trade news events with reduced size",
                    "Look for reversals at London session extremes",
                    "Follow continuation if NY agrees with London direction",
                    "Be cautious around major economic releases",
                ],
                "avoid": [
                    "Avoid entering during major news releases",
                    "Don't expect London-range breakouts in late NY",
                    "Avoid new positions after 20:00 GMT (thin market)",
                ],
            },
            "LONDON_NY_OVERLAP": {
                "approach": "AGGRESSIVE TREND TRADING",
                "do": [
                    "This is the BEST time to trade — maximum liquidity",
                    "Enter strong trend moves with confidence",
                    "Use momentum and break of structure for entries",
                    "Take profits aggressively — moves can reverse at 16:00 GMT",
                ],
                "avoid": [
                    "Don't sit on the sidelines during this window",
                    "Avoid counter-trend trades — ride the momentum",
                    "Don't overthink — the best setups are simple here",
                ],
            },
        }

        return base_recs.get(session, base_recs["LONDON"])

    def _get_volatility_expectation(self, session: str, pair: str) -> dict:
        """Get expected pip range for this session/pair."""
        session_data = self.SESSIONS.get(session, {})
        avg_ranges = session_data.get("avg_pip_range", {})

        expected_pips = avg_ranges.get(pair, avg_ranges.get("EURUSD", 50))

        return {
            "expected_pip_range": expected_pips,
            "session": session,
            "note": f"During {session_data.get('name', 'Unknown')}, {pair} typically moves {expected_pips} pips",
        }

    def _get_session_transition_alerts(self, current_session: str) -> list:
        """Alert about upcoming session transitions."""
        transitions = {
            "ASIAN": [
                "⏰ ALERT: London opens at 08:00 GMT. Watch for Asian range breakout.",
                "💡 TIP: Set limit orders at Asian range extremes for London breakout trade.",
            ],
            "LONDON": [
                "⏰ ALERT: NY opens at 13:00 GMT. London/NY overlap is most volatile.",
                "💡 TIP: London session often sets the daily high/low in first 2 hours.",
            ],
            "NEW_YORK": [
                "⏰ ALERT: Asian session begins at 00:00 GMT. NY late session is thin.",
                "💡 TIP: After 20:00 GMT, reduce activity — low liquidity increases slippage.",
            ],
            "LONDON_NY_OVERLAP": [
                "⏰ ALERT: London closes at 16:00 GMT. Often a reversal at London close.",
                "💡 TIP: Most daily volume occurs in this 3-hour window. Maximize this time.",
            ],
        }
        return transitions.get(current_session, [])

    def _best_session_for_pair(self, pair: str) -> dict:
        """Best session to trade a specific pair."""
        pair_sessions = {
            "EURUSD": {"best": "LONDON", "second": "LONDON_NY_OVERLAP"},
            "GBPUSD": {"best": "LONDON", "second": "LONDON_NY_OVERLAP"},
            "USDJPY": {"best": "LONDON_NY_OVERLAP", "second": "ASIAN"},
            "AUDUSD": {"best": "LONDON_NY_OVERLAP", "second": "ASIAN"},
            "NZDUSD": {"best": "ASIAN", "second": "LONDON_NY_OVERLAP"},
            "EURJPY": {"best": "LONDON_NY_OVERLAP", "second": "ASIAN"},
            "GBPJPY": {"best": "LONDON_NY_OVERLAP", "second": "LONDON"},
            "XAUUSD": {"best": "LONDON_NY_OVERLAP", "second": "NEW_YORK"},
            "USDCAD": {"best": "NEW_YORK", "second": "LONDON_NY_OVERLAP"},
        }
        return pair_sessions.get(
            pair, {"best": "LONDON_NY_OVERLAP", "second": "LONDON"}
        )
