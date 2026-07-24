"""
Confluence Score Engine
Aggregates ALL signals (patterns, candlesticks, S/R, Fib, structure, regime)
into a single compounded confidence score and directional bias.

Key insight: When 5+ independent signals align, the probability is NOT additive —
it COMPOUNDS. This engine models that compounding effect.
"""

import numpy as np

class ConfluenceEngine:
    """
    Calculates a master confluence score that compounds multiple signals.

    Philosophy: 5 weak signals all pointing bullish is MUCH stronger than
    1 strong signal. Most traders miss this because they evaluate signals in isolation.
    """

    def __init__(self):
        self.signals = []
        self.bull_score = 0.0
        self.bear_score = 0.0
        self.neutral_score = 0.0
        self.master_score = {}
        self.trade_plan = {}

    def analyze(self, pattern_results: list, candlestick_results: list,
                sr_results: dict, fib_results: dict,
                structure_results: dict, regime_results: dict,
                sltp_results: dict) -> dict:
        """
        Aggregate all signals into a compounded confluence score.
        Returns master direction, confidence, and a prioritized trade plan.
        """
        self.signals = []

        # 1. Collect geometric pattern signals
        self._collect_pattern_signals(pattern_results)

        # 2. Collect candlestick pattern signals
        self._collect_candlestick_signals(candlestick_results)

        # 3. Collect S/R proximity signals
        self._collect_sr_signals(sr_results, sltp_results)

        # 4. Collect Fib zone signals
        self._collect_fib_signals(fib_results, sltp_results)

        # 5. Collect structure signals
        self._collect_structure_signals(structure_results)

        # 6. Collect regime signals
        self._collect_regime_signals(regime_results)

        # 7. Calculate compounded scores
        self._calculate_compounded_scores()

        # 8. Determine master direction
        master = self._determine_master_direction()

        # 9. Generate trade plan
        trade_plan = self._generate_trade_plan(master, sltp_results)

        return {
            "signals": self.signals,
            "signal_count": len(self.signals),
            "bull_score": round(self.bull_score, 3),
            "bear_score": round(self.bear_score, 3),
            "neutral_score": round(self.neutral_score, 3),
            "master": master,
            "trade_plan": trade_plan,
            "confluence_breakdown": self._get_confluence_breakdown(),
        }

    def _add_signal(self, source: str, name: str, direction: str,
                    strength: float, weight: float = 1.0):
        """Add a signal to the collection."""
        self.signals.append({
            "source": source,
            "name": name,
            "direction": direction,  # BULLISH, BEARISH, NEUTRAL
            "strength": min(strength, 1.0),
            "weight": weight,
            "weighted_impact": strength * weight,
        })

    def _collect_pattern_signals(self, patterns: list):
        """Collect signals from geometric chart patterns."""
        for p in patterns:
            direction = "BULLISH" if "BULLISH" in p.get("type", "") else \
                        "BEARISH" if "BEARISH" in p.get("type", "") else "NEUTRAL"
            self._add_signal(
                "Chart Pattern", p["name"], direction,
                p.get("confidence", 0.5), weight=1.5  # Patterns are high-weight
            )

    def _collect_candlestick_signals(self, candlesticks: list):
        """Collect signals from individual candlestick patterns."""
        signal_map = {"BUY": "BULLISH", "SELL": "BEARISH", "REVERSAL_POSSIBLE": "NEUTRAL", "WAIT": "NEUTRAL"}
        for c in candlesticks[:5]:  # Top 5 most recent/relevant
            direction = signal_map.get(c.get("signal", "WAIT"), "NEUTRAL")
            self._add_signal(
                "Candlestick", c["name"], direction,
                c.get("confidence", 0.5), weight=1.0
            )

    def _collect_sr_signals(self, sr: dict, sltp: dict):
        """Collect signals based on price proximity to S/R."""
        current = sltp.get("current_price", 0)
        supports = sr.get("support", [])
        resistances = sr.get("resistance", [])

        # Only add S/R signals if price is near the level
        for s in supports[:2]:
            level = s.get("price_level", 0)
            proximity = abs(level - current) / (current + 1e-6) if current > 0 else 1.0
            if proximity < 0.08:
                self._add_signal(
                    "Support Zone", f"Support at {level:.1f}", "BULLISH",
                    s.get("strength", 0.5) * 0.8, weight=1.2
                )

        for r in resistances[:2]:
            level = r.get("price_level", 0)
            proximity = abs(level - current) / (current + 1e-6) if current > 0 else 1.0
            if proximity < 0.08:
                self._add_signal(
                    "Resistance Zone", f"Resistance at {level:.1f}", "BEARISH",
                    r.get("strength", 0.5) * 0.8, weight=1.2
                )

    def _collect_fib_signals(self, fib: dict, sltp: dict):
        """Collect signals from Fibonacci levels."""
        golden_zone = fib.get("golden_zone", {})
        current = sltp.get("current_price", 0)
        if golden_zone:
            trend = fib.get("trend", "RANGING")
            direction = "BULLISH" if trend in ["UPTREND", "RANGING"] else "BEARISH"
            # Check if price is actually within the golden zone
            upper = golden_zone.get("upper", 0)
            lower = golden_zone.get("lower", 0)
            in_zone = lower <= current <= upper if current > 0 and upper > lower else True
            if in_zone:
                self._add_signal(
                    "Fibonacci", "Golden Zone (61.8%-78.6%)", direction,
                    0.75, weight=1.8
                )

        # Check individual key Fib levels
        for ratio, level in fib.get("retracements", {}).items():
            if ratio in [0.618, 0.786] and level.get("importance") == "HIGH":
                fib_value = level.get("value", 0)
                proximity = abs(fib_value - current) / (current + 1e-6) if current > 0 else 1.0
                if proximity < 0.05:
                    direction = "BULLISH" if level.get("direction") == "BUY_ZONE" else "BEARISH"
                    self._add_signal(
                        "Fibonacci", f"Fib {level['label']} Retracement", direction,
                        0.6, weight=1.3
                    )

    def _collect_structure_signals(self, structure: dict):
        """Collect signals from market structure analysis."""
        trend = structure.get("trend_direction", "RANGING")
        strength = structure.get("trend_strength", 0.5)

        if trend == "UPTREND":
            self._add_signal("Market Structure", "Uptrend", "BULLISH", strength, weight=2.0)
        elif trend == "DOWNTREND":
            self._add_signal("Market Structure", "Downtrend", "BEARISH", strength, weight=2.0)
        else:
            self._add_signal("Market Structure", "Ranging/Consolidation", "NEUTRAL", 0.4, weight=1.0)

        # BOS signals
        for bos in structure.get("structure_breaks", []):
            direction = "BULLISH" if "BULLISH" in bos.get("type", "") else "BEARISH"
            self._add_signal(
                "Structure Break", bos["type"], direction,
                0.7, weight=1.5
            )

    def _collect_regime_signals(self, regime: dict):
        """Collect signals from regime classification."""
        regime_name = regime.get("regime", "UNKNOWN")
        sub_regime = regime.get("sub_regime", "")
        confidence = regime.get("confidence", 0.5)

        if regime_name == "TRENDING":
            if "UP" in sub_regime:
                self._add_signal("Market Regime", "Trending Up", "BULLISH", confidence, weight=1.3)
            elif "DOWN" in sub_regime:
                self._add_signal("Market Regime", "Trending Down", "BEARISH", confidence, weight=1.3)
        elif regime_name == "RANGING":
            self._add_signal("Market Regime", "Ranging", "NEUTRAL", confidence, weight=0.8)
        elif regime_name == "VOLATILE":
            self._add_signal("Market Regime", "Volatile", "NEUTRAL", confidence * 0.5, weight=0.5)

    def _calculate_compounded_scores(self):
        """
        COMPOUND scores instead of averaging.
        Formula: Combined confidence = 1 - (1-c1) * (1-c2) * ... * (1-cn)
        This models the way multiple independent signals COMPOUND probability.
        """
        bull_factors = []
        bear_factors = []
        neutral_factors = []

        for signal in self.signals:
            impact = signal["weighted_impact"]
            # Normalize impact to [0, 1]
            normalized = min(impact / 2.0, 1.0)  # Cap at 1.0

            if signal["direction"] == "BULLISH":
                bull_factors.append(normalized)
                bear_factors.append(normalized * 0.1)  # Small counter-signal
            elif signal["direction"] == "BEARISH":
                bear_factors.append(normalized)
                bull_factors.append(normalized * 0.1)
            else:
                neutral_factors.append(normalized)

        # Compound: P = 1 - product of (1 - p_i)
        self.bull_score = 1.0
        for f in bull_factors:
            self.bull_score *= (1.0 - f)
        self.bull_score = 1.0 - self.bull_score

        self.bear_score = 1.0
        for f in bear_factors:
            self.bear_score *= (1.0 - f)
        self.bear_score = 1.0 - self.bear_score

        self.neutral_score = 1.0
        for f in neutral_factors:
            self.neutral_score *= (1.0 - f)
        self.neutral_score = 1.0 - self.neutral_score

    def _determine_master_direction(self) -> dict:
        """Determine the master direction from compounded scores."""
        total = self.bull_score + self.bear_score + self.neutral_score

        if total < 1e-6:
            return {
                "direction": "NO_SIGNAL",
                "confidence": 0.0,
                "grade": "F",
                "strength_description": "No actionable signal detected"
            }

        bull_pct = self.bull_score / total
        bear_pct = self.bear_score / total

        if bull_pct > 0.55:
            direction = "BULLISH"
            confidence = bull_pct
        elif bear_pct > 0.55:
            direction = "BEARISH"
            confidence = bear_pct
        else:
            direction = "NEUTRAL"
            confidence = self.neutral_score / total

        # Grade the signal
        if confidence > 0.75:
            grade = "A+"
        elif confidence > 0.65:
            grade = "A"
        elif confidence > 0.55:
            grade = "B"
        elif confidence > 0.45:
            grade = "C"
        else:
            grade = "D"

        # Strength description
        n_aligning = sum(1 for s in self.signals if s["direction"] == direction)
        strength_desc = (
            f"{n_aligning} signals align {direction} "
            f"(Bull: {self.bull_score:.1%}, Bear: {self.bear_score:.1%})"
        )

        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "grade": grade,
            "strength_description": strength_desc,
            "bull_pct": round(bull_pct, 3),
            "bear_pct": round(bear_pct, 3),
        }

    def _generate_trade_plan(self, master: dict, sltp: dict) -> dict:
        """Generate a concrete trade execution plan."""
        direction = master.get("direction", "NEUTRAL")
        grade = master.get("grade", "D")

        if direction == "NEUTRAL" or grade in ["D", "F"]:
            return {
                "action": "DO NOT TRADE",
                "reason": f"Confluence grade is {grade} — signals are conflicting. Wait for clarity.",
                "alternative": "Set alerts at key S/R levels and wait for a clear breakout or reversal signal."
            }

        best_scenario = None
        for scenario in sltp.get("scenarios", []):
            if direction == "BULLISH" and scenario.get("direction") == "BUY":
                if best_scenario is None or scenario.get("risk_reward", 0) > best_scenario.get("risk_reward", 0):
                    best_scenario = scenario
            elif direction == "BEARISH" and scenario.get("direction") == "SELL":
                if best_scenario is None or scenario.get("risk_reward", 0) > best_scenario.get("risk_reward", 0):
                    best_scenario = scenario

        entry_trigger = None
        for signal in self.signals:
            if signal["source"] == "Candlestick" and signal["direction"] == direction:
                entry_trigger = signal["name"]
                break

        # Find confluence factors (what's aligned)
        aligned_signals = [s for s in self.signals if s["direction"] == direction]
        confluence_factors = [f"✅ {s['source']}: {s['name']}" for s in aligned_signals]

        # Find opposing factors
        opposing = [s for s in self.signals if s["direction"] != direction and s["direction"] != "NEUTRAL"]
        risk_factors = [f"⚠️ {s['source']}: {s['name']} ({s['direction']})" for s in opposing]

        plan = {
            "action": f"{'BUY' if direction == 'BULLISH' else 'SELL'}",
            "confluence_grade": grade,
            "confidence": master.get("confidence", 0),
            "entry_trigger": entry_trigger or "Break of key level",
            "scenario": best_scenario,
            "confluence_factors": confluence_factors,
            "risk_factors": risk_factors,
            "position_advice": self._get_position_advice(grade),
            "execution_steps": self._get_execution_steps(direction, grade, best_scenario),
        }

        return plan

    def _get_position_advice(self, grade: str) -> str:
        """Get position sizing advice based on confluence grade."""
        advice = {
            "A+": "Full position (2% risk). Multiple high-confidence signals aligned.",
            "A": "Standard position (1.5% risk). Strong confluence of signals.",
            "B": "Reduced position (1% risk). Decent confluence but some uncertainty.",
            "C": "Small position (0.5% risk). Weak confluence — consider skipping.",
            "D": "No trade recommended. Conflicting signals.",
        }
        return advice.get(grade, "No trade — wait for better setup.")

    def _get_execution_steps(self, direction: str, grade: str, scenario: dict) -> list:
        """Generate step-by-step execution plan."""
        steps = [
            f"1️⃣ DIRECTION: {'LONG (BUY)' if direction == 'BULLISH' else 'SHORT (SELL)'}",
            f"2️⃣ CONFLUENCE: Grade {grade} — {self._get_position_advice(grade)}",
            f"3️⃣ WAIT: For price to reach your entry zone or trigger candle",
        ]

        if scenario:
            steps.extend([
                f"4️⃣ ENTRY: At {scenario.get('entry', 'N/A'):.2f} (or on trigger candle close)",
                f"5️⃣ STOP LOSS: At {scenario.get('sl', 'N/A'):.2f}",
                f"6️⃣ TAKE PROFIT: At {scenario.get('tp', 'N/A'):.2f}",
                f"7️⃣ RISK:REWARD = 1:{scenario.get('risk_reward', 0):.2f}",
            ])
        else:
            steps.extend([
                "4️⃣ ENTRY: At key Fib level or S/R bounce",
                "5️⃣ STOP LOSS: Beyond the invalidation point",
                "6️⃣ TAKE PROFIT: At next opposing S/R or Fib extension",
            ])

        steps.extend([
            "8️⃣ CONFIRM: Check for volume/momentum confirmation",
            "9️⃣ MANAGE: Trail stop if trend continues, don't move SL against you",
            "🔟 REVIEW: If stopped out, re-analyze — don't revenge trade"
        ])

        return steps

    def _get_confluence_breakdown(self) -> dict:
        """Get a breakdown of confluence by source category."""
        categories = {}
        for signal in self.signals:
            source = signal["source"]
            if source not in categories:
                categories[source] = {"bullish": 0, "bearish": 0, "neutral": 0, "signals": []}
            direction = signal["direction"].lower()
            categories[source][direction] = categories[source].get(direction, 0) + 1
            categories[source]["signals"].append(signal["name"])

        return categories

