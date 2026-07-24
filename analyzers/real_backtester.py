"""
Real Backtester
===============
Backtests detected patterns against the historical trade database.
Uses ACTUAL measured win rates, NOT synthetic data.

This is what makes the system honest:
- "Head & Shoulders" claims 80%? Let's check: 198/342 = 57.9% ACTUAL
- Confluence Grade A claims 85%? Let's check: measured 68.2% ACTUAL
"""

import numpy as np

from analyzers.trade_database import TradeDatabase


class RealBacktester:
    """
    Backtests current analysis against the historical trade database.
    Returns MEASURED (not estimated) statistics.
    """

    def __init__(self, db: TradeDatabase = None):
        self.db = db or TradeDatabase()

    def backtest_current(
        self,
        pattern_results: list,
        confluence_grade: str,
        regime: str,
        session: str,
        pair: str = "EURUSD",
    ) -> dict:
        """
        Backtest the current analysis against historical data.
        Returns what ACTUALLY happened when similar setups occurred.
        """

        # ── 1. Backtest each detected pattern ──
        pattern_backtests = []
        for p in pattern_results:
            stats = self.db.get_pattern_stats(p.get("name", ""))
            if stats:
                pattern_backtests.append(
                    {
                        "pattern": p["name"],
                        "heuristic_confidence": p.get("confidence", 0),
                        "measured_win_rate": stats["win_rate"],
                        "measured_profit_factor": stats["profit_factor"],
                        "sample_size": stats["total_occurrences"],
                        "avg_win_pips": stats["avg_win_pips"],
                        "avg_loss_pips": stats["avg_loss_pips"],
                        "overconfidence": round(
                            p.get("confidence", 0) - stats["win_rate"], 3
                        ),
                        "verdict": self._pattern_verdict(
                            p.get("confidence", 0),
                            stats["win_rate"],
                            stats["total_occurrences"],
                        ),
                    }
                )

        # ── 2. Backtest confluence grade ──
        grade_stats = self.db.get_win_rate_by_grade()
        grade_backtest = grade_stats.get(confluence_grade, {})

        # ── 3. Backtest pattern combos ──
        combo_names = [p["name"] for p in pattern_results[:3]]
        combo_backtest = (
            self.db.get_combo_stats(combo_names) if len(combo_names) >= 2 else None
        )

        # ── 4. Backtest by regime + session ──
        regime_session_stats = self.db.get_win_rate_by_regime_session()
        rs_key = f"{regime}_{session}"
        rs_backtest = regime_session_stats.get(rs_key, {})

        # ── 5. Get Kelly params for this setup type ──
        setup_type = self._classify_setup(pattern_results, confluence_grade)
        kelly_params = self.db.get_kelly_params(setup_type)

        # ── 6. Get calibration for heuristic score ──
        avg_heuristic = (
            np.mean([p.get("confidence", 0.5) for p in pattern_results])
            if pattern_results
            else 0.5
        )
        calibration = self.db.get_calibration(avg_heuristic)

        # ── 7. Overall backtest verdict ──
        overall = self._overall_verdict(
            pattern_backtests, grade_backtest, kelly_params, calibration
        )

        return {
            "pattern_backtests": pattern_backtests,
            "grade_backtest": {
                "grade": confluence_grade,
                "measured_win_rate": grade_backtest.get("win_rate", 0),
                "sample_size": grade_backtest.get("total_trades", 0),
                "avg_win_pips": grade_backtest.get("avg_win_pips", 0),
                "avg_loss_pips": grade_backtest.get("avg_loss_pips", 0),
            },
            "combo_backtest": combo_backtest,
            "regime_session_backtest": rs_backtest,
            "kelly_params": kelly_params,
            "calibration": calibration,
            "setup_type": setup_type,
            "overall_verdict": overall,
        }

    def _pattern_verdict(self, heuristic: float, measured: float, sample: int) -> str:
        """Compare heuristic confidence to measured win rate."""
        diff = heuristic - measured
        if sample < 30:
            return f"⚠️ INSUFFICIENT DATA (only {sample} samples). Measured WR may be unreliable."
        if diff > 0.20:
            return f"🔴 SEVERELY OVERCONFIDENT: Heuristic says {heuristic:.0%} but history shows {measured:.0%}. Do NOT trust the heuristic."
        if diff > 0.10:
            return f"🟠 OVERCONFIDENT: Heuristic says {heuristic:.0%} vs {measured:.0%} actual. Reduce position size."
        if diff > 0.05:
            return f"🟡 SLIGHTLY OVERCONFIDENT: {heuristic:.0%} vs {measured:.0%}. Minor adjustment needed."
        if diff > -0.05:
            return f"✅ WELL CALIBRATED: {heuristic:.0%} vs {measured:.0%}. Heuristic is accurate."
        return f"🟢 UNDERCONFIDENT: {heuristic:.0%} vs {measured:.0%}. Signal may be stronger than scored."

    def _classify_setup(self, patterns, grade) -> str:
        """Classify the setup type for Kelly lookup."""
        if grade in ["A+", "A"]:
            return "confluence_a_grade"
        elif grade == "B":
            return "confluence_b_grade"
        elif grade == "C":
            return "confluence_c_grade"
        elif grade == "D":
            return "confluence_d_grade"

        # By pattern type
        for p in patterns:
            name = p.get("name", "").lower()
            cat = p.get("type", "").lower()
            if "diverg" in name:
                return "divergence_signal"
            if "fib" in name:
                return "fibonacci_entry"
            if "liquid" in name:
                return "liquidity_sweep"
            if "ma " in name or "cross" in name:
                return "ma_crossover"
            if "continuation" in cat:
                return "continuation_pattern"
            if "reversal" in cat:
                return "reversal_pattern"

        return "candlestick_signal"

    def _overall_verdict(
        self, pattern_backtests, grade_backtest, kelly_params, calibration
    ) -> dict:
        """Generate overall backtest verdict."""
        issues = []
        strengths = []

        # Check each pattern
        for pb in pattern_backtests:
            if pb["overconfidence"] > 0.10:
                issues.append(
                    f"{pb['pattern']}: Heuristic {pb['heuristic_confidence']:.0%} vs Measured {pb['measured_win_rate']:.0%}"
                )
            elif pb["measured_win_rate"] > 0.55:
                strengths.append(
                    f"{pb['pattern']}: {pb['measured_win_rate']:.0%} win rate over {pb['sample_size']} trades"
                )

        # Check grade
        measured_wr = grade_backtest.get("win_rate", 0)
        if measured_wr > 0:
            if measured_wr >= 0.65:
                strengths.append(
                    f"Grade {grade_backtest.get('grade','?')}: {measured_wr:.0%} measured win rate"
                )
            elif measured_wr < 0.50:
                issues.append(
                    f"Grade {grade_backtest.get('grade','?')}: Only {measured_wr:.0%} measured win rate"
                )

        # Kelly recommendation
        kelly_rec = "No Kelly data for this setup"
        if kelly_params:
            hf = kelly_params.get("half_kelly", 0)
            if hf > 0:
                kelly_rec = (
                    f"Half-Kelly suggests {hf:.1%} risk per trade for this setup"
                )
            else:
                kelly_rec = "Kelly is negative — no statistical edge for this setup"

        # Calibration
        cal_note = ""
        if calibration:
            measured_prob = calibration.get("measured_probability", 0)
            cal_note = f"Calibration: Heuristic scores in this range actually win {measured_prob:.0%} of the time"

        # Final verdict
        if len(issues) == 0 and len(strengths) > 0:
            verdict = "✅ BACKTEST VALIDATED: Historical data supports this trade."
            risk_adj = 1.0
        elif len(issues) <= 1 and len(strengths) > 0:
            verdict = (
                "🟡 PARTIALLY VALIDATED: Some overconfidence detected. Reduce size."
            )
            risk_adj = 0.75
        elif len(issues) > 0:
            verdict = "🔴 Overconfidence: heuristic scores exceed measured win rates. Reduce risk."
            risk_adj = 0.5
        else:
            verdict = "⚪ NO BACKTEST DATA: Cannot validate against history. Trade with extreme caution."
            risk_adj = 0.5

        return {
            "verdict": verdict,
            "risk_adjustment": risk_adj,
            "issues": issues,
            "strengths": strengths,
            "kelly_recommendation": kelly_rec,
            "calibration_note": cal_note,
        }
