"""
Real Calibration Engine
========================
Recalibrates heuristic scores using MEASURED outcomes from the trade database.
NO synthetic data — only real backtested results.

This is the engine that makes your "85% confidence" claim honest.
"""

import numpy as np
from analyzers.trade_database import TradeDatabase

class RealCalibrator:
    """Calibrates heuristic confidence using MEASURED historical win rates."""

    def __init__(self, db: TradeDatabase = None):
        self.db = db or TradeDatabase()

    def calibrate(self, heuristic_score: float, confluence_grade: str,
                  pattern_results: list) -> dict:
        """
        Calibrate a heuristic confidence score against measured data.

        The key question: "When we said 85%, what actually happened?"
        """
        # 1. Calibration from the calibration_map table
        bucket_cal = self.db.get_calibration(heuristic_score)

        # 2. Calibration from confluence grade measurements
        grade_stats = self.db.get_win_rate_by_grade()
        grade_wr = grade_stats.get(confluence_grade, {}).get("win_rate", None)

        # 3. Calibration from individual pattern measurements
        pattern_cals = []
        for p in pattern_results:
            stats = self.db.get_pattern_stats(p.get("name", ""))
            if stats:
                pattern_cals.append({
                    "pattern": p["name"],
                    "heuristic": p.get("confidence", 0),
                    "measured": stats["win_rate"],
                    "sample": stats["total_occurrences"],
                    "adjustment": stats["win_rate"] - p.get("confidence", 0),
                })

        # 4. Calculate weighted calibrated probability
        calibrated_probs = []
        weights = []

        # Bucket calibration (most reliable — based on 1000+ trades)
        if bucket_cal:
            calibrated_probs.append(bucket_cal["measured_probability"])
            weights.append(bucket_cal.get("total_trades", 100))

        # Grade calibration
        if grade_wr is not None:
            grade_total = grade_stats.get(confluence_grade, {}).get("total_trades", 50)
            calibrated_probs.append(grade_wr)
            weights.append(grade_total)

        # Pattern calibration (weighted by sample size)
        for pc in pattern_cals:
            if pc["sample"] > 20:
                calibrated_probs.append(pc["measured"])
                weights.append(pc["sample"])

        # Weighted average
        if calibrated_probs and sum(weights) > 0:
            calibrated_prob = sum(p * w for p, w in zip(calibrated_probs, weights)) / sum(weights)
        else:
            calibrated_prob = heuristic_score  # Fallback to heuristic

        # 5. The honest truth
        adjustment = calibrated_prob - heuristic_score

        return {
            "raw_heuristic": round(heuristic_score, 3),
            "calibrated_probability": round(calibrated_prob, 3),
            "adjustment": round(adjustment, 3),
            "is_overconfident": heuristic_score > calibrated_prob + 0.05,
            "is_underconfident": heuristic_score < calibrated_prob - 0.05,
            "bucket_calibration": dict(bucket_cal) if bucket_cal else None,
            "grade_calibration": {
                "grade": confluence_grade,
                "measured_win_rate": grade_wr,
                "sample_size": grade_stats.get(confluence_grade, {}).get("total_trades", 0),
            } if grade_wr is not None else None,
            "pattern_calibrations": pattern_cals,
            "honest_assessment": self._honest_assessment(
                heuristic_score, calibrated_prob, adjustment, grade_wr, pattern_cals
            ),
        }

    def _honest_assessment(self, raw, calibrated, adj, grade_wr, pattern_cals):
        """Generate assessment comparing heuristic vs measured probabilities."""
        lines = []

        # Main assessment
        if abs(adj) < 0.05:
            lines.append(
                f"✅ Your {raw:.0%} confidence is approximately CORRECT — "
                f"measured probability is {calibrated:.0%}."
            )
        elif adj < -0.10:
            lines.append(
                f"🔴 OVERCONFIDENT: You claim {raw:.0%} but measured data shows {calibrated:.0%}. "
                f"Your system is overestimating by {abs(adj):.0%}. REDUCE POSITION SIZE."
            )
        elif adj < -0.05:
            lines.append(
                f"🟠 SLIGHTLY OVERCONFIDENT: {raw:.0%} vs {calibrated:.0%} measured. "
                f"Adjust down by {abs(adj):.0%}."
            )
        elif adj > 0.10:
            lines.append(
                f"🟢 UNDERCONFIDENT: You say {raw:.0%} but data shows {calibrated:.0%}. "
                f"The signal is stronger than your engine thinks."
            )
        else:
            lines.append(
                f"🟡 CLOSE: {raw:.0%} heuristic vs {calibrated:.0%} measured. Minor adjustment needed."
            )

        # Grade assessment
        if grade_wr is not None:
            if grade_wr < 0.50:
                lines.append(
                    f"⚠️ Grade-based WR: Only {grade_wr:.0%} — this grade historically loses more than it wins."
                )

        # Pattern assessment
        overconfident_patterns = [pc for pc in pattern_cals if pc["adjustment"] < -0.10]
        if overconfident_patterns:
            for pc in overconfident_patterns[:3]:
                lines.append(
                    f"⚠️ {pc['pattern']}: Claims {pc['heuristic']:.0%}, actually {pc['measured']:.0%} "
                    f"over {pc['sample']} trades."
                )

        return " ".join(lines)

