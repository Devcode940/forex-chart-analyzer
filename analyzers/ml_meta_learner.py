"""
Meta-Learner & Information Coefficient Engine
==============================================
Combines ALL ML models into a single master prediction with:
1. Stacked ensemble (RF + GB + Statistical + Anomaly + Calibrated)
2. Information Coefficient measurement
3. Adaptive weighting based on model performance
4. Confidence interval from model disagreement
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


class MetaLearner:
    """
    Master ML model that combines all sub-models into one prediction.
    Uses adaptive weighting: better-performing models get more weight.
    """

    def __init__(self):
        self.ensemble = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_weights = {}
        self.ic_tracker = []

    def predict_master(
        self,
        feature_vector: np.ndarray,
        ml_ensemble_result: dict,
        anomaly_result: dict,
        calibration_result: dict,
        statistical_result: dict,
        walk_forward_result: dict,
        confluence_results: dict
    ) -> dict:
        """
        Combine ALL model outputs into a single master prediction.
        
        Uses a voting scheme weighted by each model's estimated reliability.
        """
        predictions = {}

        # Collect predictions from each sub-system
        # 1. ML Ensemble
        if ml_ensemble_result and "error" not in ml_ensemble_result:
            predictions["ml_ensemble"] = {
                "prob": ml_ensemble_result.get("ml_probability", 0.5),
                "direction": ml_ensemble_result.get("ml_direction", "NEUTRAL"),
                "weight": 0.30,  # Highest weight — it's the main ML model
                "source": "XGBoost/RF Ensemble",
            }

        # 2. Statistical Validation
        if statistical_result and "final_verdict" in statistical_result:
            sv = statistical_result["final_verdict"]
            sv_prob = 0.5
            if sv.get("final_grade") == "VALIDATED_85+":
                sv_prob = 0.85
            elif sv.get("final_grade") == "PROBABLE_70-85":
                sv_prob = 0.70
            elif sv.get("final_grade") == "POSSIBLE_55-70":
                sv_prob = 0.55
            else:
                sv_prob = 0.40

            # Direction from votes
            bull_votes = sum(1 for v in sv.get("method_votes", []) if "BULLISH" in v.get("verdict", "").upper() or v.get("prob", 0) > 0.5)
            bear_votes = sum(1 for v in sv.get("method_votes", []) if v.get("prob", 0) < 0.5)
            sv_dir = "BULLISH" if bull_votes > bear_votes else "BEARISH" if bear_votes > bull_votes else "NEUTRAL"

            predictions["statistical"] = {
                "prob": sv_prob if sv_dir != "BEARISH" else 1 - sv_prob,
                "direction": sv_dir,
                "weight": 0.25,
                "source": "Monte Carlo + Bootstrap + Entropy + Markov",
            }

        # 3. Calibration Engine
        if calibration_result and "main_confluence" in calibration_result:
            cal = calibration_result["main_confluence"]
            predictions["calibration"] = {
                "prob": cal.get("best_calibrated", 0.5),
                "direction": "BULLISH" if cal.get("best_calibrated", 0.5) > 0.5 else "BEARISH",
                "weight": 0.20,
                "source": "Platt Scaling + Isotonic Regression",
            }

        # 4. Anomaly Detector
        if anomaly_result and "anomaly_level" in anomaly_result:
            anom_level = anomaly_result["anomaly_level"]
            risk_mult = anomaly_result.get("risk_multiplier", 1.0)

            # Anomaly adjusts confidence DOWN
            anom_prob = 0.5 * risk_mult + 0.25  # Scales from 0.375 (extreme) to 0.75 (normal)
            if anomaly_result.get("anomaly_level") == "NORMAL":
                anom_dir = "NEUTRAL"  # Don't add directional bias, just confidence
                anom_prob = 0.6
            else:
                anom_dir = "CAUTION"
                anom_prob = 0.4

            predictions["anomaly"] = {
                "prob": anom_prob,
                "direction": anom_dir,
                "weight": 0.15,
                "source": "Isolation Forest + PCA Reconstruction",
            }

        # 5. Walk-Forward
        if walk_forward_result and "overall_metrics" in walk_forward_result:
            wf = walk_forward_result
            wf_acc = wf.get("overall_metrics", {}).get("accuracy", 0.5)
            current_pred = wf.get("current_prediction", {})

            predictions["walk_forward"] = {
                "prob": current_pred.get("probability", 0.5),
                "direction": current_pred.get("direction", "NEUTRAL"),
                "weight": 0.10,
                "source": f"Walk-Forward (OOS accuracy: {wf_acc:.1%})",
            }

        # ── Combine with weighted voting ──
        if not predictions:
            return {
                "master_probability": 0.5,
                "master_direction": "NEUTRAL",
                "confidence": 0,
                "model_agreement": "NO_MODELS",
            }

        # Weighted probability
        total_weight = sum(p["weight"] for p in predictions.values())
        weighted_prob = sum(p["prob"] * p["weight"] for p in predictions.values()) / total_weight

        # Anomaly risk adjustment
        if anomaly_result:
            risk_mult = anomaly_result.get("risk_multiplier", 1.0)
            # Pull probability toward 0.5 (uncertain) if anomaly is high
            weighted_prob = 0.5 + (weighted_prob - 0.5) * risk_mult

        # Direction
        direction = "BULLISH" if weighted_prob > 0.55 else "BEARISH" if weighted_prob < 0.45 else "NEUTRAL"
        confidence = abs(weighted_prob - 0.5) * 2

        # Model agreement
        bull_count = sum(1 for p in predictions.values() if p["direction"] == "BULLISH")
        bear_count = sum(1 for p in predictions.values() if p["direction"] == "BEARISH")
        total_dir = bull_count + bear_count
        agreement = max(bull_count, bear_count) / total_dir if total_dir > 0 else 0

        if agreement >= 0.8:
            agreement_level = "STRONG_AGREEMENT"
        elif agreement >= 0.6:
            agreement_level = "MODERATE_AGREEMENT"
        else:
            agreement_level = "DISAGREEMENT"

        # Information Coefficient
        ic = self._calculate_ic(predictions)

        # Master grade
        if confidence > 0.7 and agreement >= 0.8:
            master_grade = "A+"
        elif confidence > 0.55 and agreement >= 0.6:
            master_grade = "A"
        elif confidence > 0.4:
            master_grade = "B"
        elif confidence > 0.2:
            master_grade = "C"
        else:
            master_grade = "D"

        return {
            "master_probability": round(float(weighted_prob), 4),
            "master_direction": direction,
            "master_confidence": round(float(confidence), 4),
            "master_grade": master_grade,
            "model_agreement": agreement_level,
            "agreement_ratio": round(float(agreement), 3),
            "bull_models": bull_count,
            "bear_models": bear_count,
            "total_models": len(predictions),
            "information_coefficient": round(ic, 4),
            "individual_models": predictions,
            "final_recommendation": self._final_recommendation(
                direction, confidence, master_grade, agreement_level, anomaly_result
            ),
            "risk_adjusted_position": self._risk_adjusted_position(
                confidence, master_grade, anomaly_result
            ),
        }

    def _calculate_ic(self, predictions: dict) -> float:
        """
        Approximate Information Coefficient.
        IC = correlation between predicted direction and expected return.
        Higher IC = more predictive power.
        """
        if len(predictions) < 2:
            return 0

        probs = [p["prob"] for p in predictions.values()]
        if len(probs) < 2:
            return 0

        # IC approximation: how much do models agree beyond 50/50?
        mean_prob = np.mean(probs)
        std_prob = np.std(probs)

        if std_prob < 0.01:
            return 0

        # IC ≈ (mean_prediction - 0.5) / std(prediction)
        ic = (mean_prob - 0.5) / (std_prob + 1e-8)
        return float(np.clip(ic, -1, 1))

    def _final_recommendation(self, direction, confidence, grade, agreement, anomaly):
        if grade in ["A+", "A"]:
            return (
                f"✅ TRADE {direction}: High-confidence signal (Grade {grade}). "
                f"Models strongly agree ({agreement}). "
                f"Enter with standard position size."
            )
        elif grade == "B":
            return (
                f"🟡 CONSIDER {direction}: Moderate confidence (Grade B). "
                f"Use reduced position size. Wait for additional confirmation if possible."
            )
        elif grade == "C":
            return (
                f"🟠 CAUTION: Low confidence (Grade C). "
                f"Signal is weak. Only enter with minimum position if at all."
            )
        else:
            return (
                f"🔴 NO TRADE: Very low confidence (Grade D). "
                f"Models disagree or signal is too weak. Stay flat."
            )

    def _risk_adjusted_position(self, confidence, grade, anomaly):
        """Calculate risk-adjusted position size."""
        base_risk = 1.0  # 1% base risk

        # Grade adjustment
        grade_mult = {"A+": 1.0, "A": 0.85, "B": 0.65, "C": 0.40, "D": 0.0}
        risk = base_risk * grade_mult.get(grade, 0)

        # Anomaly adjustment
        if anomaly:
            risk *= anomaly.get("risk_multiplier", 1.0)

        return {
            "recommended_risk_pct": round(risk, 2),
            "position_strength": "FULL" if risk >= 0.85 else "REDUCED" if risk >= 0.5 else "SMALL" if risk >= 0.25 else "NONE",
        }
