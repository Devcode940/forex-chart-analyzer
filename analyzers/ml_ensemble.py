"""
XGBoost / Random Forest Ensemble Classifier
=============================================
Trains on synthetic labeled data generated from price patterns,
then predicts the probability of a profitable trade for the current setup.

Uses a stacked ensemble: Random Forest + Gradient Boosting → Meta-Learner.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class MLEnsemble:
    """
    Ensemble ML classifier for trade prediction.

    Phase 1: Generate synthetic training data from pattern heuristics
    Phase 2: Train Random Forest + Gradient Boosting base learners
    Phase 3: Stack with Logistic Regression meta-learner
    Phase 4: Predict on live features
    """

    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_stats = {}

    def train_and_predict(self, feature_vector: np.ndarray,
                          pattern_results: list,
                          structure_results: dict,
                          regime_results: dict,
                          confluence_results: dict) -> dict:
        """
        Full ML pipeline: generate data, train, predict.
        """
        if len(feature_vector) == 0:
            return {"error": "No features extracted"}
        X_train, y_train = self._generate_synthetic_data(n_samples=2000)
        X_aug, y_aug = self._augment_with_heuristics(
            feature_vector, pattern_results, structure_results, regime_results, confluence_results
        )
        X_train = np.vstack([X_train, X_aug])
        y_train = np.concatenate([y_train, y_aug])
        self._train_ensemble(X_train, y_train)
        prediction = self._predict(feature_vector)
        cv_score = self._cross_validate(X_train, y_train)
        importance = self._feature_importance()

        return {
            "ml_probability": prediction["probability"],
            "ml_direction": prediction["direction"],
            "ml_confidence": prediction["confidence"],
            "rf_probability": prediction["rf_probability"],
            "gb_probability": prediction["gb_probability"],
            "agreement": prediction["agreement"],
            "cv_score": cv_score,
            "feature_importance": importance,
            "training_samples": len(y_train),
            "is_trained": self.is_trained,
        }

    def _generate_synthetic_data(self, n_samples: int = 2000):
        """
        Generate synthetic training data that mimics real forex feature distributions.

        Each sample represents a "snapshot" of market features.
        Label = 1 if a long trade would have been profitable, 0 if short.

        The synthetic data encodes domain knowledge:
        - Strong uptrend + low volatility → likely long winner
        - Strong downtrend + low volatility → likely short winner
        - High volatility + low efficiency → likely loser both ways
        - High efficiency + bullish signals → long winner
        """
        np.random.seed(42)
        n_features = 50  # Match FeatureEngineer output

        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Generate base returns distribution
            trend_type = np.random.choice(["bullish", "bearish", "ranging", "volatile"], p=[0.3, 0.3, 0.25, 0.15])

            if trend_type == "bullish":
                mu, sigma = 0.002, 0.008
                y[i] = 1
            elif trend_type == "bearish":
                mu, sigma = -0.002, 0.008
                y[i] = 0
            elif trend_type == "ranging":
                mu, sigma = 0, 0.005
                y[i] = np.random.choice([0, 1])
            else:  # volatile
                mu, sigma = 0, 0.02
                y[i] = np.random.choice([0, 1], p=[0.55, 0.45])

            # Momentum features (0-9)
            ret = np.random.normal(mu, sigma, 20)
            X[i, 0] = ret[-1]           # return_1
            X[i, 1] = np.sum(ret[-5:])  # return_5
            X[i, 2] = np.sum(ret[-10:]) # return_10
            X[i, 3] = np.sum(ret)       # return_20
            X[i, 4] = np.sum(ret[-5:])  # momentum_5
            X[i, 5] = np.sum(ret[-10:]) # momentum_10
            X[i, 6] = np.sum(ret)       # momentum_20
            base_price = 1.0
            X[i, 7] = (ret[-5:].sum() / base_price * 100) if base_price > 0 else 0  # roc_5
            X[i, 8] = (ret[-10:].sum() / base_price * 100) if base_price > 0 else 0 # roc_10
            X[i, 9] = (ret.sum() / base_price * 100) if base_price > 0 else 0       # roc_20

            # Volatility features (10-19)
            X[i, 10] = np.std(ret[-5:])    # vol_5
            X[i, 11] = np.std(ret[-10:])   # vol_10
            X[i, 12] = np.std(ret)         # vol_20
            X[i, 13] = sigma * 1.5         # vol_50
            X[i, 14] = sigma * 10          # atr_approx
            X[i, 15] = X[i, 10] / (X[i, 12] + 1e-8)  # vol_ratio_5_20
            X[i, 16] = np.random.uniform(0.1, 0.5)    # upper_wick_ratio
            X[i, 17] = np.random.uniform(0.1, 0.5)    # lower_wick_ratio
            X[i, 18] = np.random.uniform(0.3, 0.9)    # body_ratio
            X[i, 19] = np.random.uniform(0.001, 0.01) # range_ratio

            # Trend features (20-29)
            r2 = max(0, min(1, abs(mu) / (sigma + 1e-8) * 0.3 + np.random.normal(0, 0.1)))
            X[i, 20] = r2                           # trend_r2
            X[i, 21] = mu / (sigma + 1e-8)          # trend_slope
            X[i, 22] = np.random.uniform(-0.1, 0.1) # trend_intercept
            X[i, 23] = mu * 100 + np.random.normal(0, 0.01)  # sma_slope_5
            X[i, 24] = mu * 80 + np.random.normal(0, 0.01)   # sma_slope_10
            X[i, 25] = mu * 50 + np.random.normal(0, 0.01)   # sma_slope_20
            X[i, 26] = np.random.normal(0.01 if trend_type == "bullish" else -0.01, 0.02)  # price_vs_sma5
            X[i, 27] = np.random.normal(0.01 if trend_type == "bullish" else -0.01, 0.02)  # price_vs_sma10
            X[i, 28] = np.random.normal(0.01 if trend_type == "bullish" else -0.01, 0.02)  # price_vs_sma20
            X[i, 29] = abs(mu) / (sigma * 2 + 1e-8)  # efficiency_ratio

            # Structure features (30-39)
            X[i, 30] = np.random.randint(1, 6)  # swing_high_count
            X[i, 31] = np.random.randint(1, 6)  # swing_low_count
            X[i, 32] = np.random.randint(1, 10)  # last_swing_high_dist
            X[i, 33] = np.random.randint(1, 10)  # last_swing_low_dist
            X[i, 34] = mu * 50 + np.random.normal(0, 0.5)  # swing_high_slope
            X[i, 35] = mu * 50 + np.random.normal(0, 0.5)  # swing_low_slope
            X[i, 36] = np.random.uniform(0.5, 5)  # channel_width
            X[i, 37] = mu * 10 + np.random.normal(0, 0.1)  # channel_slope
            X[i, 38] = np.random.randint(0, 3)  # bos_count_bull
            X[i, 39] = np.random.randint(0, 3)  # bos_count_bear

            # Statistical features (40-49)
            X[i, 40] = np.random.normal(0, 0.5)  # skewness
            X[i, 41] = np.random.normal(0, 1)    # kurtosis
            X[i, 42] = mu / (sigma + 1e-8) * np.sqrt(252)  # sharpe_approx
            X[i, 43] = mu / (sigma + 1e-8) * np.sqrt(252)  # sortino_approx
            X[i, 44] = np.random.uniform(-0.15, -0.01)  # max_drawdown
            X[i, 45] = -sigma * 1.65  # var_95
            X[i, 46] = -sigma * 2.0   # cvar_95
            X[i, 47] = np.random.uniform(0.3, 0.7)  # hurst_exponent
            X[i, 48] = np.random.uniform(0.2, 0.8)  # mean_reversion_score
            X[i, 49] = np.random.uniform(-0.2, 0.2)  # serial_correlation

        return X, y

    def _augment_with_heuristics(self, feature_vector, patterns, structure,
                                  regime, confluence):
        """Create augmented samples from heuristic signal strengths."""
        n_aug = 200
        X_aug = np.tile(feature_vector, (n_aug, 1))
        y_aug = np.zeros(n_aug, dtype=int)

        # Determine label from heuristics
        bull_score = confluence.get("bull_score", 0.5)
        bear_score = confluence.get("bear_score", 0.5)

        for i in range(n_aug):
            noise = np.random.normal(0, 0.01, X_aug.shape[1])
            X_aug[i] += noise

            # Label based on confluence with some randomness
            if bull_score > bear_score + 0.1:
                y_aug[i] = 1 if np.random.random() < 0.7 else 0
            elif bear_score > bull_score + 0.1:
                y_aug[i] = 0 if np.random.random() < 0.7 else 1
            else:
                y_aug[i] = np.random.choice([0, 1])

        return X_aug, y_aug

    def _train_ensemble(self, X, y):
        """Train the stacked ensemble."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Base learners
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )

        # Train base learners
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)

        # Generate meta-features
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        gb_proba = self.gb_model.predict_proba(X_scaled)[:, 1]
        meta_features = np.column_stack([rf_proba, gb_proba])

        # Train meta-learner
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(meta_features, y)

        self.is_trained = True
        self.training_stats = {
            "n_samples": len(y),
            "n_positive": int(np.sum(y)),
            "n_negative": int(len(y) - np.sum(y)),
            "class_balance": float(np.mean(y)),
        }

    def _predict(self, feature_vector: np.ndarray) -> dict:
        """Predict using the stacked ensemble."""
        X = feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Base predictions
        rf_proba = self.rf_model.predict_proba(X_scaled)[0, 1]
        gb_proba = self.gb_model.predict_proba(X_scaled)[0, 1]

        # Meta prediction
        meta_features = np.array([[rf_proba, gb_proba]])
        meta_proba = self.meta_model.predict_proba(meta_features)[0, 1]

        direction = "BULLISH" if meta_proba > 0.5 else "BEARISH"
        confidence = abs(meta_proba - 0.5) * 2  # Scale to [0, 1]

        return {
            "probability": round(float(meta_proba), 4),
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "rf_probability": round(float(rf_proba), 4),
            "gb_probability": round(float(gb_proba), 4),
            "agreement": "YES" if (rf_proba > 0.5) == (gb_proba > 0.5) else "NO",
        }

    def _cross_validate(self, X, y) -> dict:
        """Run cross-validation on the base models."""
        X_scaled = self.scaler.transform(X)

        try:
            rf_cv = cross_val_score(
                RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
                X_scaled, y, cv=5, scoring='accuracy'
            )
            gb_cv = cross_val_score(
                GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
                X_scaled, y, cv=5, scoring='accuracy'
            )

            return {
                "rf_cv_mean": round(float(np.mean(rf_cv)), 4),
                "rf_cv_std": round(float(np.std(rf_cv)), 4),
                "gb_cv_mean": round(float(np.mean(gb_cv)), 4),
                "gb_cv_std": round(float(np.std(gb_cv)), 4),
                "ensemble_estimate": round(float(np.mean([np.mean(rf_cv), np.mean(gb_cv)])), 4),
            }
        except Exception as e:
            return {"rf_cv_mean": 0, "rf_cv_std": 0, "gb_cv_mean": 0, "gb_cv_std": 0, "ensemble_estimate": 0, "cv_error": str(e)}

    def _feature_importance(self) -> list:
        """Get feature importance from both models."""
        if not self.is_trained:
            return []

        rf_imp = self.rf_model.feature_importances_
        gb_imp = self.gb_model.feature_importances_

        # Average importance from both models
        avg_imp = (rf_imp + gb_imp) / 2

        top_indices = np.argsort(avg_imp)[::-1][:10]

        return [
            {
                "feature": self._feature_name(idx),
                "importance": round(float(avg_imp[idx]), 4),
                "rf_importance": round(float(rf_imp[idx]), 4),
                "gb_importance": round(float(gb_imp[idx]), 4),
            }
            for idx in top_indices
        ]

    def _feature_name(self, idx: int) -> str:
        names = FeatureEngineer.FEATURE_NAMES if idx < len(FeatureEngineer.FEATURE_NAMES) else [f"feature_{i}" for i in range(50)]
        return names[idx] if idx < len(names) else f"feature_{idx}"

# Lazy import for feature names
from analyzers.ml_feature_engineer import FeatureEngineer

