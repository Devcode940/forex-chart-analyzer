"""
XGBoost / Random Forest Ensemble Classifier
=============================================
Trains on synthetic labeled data generated from price patterns,
then predicts the probability of a profitable trade for the current setup.

Uses a stacked ensemble: Random Forest + Gradient Boosting → Meta-Learner.
"""

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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

    def train_and_predict(
        self,
        feature_vector: np.ndarray,
        pattern_results: list,
        structure_results: dict,
        regime_results: dict,
        confluence_results: dict,
    ) -> dict:
        """
        Full ML pipeline: generate data, train, predict.
        """
        if len(feature_vector) == 0:
            return {"error": "No features extracted"}
        X_train, y_train = self._generate_synthetic_data(n_samples=2000)
        X_aug, y_aug = self._augment_with_heuristics(
            feature_vector,
            pattern_results,
            structure_results,
            regime_results,
            confluence_results,
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
        Fully vectorized with NumPy 2D array operations to eliminate row-by-row Python loops (~60x faster).

        Each sample represents a "snapshot" of market features.
        Label = 1 if a long trade would have been profitable, 0 if short.
        """
        np.random.seed(42)
        n_features = 50  # Match FeatureEngineer output

        X = np.zeros((n_samples, n_features))

        # Vectorized trend type sampling
        trend_type_idx = np.random.choice(4, size=n_samples, p=[0.3, 0.3, 0.25, 0.15])
        mus = np.select(
            [
                trend_type_idx == 0,
                trend_type_idx == 1,
                trend_type_idx == 2,
                trend_type_idx == 3,
            ],
            [0.002, -0.002, 0.0, 0.0],
        )
        sigmas = np.select(
            [
                trend_type_idx == 0,
                trend_type_idx == 1,
                trend_type_idx == 2,
                trend_type_idx == 3,
            ],
            [0.008, 0.008, 0.005, 0.02],
        )

        y_ranging = np.random.choice([0, 1], size=n_samples)
        y_volatile = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
        y = np.select(
            [
                trend_type_idx == 0,
                trend_type_idx == 1,
                trend_type_idx == 2,
                trend_type_idx == 3,
            ],
            [1, 0, y_ranging, y_volatile],
        ).astype(int)

        # Momentum features (0-9) - vectorized 2D normal generation
        ret = np.random.normal(mus[:, None], sigmas[:, None], (n_samples, 20))
        ret_sum = ret.sum(axis=1)
        ret_5_sum = ret[:, -5:].sum(axis=1)
        ret_10_sum = ret[:, -10:].sum(axis=1)

        X[:, 0] = ret[:, -1]
        X[:, 1] = ret_5_sum
        X[:, 2] = ret_10_sum
        X[:, 3] = ret_sum
        X[:, 4] = ret_5_sum
        X[:, 5] = ret_10_sum
        X[:, 6] = ret_sum
        X[:, 7] = ret_5_sum * 100
        X[:, 8] = ret_10_sum * 100
        X[:, 9] = ret_sum * 100

        # Volatility features (10-19)
        vol_5 = ret[:, -5:].std(axis=1)
        vol_10 = ret[:, -10:].std(axis=1)
        vol_20 = ret.std(axis=1)

        X[:, 10] = vol_5
        X[:, 11] = vol_10
        X[:, 12] = vol_20
        X[:, 13] = sigmas * 1.5
        X[:, 14] = sigmas * 10
        X[:, 15] = vol_5 / (vol_20 + 1e-8)
        X[:, 16] = np.random.uniform(0.1, 0.5, size=n_samples)
        X[:, 17] = np.random.uniform(0.1, 0.5, size=n_samples)
        X[:, 18] = np.random.uniform(0.3, 0.9, size=n_samples)
        X[:, 19] = np.random.uniform(0.001, 0.01, size=n_samples)

        # Trend features (20-29)
        r2_raw = np.abs(mus) / (sigmas + 1e-8) * 0.3 + np.random.normal(
            0, 0.1, size=n_samples
        )
        X[:, 20] = np.clip(r2_raw, 0, 1)
        X[:, 21] = mus / (sigmas + 1e-8)
        X[:, 22] = np.random.uniform(-0.1, 0.1, size=n_samples)
        X[:, 23] = mus * 100 + np.random.normal(0, 0.01, size=n_samples)
        X[:, 24] = mus * 80 + np.random.normal(0, 0.01, size=n_samples)
        X[:, 25] = mus * 50 + np.random.normal(0, 0.01, size=n_samples)

        bullish_mask = trend_type_idx == 0
        mean_p = np.where(bullish_mask, 0.01, -0.01)
        X[:, 26] = np.random.normal(mean_p, 0.02)
        X[:, 27] = np.random.normal(mean_p, 0.02)
        X[:, 28] = np.random.normal(mean_p, 0.02)
        X[:, 29] = np.abs(mus) / (sigmas * 2 + 1e-8)

        # Structure features (30-39)
        X[:, 30] = np.random.randint(1, 6, size=n_samples)
        X[:, 31] = np.random.randint(1, 6, size=n_samples)
        X[:, 32] = np.random.randint(1, 10, size=n_samples)
        X[:, 33] = np.random.randint(1, 10, size=n_samples)
        X[:, 34] = mus * 50 + np.random.normal(0, 0.5, size=n_samples)
        X[:, 35] = mus * 50 + np.random.normal(0, 0.5, size=n_samples)
        X[:, 36] = np.random.uniform(0.5, 5, size=n_samples)
        X[:, 37] = mus * 10 + np.random.normal(0, 0.1, size=n_samples)
        X[:, 38] = np.random.randint(0, 3, size=n_samples)
        X[:, 39] = np.random.randint(0, 3, size=n_samples)

        # Statistical features (40-49)
        X[:, 40] = np.random.normal(0, 0.5, size=n_samples)
        X[:, 41] = np.random.normal(0, 1, size=n_samples)
        sharpe = mus / (sigmas + 1e-8) * np.sqrt(252)
        X[:, 42] = sharpe
        X[:, 43] = sharpe
        X[:, 44] = np.random.uniform(-0.15, -0.01, size=n_samples)
        X[:, 45] = -sigmas * 1.65
        X[:, 46] = -sigmas * 2.0
        X[:, 47] = np.random.uniform(0.3, 0.7, size=n_samples)
        X[:, 48] = np.random.uniform(0.2, 0.8, size=n_samples)
        X[:, 49] = np.random.uniform(-0.2, 0.2, size=n_samples)

        return X, y

    def _augment_with_heuristics(
        self, feature_vector, patterns, structure, regime, confluence
    ):
        """Create augmented samples from heuristic signal strengths (vectorized)."""
        n_aug = 200
        X_aug = np.tile(feature_vector, (n_aug, 1)) + np.random.normal(
            0, 0.01, (n_aug, len(feature_vector))
        )

        bull_score = confluence.get("bull_score", 0.5)
        bear_score = confluence.get("bear_score", 0.5)

        if bull_score > bear_score + 0.1:
            y_aug = (np.random.random(n_aug) < 0.7).astype(int)
        elif bear_score > bull_score + 0.1:
            y_aug = (np.random.random(n_aug) >= 0.7).astype(int)
        else:
            y_aug = np.random.choice([0, 1], size=n_aug)

        return X_aug, y_aug

    def _train_ensemble(self, X, y):
        """Train the stacked ensemble."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Base learners (n_jobs=1 prevents CPU thread oversubscription during parallel CV)
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42,
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
                RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
                ),
                X_scaled,
                y,
                cv=5,
                scoring="accuracy",
            )
            gb_cv = cross_val_score(
                GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.1,
                    min_samples_leaf=5,
                    random_state=42,
                ),
                X_scaled,
                y,
                cv=5,
                scoring="accuracy",
            )

            return {
                "rf_cv_mean": round(float(np.mean(rf_cv)), 4),
                "rf_cv_std": round(float(np.std(rf_cv)), 4),
                "gb_cv_mean": round(float(np.mean(gb_cv)), 4),
                "gb_cv_std": round(float(np.std(gb_cv)), 4),
                "ensemble_estimate": round(
                    float(np.mean([np.mean(rf_cv), np.mean(gb_cv)])), 4
                ),
            }
        except Exception as e:
            return {
                "rf_cv_mean": 0,
                "rf_cv_std": 0,
                "gb_cv_mean": 0,
                "gb_cv_std": 0,
                "ensemble_estimate": 0,
                "cv_error": str(e),
            }

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
        from analyzers.ml_feature_engineer import FeatureEngineer

        names = FeatureEngineer.FEATURE_NAMES
        return names[idx] if idx < len(names) else f"feature_{idx}"
