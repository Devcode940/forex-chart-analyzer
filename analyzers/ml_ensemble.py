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
        Vectorized with NumPy for maximum performance.
        """
        np.random.seed(42)
        n_features = 50  # Match FeatureEngineer output

        trend_types = np.random.choice(
            [0, 1, 2, 3], size=n_samples, p=[0.3, 0.3, 0.25, 0.15]
        )

        y = np.zeros(n_samples, dtype=int)
        y[trend_types == 0] = 1
        y[trend_types == 1] = 0
        n_rang = np.sum(trend_types == 2)
        if n_rang > 0:
            y[trend_types == 2] = np.random.choice([0, 1], size=n_rang)
        n_vol = np.sum(trend_types == 3)
        if n_vol > 0:
            y[trend_types == 3] = np.random.choice([0, 1], size=n_vol, p=[0.55, 0.45])

        mus = np.zeros(n_samples)
        sigmas = np.zeros(n_samples)

        mus[trend_types == 0] = 0.002
        sigmas[trend_types == 0] = 0.008

        mus[trend_types == 1] = -0.002
        sigmas[trend_types == 1] = 0.008

        mus[trend_types == 2] = 0.0
        sigmas[trend_types == 2] = 0.005

        mus[trend_types == 3] = 0.0
        sigmas[trend_types == 3] = 0.02

        ret = np.random.normal(mus[:, None], sigmas[:, None], size=(n_samples, 20))

        X = np.zeros((n_samples, n_features))

        # Momentum features (0-9)
        X[:, 0] = ret[:, -1]
        X[:, 1] = np.sum(ret[:, -5:], axis=1)
        X[:, 2] = np.sum(ret[:, -10:], axis=1)
        X[:, 3] = np.sum(ret, axis=1)
        X[:, 4] = X[:, 1]
        X[:, 5] = X[:, 2]
        X[:, 6] = X[:, 3]
        X[:, 7] = X[:, 1] * 100
        X[:, 8] = X[:, 2] * 100
        X[:, 9] = X[:, 3] * 100

        # Volatility features (10-19)
        X[:, 10] = np.std(ret[:, -5:], axis=1)
        X[:, 11] = np.std(ret[:, -10:], axis=1)
        X[:, 12] = np.std(ret, axis=1)
        X[:, 13] = sigmas * 1.5
        X[:, 14] = sigmas * 10
        X[:, 15] = X[:, 10] / (X[:, 12] + 1e-8)
        X[:, 16] = np.random.uniform(0.1, 0.5, size=n_samples)
        X[:, 17] = np.random.uniform(0.1, 0.5, size=n_samples)
        X[:, 18] = np.random.uniform(0.3, 0.9, size=n_samples)
        X[:, 19] = np.random.uniform(0.001, 0.01, size=n_samples)

        # Trend features (20-29)
        r2 = np.clip(
            np.abs(mus) / (sigmas + 1e-8) * 0.3
            + np.random.normal(0, 0.1, size=n_samples),
            0,
            1,
        )
        X[:, 20] = r2
        X[:, 21] = mus / (sigmas + 1e-8)
        X[:, 22] = np.random.uniform(-0.1, 0.1, size=n_samples)
        X[:, 23] = mus * 100 + np.random.normal(0, 0.01, size=n_samples)
        X[:, 24] = mus * 80 + np.random.normal(0, 0.01, size=n_samples)
        X[:, 25] = mus * 50 + np.random.normal(0, 0.01, size=n_samples)
        sma_mean = np.where(trend_types == 0, 0.01, -0.01)
        X[:, 26] = np.random.normal(sma_mean, 0.02)
        X[:, 27] = np.random.normal(sma_mean, 0.02)
        X[:, 28] = np.random.normal(sma_mean, 0.02)
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
        X[:, 42] = mus / (sigmas + 1e-8) * np.sqrt(252)
        X[:, 43] = mus / (sigmas + 1e-8) * np.sqrt(252)
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
        n_features = len(feature_vector)
        X_aug = feature_vector + np.random.normal(0, 0.01, size=(n_aug, n_features))

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

        # Base learners - tuned estimators for fast high-accuracy training
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
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
        """Run cross-validation on the base models (parallelized with n_jobs=-1)."""
        X_scaled = self.scaler.transform(X)

        try:
            rf_cv = cross_val_score(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
                X_scaled,
                y,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            )
            gb_cv = cross_val_score(
                GradientBoostingClassifier(
                    n_estimators=80,
                    max_depth=5,
                    learning_rate=0.1,
                    min_samples_leaf=5,
                    random_state=42,
                ),
                X_scaled,
                y,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
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
