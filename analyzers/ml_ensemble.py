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

        Each sample represents a "snapshot" of market features.
        Label = 1 if a long trade would have been profitable, 0 if short.

        Optimized via 2D NumPy vectorization to bypass per-sample loop overhead.
        """
        rng = np.random.default_rng(42)
        n_features = 50  # Match FeatureEngineer output

        # 0: bullish, 1: bearish, 2: ranging, 3: volatile
        trend_idx = rng.choice(4, size=n_samples, p=[0.3, 0.3, 0.25, 0.15])
        mu_map = np.array([0.002, -0.002, 0.0, 0.0])
        sigma_map = np.array([0.008, 0.008, 0.005, 0.02])
        mu = mu_map[trend_idx]
        sigma = sigma_map[trend_idx]

        y = np.empty(n_samples, dtype=int)
        y[trend_idx == 0] = 1
        y[trend_idx == 1] = 0
        ranging_mask = trend_idx == 2
        y[ranging_mask] = rng.integers(0, 2, size=np.count_nonzero(ranging_mask))
        volatile_mask = trend_idx == 3
        y[volatile_mask] = rng.choice(
            [0, 1], size=np.count_nonzero(volatile_mask), p=[0.55, 0.45]
        )

        # Momentum features (0-9) - vectorized 2D normal sampling
        ret = rng.normal(mu[:, None], sigma[:, None], size=(n_samples, 20))
        ret_5 = ret[:, -5:]
        ret_10 = ret[:, -10:]
        sum_5 = ret_5.sum(axis=1)
        sum_10 = ret_10.sum(axis=1)
        sum_20 = ret.sum(axis=1)

        X = np.empty((n_samples, n_features))
        X[:, 0] = ret[:, -1]
        X[:, 1] = sum_5
        X[:, 2] = sum_10
        X[:, 3] = sum_20
        X[:, 4] = sum_5
        X[:, 5] = sum_10
        X[:, 6] = sum_20
        X[:, 7] = sum_5 * 100.0
        X[:, 8] = sum_10 * 100.0
        X[:, 9] = sum_20 * 100.0

        # Volatility features (10-19)
        std_5 = np.std(ret_5, axis=1)
        std_10 = np.std(ret_10, axis=1)
        std_20 = np.std(ret, axis=1)
        X[:, 10] = std_5
        X[:, 11] = std_10
        X[:, 12] = std_20
        X[:, 13] = sigma * 1.5
        X[:, 14] = sigma * 10.0
        X[:, 15] = std_5 / (std_20 + 1e-8)
        X[:, 16] = rng.uniform(0.1, 0.5, size=n_samples)
        X[:, 17] = rng.uniform(0.1, 0.5, size=n_samples)
        X[:, 18] = rng.uniform(0.3, 0.9, size=n_samples)
        X[:, 19] = rng.uniform(0.001, 0.01, size=n_samples)

        # Trend features (20-29)
        r2_raw = np.abs(mu) / (sigma + 1e-8) * 0.3 + rng.normal(0, 0.1, size=n_samples)
        X[:, 20] = np.clip(r2_raw, 0, 1)
        X[:, 21] = mu / (sigma + 1e-8)
        X[:, 22] = rng.uniform(-0.1, 0.1, size=n_samples)
        X[:, 23] = mu * 100 + rng.normal(0, 0.01, size=n_samples)
        X[:, 24] = mu * 80 + rng.normal(0, 0.01, size=n_samples)
        X[:, 25] = mu * 50 + rng.normal(0, 0.01, size=n_samples)
        center_p = np.where(trend_idx == 0, 0.01, np.where(trend_idx == 1, -0.01, 0.0))
        X[:, 26] = rng.normal(center_p, 0.02)
        X[:, 27] = rng.normal(center_p, 0.02)
        X[:, 28] = rng.normal(center_p, 0.02)
        X[:, 29] = np.abs(mu) / (sigma * 2 + 1e-8)

        # Structure features (30-39)
        X[:, 30] = rng.integers(1, 6, size=n_samples)
        X[:, 31] = rng.integers(1, 6, size=n_samples)
        X[:, 32] = rng.integers(1, 10, size=n_samples)
        X[:, 33] = rng.integers(1, 10, size=n_samples)
        X[:, 34] = mu * 50 + rng.normal(0, 0.5, size=n_samples)
        X[:, 35] = mu * 50 + rng.normal(0, 0.5, size=n_samples)
        X[:, 36] = rng.uniform(0.5, 5.0, size=n_samples)
        X[:, 37] = mu * 10 + rng.normal(0, 0.1, size=n_samples)
        X[:, 38] = rng.integers(0, 3, size=n_samples)
        X[:, 39] = rng.integers(0, 3, size=n_samples)

        # Statistical features (40-49)
        X[:, 40] = rng.normal(0, 0.5, size=n_samples)
        X[:, 41] = rng.normal(0, 1.0, size=n_samples)
        sharpe = mu / (sigma + 1e-8) * np.sqrt(252)
        X[:, 42] = sharpe
        X[:, 43] = sharpe
        X[:, 44] = rng.uniform(-0.15, -0.01, size=n_samples)
        X[:, 45] = -sigma * 1.65
        X[:, 46] = -sigma * 2.0
        X[:, 47] = rng.uniform(0.3, 0.7, size=n_samples)
        X[:, 48] = rng.uniform(0.2, 0.8, size=n_samples)
        X[:, 49] = rng.uniform(-0.2, 0.2, size=n_samples)

        return X, y

    def _augment_with_heuristics(
        self, feature_vector, patterns, structure, regime, confluence
    ):
        """Create augmented samples from heuristic signal strengths vectorized with NumPy."""
        n_aug = 200
        noise = np.random.normal(0, 0.01, (n_aug, len(feature_vector)))
        X_aug = np.tile(feature_vector, (n_aug, 1)) + noise

        # Determine label from heuristics
        bull_score = confluence.get("bull_score", 0.5)
        bear_score = confluence.get("bear_score", 0.5)

        if bull_score > bear_score + 0.1:
            y_aug = (np.random.random(n_aug) < 0.7).astype(int)
        elif bear_score > bull_score + 0.1:
            y_aug = (np.random.random(n_aug) >= 0.7).astype(int)
        else:
            y_aug = np.random.randint(0, 2, size=n_aug)

        return X_aug, y_aug

    def _train_ensemble(self, X, y):
        """Train the stacked ensemble."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Base learners (optimized hyperparameters for latency and accuracy)
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=4,
            max_features="sqrt",
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
        """Run cross-validation on the base models in parallel across folds."""
        X_scaled = self.scaler.transform(X)

        try:
            # Parallelize cross-validation folds across available CPU cores (n_jobs=-1).
            # Underlying RF estimator uses n_jobs=1 to avoid thread oversubscription.
            rf_cv = cross_val_score(
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=1,
                ),
                X_scaled,
                y,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
            )
            gb_cv = cross_val_score(
                GradientBoostingClassifier(
                    n_estimators=80,
                    max_depth=4,
                    max_features="sqrt",
                    learning_rate=0.1,
                    min_samples_leaf=5,
                    random_state=42,
                ),
                X_scaled,
                y,
                cv=5,
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
