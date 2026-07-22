"""
ML Anomaly Detector — Isolation Forest + Autoencoder
=====================================================
Detects anomalous market states (regime changes, black swans, unusual patterns).
When the market is in an anomalous state, standard pattern analysis may fail.

Isolation Forest: Unsupervised anomaly detection via random partitioning
Statistical Autoencoder: Reconstruction error as anomaly score
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MLAnomalyDetector:
    """
    Detects anomalous market states using multiple methods.
    Anomalous states = regime changes, flash events, or unusual configurations
    where normal pattern analysis may be unreliable.
    """

    def __init__(self):
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        self.baseline_stats = {}

    def detect(self, feature_vector: np.ndarray,
               price_series: dict) -> dict:
        """
        Detect if current market state is anomalous.
        Returns anomaly score, interpretation, and risk adjustment.
        """
        if len(feature_vector) == 0:
            return {"error": "No features available"}

        # Generate baseline normal market data
        X_baseline = self._generate_baseline_data(n_samples=1000)

        # Train isolation forest on baseline
        X_scaled = self.scaler.fit_transform(X_baseline)
        self.isolation_forest = IsolationForest(
            n_estimators=200, contamination=0.1,
            random_state=42, n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)

        # PCA for reconstruction anomaly
        self.pca = PCA(n_components=min(10, X_scaled.shape[1]))
        self.pca.fit(X_scaled)

        self.is_trained = True

        # Score the current state
        current_scaled = self.scaler.transform(feature_vector.reshape(1, -1))

        # Method 1: Isolation Forest
        iso_score = self.isolation_forest.decision_function(current_scaled)[0]
        iso_label = self.isolation_forest.predict(current_scaled)[0]
        # iso_label: 1 = normal, -1 = anomaly

        # Method 2: PCA Reconstruction Error
        reconstructed = self.pca.inverse_transform(self.pca.transform(current_scaled))
        reconstruction_error = float(np.mean((current_scaled - reconstructed) ** 2))

        # Method 3: Statistical Z-Score Anomaly
        z_scores = np.abs((current_scaled - self.scaler.mean_) / (self.scaler.scale_ + 1e-8))
        max_zscore = float(np.max(z_scores))
        mean_zscore = float(np.mean(z_scores))
        n_extreme = int(np.sum(z_scores > 3))  # Features > 3 std devs

        # Method 4: Price-based anomaly (GAP detection)
        price_anomaly = self._price_anomaly_score(price_series)

        # Composite anomaly score
        # Normalize each method to [0, 1] where 1 = most anomalous
        iso_anomaly = float(np.clip(1 - (iso_score + 0.5), 0, 1))  # Higher = more anomalous
        pca_anomaly = float(np.clip(reconstruction_error * 100, 0, 1))
        zscore_anomaly = float(np.clip(mean_zscore / 5, 0, 1))

        composite = (iso_anomaly * 0.35 + pca_anomaly * 0.25 +
                     zscore_anomaly * 0.25 + price_anomaly * 0.15)

        # Determine anomaly level
        if composite < 0.25:
            level = "NORMAL"
            color = "#00ff88"
            risk_mult = 1.0
        elif composite < 0.50:
            level = "ELEVATED"
            color = "#ffdd44"
            risk_mult = 0.75
        elif composite < 0.75:
            level = "HIGH"
            color = "#ff8844"
            risk_mult = 0.5
        else:
            level = "EXTREME"
            color = "#ff4444"
            risk_mult = 0.25

        return {
            "anomaly_composite_score": round(composite, 3),
            "anomaly_level": level,
            "risk_multiplier": risk_mult,
            "methods": {
                "isolation_forest": {
                    "score": round(iso_anomaly, 3),
                    "raw_score": round(float(iso_score), 4),
                    "label": "ANOMALY" if iso_label == -1 else "NORMAL",
                },
                "pca_reconstruction": {
                    "score": round(pca_anomaly, 3),
                    "reconstruction_error": round(reconstruction_error, 6),
                    "explained_variance": round(float(np.sum(self.pca.explained_variance_ratio_)), 3),
                },
                "zscore_analysis": {
                    "score": round(zscore_anomaly, 3),
                    "max_zscore": round(max_zscore, 2),
                    "mean_zscore": round(mean_zscore, 2),
                    "extreme_features": n_extreme,
                },
                "price_anomaly": {
                    "score": round(price_anomaly, 3),
                },
            },
            "interpretation": self._interpret_anomaly(level, composite, iso_label, n_extreme),
            "recommendation": self._anomaly_recommendation(level, risk_mult),
        }

    def _generate_baseline_data(self, n_samples: int = 1000) -> np.ndarray:
        """Generate baseline normal market feature data."""
        np.random.seed(42)
        n_features = 50

        X = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            regime = np.random.choice(["trend", "range", "volatile"], p=[0.5, 0.35, 0.15])

            if regime == "trend":
                mu = np.random.choice([-0.002, 0.002])
                sigma = 0.008
            elif regime == "range":
                mu = 0
                sigma = 0.005
            else:
                mu = 0
                sigma = 0.02

            ret = np.random.normal(mu, sigma, 20)
            X[i, 0] = ret[-1]
            X[i, 1] = np.sum(ret[-5:])
            X[i, 2] = np.sum(ret[-10:])
            X[i, 3] = np.sum(ret)
            X[i, 4:7] = [np.sum(ret[-5:]), np.sum(ret[-10:]), np.sum(ret)]
            X[i, 7:10] = [np.random.normal(mu * 100, 0.5) for _ in range(3)]
            X[i, 10:14] = [np.std(ret[-5:]), np.std(ret[-10:]), np.std(ret), sigma * 1.5]
            X[i, 14:20] = [sigma * 10, X[i, 10] / (X[i, 12] + 1e-8)] + [np.random.uniform(0.1, 0.5) for _ in range(4)]
            X[i, 20:30] = [abs(mu) / (sigma + 1e-8) * 0.3, mu / (sigma + 1e-8)] + [np.random.normal(0, 0.1) for _ in range(8)]
            X[i, 30] = np.random.randint(1, 6)
            X[i, 31] = np.random.randint(1, 6)
            X[i, 32] = np.random.uniform(1, 10)
            X[i, 33] = np.random.randint(1, 10)
            X[i, 34] = mu * 50 + np.random.normal(0, 0.5)
            X[i, 35] = mu * 50 + np.random.normal(0, 0.5)
            X[i, 36] = np.random.uniform(0.5, 5)
            X[i, 37] = mu * 10 + np.random.normal(0, 0.1)
            X[i, 38] = np.random.randint(0, 3)
            X[i, 39] = np.random.randint(0, 3)
            X[i, 40:50] = [np.random.normal(0, 0.5), np.random.normal(0, 1), mu / (sigma + 1e-8) * 15.87, mu / (sigma + 1e-8) * 15.87, np.random.uniform(-0.15, -0.01), -sigma * 1.65, -sigma * 2.0, np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 0.8), np.random.uniform(-0.2, 0.2)]

        return X

    def _price_anomaly_score(self, price_series: dict) -> float:
        """Detect price-based anomalies (gaps, spikes)."""
        smoothed = np.array(price_series.get("smoothed", []))

        if len(smoothed) < 5:
            return 0.0

        returns = np.diff(smoothed) / (smoothed[:-1] + 1e-8)
        recent_return = abs(returns[-1]) if len(returns) > 0 else 0
        mean_return = np.mean(np.abs(returns))
        std_return = np.std(returns)

        if std_return > 0:
            z_score = (recent_return - mean_return) / std_return
        else:
            z_score = 0

        # Anomaly if recent return is > 3 standard deviations
        return float(np.clip(z_score / 5, 0, 1))

    def _interpret_anomaly(self, level, composite, iso_label, n_extreme):
        if level == "NORMAL":
            return "Market is in a normal state. Standard pattern analysis is reliable."
        elif level == "ELEVATED":
            return f"Minor anomaly detected (score: {composite:.2f}). Some caution advised — one or more features are unusual."
        elif level == "HIGH":
            return f"Significant anomaly detected ({n_extreme} extreme features). Market may be transitioning between regimes. Pattern reliability is reduced."
        else:
            return f"EXTREME anomaly (score: {composite:.2f}). Market is in an unusual state — likely a regime change or news event. Standard analysis may be WRONG. Reduce risk dramatically."

    def _anomaly_recommendation(self, level, risk_mult):
        recs = {
            "NORMAL": f"Proceed with normal analysis. Risk multiplier: {risk_mult}× (standard sizing).",
            "ELEVATED": f"Reduce position size by {(1-risk_mult)*100:.0f}%. Verify pattern signals with additional confirmation.",
            "HIGH": f"Cut position size to {risk_mult}× normal. Wait for market to stabilize before entering. Use wider stops.",
            "EXTREME": f"Risk multiplier: {risk_mult}×. Consider staying flat. If you must trade, use minimum size and very wide stops. Pattern reliability is LOW.",
        }
        return recs.get(level, "Proceed with caution.")

