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

        # Train isolation forest on baseline (tuned n_estimators=100 for speed without loss of precision)
        X_scaled = self.scaler.fit_transform(X_baseline)
        self.isolation_forest = IsolationForest(
            n_estimators=100, contamination=0.1,
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
        """Generate baseline normal market feature data (vectorized for maximum performance)."""
        np.random.seed(42)
        n_features = 50

        # Sample market regimes in bulk
        regimes = np.random.choice(["trend", "range", "volatile"], size=n_samples, p=[0.5, 0.35, 0.15])
        mus = np.zeros(n_samples)
        sigmas = np.zeros(n_samples)

        trend_mask = (regimes == "trend")
        range_mask = (regimes == "range")
        vol_mask = (regimes == "volatile")

        mus[trend_mask] = np.random.choice([-0.002, 0.002], size=np.sum(trend_mask))
        sigmas[trend_mask] = 0.008
        sigmas[range_mask] = 0.005
        sigmas[vol_mask] = 0.02

        # Bulk generate returns (n_samples, 20)
        rets = np.random.normal(mus[:, None], sigmas[:, None], size=(n_samples, 20))

        X = np.zeros((n_samples, n_features))
        X[:, 0] = rets[:, -1]
        X[:, 1] = np.sum(rets[:, -5:], axis=1)
        X[:, 2] = np.sum(rets[:, -10:], axis=1)
        X[:, 3] = np.sum(rets, axis=1)
        X[:, 4] = X[:, 1]
        X[:, 5] = X[:, 2]
        X[:, 6] = X[:, 3]

        X[:, 7:10] = np.random.normal((mus * 100)[:, None], 0.5, size=(n_samples, 3))

        std_5 = np.std(rets[:, -5:], axis=1)
        std_10 = np.std(rets[:, -10:], axis=1)
        std_all = np.std(rets, axis=1)
        X[:, 10] = std_5
        X[:, 11] = std_10
        X[:, 12] = std_all
        X[:, 13] = sigmas * 1.5

        X[:, 14] = sigmas * 10
        X[:, 15] = std_5 / (std_all + 1e-8)
        X[:, 16:20] = np.random.uniform(0.1, 0.5, size=(n_samples, 4))

        X[:, 20] = np.abs(mus) / (sigmas + 1e-8) * 0.3
        X[:, 21] = mus / (sigmas + 1e-8)
        X[:, 22:30] = np.random.normal(0, 0.1, size=(n_samples, 8))

        X[:, 30] = np.random.randint(1, 6, size=n_samples)
        X[:, 31] = np.random.randint(1, 6, size=n_samples)
        X[:, 32] = np.random.uniform(1, 10, size=n_samples)
        X[:, 33] = np.random.randint(1, 10, size=n_samples)
        X[:, 34] = mus * 50 + np.random.normal(0, 0.5, size=n_samples)
        X[:, 35] = mus * 50 + np.random.normal(0, 0.5, size=n_samples)
        X[:, 36] = np.random.uniform(0.5, 5, size=n_samples)
        X[:, 37] = mus * 10 + np.random.normal(0, 0.1, size=n_samples)
        X[:, 38] = np.random.randint(0, 3, size=n_samples)
        X[:, 39] = np.random.randint(0, 3, size=n_samples)

        X[:, 40] = np.random.normal(0, 0.5, size=n_samples)
        X[:, 41] = np.random.normal(0, 1, size=n_samples)
        X[:, 42] = mus / (sigmas + 1e-8) * 15.87
        X[:, 43] = mus / (sigmas + 1e-8) * 15.87
        X[:, 44] = np.random.uniform(-0.15, -0.01, size=n_samples)
        X[:, 45] = -sigmas * 1.65
        X[:, 46] = -sigmas * 2.0
        X[:, 47] = np.random.uniform(0.3, 0.7, size=n_samples)
        X[:, 48] = np.random.uniform(0.2, 0.8, size=n_samples)
        X[:, 49] = np.random.uniform(-0.2, 0.2, size=n_samples)

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

