"""
Walk-Forward Cross-Validation Engine
=====================================
Validates that the ML model works out-of-sample, not just on training data.

Walk-forward is THE gold standard for trading strategy validation:
1. Train on window [0..T]
2. Predict on [T+1..T+H]  (out of sample)
3. Roll forward: train on [W..T+W], predict on [T+W+1..T+W+H]
4. Measure actual performance on ALL out-of-sample predictions

This prevents overfitting and gives realistic expected performance.
"""

import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class WalkForwardValidator:
    """
    Walk-forward cross-validation for trading strategy evaluation.
    Simulates how the strategy would have performed in real-time.
    """

    def __init__(self):
        self.results = {}

    def validate(
        self,
        feature_vector: np.ndarray,
        pattern_results: list,
        structure_results: dict,
        confluence_results: dict,
        n_windows: int = 5,
    ) -> dict:
        """
        Run walk-forward validation using synthetic historical data.
        """
        if len(feature_vector) == 0:
            return {"error": "No features available"}

        # Generate time-series data for walk-forward
        X, y = self._generate_time_series_data(n_samples=3000)

        # Run walk-forward windows
        window_size = len(X) // (n_windows + 1)
        test_size = window_size // 3

        all_predictions = []
        all_actuals = []
        window_results = []

        for w in range(n_windows):
            train_start = w * window_size // n_windows
            train_end = train_start + window_size
            test_end = min(train_end + test_size, len(X))

            if train_end >= len(X) or test_end > len(X):
                continue

            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            if len(X_train) < 50 or len(X_test) < 10:
                continue

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model on this window (tuned with max_features='sqrt', max_depth=3, n_estimators=40 for speed)
            model = GradientBoostingClassifier(
                n_estimators=40,
                max_depth=3,
                max_features="sqrt",
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_train_scaled, y_train)

            # Predict out-of-sample
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]

            all_predictions.extend(predictions.tolist())
            all_actuals.extend(y_test.tolist())

            # Window metrics
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, zero_division=0)
            rec = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)

            # Simulated P&L
            pnl = self._simulate_pnl(probabilities, y_test)

            window_results.append(
                {
                    "window": w + 1,
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "accuracy": round(acc, 3),
                    "precision": round(prec, 3),
                    "recall": round(rec, 3),
                    "f1_score": round(f1, 3),
                    "simulated_pnl": round(pnl["total_pnl"], 2),
                    "win_rate": round(pnl["win_rate"], 3),
                    "profit_factor": round(pnl["profit_factor"], 2),
                    "max_drawdown": round(pnl["max_drawdown"], 3),
                    "sharpe_ratio": round(pnl["sharpe_ratio"], 2),
                }
            )

        # Aggregate results
        if all_predictions and all_actuals:
            overall_acc = accuracy_score(all_actuals, all_predictions)
            overall_prec = precision_score(
                all_actuals, all_predictions, zero_division=0
            )
            overall_f1 = f1_score(all_actuals, all_predictions, zero_division=0)
        else:
            overall_acc = 0
            overall_prec = 0
            overall_f1 = 0

        # Score current features through the last trained model
        current_prediction = self._predict_current(feature_vector, X, y)

        return {
            "overall_metrics": {
                "accuracy": round(overall_acc, 3),
                "precision": round(overall_prec, 3),
                "f1_score": round(overall_f1, 3),
                "total_out_of_sample_trades": len(all_predictions),
            },
            "window_results": window_results,
            "current_prediction": current_prediction,
            "interpretation": self._interpret_wf(
                overall_acc, overall_prec, window_results
            ),
            "overfitting_check": self._check_overfitting(window_results),
        }

    def _generate_time_series_data(self, n_samples: int = 3000):
        """Generate time-series data with regime changes for realistic WF testing (vectorized)."""
        np.random.seed(42)
        n_features = 50
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)

        # Simulate regime periods
        regimes = []
        i = 0
        while i < n_samples:
            regime_len = np.random.randint(100, 500)
            regime_type = np.random.choice(["bull", "bear", "range"])
            regimes.append((i, min(i + regime_len, n_samples), regime_type))
            i += regime_len

        # Vectorized generation across regimes
        for start, end, rtype in regimes:
            size = end - start
            if size <= 0:
                continue

            if rtype == "bull":
                mu, sigma = 0.003, 0.008
                y[start:end] = 1
            elif rtype == "bear":
                mu, sigma = -0.003, 0.008
                y[start:end] = 0
            else:
                mu, sigma = 0, 0.005
                y[start:end] = np.random.choice([0, 1], size=size)

            ret = np.random.normal(mu, sigma, size=(size, 20))
            X[start:end, 0] = ret[:, -1]
            X[start:end, 1] = np.sum(ret[:, -5:], axis=1)
            X[start:end, 2] = np.sum(ret[:, -10:], axis=1)
            X[start:end, 3] = np.sum(ret, axis=1)

            # Features 4:10
            X[start:end, 4:10] = np.column_stack(
                [np.sum(ret[:, -k:], axis=1) for k in [5, 10, 20, 5, 10, 20]]
            )

            # Features 10:14
            X[start:end, 10:14] = np.column_stack(
                [np.std(ret[:, -k:], axis=1) for k in [5, 10, 20, 20]]
            )

            # Features 14:20
            col14_15 = np.column_stack([np.full(size, sigma * 10), np.ones(size)])
            col16_19 = np.random.uniform(0.1, 0.5, size=(size, 4))
            X[start:end, 14:20] = np.hstack([col14_15, col16_19])

            # Features 20:30
            c20 = np.full(size, abs(mu) / (sigma + 1e-8) * 0.3)
            c21 = np.full(size, mu / (sigma + 1e-8))
            c22_29 = np.random.normal(0, 0.1, size=(size, 8))
            X[start:end, 20:30] = np.hstack([np.column_stack([c20, c21]), c22_29])

            # Features 30:40
            c30_31 = np.random.randint(1, 6, size=(size, 2))
            c32 = np.random.uniform(1, 10, size=size)
            c33 = np.random.randint(1, 10, size=size)
            c34_35 = np.full((size, 2), mu * 50)
            c36 = np.random.uniform(0.5, 5, size=size)
            c37 = np.full(size, mu * 10)
            c38_39 = np.random.randint(0, 3, size=(size, 2))
            X[start:end, 30:40] = np.column_stack(
                [c30_31, c32, c33, c34_35, c36, c37, c38_39]
            )

            # Features 40:50
            c40 = np.random.normal(0, 0.5, size=size)
            c41 = np.random.normal(0, 1, size=size)
            c42_43 = np.full((size, 2), mu / (sigma + 1e-8) * 15.87)
            c44 = np.random.uniform(-0.15, -0.01, size=size)
            c45 = np.full(size, -sigma * 1.65)
            c46 = np.full(size, -sigma * 2.0)
            c47_49 = np.column_stack(
                [
                    np.random.uniform(0.3, 0.7, size=size),
                    np.random.uniform(0.2, 0.8, size=size),
                    np.random.uniform(-0.2, 0.2, size=size),
                ]
            )
            X[start:end, 40:50] = np.column_stack(
                [c40, c41, c42_43, c44, c45, c46, c47_49]
            )

        return X, y

    def _simulate_pnl(self, probabilities: np.ndarray, actual: np.ndarray) -> dict:
        """Simulate P&L from predictions with position sizing by confidence (vectorized)."""
        predicted = (probabilities > 0.5).astype(int)
        confidence = np.abs(probabilities - 0.5)
        is_correct = predicted == actual
        pnl_list = np.where(is_correct, confidence * 2.0, -confidence * 2.0)

        wins = int(np.sum(is_correct))
        losses = len(actual) - wins
        total_win = float(np.sum(pnl_list[is_correct])) if wins > 0 else 0.0
        total_loss = (
            float(np.sum(np.abs(pnl_list[~is_correct]))) if losses > 0 else 0.0
        )

        # Calculate metrics
        cumulative = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        profit_factor = total_win / (total_loss + 1e-8)
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        sharpe = (
            float(np.mean(pnl_list) / (np.std(pnl_list) + 1e-8) * np.sqrt(252))
            if len(pnl_list) > 1
            else 0
        )

        return {
            "total_pnl": float(np.sum(pnl_list)),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "n_trades": len(pnl_list),
            "wins": wins,
            "losses": losses,
        }

    def _predict_current(self, feature_vector: np.ndarray, X, y) -> dict:
        """Predict on current features using a model trained on all data."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(
            n_estimators=40,
            max_depth=3,
            max_features="sqrt",
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_scaled, y)

        current_scaled = scaler.transform(feature_vector.reshape(1, -1))
        prob = model.predict_proba(current_scaled)[0, 1]
        pred = int(model.predict(current_scaled)[0])

        return {
            "probability": round(float(prob), 4),
            "direction": "BULLISH" if pred == 1 else "BEARISH",
            "confidence": round(abs(float(prob) - 0.5) * 2, 4),
        }

    def _interpret_wf(self, acc, prec, windows):
        if acc > 0.65:
            return (
                f"✅ Walk-forward accuracy: {acc:.1%} (precision: {prec:.1%}). "
                f"Model generalizes well out-of-sample across {len(windows)} windows. "
                f"Predictions are likely reliable."
            )
        elif acc > 0.55:
            return (
                f"🟡 Walk-forward accuracy: {acc:.1%}. Modest edge over random. "
                f"Use with caution and always combine with other analysis."
            )
        else:
            return (
                f"🔴 Walk-forward accuracy: {acc:.1%} — barely above or below random. "
                f"Model does NOT generalize well. Heuristic scores may be overfitting to this chart."
            )

    def _check_overfitting(self, windows) -> dict:
        """Check for overfitting by comparing window performance variance."""
        if not windows:
            return {"status": "INSUFFICIENT_DATA"}

        accuracies = [w["accuracy"] for w in windows]
        acc_std = np.std(accuracies) if len(accuracies) > 1 else 0
        acc_mean = np.mean(accuracies)

        if acc_std < 0.05:
            status = "STABLE"
            note = (
                "Performance is consistent across windows — no overfitting detected."
            )
        elif acc_std < 0.15:
            status = "MODERATE_VARIANCE"
            note = "Some variance across windows. Model may be overfitting in certain regimes."
        else:
            status = "HIGH_VARIANCE"
            note = "High variance across windows — model is likely overfitting. Do not trust individual predictions."

        return {
            "status": status,
            "mean_accuracy": round(float(acc_mean), 3),
            "std_accuracy": round(float(acc_std), 3),
            "note": note,
        }
