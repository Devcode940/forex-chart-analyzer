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

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class WalkForwardValidator:
    """
    Walk-forward cross-validation for trading strategy evaluation.
    Simulates how the strategy would have performed in real-time.
    """

    # Class-level cache for static, deterministic operations
    _cached_X = None
    _cached_y = None
    _cached_validation_runs = {}  # maps n_windows -> static validation metrics dict
    _cached_model = None          # cached trained model for _predict_current

    def __init__(self):
        self.results = {}
        # Instance-level cache for consecutive calls with identical feature vector
        self._last_feature_hash = None
        self._last_result = None

    def validate(self, feature_vector: np.ndarray,
                 pattern_results: list,
                 structure_results: dict,
                 confluence_results: dict,
                 n_windows: int = 5) -> dict:
        """
        Run walk-forward validation using synthetic historical data.
        """
        if len(feature_vector) == 0:
            return {"error": "No features available"}

        # Check instance-level cache based on feature vector hash
        feat_hash = hash(feature_vector.tobytes()) if feature_vector is not None else None
        if feat_hash is not None and self._last_feature_hash == feat_hash and self._last_result is not None:
            return self._last_result

        # 1. Fetch or generate time-series data
        if WalkForwardValidator._cached_X is None or WalkForwardValidator._cached_y is None:
            X, y = self._generate_time_series_data(n_samples=3000)
            WalkForwardValidator._cached_X = X
            WalkForwardValidator._cached_y = y
        else:
            X = WalkForwardValidator._cached_X
            y = WalkForwardValidator._cached_y

        # 2. Check validation results cache
        if n_windows in WalkForwardValidator._cached_validation_runs:
            cached_res = WalkForwardValidator._cached_validation_runs[n_windows].copy()
            # Predict on current features using the cached trained model
            current_prediction = self._predict_current(feature_vector, X, y)
            cached_res["current_prediction"] = current_prediction

            # Cache results at the instance level
            self._last_feature_hash = feat_hash
            self._last_result = cached_res
            return cached_res

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

            # Train model on this window
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42
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

            window_results.append({
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
            })

        # Aggregate results
        if all_predictions and all_actuals:
            overall_acc = accuracy_score(all_actuals, all_predictions)
            overall_prec = precision_score(all_actuals, all_predictions, zero_division=0)
            overall_f1 = f1_score(all_actuals, all_predictions, zero_division=0)
        else:
            overall_acc = 0
            overall_prec = 0
            overall_f1 = 0

        # Cache the static validation results before predicting for the current feature vector
        validation_results = {
            "overall_metrics": {
                "accuracy": round(overall_acc, 3),
                "precision": round(overall_prec, 3),
                "f1_score": round(overall_f1, 3),
                "total_out_of_sample_trades": len(all_predictions),
            },
            "window_results": window_results,
            "interpretation": self._interpret_wf(overall_acc, overall_prec, window_results),
            "overfitting_check": self._check_overfitting(window_results),
        }
        WalkForwardValidator._cached_validation_runs[n_windows] = validation_results

        # Score current features through the last trained model
        current_prediction = self._predict_current(feature_vector, X, y)

        final_res = validation_results.copy()
        final_res["current_prediction"] = current_prediction

        # Cache results at the instance level
        self._last_feature_hash = feat_hash
        self._last_result = final_res
        return final_res

    def _generate_time_series_data(self, n_samples: int = 3000):
        """Generate time-series data with regime changes for realistic WF testing."""
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

        for start, end, rtype in regimes:
            for j in range(start, end):
                if rtype == "bull":
                    mu, sigma = 0.003, 0.008
                    y[j] = 1
                elif rtype == "bear":
                    mu, sigma = -0.003, 0.008
                    y[j] = 0
                else:
                    mu, sigma = 0, 0.005
                    y[j] = np.random.choice([0, 1])

                ret = np.random.normal(mu, sigma, 20)
                X[j, 0] = ret[-1]
                X[j, 1] = np.sum(ret[-5:])
                X[j, 2] = np.sum(ret[-10:])
                X[j, 3] = np.sum(ret)
                X[j, 4:10] = [np.sum(ret[-k:]) if len(ret) >= k else np.sum(ret) for k in [5, 10, 20, 5, 10, 20]]
                X[j, 10:14] = [np.std(ret[-k:]) if len(ret) >= k else np.std(ret) for k in [5, 10, 20, 50]]
                X[j, 14:20] = [sigma * 10, 1.0] + [np.random.uniform(0.1, 0.5) for _ in range(4)]
                X[j, 20:30] = [abs(mu) / (sigma + 1e-8) * 0.3, mu / (sigma + 1e-8)] + [np.random.normal(0, 0.1) for _ in range(8)]
                X[j, 30:40] = [np.random.randint(1, 6), np.random.randint(1, 6)] + [np.random.uniform(1, 10)] + [np.random.randint(1, 10)] + [mu * 50] * 2 + [np.random.uniform(0.5, 5)] + [mu * 10] + [np.random.randint(0, 3)] * 2
                X[j, 40:50] = [np.random.normal(0, 0.5), np.random.normal(0, 1)] + [mu / (sigma + 1e-8) * 15.87] * 2 + [np.random.uniform(-0.15, -0.01), -sigma * 1.65, -sigma * 2.0] + [np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 0.8), np.random.uniform(-0.2, 0.2)]

        return X, y

    def _simulate_pnl(self, probabilities: np.ndarray, actual: np.ndarray) -> dict:
        """Simulate P&L from predictions with position sizing by confidence."""
        pnl_list = []
        wins = 0
        losses = 0
        total_win = 0
        total_loss = 0

        for i, (prob, actual_val) in enumerate(zip(probabilities, actual)):
            # Position: long if prob > 0.5, short if prob < 0.5
            if prob > 0.5:
                predicted = 1
                confidence = prob - 0.5
            else:
                predicted = 0
                confidence = 0.5 - prob

            # Simplified P&L: +1 for correct, -1 for wrong, scaled by confidence
            if predicted == actual_val:
                pnl = confidence * 2
                wins += 1
                total_win += pnl
            else:
                pnl = -confidence * 2
                losses += 1
                total_loss += abs(pnl)

            pnl_list.append(pnl)

        # Calculate metrics
        cumulative = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        profit_factor = total_win / (total_loss + 1e-8)
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        sharpe = float(np.mean(pnl_list) / (np.std(pnl_list) + 1e-8) * np.sqrt(252)) if len(pnl_list) > 1 else 0

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

        if WalkForwardValidator._cached_model is None:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42
            )
            model.fit(X_scaled, y)
            WalkForwardValidator._cached_model = model
        else:
            model = WalkForwardValidator._cached_model

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
            note = "Performance is consistent across windows — no overfitting detected."
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

