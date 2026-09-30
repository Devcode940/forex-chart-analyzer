## 2025-05-20 - Vectorized Synthetic Data Generation & Cross-Validation Threading in MLEnsemble

**Learning:** Iterative Python `for` loops in synthetic sample generation and heuristic augmentation create a massive bottleneck (~500ms). Furthermore, passing `n_jobs=-1` to both `cross_val_score` and `RandomForestClassifier` simultaneously causes severe CPU thread oversubscription, slowing down cross-validation.
**Action:** Vectorize synthetic data generation using 2D NumPy array operations and set `n_jobs=1` on base estimators when cross-validating with `cross_val_score(..., n_jobs=-1)`.
