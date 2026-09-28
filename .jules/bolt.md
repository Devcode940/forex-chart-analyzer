# Bolt's Performance Journal

## 2025-05-18 - Thread Oversubscription in Parallel Cross-Validation & Synthetic Vectorization

**Learning:** When using `scikit-learn`'s `cross_val_score(..., n_jobs=-1)` alongside base estimators that also use parallel worker threads (e.g., `RandomForestClassifier(n_jobs=-1)`), heavy CPU thread contention and process spawning overhead significantly slow down execution (~37.7s for `MLEnsemble.train_and_predict`). Setting `n_jobs=1` on the underlying estimator while allowing `cross_val_score` to manage `n_jobs=-1` (or vice versa) prevents oversubscription. Additionally, vectorizing synthetic data generation across samples using 2D NumPy operations reduces synthetic data generation latency from ~500ms to ~8ms.

**Action:** Always set `n_jobs=1` on estimators passed into `cross_val_score(..., n_jobs=-1)` and vectorize synthetic feature creation with 2D NumPy matrix operations instead of iterative Python loops over samples.
