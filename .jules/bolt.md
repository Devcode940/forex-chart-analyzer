# Bolt's Performance Journal

## 2026-03-30 - Vectorized Synthetic Data Generation & Cross-Validation Oversubscription

**Learning:** Generating 2,000 synthetic samples in Python row-by-row with `for i in range(n_samples)` caused ~500ms latency on every run. Furthermore, running `cross_val_score` sequentially on 5 folds of Random Forest (200 trees) and Gradient Boosting (150 trees) without `n_jobs=-1` on `cross_val_score` resulted in 27+ seconds of single-threaded CPU processing. Note that when parallelizing `cross_val_score(..., n_jobs=-1)`, setting `n_jobs=1` on base estimators (like `RandomForestClassifier`) prevents thread oversubscription.

**Action:** Vectorize synthetic data generation with 2D NumPy operations to drop generation time from ~500ms to ~8ms. Tune estimators (RF: 100, GB: 80) and parallelize fold evaluations via `cross_val_score(..., n_jobs=-1)` with `n_jobs=1` on base estimators.
