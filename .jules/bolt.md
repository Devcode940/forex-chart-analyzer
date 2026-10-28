## 2026-03-30 - Vectorize Synthetic Data Generation and Prevent Thread Oversubscription in ML Cross Validation

**Learning:** Generating synthetic datasets row-by-row with Python scalar loops (`for i in range(n_samples)`) creates significant runtime overhead (~520ms). Vectorizing array sampling with 2D NumPy operations reduces generation latency to ~8ms. Furthermore, passing `n_jobs=-1` on internal estimator models during parallel `cross_val_score` causes severe CPU thread oversubscription on multi-core systems, degrading cross-validation performance. Setting `n_jobs=1` on base estimators during CV avoids thread contention.

**Action:** Vectorize synthetic data generation with bulk 2D NumPy matrix operations and explicitly set `n_jobs=1` on estimators passed to `cross_val_score`.
