## 2025-05-18 - Optimizing Ensemble ML Training and Cross-Validation

**Learning:** GradientBoostingClassifier without `max_features` defaults to searching all features at every split, which creates a severe bottleneck during CV and fitting. Vectorizing synthetic data generation with 2D NumPy matrices and setting `max_features='sqrt'` on GradientBoostingClassifier while parallelizing `cross_val_score(..., n_jobs=-1)` (with estimator `n_jobs=1`) reduces ensemble pipeline latency by ~88% (from ~42.8s to ~4.9s).

**Action:** Always set `max_features='sqrt'` for tree boosting on multi-feature datasets and ensure parallel `cross_val_score` uses estimators with `n_jobs=1` to prevent CPU oversubscription.
