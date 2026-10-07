# Bolt's Performance Journal

## 2026-10-07 - Cross-Validation Oversubscription & Vectorized Data Generation
**Learning:** Default sequential `cross_val_score` on tree ensembles like `GradientBoostingClassifier` without `max_features='sqrt'` causes massive CPU execution delays. Parallelizing `cross_val_score(..., n_jobs=-1)` while enforcing `n_jobs=1` on internal estimators prevents CPU thread thrashing and dramatically speeds up validation. Additionally, generating synthetic feature vectors row-by-row in Python loops introduces significant latency that vectorizing with 2D NumPy array operations completely eliminates.
**Action:** When working with scikit-learn cross-validation, set `n_jobs=-1` on `cross_val_score` and `n_jobs=1` on underlying estimators, and always vectorize sample generation using NumPy array indexing.
