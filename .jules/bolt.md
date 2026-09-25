## 2025-05-18 - Avoid Thread Oversubscription in Parallel Cross-Validation

**Learning:** When using `cross_val_score(..., n_jobs=-1)` in scikit-learn, underlying estimators like `RandomForestClassifier` must explicitly set `n_jobs=1`. If both `cross_val_score` and `RandomForestClassifier` attempt to use all CPU cores concurrently (`n_jobs=-1`), CPU thread context switching overhead dominates runtime.
**Action:** When parallelizing cross-validation across folds, always pass `n_jobs=1` to internal ensemble estimators.
