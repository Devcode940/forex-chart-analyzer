## 2025-05-18 - Avoid thread oversubscription when parallelizing cross-validation
**Learning:** Setting `n_jobs=-1` on both `cross_val_score` and the underlying estimators (e.g. `RandomForestClassifier`) causes thread oversubscription and severe CPU contention on multi-core systems, slowing execution down.
**Action:** When parallelizing `cross_val_score(..., n_jobs=-1)`, explicitly set `n_jobs=1` on the base estimators passed into cross-validation.
