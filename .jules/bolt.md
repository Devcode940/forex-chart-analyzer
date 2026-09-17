## 2026-03-30 - Parallel Cross-Validation & Thread Oversubscription
**Learning:** Setting `n_jobs=-1` simultaneously on `cross_val_score` and underlying estimator models (such as `RandomForestClassifier`) causes CPU thread oversubscription, slowing execution due to context switching context overhead. Passing `n_jobs=1` to the base estimator when running parallel cross-validation (`n_jobs=-1` on `cross_val_score`) yields optimal core utilization.
**Action:** Always set `n_jobs=1` on base estimators passed into parallel cross-validation routines.
