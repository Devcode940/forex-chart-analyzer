## 2025-05-20 - Cross-Validation Parallelization & Vectorized Synthetic Data in MLEnsemble

**Learning:** Running `cross_val_score` sequentially with unparallelized `RandomForestClassifier` and `GradientBoostingClassifier` models on synthetic dataset generation with Python `for` loops caused `MLEnsemble` to take over 39s (~78% of overall pipeline runtime). Setting `n_jobs=-1` on `cross_val_score` while passing `n_jobs=1` to the inner estimators prevents thread oversubscription while enabling parallel fold execution. Vectorizing synthetic feature matrix generation with 2D NumPy operations reduces data generation latency from ~500ms to ~7ms.

**Action:** When parallelizing cross-validation on multi-threaded scikit-learn models, always set `n_jobs=1` on the underlying estimators and `n_jobs=-1` on `cross_val_score` to avoid thread contention while maximizing multi-core throughput.
