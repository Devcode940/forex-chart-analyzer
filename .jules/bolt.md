## 2026-03-31 - Thread Oversubscription in Parallel Cross-Validation & Synthetic Vectorization

**Learning:** When vectorizing synthetic training data generation, using 2D NumPy array operations reduced generation time from ~500ms to ~8ms (~88x speedup). In scikit-learn cross-validation, using `cross_val_score(..., n_jobs=-1)` while leaving underlying estimators like `RandomForestClassifier` with default multi-threading leads to severe thread oversubscription penalties. Passing `n_jobs=1` to the underlying estimator when parallelizing `cross_val_score` prevents CPU thread contention and accelerates execution.

**Action:** Always set `n_jobs=1` on base estimators whenever `cross_val_score` or `GridSearchCV` handles outer parallelism with `n_jobs=-1`.
