## 2026-07-24 - Vectorization and Tree Parameter Tuning in MLEnsemble

**Learning:** `GradientBoostingClassifier` defaults to `max_features=None`, evaluating all features at every split. Combining `max_features='sqrt'` with vectorized 2D NumPy synthetic data generation and setting `n_jobs=1` on base estimators while setting `n_jobs=-1` on `cross_val_score` prevents CPU thread oversubscription while dramatically accelerating cross-validation folds in parallel.
**Action:** When working with scikit-learn ensemble models in Python, vectorize synthetic matrix creation in NumPy, set `max_features='sqrt'` on tree models, and pass `n_jobs=1` to base models when parallelizing cross-validation via `cross_val_score(..., n_jobs=-1)`.
