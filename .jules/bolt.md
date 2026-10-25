## 2026-07-24 - ML Ensemble Training Vectorization & CV Parallelization

**Learning:** `MLEnsemble` model training and cross-validation was heavily bottlenecked by iterative sample generation in Python loops and sequential `cross_val_score` execution. Vectorizing synthetic data generation with 2D NumPy operations and passing `n_jobs=-1` to `cross_val_score` while setting `n_jobs=1` on base estimators (`RandomForestClassifier`, `GradientBoostingClassifier`) avoids thread oversubscription while reducing execution time from ~38.7s to ~5.5s (~85% speedup).

**Action:** Always check ML models for iterative synthetic data creation and parallelize `cross_val_score` folds while keeping base estimators at `n_jobs=1` to prevent CPU oversubscription.
