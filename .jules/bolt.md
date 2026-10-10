# Bolt's Journal

## 2025-05-15 - Vectorizing synthetic generation and optimizing ensemble CV in MLEnsemble
**Learning:** Synthetic training data generation in `MLEnsemble` had heavy row-by-row iteration in Python, and cross-validation was slow due to sequential execution with excessive estimators (200 RF, 150 GB). Vectorizing generation with NumPy arrays and parallelizing 5-fold CV with tuned estimators reduces MLEnsemble latency significantly (~38s down to ~12s). Avoid setting `n_jobs=-1` concurrently on both `cross_val_score` and internal estimators like `RandomForestClassifier` to prevent thread oversubscription.
**Action:** Vectorize array generation in NumPy and pass `n_jobs=1` to base models when `cross_val_score` uses `n_jobs=-1`.
