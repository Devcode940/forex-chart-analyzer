# Bolt Performance Journal

## 2025-05-20 - Vectorize Synthetic Data Generation and CV Parallelization in ML Ensemble
**Learning:** Synthetic training data generation in `MLEnsemble` using Python `for` loops took ~500ms and sequential cross-validation took ~31s. Vectorizing sample matrix creation with NumPy array operations dropped generation latency to ~8ms. Setting `n_jobs=-1` on `cross_val_score` while setting `n_jobs=1` on internal estimator instances parallelizes cross-validation folds cleanly without CPU thread oversubscription contention.
**Action:** Vectorize sample generation for synthetic ML datasets using 2D NumPy operations, and set `n_jobs=1` on estimators passed to `cross_val_score(..., n_jobs=-1)`.
