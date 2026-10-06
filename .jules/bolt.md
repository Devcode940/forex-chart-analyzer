# Bolt Performance Journal

## 2026-10-06 - Vectorize Synthetic Data Generation and Tune ML Ensemble Folds

**Learning:** Python `for` loops generating synthetic sample features element-by-element introduce massive execution latency (~515ms for 2000 samples). Vectorizing generation across 2D NumPy matrices reduces latency to ~13ms (~37x faster). Furthermore, setting `n_jobs=-1` on `cross_val_score` while keeping internal base estimators at `n_jobs=1` prevents CPU thread oversubscription while parallelizing cross-validation fold evaluation.

**Action:** Always vectorize synthetic dataset construction in NumPy and pass `n_jobs=1` to base estimators when parallelizing top-level cross-validation folds.
