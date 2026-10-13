## 2025-03-20 - Vectorizing Synthetic Training Data & CV Parallelization in ML Ensemble

**Learning:** Synthetic training data generation and heuristic augmentation in `analyzers/ml_ensemble.py` were bottlenecked by Python element-by-element loops (~500ms). Furthermore, `cross_val_score` defaults to single-thread sequentially while `RandomForestClassifier` defaulted to `n_jobs=-1` inside CV workers, causing severe CPU thread oversubscription.
**Action:** Vectorized 2D NumPy array operations for synthetic generation and heuristic noise addition (reducing generation time to ~25ms). Explicitly set `n_jobs=-1` on `cross_val_score` and `n_jobs=1` on base `RandomForestClassifier` estimators, cutting full `train_and_predict` latency from ~40s to ~13.6s (~3x speedup).
