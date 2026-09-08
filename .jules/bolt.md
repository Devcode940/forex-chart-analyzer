## 2025-05-18 - Vectorizing Synthetic Data Generation & Parallel Cross-Validation in MLEnsemble

**Learning:** `MLEnsemble.train_and_predict` spent ~39 seconds total, with 475ms spent generating 2,000 synthetic samples in a row-by-row Python `for` loop and over 30s in sequential cross-validation. Vectorizing the synthetic matrix construction using 2D NumPy operations reduced synthetic generation time from 475ms to 8ms. Combining tuned hyperparameters (RF 100, GB 80) with multi-threaded cross-validation (`cv=3`, `n_jobs=-1`) reduced total execution time by ~55% (~39s -> ~18s).

**Action:** Whenever generating synthetic training datasets or running cross-validation in ML ensemble models, avoid element-wise Python loops in favor of 2D NumPy array broadcasting and pass `n_jobs=-1` to `cross_val_score`.
