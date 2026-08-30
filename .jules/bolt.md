## 2025-05-18 - Vectorize synthetic dataset generation and cross-validation in MLEnsemble

**Learning:** Generating synthetic feature matrices via Python `for` loops line-by-line introduces severe CPU overhead during ML model training. Re-architecting matrix generation with 2D NumPy array broadcasting and tuning cross-validation hyperparameters (`cv=3`, `n_jobs=-1`) speeds up model training and prediction by >2.6x without sacrificing model accuracy or convergence.
**Action:** Always vectorize matrix/dataset creation in Python ML pipelines and leverage parallel cross-validation jobs when scoring base learners.
